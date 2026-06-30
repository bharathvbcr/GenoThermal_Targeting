"""
Geno-Thermal Targeting — Model Context Protocol (MCP) server.

Exposes the project's discovery -> fold/dock -> ligand-design -> verify loop as
callable tools so the Claude Science coordinating agent (or any MCP client) can run
the whole pipeline in plain language. Every tool wraps EXISTING project code:

    discover_target        -> alphagenome_utils.AlphaGenomeClient        (Phase 1)
    design_ligands         -> boltz_designer.py  (Boltz-2 fold + affinity, Phase 2)
    design_thermal_switch  -> hard_mode.thermo_fold.ThermoSwitchOptimizer (Phase 5)
    verify_with_bionemo    -> NVIDIA BioNeMo Boltz-2 NIM, independent cross-check
    run_full_pipeline      -> run_pipeline.py   (the full 12-phase orchestration)
    design_promoter_flash  -> hard_mode/evolver.py (GA promoter design, fitness fanned
                               out on the RunPod Flash fleet when use_flash=True)
    screen_and_verify      -> chains discover_target -> design_ligands ->
                               (design_promoter_flash) -> verify_with_bionemo into one
                               call; the headline demo artifact
    kill_flash_endpoints   -> `flash undeploy --all --force` (gated by confirm=True),
                               the destructive half of the reliability demo

Design rules (hackathon-grade robustness):
  * No tool raises to the client — every failure returns a structured {"ok": False,...}.
  * Heavy/optional deps (GPU folding, NVIDIA NIM) degrade to a clearly-flagged
    second opinion instead of crashing a live demo.
  * Pure stdlib + the project's own modules + `mcp`. No new heavy dependency.

Run standalone for a smoke test:
    python mcp_geno_thermal.py --selftest

Run as an MCP stdio server (how Claude Science launches it):
    python mcp_geno_thermal.py
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from typing import Any

try:
    from mcp.server.fastmcp import Context, FastMCP
except ModuleNotFoundError as e:
    raise SystemExit(
        f"{e}\n\n"
        "mcp_geno_thermal.py must run under the project's .venv-flash interpreter — the only "
        "environment with `mcp`/`numpy`/`pandas`/`alphagenome` installed together. Run it as:\n"
        "    .venv-flash/bin/python mcp_geno_thermal.py [--selftest]\n"
        f"(you ran it under: {sys.executable})"
    ) from e

# Resolve the project root so tools work regardless of the client's CWD.
ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable  # same interpreter -> same venv/deps as this server


def _load_dotenv(path: str | None = None) -> None:
    """Minimal stdlib .env loader: Claude Science launches us via .mcp.json, which does
    NOT source the shell or .env. Read KEY=VALUE lines so ALPHAGENOME_API_KEY /
    NVIDIA_API_KEY / RUNPOD_API_KEY reach the tools. Never overrides an already-set var."""
    path = path or os.path.join(ROOT, ".env")
    if not os.path.exists(path):
        return
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key, val = key.strip(), val.strip().strip('"').strip("'")
                if key and val and key not in os.environ:
                    os.environ[key] = val
    except OSError:
        pass


_load_dotenv()

# .mcp.json sets GENOTHERMAL_LOG_LEVEL, but FastMCP only listens to its own log_level
# kwarg / FASTMCP_LOG_LEVEL env var — a project module's logging.basicConfig() call is a
# no-op once FastMCP's own RichHandler has already attached to the root logger. Passing
# the level through here is what actually quiets the server.
_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
_log_level = os.environ.get("GENOTHERMAL_LOG_LEVEL", "INFO").upper()
if _log_level not in _LOG_LEVELS:
    _log_level = "INFO"

mcp = FastMCP("geno-thermal-targeting", log_level=_log_level)

# Real GRCh38 reference windows for the default discovery example. The EGFR locus
# (chr7, oncogene TSS — amplified/overexpressed across many tumors) carries genuine
# CAGE/TSS signal; the gene-desert control (chr2 intergenic) is transcriptionally
# silent. So the *real* AlphaGenome API produces a true SUPER_ENHANCER delta here,
# rather than noise on a short toy sequence padded with N. Provenance is in each
# FASTA header (coordinates + Ensembl GRCh38 source).
DEFAULT_DISCOVERY_FASTA = "data/sample_data/egfr_super_enhancer.fasta"
DEFAULT_CONTROL_FASTA = "data/sample_data/gene_desert_control.fasta"

# EGFR ectodomain fragment — same default target the project's designers use.
DEFAULT_TARGET = (
    "LEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVLGNLEITYVQRNYDLSFLKTIQEVAGYVLIALN"
    "TVERIPLENLQIIRGNMYYENSYALAVLSNYDANKTGLKELPMRNLQEILHGAVRFSNNPALCNVESIQWR"
    "DIVSSDFLSNMSMDFQNHLGSCQKCDPSCPNGSCWGAGEENCQKLTKIICAQQCSGRCRGKSPSDCCHNQC"
    "AAGCTGPRESDCLVCRKFRDEATCKDTCPPLMLYNPTTYQMDVNPEGKYSFGATCVKKCPRNYVVTDHGSCV"
    "RACGADSYEMEEDGVRKC"
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
async def _run_cli_async(
    args: list[str],
    ctx: Context | None = None,
    env: dict[str, str] | None = None,
    timeout: int = 1800,
    live: bool = True,
) -> dict[str, Any]:
    """Run a project CLI phase as a subprocess, STREAMING its stdout/stderr back to the
    MCP client as progress notifications as they're produced — not just a log tail after
    the whole call completes. This is what lets the live "[phase] in-flight: N" \\r-ticker
    (flash_metrics.FanoutMetrics, gated by GENOTHERMAL_LIVE) reach a judge driving the
    demo through Claude, instead of only the bare terminal a blocking subprocess.run with
    capture_output=True would swallow it into. Never raises."""
    cmd = [PY, *args]
    run_env = dict(env) if env is not None else dict(os.environ)
    if live:
        run_env["GENOTHERMAL_LIVE"] = "1"
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=ROOT, env=run_env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "command": " ".join(args)}

    lines: list[str] = []
    buf = b""
    step = 0

    async def _pump() -> None:
        nonlocal buf, step
        assert proc.stdout is not None
        while True:
            chunk = await proc.stdout.read(256)
            if not chunk:
                break
            buf += chunk
            # The live ticker overwrites its line with \r, not \n — split on either so
            # each tick is forwarded as its own progress update instead of being buffered
            # until the next real newline.
            while True:
                m = re.search(rb"[\r\n]", buf)
                if not m:
                    break
                raw, buf = buf[: m.start()], buf[m.end():]
                text = raw.decode(errors="replace").strip()
                if not text:
                    continue
                lines.append(text)
                step += 1
                if ctx is not None:
                    try:
                        await ctx.report_progress(step, None, text[:300])
                    except Exception:
                        pass

    try:
        await asyncio.wait_for(_pump(), timeout=timeout)
        if buf.strip():
            lines.append(buf.decode(errors="replace").strip())
        returncode = await asyncio.wait_for(proc.wait(), timeout=30)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return {"ok": False, "error": f"timeout after {timeout}s", "command": " ".join(args)}
    except Exception as e:
        proc.kill()
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "command": " ".join(args)}

    return {
        "ok": returncode == 0,
        "exit_code": returncode,
        "command": " ".join(args),
        "log_tail": "\n".join(lines[-40:]),
    }


def _read_csv_records(path: str) -> list[dict[str, Any]]:
    import csv

    abspath = path if os.path.isabs(path) else os.path.join(ROOT, path)
    if not os.path.exists(abspath):
        return []
    with open(abspath, newline="") as f:
        return list(csv.DictReader(f))


def _sequence_complexity(seq: str) -> dict[str, Any]:
    """Cheap sequence-complexity metrics used to reject non-specific / negative-control
    binders (homopolymers like poly-Ala, short tandem repeats). For such low-complexity
    sequences a structure predictor's confidence is NOT evidence of specific binding, so
    high agreement between two folds must not be read as corroboration."""
    seq = (seq or "").strip().upper()
    n = len(seq)
    if n == 0:
        return {"length": 0, "distinct_fraction": 0.0, "max_residue_fraction": 1.0,
                "is_low_complexity": True}
    distinct_fraction = round(len(set(seq)) / n, 3)
    max_residue_fraction = round(max(seq.count(a) for a in set(seq)) / n, 3)
    is_low_complexity = (distinct_fraction < 0.4) or (max_residue_fraction > 0.6)
    return {
        "length": n,
        "distinct_fraction": distinct_fraction,
        "max_residue_fraction": max_residue_fraction,
        "is_low_complexity": bool(is_low_complexity),
    }


# --------------------------------------------------------------------------- #
# Phase 1 — Genomic discovery (AlphaGenome)
# --------------------------------------------------------------------------- #
@mcp.tool()
def discover_target(
    target_gene: str = "EGFR",
    mutated_seq: str | None = None,
    input_fasta: str | None = None,
    control_fasta: str | None = None,
) -> dict[str, Any]:
    """Score a candidate genomic target with AlphaGenome (expression + epigenetic state).

    Classifies the locus as SUPER_ENHANCER vs NORMAL and returns a confidence — the
    first gate of the pipeline. The default candidate/control are REAL GRCh38 windows
    (EGFR oncogene promoter vs an intergenic gene desert), so the live AlphaGenome API
    yields a genuine SUPER_ENHANCER delta rather than noise. Falls back to the project's
    deterministic local model when no ALPHAGENOME_API_KEY is set, so it always returns a
    result. The returned `mode` states which produced the numbers, and `provenance`
    records which loci were compared.

    Args:
        target_gene: Gene symbol to score (default EGFR).
        mutated_seq: DNA sequence to evaluate. If omitted, read from input_fasta
            (default: the real EGFR promoter window).
        input_fasta: Path to a FASTA file (relative to the project root) for the
            candidate locus when mutated_seq is not given.
        control_fasta: Path to a FASTA file for the baseline/normal locus
            (default: the real gene-desert control).
    """
    try:
        from alphagenome_utils import AlphaGenomeClient

        client = AlphaGenomeClient()

        def _load(path: str) -> str:
            return client.parse_fasta(path if os.path.isabs(path) else os.path.join(ROOT, path))

        # Baseline ("normal") locus: a transcriptionally silent gene desert by default.
        control_src = control_fasta or DEFAULT_CONTROL_FASTA
        normal_seq = _load(control_src)

        # Candidate ("mutated"/tumor) locus.
        if mutated_seq is None:
            candidate_src = input_fasta or DEFAULT_DISCOVERY_FASTA
            mutated_seq = _load(candidate_src)
        else:
            candidate_src = "inline:mutated_seq"

        result = client.get_expression_score(
            gene_id=target_gene, normal_seq=normal_seq, mutated_seq=mutated_seq
        )
        result["ok"] = True
        result["mode"] = getattr(client, "_mode", "UNKNOWN")
        result["provenance"] = {"candidate_locus": candidate_src, "control_locus": control_src}
        return result
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


# --------------------------------------------------------------------------- #
# Phase 2 — Ligand engineering (Boltz-2 fold + affinity)
# --------------------------------------------------------------------------- #
@mcp.tool()
async def design_ligands(
    candidates_file: str = "data/sample_data/candidates.csv",
    target_seq: str | None = None,
    output_csv: str = "outputs/reports/candidate_library_v2.csv",
    use_flash: bool = True,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Fold/dock peptide & small-molecule candidates against the target (Boltz-2).

    Drives boltz_designer.py, which fans candidates out to the RunPod Flash GPU fleet
    (use_flash=True, the default — this is the tool's whole value-add for today's GPU
    fan-out judging criterion) or folds locally, returning interface confidence
    (ipTM*100), a binding class, and predicted affinity. Reads back the produced library
    and streams the run's live output back as progress notifications as it happens.

    Args:
        candidates_file: CSV with 'name' and 'seq'/'smiles' columns.
        target_seq: Receptor amino-acid sequence; defaults to the EGFR ectodomain.
        output_csv: Where boltz_designer writes the ranked library.
        use_flash: Fan out on the RunPod Flash GPU fleet (default True). False forces an
            in-process local fold, which silently skips (see the returned `stale` flag)
            when no local boltz/torch toolchain is installed.
    """
    args = ["boltz_designer.py", "--candidates_file", candidates_file, "--output_csv", output_csv]
    if target_seq:
        args += ["--target_seq", target_seq]
    if not use_flash:
        args += ["--local"]
    env = dict(os.environ)
    if use_flash:
        env["GENOTHERMAL_FLASH"] = "1"
    else:
        env.pop("GENOTHERMAL_FLASH", None)

    # boltz_designer.py deliberately LEAVES output_csv untouched (keeping the committed
    # library) when folding is skipped (no local boltz/torch toolchain) or all jobs fail —
    # so a successful subprocess exit does not by itself mean fresh fold results were
    # written. Compare mtimes around the run to tell a live fold from a stale read-back.
    abspath = output_csv if os.path.isabs(output_csv) else os.path.join(ROOT, output_csv)
    mtime_before = os.path.getmtime(abspath) if os.path.exists(abspath) else None
    metrics_path = os.path.join(ROOT, "flash_metrics.json")
    metrics_mtime_before = os.path.getmtime(metrics_path) if os.path.exists(metrics_path) else None

    run = await _run_cli_async(args, ctx=ctx, env=env)

    mtime_after = os.path.getmtime(abspath) if os.path.exists(abspath) else None
    if mtime_after is None:
        freshness = "missing"
    elif mtime_before is None or mtime_after > mtime_before:
        freshness = "fresh"
    else:
        freshness = "stale"
    records = _read_csv_records(output_csv)
    run["library_size"] = len(records)
    run["candidates"] = records[:25]
    run["data_freshness"] = freshness
    run["stale"] = freshness == "stale"
    if freshness == "stale":
        run["note"] = (
            "No new fold results were written this call (no local boltz/torch toolchain and "
            "use_flash=False, or all fold jobs failed) — returning the PREVIOUSLY COMMITTED "
            "library, not a live fold. Set use_flash=True, or run with a local boltz+torch "
            "toolchain, for fresh results."
        )

    # boltz_designer.py records a 'fold-boltz2' FanoutMetrics phase when it fans candidates
    # out to Flash (boltz_designer.py:_fold_remote); surface it the same way
    # design_promoter_flash surfaces 'ga-fitness', so the Boltz-2 screen isn't the one
    # Flash-touching tool with no cost/autoscale numbers reaching the judge.
    flash_engaged = bool(use_flash and _flash_metrics_engaged(metrics_mtime_before, "fold-boltz2"))
    run["flash_autoscaling"] = _read_flash_metrics("fold-boltz2") if flash_engaged else None
    run["flash_scaling_chart"] = await asyncio.to_thread(_render_flash_dashboard) if flash_engaged else None
    return run


# --------------------------------------------------------------------------- #
# Phase 5 — Thermo-switch protein design (the "thermal targeting" core)
# --------------------------------------------------------------------------- #
@mcp.tool()
def design_thermal_switch(
    scaffold: str = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ",
    generations: int = 30,
) -> dict[str, Any]:
    """Evolve a protein that folds at 37°C but unfolds at 43°C (a thermal switch).

    Runs the project's ThermoSwitchOptimizer GA and reports the best sequence, its
    melting temperature, and the pLDDT differential between body and hyperthermia
    temperature — the mechanism that makes the therapeutic tumor-selective.

    Args:
        scaffold: Starting amino-acid sequence to optimise.
        generations: GA generations (lower = faster demo).
    """
    try:
        sys.path.insert(0, os.path.join(ROOT, "hard_mode"))
        from hard_mode.thermo_fold import ThermoSwitchOptimizer, ProteinPhysicsOracle

        opt = ThermoSwitchOptimizer(scaffold)
        opt.generations = int(generations)
        best_seq, tm = opt.run()
        oracle = ProteinPhysicsOracle()
        plddt_37 = oracle.predict_plddt(best_seq, 37.0)
        plddt_43 = oracle.predict_plddt(best_seq, 43.0)
        return {
            "ok": True,
            "best_sequence": best_seq,
            "melting_temp_c": round(float(tm), 2),
            "plddt_37c": round(float(plddt_37), 1),
            "plddt_43c": round(float(plddt_43), 1),
            "switch_delta": round(float(plddt_37 - plddt_43), 1),
            "interpretation": "Higher switch_delta = sharper folded->unfolded transition between body and hyperthermia temperature.",
        }
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


# --------------------------------------------------------------------------- #
# Independent verifier — NVIDIA BioNeMo Boltz-2 NIM
# --------------------------------------------------------------------------- #
@mcp.tool()
def verify_with_bionemo(
    target_seq: str,
    binder_seq: str,
    project_plddt: float | None = None,
) -> dict[str, Any]:
    """Independently re-fold a target+binder complex on NVIDIA's BioNeMo Boltz-2 NIM.

    This is the *adversarial verifier* step: a second, independent structure prediction
    that cross-checks the project's own fold. When the project and BioNeMo confidences
    agree the binding call is corroborated; when they diverge the candidate is flagged
    for review. Requires NVIDIA_API_KEY (from build.nvidia.com); without it, returns a
    clearly-labelled local second opinion so the demo never blocks.

    Args:
        target_seq: Receptor amino-acid sequence.
        binder_seq: Designed binder/peptide amino-acid sequence.
        project_plddt: The project's own interface confidence (ipTM*100) to compare against.
    """
    api_key = os.environ.get("NVIDIA_API_KEY") or os.environ.get("NGC_API_KEY")
    bionemo_conf: float | None = None
    source = "bionemo_boltz2_nim"
    detail = ""

    if api_key:
        url = "https://health.api.nvidia.com/v1/biology/mit/boltz2/predict"
        payload = {
            "polymers": [
                {"id": "A", "molecule_type": "protein", "sequence": target_seq},
                {"id": "B", "molecule_type": "protein", "sequence": binder_seq},
            ],
            "recycling_steps": 3,
            "sampling_steps": 50,
            "diffusion_samples": 1,
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = json.loads(resp.read().decode())
            # NIM returns per-structure confidences; pull the best available scalar.
            conf = (
                data.get("confidence_scores")
                or data.get("confidences")
                or data.get("plddt")
            )
            if isinstance(conf, list) and conf:
                conf = conf[0]
            if isinstance(conf, dict):
                conf = conf.get("complex_plddt") or conf.get("iptm") or conf.get("plddt")
            if conf is not None:
                bionemo_conf = float(conf) * (100.0 if float(conf) <= 1.0 else 1.0)
            detail = "BioNeMo Boltz-2 NIM prediction succeeded."
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as e:
            source = "local_second_opinion"
            detail = f"BioNeMo NIM unreachable ({type(e).__name__}: {e}); used local second opinion."

    if bionemo_conf is None:
        # Independent local re-score: a deterministic biophysical heuristic distinct from
        # the project's folding path, so it functions as a genuine second opinion.
        source = source if source == "local_second_opinion" else "local_second_opinion_no_key"
        hydroph = sum(binder_seq.count(a) for a in "AILMFWVY") / max(len(binder_seq), 1)
        charge = sum(binder_seq.count(a) for a in "KR") - sum(binder_seq.count(a) for a in "DE")
        length_term = min(len(binder_seq) / 20.0, 1.0)
        # Sequence complexity: a real interface verifier distrusts low-diversity binders
        # (homopolymers like poly-Ala — common negative controls). Penalise them so the
        # offline fallback doesn't "validate" a control the way a naive hydrophobicity term would.
        complexity = len(set(binder_seq)) / max(len(binder_seq), 1)
        complexity_penalty = max(0.0, (0.5 - complexity)) * 80.0
        bionemo_conf = round(
            max(0.0, min(100.0,
                40.0 + 45.0 * hydroph + 5.0 * length_term - abs(charge) * 2.0 - complexity_penalty)), 1
        )
        if not detail:
            detail = "No NVIDIA_API_KEY set; used deterministic local second opinion (set the key to call the real BioNeMo Boltz-2 NIM)."

    verdict = "UNCOMPARED"
    agreement = None
    if project_plddt is not None:
        agreement = round(100.0 - abs(project_plddt - bionemo_conf), 1)
        if abs(project_plddt - bionemo_conf) <= 12.0:
            verdict = "CORROBORATED"
        elif abs(project_plddt - bionemo_conf) <= 25.0:
            verdict = "WEAK_AGREEMENT"
        else:
            verdict = "DIVERGENT_FLAG_FOR_REVIEW"

    # Specificity guard — applies to BOTH the real BioNeMo NIM and the local second
    # opinion. Structure-prediction confidence is unreliable and non-specific for
    # low-complexity binders (homopolymers / short repeats — classic negative controls).
    # Reject them regardless of how well the two folds agree, so the verifier can never
    # "corroborate" a poly-Ala control just because both models are confident on it.
    seq_complexity = _sequence_complexity(binder_seq)
    if seq_complexity["is_low_complexity"]:
        verdict = "REJECTED_LOW_COMPLEXITY"
        detail += (
            f" Binder rejected as low-complexity (distinct_residue_fraction="
            f"{seq_complexity['distinct_fraction']}, max_residue_fraction="
            f"{seq_complexity['max_residue_fraction']}): fold-confidence agreement is not "
            f"evidence of specific binding for such sequences."
        )

    return {
        "ok": True,
        "verifier_source": source,
        "bionemo_confidence": bionemo_conf,
        "project_plddt": project_plddt,
        "agreement_pct": agreement,
        "verdict": verdict,
        "low_complexity": seq_complexity["is_low_complexity"],
        "complexity": seq_complexity,
        "detail": detail,
    }


# --------------------------------------------------------------------------- #
# Full pipeline
# --------------------------------------------------------------------------- #
@mcp.tool()
async def run_full_pipeline(
    smoke: bool = True, flash: bool = True, ctx: Context | None = None
) -> dict[str, Any]:
    """Run the entire 12-phase Geno-Thermal pipeline via run_pipeline.py.

    Streams the run's live output back as progress notifications as it happens instead
    of blocking silently until the whole (up to 1-hour) run finishes.

    Args:
        smoke: Tiny GA/PPO/folding workload for a <1-min end-to-end demo.
        flash: Fan compute out on the RunPod Flash serverless GPU fleet (default True —
            this is the run's whole point for today's GPU fan-out judging criterion).
    """
    args = ["run_pipeline.py", "--keep-going"]
    if smoke:
        args.append("--smoke")
    if flash:
        args.append("--flash")
    metrics_path = os.path.join(ROOT, "flash_metrics.json")
    mtime_before = os.path.getmtime(metrics_path) if os.path.exists(metrics_path) else None
    run = await _run_cli_async(args, ctx=ctx, timeout=3600)
    if flash and _flash_metrics_engaged(mtime_before):
        run["flash_scaling_chart"] = await asyncio.to_thread(_render_flash_dashboard)
    return run


# --------------------------------------------------------------------------- #
# GA promoter design — fans the fitness scoring out on the RunPod Flash fleet
# --------------------------------------------------------------------------- #
def _read_flash_metrics(phase: str = "ga-fitness") -> dict[str, Any] | None:
    """Surface the autoscaling story from flash_metrics.json (peak workers, cost, speedup)."""
    path = os.path.join(ROOT, "flash_metrics.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            phases = json.load(f)
    except (OSError, ValueError):
        return None
    # flash_metrics.json is append-only (one entry per phase per run, newest last) — take
    # the LATEST matching entry, not the first, so a rehearsal run's numbers can't surface
    # during the live demo.
    matching = [p for p in phases if p.get("phase") == phase]
    rec = matching[-1] if matching else (phases[-1] if phases else None)
    if not rec:
        return None
    return {
        "phase": rec.get("phase"),
        "resource": rec.get("resource"),
        "jobs_ok": rec.get("n_ok"),
        "jobs_total": rec.get("n_jobs"),
        "peak_workers_inflight": rec.get("peak_inflight"),
        "autoscale": f"0 -> {rec.get('peak_inflight')} -> 0",
        "wall_s": rec.get("wall_s"),
        "compute_seconds": rec.get("gpu_seconds"),
        "speedup_vs_serial": rec.get("speedup_vs_serial"),
        "est_cost_usd": rec.get("est_cost_usd"),
    }


def _flash_metrics_engaged(mtime_before: float | None, phase: str | None = None) -> bool:
    """True if flash_metrics.json shows Flash genuinely produced at least one real result
    since mtime_before — not just that a Flash dispatch was ATTEMPTED. A cold/undeployed
    endpoint can time out on every job (n_ok=0), in which case the caller (evolver.py /
    boltz_designer.py) falls back to the local threadpool for ALL scoring even though a
    metrics entry still got appended — reporting mode='FLASH' in that case is exactly the
    silent-fallback dishonesty this check exists to prevent (verified live against a real,
    momentarily-cold RunPod endpoint: 0/8 jobs succeeded, metrics file still updated).
    Pass `phase` to additionally require that phase's `jobs_ok` > 0; omit it to only check
    that the file moved at all (e.g. run_full_pipeline, which doesn't report a single
    phase's mode and just wants to know whether to bother rendering the chart)."""
    path = os.path.join(ROOT, "flash_metrics.json")
    mtime_after = os.path.getmtime(path) if os.path.exists(path) else None
    if mtime_after is None or (mtime_before is not None and mtime_after <= mtime_before):
        return False
    if phase is None:
        return True
    rec = _read_flash_metrics(phase)
    return bool(rec and (rec.get("jobs_ok") or 0) > 0)


def _render_flash_dashboard() -> str | None:
    """Best-effort render of outputs/figures/flash_scaling.png from flash_metrics.json —
    "literally the slides" per FLASH_HACKATHON_NOTES.md. Returns the path on success so a
    judge driving the demo through Claude can see the chart, not just scalar metrics.
    Blocking (matplotlib); call via asyncio.to_thread from async tools."""
    try:
        proc = subprocess.run(
            [PY, "flash_dashboard.py"], cwd=ROOT, capture_output=True, text=True, timeout=60,
        )
    except Exception:
        return None
    out_path = os.path.join(ROOT, "outputs/figures/flash_scaling.png")
    return out_path if proc.returncode == 0 and os.path.exists(out_path) else None


def _read_ga_best() -> dict[str, Any] | None:
    """Best promoter row from the GA evolution log (last generation = most evolved)."""
    rows = _read_csv_records("outputs/reports/evolution_log.csv")
    if not rows:
        return None
    best = rows[-1]
    return {
        "generation": best.get("Generation"),
        "best_fitness": best.get("Best_Fitness"),
        "tumor_score": best.get("Tumor_Score"),
        "normal_score": best.get("Normal_Score"),
        "heat_score": best.get("Heat_Score"),
    }


@mcp.tool()
async def design_promoter_flash(
    use_flash: bool = True, smoke: bool = True, ctx: Context | None = None
) -> dict[str, Any]:
    """Evolve a hyperthermia-gated tumor promoter, fanning fitness scoring out on RunPod Flash.

    Runs the genetic algorithm (hard_mode/evolver.py). With use_flash=True (the default)
    the per-individual fitness scoring is dispatched to the deployed `genothermal-fitness`
    Flash endpoint, whose worker fleet autoscales 0 -> N -> 0. Returns the best evolved
    promoter's scores AND the live autoscaling metrics (peak concurrent workers, speedup,
    estimated cost) — so the tool call itself demonstrates serverless GPU/CPU fan-out, not
    just a local loop. Streams the live "[ga-fitness] in-flight: N" ticker back as
    progress notifications as it happens, instead of only returning it after the fact.

    Args:
        use_flash: Dispatch fitness scoring to the RunPod Flash fleet (needs RUNPOD_API_KEY
            and the genothermal-fitness endpoint deployed). Default True — this is the
            tool's whole reason to exist; pass False to force the local threadpool.
        smoke: Small/fast GA (pop=8, 3 gens) suitable for an interactive call.
    """
    env = dict(os.environ)
    if use_flash:
        env["GENOTHERMAL_FLASH"] = "1"
    else:
        env.pop("GENOTHERMAL_FLASH", None)
    if smoke:
        env["GENOTHERMAL_SMOKE"] = "1"

    # evolver.py only writes a "ga-fitness" flash_metrics.json entry when its
    # GeneticOptimizer actually dispatched at least one generation to the Flash fleet
    # (see hard_mode/evolver.py:_score_via_flash / FanoutMetrics.save). A mtime bump on
    # this file is therefore ground truth that Flash genuinely engaged this call — unlike
    # echoing the requested use_flash flag, which can't tell a real run from a silent
    # fallback (no RUNPOD_API_KEY, endpoint not deployed, flash_fitness import failure).
    metrics_path = os.path.join(ROOT, "flash_metrics.json")
    mtime_before = os.path.getmtime(metrics_path) if os.path.exists(metrics_path) else None

    run = await _run_cli_async(["hard_mode/evolver.py"], ctx=ctx, env=env, timeout=900)

    flash_engaged = bool(use_flash and _flash_metrics_engaged(mtime_before, "ga-fitness"))
    metrics = _read_flash_metrics("ga-fitness") if flash_engaged else None
    chart = await asyncio.to_thread(_render_flash_dashboard) if flash_engaged else None
    return {
        "ok": run.get("ok", False),
        "mode": "FLASH" if flash_engaged else "LOCAL",
        "requested_flash": use_flash,
        "best_promoter": _read_ga_best(),
        "flash_autoscaling": metrics,
        "flash_scaling_chart": chart,
        "log_tail": run.get("log_tail") or run.get("error"),
        "note": (
            "Fitness scoring ran on the RunPod Flash fleet (autoscaled 0->N->0)."
            if flash_engaged else
            "Ran on the local threadpool"
            + (
                " (use_flash=True was requested but Flash fan-out did not engage this run — "
                "check RUNPOD_API_KEY, the genothermal-fitness endpoint deployment, and the "
                "flash_fitness import)."
                if use_flash else "."
            )
        ),
    }


@mcp.tool()
async def kill_flash_endpoints(confirm: bool = False) -> dict[str, Any]:
    """Undeploy ALL RunPod Flash endpoints on this account (`flash undeploy --all --force`).

    This is the destructive half of the hackathon's reliability demo (FLASH_HACKATHON_NOTES.md:
    kill an endpoint mid-run and watch the Flash-touching tools auto-fall-back to the local
    threadpool and finish anyway). Affects EVERY deployed Flash endpoint on this RunPod
    account, not just this project's — gated behind confirm=True so it can never fire from
    an ambiguous or accidental prompt.

    Args:
        confirm: Must be explicitly True to actually undeploy anything. False (default) is
            a safe no-op that just explains what would happen.
    """
    if not confirm:
        return {
            "ok": True,
            "executed": False,
            "note": "No-op: pass confirm=True to actually run `flash undeploy --all --force`. "
                    "This affects EVERY deployed Flash endpoint on this RunPod account.",
        }
    flash_bin = os.path.join(os.path.dirname(PY), "flash")
    if not os.path.exists(flash_bin):
        return {"ok": False, "error": f"flash CLI not found at {flash_bin}"}
    try:
        proc = await asyncio.create_subprocess_exec(
            flash_bin, "undeploy", "--all", "--force", cwd=ROOT,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=120)
    except asyncio.TimeoutError:
        return {"ok": False, "error": "flash undeploy timed out after 120s"}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    tail = "\n".join(stdout.decode(errors="replace").splitlines()[-40:])
    return {
        "ok": proc.returncode == 0,
        "exit_code": proc.returncode,
        "log_tail": tail,
        "note": "Undeployed ALL Flash endpoints on this account. The next Flash-touching "
                "tool call will auto-fall-back to the local threadpool (the reliability "
                "beat) until endpoints are redeployed with `flash deploy`.",
    }


# --------------------------------------------------------------------------- #
# Orchestration — the full adversarial loop in one call
# --------------------------------------------------------------------------- #
@mcp.tool()
async def screen_and_verify(
    target_gene: str = "EGFR",
    candidates_file: str = "data/sample_data/candidates.csv",
    target_seq: str | None = None,
    use_flash: bool = True,
    enforce_gate: bool = True,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Run the entire loop in one call: discover -> design -> independently verify.

    1) Scores the genomic target (AlphaGenome). 2) Folds/docks the candidate library
    (Boltz-2). 3) For each peptide binder, independently re-folds the complex on NVIDIA
    BioNeMo and emits a verdict. Returns a single ranked report where only candidates
    the independent verifier corroborates are marked validated — the headline artifact
    for a demo.

    The discovery gate is ENFORCED: if the locus is not SUPER_ENHANCER the run stops
    before design/verification (matching the skill's "proceed only if SUPER_ENHANCER"
    rule), so the report can never read "gate FAILED" yet still list validated
    candidates. Pass enforce_gate=False to deliberately run past a failed gate.

    Args:
        target_gene: Gene to score in the discovery gate.
        candidates_file: CSV with 'name' and 'seq' (peptide) / 'smiles' columns.
        target_seq: Receptor sequence; defaults to the EGFR ectodomain.
        use_flash: Fan folding out across the RunPod Flash GPU fleet (default True —
            this is the headline demo artifact's whole point for today's GPU fan-out
            judging criterion; pass False to force an entirely local, non-Flash run).
        enforce_gate: Block the pipeline when the discovery gate fails (default True).
    """
    tgt = target_seq or DEFAULT_TARGET
    report: dict[str, Any] = {"ok": True, "target_gene": target_gene}

    # 1) Discovery gate
    disc = discover_target(target_gene=target_gene)
    report["discovery"] = disc
    cls = (disc.get("predictions") or {}).get("classification")
    report["discovery_gate_passed"] = cls == "SUPER_ENHANCER"

    # Enforce the gate: a failed discovery call must stop the loop, not be ignored.
    if enforce_gate and not report["discovery_gate_passed"]:
        report["status"] = "BLOCKED_AT_DISCOVERY_GATE"
        report["verified_candidates"] = []
        report["validated_count"] = 0
        report["summary"] = (
            f"target {target_gene}: discovery gate FAILED (classification={cls!r}); "
            f"pipeline BLOCKED before design/verification. Supply a tumor-active locus, "
            f"or pass enforce_gate=False to override."
        )
        return report

    # 2) Design / fold
    design = await design_ligands(
        candidates_file=candidates_file, target_seq=tgt, use_flash=use_flash, ctx=ctx
    )
    report["design"] = {
        k: design.get(k)
        for k in ("ok", "library_size", "log_tail", "data_freshness", "stale", "flash_autoscaling")
    }
    # Surface a stale committed library at the top level too, so it can't be missed by only
    # reading report["summary"] or report["validated_count"].
    report["stale_data_warning"] = bool(design.get("stale"))
    report["flash_scaling_chart"] = design.get("flash_scaling_chart")

    # 2b) GA promoter design — when use_flash, the fitness fan-out runs on the RunPod Flash
    # fleet, so the report carries the live 0->N->0 autoscaling metrics.
    if use_flash:
        ga = await design_promoter_flash(use_flash=True, smoke=True, ctx=ctx)
        report["ga_promoter_design"] = {
            "mode": ga.get("mode"),
            "best_promoter": ga.get("best_promoter"),
            "flash_autoscaling": ga.get("flash_autoscaling"),
            "note": ga.get("note"),
        }
        report["flash_scaling_chart"] = ga.get("flash_scaling_chart") or report["flash_scaling_chart"]

    # 3) Independent verification per peptide candidate
    seqs = {r.get("name", ""): r.get("seq", "") for r in _read_csv_records(candidates_file)}
    library = design.get("candidates", []) or []
    verified: list[dict[str, Any]] = []
    seen_jobs: set[str] = set()
    for row in library:
        job = str(row.get("job_name", ""))
        if job in seen_jobs:  # the committed CSV can carry duplicate rows from prior runs
            continue
        seen_jobs.add(job)
        plddt = row.get("plddt_score")
        plddt = float(plddt) if plddt not in (None, "") else None
        # Match the produced library row back to a candidate sequence by name substring.
        binder = ""
        for name, seq in seqs.items():
            tag = name.lower().replace(" ", "_")
            if seq and (tag in job.lower() or job.lower().endswith(tag)):
                binder = seq
                break
        if not binder:
            continue
        v = verify_with_bionemo(target_seq=tgt, binder_seq=binder, project_plddt=plddt)
        verified.append({
            "name": job,
            "binder_seq": binder,
            "project_plddt": plddt,
            "bionemo_confidence": v.get("bionemo_confidence"),
            "verdict": v.get("verdict"),
            "validated": v.get("verdict") == "CORROBORATED",
        })

    # Rank: corroborated first, then by mean of the two confidences.
    def _key(x: dict[str, Any]) -> tuple:
        conf = [c for c in (x["project_plddt"], x["bionemo_confidence"]) if c is not None]
        return (x["validated"], sum(conf) / len(conf) if conf else 0.0)

    verified.sort(key=_key, reverse=True)
    report["verified_candidates"] = verified
    report["validated_count"] = sum(1 for v in verified if v["validated"])
    report["summary"] = (
        f"target {target_gene}: gate={'PASS' if report['discovery_gate_passed'] else 'FAIL'}, "
        f"{report['validated_count']}/{len(verified)} candidates independently corroborated."
        + (
            " WARNING: candidate library is STALE (no fresh fold ran this call — verifying "
            "the previously committed library, not a live run)."
            if report["stale_data_warning"] else ""
        )
    )
    return report


# --------------------------------------------------------------------------- #
# Entry
# --------------------------------------------------------------------------- #
async def _selftest() -> int:
    print("== discover_target ==")
    print(json.dumps(discover_target(target_gene="EGFR"), indent=2)[:600])
    print("\n== design_thermal_switch (5 gens) ==")
    print(json.dumps(design_thermal_switch(generations=5), indent=2)[:600])
    print("\n== verify_with_bionemo ==")
    print(json.dumps(
        verify_with_bionemo(target_seq="LEEKKVCQGT", binder_seq="YHWYGYTPQNVI", project_plddt=82.0),
        indent=2,
    ))
    print("\n== kill_flash_endpoints (no-op, confirm=False) ==")
    print(json.dumps(await kill_flash_endpoints(), indent=2))
    # Forces use_flash=False here (despite the tool's demo-friendly default of True) so the
    # smoke test stays fast/local and doesn't depend on a deployed Flash endpoint.
    print("\n== screen_and_verify (full loop, local) ==")
    rep = await screen_and_verify(target_gene="EGFR", use_flash=False)
    print("summary:", rep.get("summary"))
    for v in rep.get("verified_candidates", []):
        print(f"  - {v['name']}: project={v['project_plddt']} bionemo={v['bionemo_confidence']} "
              f"-> {v['verdict']} (validated={v['validated']})")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(asyncio.run(_selftest()))
    mcp.run()
