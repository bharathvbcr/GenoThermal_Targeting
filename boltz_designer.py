"""
Phase 2 driver (Flash edition) — drop-in replacement for ligand_designer.py.

Old flow:  generate AlphaFold Server JSON -> upload to alphafoldserver.com by hand
           -> download result ZIPs -> re-run to parse.   (manual, serial, rate-limited)

New flow:  fan every candidate out to the Boltz-2 Flash endpoint concurrently;
           the GPU fleet autoscales 0 -> N, folds in parallel, returns structures +
           binding affinity, then scales to zero. One command, no web UI.

Writes the same `outputs/reports/candidate_library.csv` (job_name, plddt_score, pae_score,
binding_class, structure_path) plus affinity columns for small-molecule candidates, and saves
each returned structure into outputs/predicted_structures/.

Usage:
    python boltz_designer.py                         # fan out on Flash
    python boltz_designer.py --local                 # fold in-process (needs local GPU)
    python boltz_designer.py --candidates_file my.csv --output_csv lib.csv
"""

import os
import asyncio
import argparse
import logging

import pandas as pd

from env_utils import load_dotenv

# Load .env so RUNPOD_API_KEY reaches the Flash SDK when this phase is run standalone.
load_dotenv()

import flash_boltz

_LEVEL = getattr(logging, os.environ.get("GENOTHERMAL_LOG_LEVEL", "INFO").upper(), logging.INFO)
logging.basicConfig(
    level=_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("boltz_designer.log"), logging.StreamHandler()],
)
logger = logging.getLogger("BoltzDesigner")

DEFAULT_TARGET = (  # EGFR ectodomain fragment (same default as ligand_designer.py)
    "LEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVLGNLEITYVQRNYDLSFLKTIQEVAGYVLIALN"
    "TVERIPLENLQIIRGNMYYENSYALAVLSNYDANKTGLKELPMRNLQEILHGAVRFSNNPALCNVESIQWR"
    "DIVSSDFLSNMSMDFQNHLGSCQKCDPSCPNGSCWGAGEENCQKLTKIICAQQCSGRCRGKSPSDCCHNQC"
    "AAGCTGPRESDCLVCRKFRDEATCKDTCPPLMLYNPTTYQMDVNPEGKYSFGATCVKKCPRNYVVTDHGSCV"
    "RACGADSYEMEEDGVRKC"
)

DEFAULT_CANDIDATES = [
    {"name": "GE11 (EGF Mimic)", "seq": "YHWYGYTPQNVI"},
    {"name": "RGD (Integrin binder)", "seq": "ACDCRGDCFC"},
    {"name": "Poly-Alanine (Neg Control)", "seq": "AAAAAAAAAA"},
]

PDB_DIR = "outputs/predicted_structures"


def _save_structure(record: dict) -> str:
    """Write the returned CIF text to outputs/predicted_structures/ and return its path."""
    os.makedirs(PDB_DIR, exist_ok=True)
    cif = record.pop("structure_cif", "")
    if not cif:
        logger.warning("No structure_cif in record for job '%s'; skipping save.", record.get("job_name"))
        return ""
    path = os.path.join(PDB_DIR, f"{record['job_name']}_model_0.cif")
    with open(path, "w") as f:
        f.write(cif)
    logger.debug("Structure saved: %s (%d chars)", path, len(cif))
    return path


async def _fold_remote(target_seq: str, candidates: list, use_msa_server: bool,
                       timeout_s: int = 900) -> list:
    """Fan candidates out across the Flash GPU fleet and gather results.
    Each job is bounded by a timeout; failed/timed-out folds are dropped and flagged
    rather than crashing the whole phase (cold-start / eviction / MSA-server hangs)."""
    if not flash_boltz.FLASH_AVAILABLE:
        raise RuntimeError("runpod_flash not installed. Use --local, or `pip install runpod-flash`.")
    from flash_metrics import FanoutMetrics

    ep = flash_boltz.fold_endpoint
    metrics = FanoutMetrics(phase="fold-boltz2", resource="A100_80GB")
    logger.info("Dispatching %d folding jobs to Flash (workers scale 0->N)...", len(candidates))

    async def _await(c):
        rec = metrics.start()
        try:
            # Decorator endpoint -> await it directly (returns the result dict), not .run()/job.wait().
            out = await asyncio.wait_for(
                ep({"target_seq": target_seq, "candidate": c, "use_msa_server": use_msa_server}),
                timeout=timeout_s)
            # SDK gotcha: if a job outlasts RunPod's own internal /runsync window, the
            # runpod_flash sentinel returns the raw job-status envelope (status/id/
            # workerId/delayTime, no "job_name") instead of raising. Treat that as a
            # failure rather than silently writing a result-shaped row with no result.
            if not isinstance(out, dict) or "job_name" not in out:
                status = out.get("status") if isinstance(out, dict) else type(out).__name__
                raise RuntimeError(
                    f"non-terminal Flash response (status={status!r}); job was still "
                    f"running past RunPod's own sync window")
            metrics.done(rec, ok=True)
            return out
        except Exception as e:  # timeout, worker eviction, transport error
            logger.warning("Fold job failed/timed out (%s); dropping that candidate.", e)
            metrics.done(rec, ok=False)
            return None

    outputs = await asyncio.gather(*(_await(c) for c in candidates))
    metrics.save()
    ok = [o for o in outputs if o]
    logger.info("_fold_remote: %d/%d jobs succeeded.", len(ok), len(outputs))
    if len(ok) < len(outputs):
        logger.warning("%d/%d fold jobs failed and were dropped.", len(outputs) - len(ok), len(outputs))
    return ok


def _local_fold_toolchain_missing() -> str:
    """Return a human reason if the in-process Boltz-2 toolchain isn't available locally, else ''.
    The heavy fold deps (the `boltz` CLI + a CUDA `torch`) live on the Flash worker's
    @Endpoint(dependencies=...), not in the driver venv — so locally we skip rather than crash."""
    import shutil
    import importlib.util
    if shutil.which("boltz") is None:
        return "the `boltz` CLI is not installed locally"
    if importlib.util.find_spec("torch") is None:
        return "`torch` is not installed locally"
    return ""


def _fold_local(target_seq: str, candidates: list, use_msa_server: bool) -> list:
    """Fold in-process — same fold_complex the worker runs (needs a local GPU + boltz).
    Degrades to a clean skip (returns []) when the local toolchain is absent, so a driver-only
    machine doesn't crash; main() then preserves the committed library and the pipeline continues."""
    reason = _local_fold_toolchain_missing()
    if reason:
        logger.warning("SKIPPING local Boltz-2 fold — %s. Set GENOTHERMAL_FLASH=1 to fold on a "
                       "Flash GPU worker (deps ship with the endpoint); the committed library is kept.", reason)
        return []
    logger.info("Folding %d candidates locally...", len(candidates))
    return [flash_boltz.fold_complex(target_seq, c, use_msa_server) for c in candidates]


def main():
    parser = argparse.ArgumentParser(description="Geno-Thermal Phase 2 — Boltz-2 on Flash")
    parser.add_argument("--target_seq", type=str, default=None)
    parser.add_argument("--candidates_file", type=str, default="data/sample_data/candidates.csv",
                        help="CSV with 'name' and 'seq' (and optional 'smiles') columns.")
    parser.add_argument("--output_csv", type=str, default="outputs/reports/candidate_library.csv")
    parser.add_argument("--local", action="store_true",
                        help="Fold in-process instead of on Flash.")
    parser.add_argument("--no_msa_server", action="store_true",
                        help="Disable the public MSA server (provide your own MSA to fully self-host).")
    args = parser.parse_args()

    target_seq = args.target_seq or DEFAULT_TARGET
    if not args.target_seq:
        logger.info("Using default EGFR target (%d aa)", len(target_seq))

    try:
        df = pd.read_csv(args.candidates_file)
        if "name" not in df.columns or not ({"seq", "smiles"} & set(df.columns)):
            raise ValueError("CSV must contain 'name' and at least one of 'seq' (peptide) or 'smiles' (small molecule).")
        candidates = df.to_dict("records")
        logger.info("Loaded %d candidates from %s", len(candidates), args.candidates_file)
    except Exception as e:
        logger.warning("Could not read candidates (%s); using defaults.", e)
        candidates = DEFAULT_CANDIDATES

    if os.environ.get("GENOTHERMAL_SMOKE"):
        candidates = candidates[:1]
        logger.info("Smoke mode: folding only the first candidate.")

    # Route to Flash only when it's actually requested AND the SDK is importable; otherwise fold
    # locally (which self-skips if the local toolchain is absent). This keeps every entry point —
    # `make screen`, a bare `python boltz_designer.py`, the pipeline — from crashing on a driver box.
    want_flash = bool(os.environ.get("GENOTHERMAL_FLASH")) and not args.local
    use_flash = want_flash and flash_boltz.FLASH_AVAILABLE
    if want_flash and not flash_boltz.FLASH_AVAILABLE:
        logger.warning("GENOTHERMAL_FLASH set but runpod_flash is unavailable — folding locally instead.")
    use_msa_server = not args.no_msa_server
    logger.info("Fold mode: %s | use_msa_server=%s | n_candidates=%d",
                "FLASH" if use_flash else "LOCAL", use_msa_server, len(candidates))
    if use_flash:
        records = asyncio.run(_fold_remote(target_seq, candidates, use_msa_server))
    else:
        records = _fold_local(target_seq, candidates, use_msa_server)
    logger.info("Fold complete: %d records returned.", len(records))

    # Guard the (often git-tracked) output CSV. A flaky live Flash run can return FEWER
    # records than candidates (drop-and-flag) — never let a partial/empty fan-out clobber
    # a committed library the selectivity panel + summary report depend on downstream.
    if not records:
        logger.warning("No fold results returned (local toolchain skipped, or all jobs "
                       "failed/dropped) — keeping %s untouched so downstream phases use the "
                       "committed library.", args.output_csv)
        return

    for r in records:
        r["structure_path"] = _save_structure(r)

    results_df = pd.DataFrame(records)
    if len(records) < len(candidates) and os.path.exists(args.output_csv):
        sidecar = os.path.splitext(args.output_csv)[0] + ".partial.csv"
        results_df.to_csv(sidecar, index=False)
        logger.warning("Only %d/%d folds succeeded — wrote partial results to %s and LEFT "
                       "%s untouched to protect the committed library.",
                       len(records), len(candidates), sidecar, args.output_csv)
    else:
        results_df.to_csv(args.output_csv, index=False)
        logger.info("Saved %d results to %s", len(results_df), args.output_csv)

    if "plddt_score" in results_df.columns and len(results_df):
        best = results_df.loc[results_df["plddt_score"].idxmax()]
        logger.info("Best by interface confidence: %s (ipTM*100=%.1f, %s)",
                    best["job_name"], best["plddt_score"], best["binding_class"])
    if "affinity_pred_value" in results_df.columns:
        ranked = results_df.dropna(subset=["affinity_pred_value"])
        if len(ranked):  # lower predicted log(IC50) = stronger binder
            strongest = ranked.loc[ranked["affinity_pred_value"].idxmin()]
            logger.info("Strongest predicted affinity: %s (pred log-IC50=%.2f)",
                        strongest["job_name"], strongest["affinity_pred_value"])


if __name__ == "__main__":
    main()
