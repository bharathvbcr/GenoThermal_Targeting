"""
Phase 2 (Flash edition): Boltz-2 folding + binding-affinity on RunPod Flash.

Replaces the manual AlphaFold Server upload/download loop in `alphafold_utils.py`
with an open-weight (MIT-licensed) Boltz-2 model that runs on an autoscaling
serverless GPU. Each (target, candidate) complex is one job; the GA / candidate
library fans them out and the worker fleet scales 0 -> N on demand, then to zero.

Two prediction modes, chosen per candidate:
  * peptide binder  (candidate has `seq`)   -> fold complex, rank by interface ipTM
  * small molecule  (candidate has `smiles`) -> fold + Boltz-2 affinity head

The returned record is schema-compatible with AlphaFoldClient.parse_all_results()
(`job_name`, `plddt_score`, `pae_score`, `structure_cif`) and adds affinity fields.

Run modes:
  * remote (default): the @Endpoint decorator ships `fold_complex` to a Flash GPU.
  * local: `python flash_boltz.py --selftest` runs the same fold in-process
    (needs a local GPU + `pip install boltz`), so the logic is testable off-Flash.
"""

import os
import re
import json
import glob
import shutil
import time
import logging
import tempfile
import threading
import subprocess

# Honor GENOTHERMAL_LOG_LEVEL (e.g. DEBUG) so the per-line debug logs below are reachable
# at the venue without editing source. Defaults to INFO.
_LEVEL = getattr(logging, os.environ.get("GENOTHERMAL_LOG_LEVEL", "INFO").upper(), logging.INFO)
logging.basicConfig(
    level=_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("FlashBoltz")


def _log_gpu(stage: str):
    """Best-effort GPU telemetry (device + memory) for the RunPod worker. Self-contained:
    uses torch (already a worker dep) and is a no-op off-CUDA, so it never breaks a CPU run."""
    try:
        import torch
        if not getattr(torch, "cuda", None) or not torch.cuda.is_available():
            logger.info("[gpu:%s] CUDA not available (CPU worker).", stage)
            return
        i = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(i)
        total = props.total_memory / 1e9
        alloc = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        peak = torch.cuda.max_memory_allocated(i) / 1e9
        logger.info("[gpu:%s] %s (cc %d.%d) | alloc=%.2fGB reserved=%.2fGB peak=%.2fGB / total=%.1fGB (%.0f%% peak)",
                    stage, props.name, props.major, props.minor,
                    alloc, reserved, peak, total, 100.0 * peak / total if total else 0.0)
    except Exception as e:  # torch missing / driver hiccup — telemetry is never load-bearing
        logger.debug("[gpu:%s] telemetry unavailable: %s", stage, e)


class _GpuSampler:
    """Background GPU-UTILIZATION sampler (SM occupancy % + memory via NVML). Runs on a daemon
    thread for the lifetime of a `with` block, then logs mean/peak utilization. Self-contained
    and fully guarded: a no-op when pynvml or a GPU is absent, so it never breaks a CPU run.
    Memory telemetry (_log_gpu) shows footprint; this shows whether the GPU is actually BUSY —
    the signal you need to tell 'right-sized' from 'idle/over-provisioned'."""

    def __init__(self, stage, interval=2.0):
        self.stage, self.interval = stage, interval
        self._stop = threading.Event()
        self._thread = None
        self._nvml = None
        self._handle = None
        self.samples = []  # (util_pct, mem_used_gb)

    def __enter__(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            logger.debug("[gpu-util:%s] NVML sampler started (interval=%.1fs)", self.stage, self.interval)
        except Exception as e:  # no pynvml / no GPU / no driver — sampling is best-effort
            logger.debug("[gpu-util:%s] sampler unavailable (%s); skipping.", self.stage, e)
        return self

    def _run(self):
        while not self._stop.wait(self.interval):
            try:
                u = self._nvml.nvmlDeviceGetUtilizationRates(self._handle)
                m = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
                self.samples.append((u.gpu, m.used / 1e9))
            except Exception:
                break

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._nvml:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
        if self.samples:
            us = [u for u, _ in self.samples]
            ms = [m for _, m in self.samples]
            logger.info("[gpu-util:%s] %d samples | SM util mean=%.0f%% peak=%d%% | mem used mean=%.2fGB peak=%.2fGB",
                        self.stage, len(self.samples), sum(us) / len(us), max(us),
                        sum(ms) / len(ms), max(ms))
        else:
            logger.debug("[gpu-util:%s] no utilization samples collected.", self.stage)
        return False  # never suppress exceptions from the wrapped block

# Map interface CONFIDENCE (ipTM*100) -> a binding-confidence bucket. Mirrors
# AlphaFoldClient.classify_binding thresholds so downstream reporting is unchanged.
# HONEST LABEL: these buckets reflect predicted interface CONFIDENCE, not measured
# binding affinity — "STRONG_BINDER" means "high-confidence interface", not high Kd/IC50.
# (Measured affinity for small molecules comes from the separate Boltz-2 affinity head.)
def classify_binding(iptm_x100: float) -> str:
    if iptm_x100 >= 80:
        return "STRONG_BINDER"
    if iptm_x100 >= 60:
        return "MODERATE_BINDER"
    if iptm_x100 >= 40:
        return "WEAK_BINDER"
    return "NON_BINDER"


def _sanitize(name: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_\-]", "_", name)
    return re.sub(r"__+", "_", clean).strip("_") or "job"


def _ligand_smiles(candidate: dict):
    """Return a clean SMILES string if this candidate is a small molecule, else None.
    Guards against pandas turning a blank CSV cell into float('nan') (which is truthy)."""
    smiles = candidate.get("smiles")
    if isinstance(smiles, str) and smiles.strip():
        return smiles.strip()
    return None


def _build_yaml(target_seq: str, candidate: dict, job_name: str) -> str:
    """Build a Boltz-2 input YAML for a target + one candidate binder."""
    lines = ["version: 1", "sequences:",
             "  - protein:", "      id: A", f"      sequence: {target_seq}"]
    smiles = _ligand_smiles(candidate)
    if smiles:  # small molecule -> ligand chain + affinity head
        lines += ["  - ligand:", "      id: B", f"      smiles: '{smiles}'"]
        affinity = ["properties:", "  - affinity:", "      binder: B"]
    else:       # peptide -> second protein chain, rank by interface confidence
        lines += ["  - protein:", "      id: B", f"      sequence: {candidate['seq']}"]
        affinity = []
    return "\n".join(lines + affinity) + "\n"


def fold_complex(target_seq: str, candidate: dict, use_msa_server: bool = True) -> dict:
    """
    Fold one target+candidate complex with Boltz-2 and return a result record.
    This is the unit of work that Flash fans out across the GPU fleet.
    """
    import torch  # noqa: F401  (worker import; confirms CUDA on the box)
    logger.info("fold_complex: job='%s', target_len=%d, use_msa=%s",
                candidate.get("name", "candidate"), len(target_seq), use_msa_server)
    logger.debug("fold_complex: CUDA available=%s", torch.cuda.is_available() if hasattr(torch, "cuda") else "unknown")
    # Reset peak counter so the telemetry below reflects THIS fold only (right-size the GPU).
    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass
    _log_gpu("fold-start")

    job_name = _sanitize(candidate.get("name", "candidate"))
    workdir = tempfile.mkdtemp(prefix=f"boltz_{job_name}_")
    logger.debug("fold_complex: workdir=%s", workdir)
    yaml_path = os.path.join(workdir, f"{job_name}.yaml")
    out_dir = os.path.join(workdir, "out")
    with open(yaml_path, "w") as f:
        yaml_content = _build_yaml(target_seq, candidate, job_name)
        f.write(yaml_content)
    logger.debug("fold_complex: wrote YAML to %s (%d chars)", yaml_path, len(yaml_content))

    cmd = ["boltz", "predict", yaml_path, "--out_dir", out_dir,
           "--output_format", "mmcif"]
    if use_msa_server:
        cmd.append("--use_msa_server")  # public MSA server; swap for local MSA to fully self-host
    logger.info("Running: %s", " ".join(cmd))
    _t_fold = time.time()
    try:
        with _GpuSampler("boltz-fold"):  # sample SM utilization while boltz runs
            proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        # Surface boltz's own diagnostics instead of dying with an opaque traceback.
        logger.error("boltz predict FAILED for '%s' (exit %s) after %.1fs", job_name, e.returncode, time.time() - _t_fold)
        if e.stdout:
            logger.error("boltz stdout (tail):\n%s", e.stdout[-2000:])
        if e.stderr:
            logger.error("boltz stderr (tail):\n%s", e.stderr[-2000:])
        raise
    _fold_s = time.time() - _t_fold
    logger.info("fold_complex: boltz predict completed for '%s' in %.1fs.", job_name, _fold_s)
    if proc.stderr:  # boltz emits progress/warnings on stderr even on success
        logger.debug("boltz stderr (tail):\n%s", proc.stderr[-1000:])
    _log_gpu("fold-done")  # peak memory here tells you if the A100-80GB is over/under-provisioned

    pred_dir = os.path.join(out_dir, f"boltz_results_{job_name}", "predictions", job_name)
    logger.debug("fold_complex: pred_dir=%s", pred_dir)

    cif_files = sorted(glob.glob(os.path.join(pred_dir, f"{job_name}_model_*.cif")))
    if not cif_files:
        logger.warning("fold_complex: no CIF files found for job '%s' in %s", job_name, pred_dir)
    else:
        logger.debug("fold_complex: found %d CIF file(s), using %s", len(cif_files), cif_files[0])
    structure_cif = open(cif_files[0]).read() if cif_files else ""

    conf = {}
    conf_files = sorted(glob.glob(os.path.join(pred_dir, f"confidence_{job_name}_model_*.json")))
    if conf_files:
        conf = json.load(open(conf_files[0]))
        logger.debug("fold_complex: confidence file loaded: %s", conf_files[0])
    else:
        logger.warning("fold_complex: no confidence JSON found for '%s'; defaulting iptm=0.", job_name)
    # ipTM is the interface-quality score for the complex; fall back to ptm/confidence.
    iptm = conf.get("iptm", conf.get("ptm", conf.get("confidence_score", 0.0)))
    plddt_score = float(iptm) * 100.0

    binding_class = classify_binding(plddt_score)
    record = {
        "job_name": job_name,
        "plddt_score": plddt_score,          # interface ipTM*100, drop-in for the old pLDDT field
        # Honest label: this is Boltz's predicted distance error (PDE), not AlphaFold PAE.
        "complex_pde": float(conf.get("complex_pde", conf.get("complex_iplddt", 0.0))),
        "binding_class": binding_class,
        "structure_cif": structure_cif,      # returned as text; driver writes it to disk
        "mode": "ligand" if _ligand_smiles(candidate) else "peptide",
        "fold_seconds": round(_fold_s, 1),   # worker compute time (cold-start excluded); for optimization tracking
    }
    logger.info("fold_complex result: job=%s, ipTM*100=%.1f, class=%s, cif_len=%d, fold=%.1fs",
                job_name, plddt_score, binding_class, len(structure_cif), _fold_s)

    aff_files = glob.glob(os.path.join(pred_dir, f"affinity_{job_name}.json"))
    if aff_files:  # only present for small-molecule ligands
        aff = json.load(open(aff_files[0]))
        # affinity_pred_value: predicted log(IC50) in uM, LOWER = stronger binder
        record["affinity_pred_value"] = float(aff.get("affinity_pred_value", 0.0))
        record["affinity_probability"] = float(aff.get("affinity_probability_binary", 0.0))
        logger.info("Affinity: pred_log_IC50=%.4f, probability=%.4f",
                    record["affinity_pred_value"], record["affinity_probability"])

    shutil.rmtree(workdir, ignore_errors=True)
    logger.debug("Cleaned up workdir %s", workdir)
    return record



def _ensure_boltz():
    """Docker-free: install boltz on the Flash worker at runtime. Flash's BUILD pip can't
    resolve boltz, but the WORKER has normal PyPI access. Cached on the warm worker; point
    BOLTZ_CACHE at a mounted network volume to persist weights across cold starts."""
    import shutil, subprocess, sys as _sys
    if shutil.which("boltz"):
        return
    subprocess.run([_sys.executable, "-m", "pip", "install", "-q", "boltz==2.1.1", "pynvml"],
                   check=True)

# --- Flash endpoint -------------------------------------------------------
# Guarded import so the module stays importable (and --selftest works) without
# the SDK installed. On Flash, `fold_endpoint` ships to a GPU worker.
try:
    from runpod_flash import Endpoint, GpuType

    @Endpoint(
        name="genothermal-boltz2",
        gpu=GpuType.NVIDIA_A100_80GB_PCIe,   # Boltz-2 wants ~A100-class; 80GB is comfortable
        workers=(0, 2),                     # scale 0 -> 20, then back to zero
        dependencies=[],   # Docker-free: boltz installed at runtime on the worker    # boltz pins its own torch; pynvml enables SM-util sampling
        idle_timeout=120,                    # reuse warm boxes across a candidate burst (multi-GB weight reload is slow)
    )
    async def fold_endpoint(payload: dict) -> dict:
        """payload = {target_seq, candidate, use_msa_server?}"""
        _ensure_boltz()
        return fold_complex(
            payload["target_seq"],
            payload["candidate"],
            payload.get("use_msa_server", True),
        )

    FLASH_AVAILABLE = True
except (ImportError, AttributeError, ValueError):  # bad GpuType/cpu flavor degrades to local, not crash
    fold_endpoint = None
    FLASH_AVAILABLE = False
    logger.warning("runpod_flash unavailable — remote folding disabled (local --selftest still works).")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--selftest", action="store_true",
                   help="Fold a tiny target+peptide complex locally (needs GPU + boltz).")
    args = p.parse_args()
    if args.selftest:
        rec = fold_complex(
            "LEEKKVCQGTSNKLTQLGTFEDHFLSLQRM",
            {"name": "GE11", "seq": "YHWYGYTPQNVI"},
        )
        logger.info("Self-test result:\n%s",
                    json.dumps({k: v for k, v in rec.items() if k != "structure_cif"}, indent=2))
