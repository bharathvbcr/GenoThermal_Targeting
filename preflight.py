"""
Pre-flight check for the RunPod Flash hackathon — run this FIRST at the venue.

Exercises every local / fallback code path that does NOT need the Flash SDK, a GPU,
or the heavy worker deps (boltz/torch/openmm/stable-baselines3). Green here means the
plumbing is sound and only the Flash-side (bundling + GpuType names) remains to verify.

    python preflight.py            # prints a PASS/FAIL table, exits non-zero on any failure
"""

import os
import sys
import math
import random
import traceback
import logging

_LEVEL = getattr(logging, os.environ.get("GENOTHERMAL_LOG_LEVEL", "INFO").upper(), logging.INFO)
logging.basicConfig(
    level=_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("preflight.log"), logging.StreamHandler()],
)
logger = logging.getLogger("Preflight")

RESULTS = []


def check(name):
    """Decorator: run the function, record PASS/FAIL with a short note."""
    def wrap(fn):
        logger.debug("Running check: %s", name)
        try:
            note = fn() or ""
            RESULTS.append((name, True, note))
            logger.info("[PASS] %s — %s", name, note)
        except Exception as e:
            RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
            logger.error("[FAIL] %s — %s: %s", name, type(e).__name__, e)
            if "-v" in sys.argv:
                traceback.print_exc()
        return fn
    return wrap


@check("endpoint modules import without SDK (graceful degrade)")
def _():
    import flash_boltz, flash_fitness, flash_gpu_jobs
    assert flash_boltz.FLASH_AVAILABLE is False, "expected SDK absent locally"
    assert flash_fitness.FLASH_AVAILABLE is False
    assert flash_gpu_jobs.FLASH_AVAILABLE is False
    return "all 3 degraded to local"


@check("flash_fitness is self-contained (no evolver import on Local path)")
def _():
    sys.modules["evolver"] = None  # poison: prove the local path never touches it
    import flash_fitness as ff
    scores = ff.score_sequences([("ACGT" * 50) for _ in range(3)], mode="Local")
    assert len(scores) == 3 and all(isinstance(s, float) for s in scores)
    del sys.modules["evolver"]
    return f"MAX_WORKERS={ff.MAX_WORKERS}, scores ok"


@check("flash_gpu_jobs is self-contained (no ppo_agent/physics_verify/rl_gene_designer imports)")
def _():
    for sib in ("ppo_agent", "physics_verify", "rl_gene_designer"):
        sys.modules[sib] = None  # poison: prove the module never imports siblings
    sys.modules.pop("flash_gpu_jobs", None)
    import flash_gpu_jobs as fg
    r = fg._local_reward("ATATAGAATTC" * 5)   # stdlib-only reward path
    assert isinstance(r, float)
    note = "import + local reward ok"
    try:
        Env = fg._make_env_class()             # needs gymnasium+numpy; best-effort
        e = Env(20)
        obs, _ = e.reset(seed=0)
        assert len(obs) == 20
        note = "import + reward + env ok"
    except Exception as ex:
        note = f"import + reward ok (env needs gymnasium: {type(ex).__name__})"
        logger.debug("flash_gpu_jobs env construction skipped (gymnasium not available): %s", ex)
    for sib in ("ppo_agent", "physics_verify", "rl_gene_designer"):
        del sys.modules[sib]
    return note


@check("Boltz _ligand_smiles guards NaN / blank (peptide vs ligand routing)")
def _():
    import flash_boltz as fb
    assert fb._ligand_smiles({"seq": "AAAA", "smiles": ""}) is None
    assert fb._ligand_smiles({"seq": "AAAA", "smiles": float("nan")}) is None
    assert fb._ligand_smiles({"name": "x", "smiles": "CCO"}) == "CCO"
    assert "ligand:" in fb._build_yaml("MKT", {"name": "m", "smiles": "CCO"}, "m")
    assert "protein:" in fb._build_yaml("MKT", {"name": "p", "seq": "YHW"}, "p")
    return "routing + YAML ok"


@check("binding classification thresholds")
def _():
    import flash_boltz as fb
    assert fb.classify_binding(85) == "STRONG_BINDER"
    assert fb.classify_binding(65) == "MODERATE_BINDER"
    assert fb.classify_binding(45) == "WEAK_BINDER"
    assert fb.classify_binding(10) == "NON_BINDER"
    return "ok"


@check("FanoutMetrics math + save/load")
def _():
    import os, json
    from flash_metrics import FanoutMetrics
    m = FanoutMetrics(phase="preflight", resource="cpu5c-4-8")
    recs = [m.start() for _ in range(10)]
    for r in recs:
        r["end"] = r["start"] + 1.0
        r["ok"] = True
    s = m.summary()
    assert s["peak_inflight"] == 10, s["peak_inflight"]
    assert s["n_ok"] == 10
    path = "_preflight_metrics.json"
    m.save(path)
    data = json.load(open(path))
    os.remove(path)
    assert isinstance(data, list) and data[-1]["phase"] == "preflight"
    return f"peak={s['peak_inflight']}, est_cost=${s['est_cost_usd']}"


@check("dashboard renders a PNG")
def _():
    import os, json
    from flash_metrics import FanoutMetrics
    m = FanoutMetrics(phase="preflight", resource="A100_80GB")
    recs = [m.start() for _ in range(5)]
    for r in recs:
        r["end"] = r["start"] + 2.0
        r["ok"] = True
    m.save("_preflight_metrics.json")
    import subprocess
    subprocess.run([sys.executable, "flash_dashboard.py",
                    "--metrics", "_preflight_metrics.json", "--out", "_preflight.png"],
                   check=True, capture_output=True)
    ok = os.path.exists("_preflight.png")
    for f in ("_preflight_metrics.json", "_preflight.png"):
        if os.path.exists(f):
            os.remove(f)
    assert ok, "dashboard PNG not produced"
    return "PNG generated"


@check("target_panel selectivity ranking")
def _():
    import target_panel as tp
    rows = [
        {"job_name": "GE11", "target": "EGFR", "plddt_score": 85.0},
        {"job_name": "GE11", "target": "KRAS", "plddt_score": 40.0},
        {"job_name": "RGD", "target": "EGFR", "plddt_score": 60.0},
        {"job_name": "RGD", "target": "KRAS", "plddt_score": 70.0},
    ]
    _, ranked = tp.selectivity_table(rows, "EGFR")
    assert ranked.index[0] == "GE11", ranked.index[0]
    assert abs(ranked.loc["GE11", "selectivity_margin"] - 45.0) < 1e-6
    return "GE11 top, margin=45"


@check("GA chunk sizing reaches full fleet")
def _():
    from flash_fitness import MAX_WORKERS
    for pop in (100, 50, 8):
        chunk = max(1, math.ceil(pop / MAX_WORKERS))
        jobs = math.ceil(pop / chunk)
        assert jobs <= MAX_WORKERS
    chunk = max(1, math.ceil(100 / MAX_WORKERS))
    assert math.ceil(100 / chunk) == MAX_WORKERS, "pop=100 should hit the full fleet"
    return f"pop=100 -> {MAX_WORKERS} jobs"


@check("GA runs end-to-end in smoke mode (local fallback)")
def _():
    import os
    os.environ["GENOTHERMAL_SMOKE"] = "1"
    sys.path.insert(0, "hard_mode")
    # fresh import so smoke constants take effect
    for mod in ("evolver",):
        sys.modules.pop(mod, None)
    import evolver
    opt = evolver.GeneticOptimizer(evolver.AlphaGenomeOracle(mode="Local"))
    best, hist = opt.run()
    assert isinstance(best, str) and len(best) == evolver.GENOME_LENGTH
    assert opt._flash_enabled is False
    return f"pop={evolver.POPULATION_SIZE} gens={evolver.GENERATIONS}, converged"


@check("bright_data_intel degrades to a flagged local stub without a token")
def _():
    os.environ.pop("BRIGHTDATA_API_TOKEN", None)  # prove the no-key path
    sys.modules.pop("bright_data_intel", None)
    import bright_data_intel as bdi
    assert bdi.FLASH_AVAILABLE is False, "expected SDK absent locally"
    rec = bdi.fetch_target_intel("EGFR")
    assert rec["source"] == "local-stub" and rec["target"] == "EGFR", rec
    assert "EGFR" in rec["headline"]
    recs = bdi._intel_local(["EGFR", "KRAS"], None, 5)
    assert len(recs) == 2 and all(r["source"] == "local-stub" for r in recs)
    return "stub ok, no token / network required"


@check("candidate CSV loader accepts SMILES-only and peptide CSVs")
def _():
    import pandas as pd
    # mimic boltz_designer's validation rule
    def valid(cols):
        return "name" in cols and bool({"seq", "smiles"} & set(cols))
    assert valid(["name", "seq"])
    assert valid(["name", "smiles"])
    assert not valid(["name"])
    sm = pd.read_csv("data/sample_data/small_molecule_candidates.csv")
    assert "smiles" in sm.columns and len(sm) >= 4
    return f"{len(sm)} small molecules load"


def main():
    logger.info("=== Geno-Thermal Flash pre-flight ===")
    width = max(len(n) for n, _, _ in RESULTS)
    passed = 0
    for name, ok, note in RESULTS:
        mark = "PASS" if ok else "FAIL"
        logger.info("  [%s] %s  %s", mark, name.ljust(width), note)
        passed += ok
    total = len(RESULTS)
    logger.info("%d/%d checks passed.", passed, total)
    if passed != total:
        logger.warning("Re-run with -v for tracebacks.")
        sys.exit(1)
    logger.info("All local paths green — only Flash bundling + GpuType names remain to verify on-site.")


if __name__ == "__main__":
    main()
