"""
Phase 4 accelerator: GA fitness scoring as a RunPod Flash fan-out endpoint.

The genetic algorithm in hard_mode/evolver.py evaluates ~POP*GENERATIONS sequences
(100 x 50 = 5,000). Today that's throttled to API_MAX_WORKERS=3 with a 1s delay to
be polite to the AlphaGenome API. This endpoint removes the bottleneck: each batch
of sequences is scored on its own worker and the fleet scales 0 -> N -> 0.

Scoring backend (swappable):
  * default  -> the project's AlphaGenomeOracle (AlphaGenome API or local motif scan).
  * upgrade  -> load an OPEN-WEIGHT DNA expression model (Borzoi / Enformer) on a GPU
    worker for a fully self-hosted, commercial-clean expression oracle. The hook is
    marked below; flip `gpu=` on the decorator when you enable it.

The endpoint returns only `scores` (the expensive part). Cheap motif `props` are
recomputed driver-side, so the GA's results schema {seq: (fitness, props)} is unchanged.

HONESTY NOTE (state to judges): the `mode='Local'` worker fitness below is a fast regex
MOTIF-COUNT heuristic with a small random jitter (NOT deterministic, NOT a learned model) —
a stand-in for a real expression oracle so the fan-out is demonstrable without burning the
AlphaGenome API across 50 workers. It scores motif presence, not measured expression.
"""

import os
import logging

_LEVEL = getattr(logging, os.environ.get("GENOTHERMAL_LOG_LEVEL", "INFO").upper(), logging.INFO)
logging.basicConfig(
    level=_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("FlashFitness")

# Fleet width — shared by the decorator and the GA so batch sizing targets the full fleet.
MAX_WORKERS = 8  # restore after boltz quota experiment

# --- Self-contained local scoring ----------------------------------------
# Inlined from hard_mode/evolver.AlphaGenomeOracle so this endpoint ships NO sibling
# repo imports to the worker (the #1 demo-killer: Flash may not bundle local source).
# Keep these constants in sync with evolver.py if that fitness function changes.
import re as _re
import random as _random

_TUMOR_MOTIFS = [r"AGAACA", r"GGATCTT", r"CACGTG"]
_NORMAL_MOTIFS = [r"TATAAA", r"CCAAT", r"GCGCGC"]
_HEAT_MOTIFS = [r"GAA..TTC", r"TTC..GAA"]
_W_TUMOR, _W_NORMAL, _W_HEAT, _W_GC = 1.5, 2.0, 1.2, 0.5


def _count(seq, motifs):
    return sum(len(_re.findall(p, seq)) for p in motifs)


def _local_score(seq):
    tumor = _count(seq, _TUMOR_MOTIFS) * 20.0
    normal = _count(seq, _NORMAL_MOTIFS) * 15.0
    heat = _count(seq, _HEAT_MOTIFS) * 25.0
    gc = (seq.count("G") + seq.count("C")) / max(len(seq), 1)
    gc_penalty = abs(0.55 - gc) * 100.0
    score = (tumor * _W_TUMOR) - (normal * _W_NORMAL) + (heat * _W_HEAT) - (gc_penalty * _W_GC)
    score += _random.uniform(-1, 1)
    result = max(0.0, score)
    logger.debug("_local_score: tumor=%.1f, normal=%.1f, heat=%.1f, gc=%.3f, raw=%.3f -> %.3f",
                 tumor, normal, heat, gc, score, result)
    return result


def score_sequences(sequences: list, api_key: str = None, mode: str = "Auto") -> list:
    """Score a batch of promoter sequences. This is the unit Flash fans out.

    mode='Local' (what the GA sends to the fleet) uses the inlined motif scorer above —
    fully self-contained, no sibling imports. Other modes fall back to the project oracle.
    """
    logger.info("score_sequences: n=%d, mode=%s", len(sequences), mode)
    if mode != "Local":
        # API/Auto path needs alphagenome_utils shipped to the worker; the GA avoids this
        # by always sending mode='Local' (never forwarding the API key to N workers).
        import sys
        _root = os.path.abspath(os.path.dirname(__file__))
        if _root not in sys.path:
            sys.path.insert(0, _root)
        sys.path.insert(0, os.path.join(_root, "hard_mode"))
        from evolver import AlphaGenomeOracle
        logger.info("Using AlphaGenomeOracle for scoring (mode=%s)", mode)
        oracle = AlphaGenomeOracle(api_key=api_key, mode=mode)
        scores = [float(oracle.score(s)) for s in sequences]
        logger.info("Scoring complete: %d scores, mean=%.3f", len(scores), sum(scores) / max(len(scores), 1))
        return scores

    # --- OPEN-MODEL HOOK: swap _local_score for a Borzoi/Enformer GPU scorer here ---
    logger.debug("Using inlined local motif scorer for %d sequences", len(sequences))
    scores = [float(_local_score(s)) for s in sequences]
    logger.info("Local scoring complete: %d scores, mean=%.3f", len(scores), sum(scores) / max(len(scores), 1))
    return scores


try:
    from runpod_flash import Endpoint, CpuInstanceType  # GpuType when using the GPU/Borzoi path

    @Endpoint(
        name="genothermal-fitness",
        cpu=CpuInstanceType.CPU5C_4_8,  # 4 vCPU/8GB; motif scoring is CPU-bound (swap to gpu= for Borzoi)
        workers=(0, MAX_WORKERS),  # the bottleneck buster: 3 -> MAX_WORKERS concurrent
        dependencies=[],         # local scorer is pure stdlib -> instant cold start; add "alphagenome" for API mode
        idle_timeout=15,
    )
    async def score_endpoint(payload: dict) -> dict:
        """payload = {sequences: [...], api_key?, mode?}

        Defaults to mode='Local': this endpoint ships with dependencies=[] and NO
        sibling source, so the 'Local' inlined motif scorer is the only path the worker
        can actually serve. A caller must explicitly opt into mode='API'/'Auto' AND the
        endpoint must be redeployed with dependencies=["alphagenome"] (+ the oracle source
        bundled) for that branch to work — otherwise it crashes on the worker.
        """
        scores = score_sequences(
            payload["sequences"],
            payload.get("api_key"),
            payload.get("mode", "Local"),
        )
        return {"scores": scores}

    FLASH_AVAILABLE = True
except (ImportError, AttributeError, ValueError):  # bad GpuType/cpu flavor degrades to local, not crash
    score_endpoint = None
    FLASH_AVAILABLE = False
    logger.warning("runpod_flash unavailable — GA fan-out disabled (local threading still works).")


if __name__ == "__main__":
    # Quick local check of the scoring unit (no Flash needed).
    import random
    pop = ["".join(random.choices("ACGT", k=200)) for _ in range(4)]
    logger.info("Sample scores: %s", score_sequences(pop, mode="Local"))
