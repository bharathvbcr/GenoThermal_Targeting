"""Shared stdlib .env loader for the standalone CLI phases.

The MCP server (`mcp_geno_thermal.py`) has its own `_load_dotenv()` because Claude
Science launches it via `.mcp.json`, which does NOT source the shell or `.env`. The
CLI phases driven by `run_pipeline.py` had the SAME problem from a different angle:
`run_pipeline.py` shells out `python <phase>.py` as subprocesses, and none of those
phases loaded `.env`. So `ALPHAGENOME_API_KEY` / `NVIDIA_API_KEY` / `BRIGHTDATA_API_TOKEN`
/ `RUNPOD_API_KEY` never reached them, and every phase silently fell back to its
synthetic path even when a valid key sat in `.env` — which is why the committed
`target_report.json` was synthetic despite a real key being present.

This module centralises that loader so `run_pipeline.py` (once, before fanning out to
children that inherit `os.environ`) and each phase (for standalone runs) load the same
file the same way. Never overrides an already-set variable, so a real shell export or
an env var injected by the orchestrator always wins.
"""

import os

# Repo root = the directory this file lives in (phases run from the repo root, but
# anchoring to __file__ keeps it correct regardless of the caller's CWD).
ROOT = os.path.dirname(os.path.abspath(__file__))


def load_dotenv(path: str | None = None) -> bool:
    """Load KEY=VALUE lines from `.env` into os.environ. Returns True if a file was read.

    Mirrors the loader in mcp_geno_thermal.py: skips blanks/comments, strips matching
    surrounding quotes, and never clobbers a variable that is already set."""
    path = path or os.path.join(ROOT, ".env")
    if not os.path.exists(path):
        return False
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
        return False
    return True
