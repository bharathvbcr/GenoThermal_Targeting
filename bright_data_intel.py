"""
Phase 1.5 (Flash edition): LIVE target intelligence via Bright Data, fanned out on RunPod Flash.

For each oncogene target we query Bright Data's SERP API for the latest literature / known
inhibitors (optionally scraping the top hit to markdown via Web Unlocker). Each target is one
job; the worker fleet scales 0 -> N -> 0 and records FanoutMetrics for the SAME dashboard —
the autoscaling story, now over web-data calls. This is the pipeline that uses BOTH hackathon
sponsors at once: RunPod Flash (the fan-out) + Bright Data (the web data).

Architecture note: the pipeline phase calls Bright Data's REST API directly from the worker
(`requests` -> https://api.brightdata.com/request). The Bright Data MCP server in .mcp.json is
the complementary path for the interactive agent — same account/token, different entry point.

Graceful degradation (so preflight / offline / no-key runs stay green):
  * No BRIGHTDATA_API_TOKEN, no network, or ANY error  -> a deterministic LOCAL STUB summary,
    clearly flagged source="local-stub". This function NEVER raises and NEVER blocks the pipeline.
  * With a token -> real Bright Data SERP (and optional Unlocker scrape), source="brightdata".

    python bright_data_intel.py                       # local (stub unless token set), all targets
    GENOTHERMAL_FLASH=1 python bright_data_intel.py    # fan out across the Flash fleet
    python bright_data_intel.py --targets EGFR KRAS    # explicit targets
"""

import os
import re
import json
import logging

from env_utils import load_dotenv

# Load .env so BRIGHTDATA_API_TOKEN reaches the SERP call when run standalone. On a Flash
# worker there is no .env (load returns False) and the token arrives via Flash's env, so
# this is a harmless no-op there.
load_dotenv()

_LEVEL = getattr(logging, os.environ.get("GENOTHERMAL_LOG_LEVEL", "INFO").upper(), logging.INFO)
logging.basicConfig(
    level=_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("bright_data_intel.log"), logging.StreamHandler()],
)
logger = logging.getLogger("BrightDataIntel")

MAX_WORKERS = 16                      # web-data calls are I/O-bound; modest fleet width
BRIGHTDATA_ENDPOINT = "https://api.brightdata.com/request"

# Default oncogene panel (matches data/sample_data/targets.csv, stripped of domain suffixes).
DEFAULT_TARGETS = ["EGFR", "KRAS", "HER2", "BRAF"]

# Canned, source-flagged fallbacks so the phase is meaningful offline / without a key.
# These are STATIC reference notes (clearly local-stub), NOT live data.
_STUB_NOTES = {
    "EGFR": "EGFR (ErbB1) — TKIs: gefitinib, erlotinib, osimertinib; ectodomain peptide-binder target.",
    "KRAS": "KRAS — historically 'undruggable'; KRAS-G12C inhibitors sotorasib/adagrasib.",
    "HER2": "HER2 (ERBB2) — trastuzumab/pertuzumab; ECD a classic antibody/peptide target.",
    "BRAF": "BRAF — V600E kinase-domain inhibitors vemurafenib/dabrafenib; intracellular (small-molecule).",
}


def _normalize(target: str) -> str:
    """Strip domain suffixes used in the fold panel (HER2_ECD -> HER2, BRAF_KD -> BRAF)."""
    return target.split("_")[0].upper()


# Pull real external result links out of Google SERP markdown ([title](url)), skipping Google's
# own nav/asset links. Good enough to surface the top organic hits for LLM intel.
_LINK_RE = re.compile(r"\[([^\]]{2,160})\]\((https?://[^)]+)\)")
_SKIP_HOSTS = ("google.com", "gstatic.com", "googleusercontent.com", "youtube.com/redirect")
# Google SERP markdown link anchors that aren't real titles -> fall back to the source domain.
_GENERIC = {"read more", "cached", "translate this page", "more results", "images", "maps",
            "news", "videos", "shopping", "books", "flights", "sign in", "settings", "more",
            "next", "previous", "view all", "feedback", "about", "privacy", "terms"}


def _domain(url: str) -> str:
    try:
        import urllib.parse
        return urllib.parse.urlparse(url).netloc.replace("www.", "") or url[:40]
    except Exception:
        return url[:40]


def _parse_markdown_results(md: str, n: int) -> list:
    results, seen = [], set()
    for text, url in _LINK_RE.findall(md):
        url = url.split("#:~:")[0]                       # strip Google text-fragment suffix
        if any(h in url for h in _SKIP_HOSTS) or url in seen:
            continue
        title = " ".join(text.split())
        if title.lower() in _GENERIC or len(title) < 6:  # generic anchor -> use the source domain
            title = _domain(url)
        seen.add(url)
        results.append({"title": title, "url": url, "snippet": None})
        if len(results) >= n:
            break
    return results


def _local_stub(target: str) -> dict:
    base = _normalize(target)
    return {
        "target": target,
        "source": "local-stub",
        "query": f"{base} inhibitor cancer therapeutic latest research",
        "headline": _STUB_NOTES.get(base, f"{base} — oncogenic target (no live data; local stub)."),
        "results": [],
        "note": "LOCAL STUB — set BRIGHTDATA_API_TOKEN for live Bright Data SERP results.",
    }


def fetch_target_intel(target: str, api_token: str = None, n_results: int = 5,
                       timeout_s: int = 30) -> dict:
    """Return a target-intel record. Real Bright Data SERP when a token is present, else a
    flagged local stub. Defensive: any failure (no key, no net, parse error) degrades to stub."""
    token = api_token or os.environ.get("BRIGHTDATA_API_TOKEN")
    if not token:
        logger.info("[%s] no BRIGHTDATA_API_TOKEN -> local stub.", target)
        return _local_stub(target)

    base = _normalize(target)
    query = f"{base} inhibitor cancer therapeutic latest research"
    try:
        import requests
        import urllib.parse
        zone = os.environ.get("BRIGHTDATA_SERP_ZONE", "serp_api1")
        # data_format=markdown returns the SERP rendered as clean markdown (works across SERP
        # zones and is ideal for LLM intel); we pull the external result links out of it.
        search_url = "https://www.google.com/search?" + urllib.parse.urlencode({"q": query})
        logger.info("[%s] Bright Data SERP (zone=%s): %s", target, zone, query)
        resp = requests.post(
            BRIGHTDATA_ENDPOINT,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={"zone": zone, "url": search_url, "format": "raw", "data_format": "markdown"},
            timeout=timeout_s,
        )
        resp.raise_for_status()
        md = resp.text
        results = _parse_markdown_results(md, n_results)
        headline = results[0]["title"] if results else f"{base}: SERP returned ({len(md)} chars), no parsed links."
        logger.info("[%s] Bright Data returned %d chars, %d result link(s).", target, len(md), len(results))
        return {"target": target, "source": "brightdata", "query": query,
                "headline": headline, "results": results, "serp_chars": len(md)}
    except Exception as e:  # no net / bad zone / parse error / rate limit — never break the phase
        logger.warning("[%s] Bright Data call failed (%s) -> local stub.", target, e)
        stub = _local_stub(target)
        stub["error"] = str(e)
        return stub


# --- Flash endpoint -------------------------------------------------------
try:
    from runpod_flash import Endpoint, CpuInstanceType

    @Endpoint(
        name="genothermal-intel",
        cpu=CpuInstanceType.CPU5C_4_8,  # 4 vCPU/8GB; I/O-bound web calls, CPU flavor is plenty
        workers=(0, MAX_WORKERS),    # fan 0 -> N -> 0 over the target panel
        dependencies=["requests"],   # the only worker dep; token comes in the payload
        idle_timeout=15,
    )
    async def intel_endpoint(payload: dict) -> dict:
        """payload = {target, api_token?, n_results?}"""
        return fetch_target_intel(payload["target"], payload.get("api_token"),
                                  payload.get("n_results", 5))

    FLASH_AVAILABLE = True
except (ImportError, AttributeError, ValueError):  # bad flavor / no SDK -> local, not crash
    intel_endpoint = None
    FLASH_AVAILABLE = False
    logger.warning("runpod_flash unavailable — intel fan-out disabled (local path still works).")


def _intel_remote(targets: list, api_token: str, n_results: int, timeout_s: int = 120) -> list:
    """Fan target-intel jobs across the Flash fleet; drop-and-flag failures, record metrics."""
    import asyncio
    from flash_metrics import FanoutMetrics

    metrics = FanoutMetrics(phase="bright-intel", resource="cpu5c-4-8")
    logger.info("Submitting %d intel jobs to Flash (workers scale 0->%d)...", len(targets), MAX_WORKERS)

    async def _run():
        async def _await(target):
            rec = metrics.start()
            try:
                # Decorator endpoint -> await directly (returns the result dict), not .run()/job.wait().
                result = await asyncio.wait_for(
                    intel_endpoint({"target": target, "api_token": api_token, "n_results": n_results}),
                    timeout=timeout_s)
                metrics.done(rec, ok=True)
                return result
            except Exception as e:
                # A dead/cold worker shouldn't downgrade us to FAKE data when a real local
                # Bright Data call still works. Try the live local REST call first; that
                # function only stubs if IT also fails (no token / no network).
                logger.warning("Intel job for %s failed on the worker (%s); retrying via a "
                               "live LOCAL Bright Data call.", target, e)
                metrics.done(rec, ok=False)
                return fetch_target_intel(target, api_token, n_results)

        return await asyncio.gather(*(_await(t) for t in targets))

    out = asyncio.run(_run())
    metrics.save()
    return out


def _intel_local(targets: list, api_token: str, n_results: int) -> list:
    logger.info("Fetching intel for %d target(s) locally...", len(targets))
    return [fetch_target_intel(t, api_token, n_results) for t in targets]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Bright Data target intelligence (Flash or local).")
    parser.add_argument("--targets", nargs="*", default=None,
                        help="Target names (default: EGFR KRAS HER2 BRAF).")
    parser.add_argument("--n_results", type=int, default=5)
    parser.add_argument("--out", default="outputs/reports/target_intel.json")
    parser.add_argument("--local", action="store_true", help="Force local (no Flash fan-out).")
    args = parser.parse_args()

    targets = args.targets or DEFAULT_TARGETS
    if os.environ.get("GENOTHERMAL_SMOKE"):
        targets = targets[:1]
        logger.info("Smoke mode: querying only the first target.")

    api_token = os.environ.get("BRIGHTDATA_API_TOKEN")
    use_flash = bool(os.environ.get("GENOTHERMAL_FLASH")) and FLASH_AVAILABLE and not args.local
    logger.info("Mode: %s | targets=%s | token=%s", "FLASH" if use_flash else "LOCAL",
                targets, "set" if api_token else "absent (stub)")

    records = (_intel_remote(targets, api_token, args.n_results) if use_flash
               else _intel_local(targets, api_token, args.n_results))

    with open(args.out, "w") as f:
        json.dump(records, f, indent=2)
    n_live = sum(1 for r in records if r.get("source") == "brightdata")
    logger.info("Wrote %s — %d/%d target(s) with LIVE Bright Data data (%d local-stub).",
                args.out, n_live, len(records), len(records) - n_live)
    for r in records:
        logger.info("  - %-10s [%s] %s", r["target"], r["source"], r.get("headline", "")[:90])


if __name__ == "__main__":
    main()
