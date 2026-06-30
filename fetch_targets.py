"""
Build data/sample_data/targets.csv from EXACT UniProt sequences (no hand-copying -> no transcription errors),
with optional DOMAIN TRIMMING so the panel compares like-for-like regions.

Each target: (name, accession, domain_range). domain_range is (start, end) in 1-indexed UniProt
residue numbering (inclusive), or None for full length. Add/adjust entries and re-run:
    python fetch_targets.py

Default panel uses binding-relevant domains so sizes are comparable to the EGFR ectodomain fragment:
  * HER2_ECD — extracellular domain, res 23-652 (signal 1-22, TM ~653-675)  [UniProt P04626]
  * BRAF_KD  — protein kinase domain, res 457-717                            [UniProt P15056]
  * KRAS     — full (189 aa, already small)                                  [UniProt P01116]
  * EGFR     — the pipeline's ectodomain fragment (matches boltz_designer DEFAULT_TARGET)
Set a target's domain_range to None to use full length instead.
"""

import os
import csv
import time
import logging
import urllib.error
import urllib.request

from boltz_designer import DEFAULT_TARGET  # EGFR ectodomain fragment (single source of truth)

logging.basicConfig(
    level=getattr(logging, os.environ.get("GENOTHERMAL_LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("fetch_targets.log"), logging.StreamHandler()],
)
logger = logging.getLogger("FetchTargets")

# (name, UniProt accession, domain_range | None)
TARGETS = [
    ("KRAS", "P01116", None),
    ("HER2_ECD", "P04626", (23, 652)),
    ("BRAF_KD", "P15056", (457, 717)),
]

OUT = "data/sample_data/targets.csv"


def fetch(acc):
    url = f"https://rest.uniprot.org/uniprotkb/{acc}.fasta"
    logger.info("Fetching UniProt accession %s from %s (timeout=30s)", acc, url)
    _t0 = time.time()
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            status = getattr(r, "status", r.getcode())
            payload = r.read().decode()
        elapsed = time.time() - _t0
        logger.info("UniProt %s: HTTP %s, %d bytes in %.2fs", acc, status, len(payload), elapsed)
    except urllib.error.HTTPError as e:
        logger.error("UniProt %s: HTTP error %s %s after %.2fs", acc, e.code, e.reason, time.time() - _t0)
        raise
    except (urllib.error.URLError, TimeoutError) as e:
        logger.error("UniProt %s: network error after %.2fs: %s", acc, time.time() - _t0, e)
        raise
    lines = payload.splitlines()
    seq = "".join(l for l in lines if not l.startswith(">"))
    if not seq:
        logger.error("UniProt %s: empty sequence (response had %d lines)", acc, len(lines))
        raise RuntimeError(f"No sequence returned for {acc}")
    logger.info("Fetched %s: %d aa", acc, len(seq))
    return seq


def trim(seq, domain_range):
    if domain_range is None:
        logger.debug("No domain trim requested, using full length (%d aa)", len(seq))
        return seq
    start, end = domain_range          # 1-indexed inclusive
    if not (1 <= start <= end <= len(seq)):
        raise ValueError(f"domain {domain_range} out of range for {len(seq)} aa sequence")
    trimmed = seq[start - 1:end]
    logger.debug("Trimmed to res %d-%d: %d aa", start, end, len(trimmed))
    return trimmed


def main():
    logger.info("--- Starting FetchTargets ---")
    rows = [("EGFR", DEFAULT_TARGET)]
    logger.info("EGFR: using DEFAULT_TARGET (%d aa)", len(DEFAULT_TARGET))
    for name, acc, dom in TARGETS:
        logger.info("Processing %s (accession=%s)", name, acc)
        seq = trim(fetch(acc), dom)
        rows.append((name, seq))
        tag = f"res {dom[0]}-{dom[1]}" if dom else "full length"
        logger.info("%s (%s, %s): %d aa", name, acc, tag, len(seq))
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "seq"])
        w.writerows(rows)
    logger.info("Wrote %s with %d targets.", OUT, len(rows))


if __name__ == "__main__":
    main()
