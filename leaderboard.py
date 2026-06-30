"""
Unified design leaderboard — ONE ranked board across both binder modalities.

The Boltz-2 screen emits peptide and small-molecule hits into a single candidate library
(distinguished by the `mode` column: 'peptide' vs 'ligand'). This collapses them into one
ranked table for the novelty beat:

  * primary rank  = interface CONFIDENCE (plddt_score = ipTM*100), common to every row.
  * for ligands   = the affinity head's predicted log(IC50) is shown alongside
                    (LOWER = stronger binder) and used to rank within the ligand group.

HONESTY: plddt_score is interface confidence (ipTM*100), NOT measured affinity — see
METHODS.md. The board is robust to the older AlphaFold-era library schema (no `mode`/
affinity columns): those rows are treated as peptide/confidence-only.

    python leaderboard.py                                  # reads outputs/reports/candidate_library.csv
    python leaderboard.py --input outputs/reports/candidate_library_v2.csv --out outputs/reports/leaderboard.csv
"""

import os
import argparse
import logging

import pandas as pd

_LEVEL = getattr(logging, os.environ.get("GENOTHERMAL_LOG_LEVEL", "INFO").upper(), logging.INFO)
logging.basicConfig(
    level=_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("leaderboard.log"), logging.StreamHandler()],
)
logger = logging.getLogger("Leaderboard")

DEFAULT_INPUTS = ["outputs/reports/candidate_library.csv", "outputs/reports/candidate_library_v2.csv"]


def _classify(conf):
    """Confidence bucket (mirrors flash_boltz.classify_binding). Interface confidence, NOT affinity."""
    if conf >= 80:
        return "STRONG_BINDER"
    if conf >= 60:
        return "MODERATE_BINDER"
    if conf >= 40:
        return "WEAK_BINDER"
    return "NON_BINDER"


def build_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a candidate library (either schema) into one ranked board."""
    out = pd.DataFrame()
    out["name"] = df.get("job_name", df.get("name", pd.Series(range(len(df)))).astype(str))
    out["mode"] = df["mode"] if "mode" in df.columns else "peptide"  # old libs were peptide docks
    out["confidence"] = pd.to_numeric(df.get("plddt_score"), errors="coerce")  # ipTM*100
    out["binding_class"] = (df["binding_class"] if "binding_class" in df.columns
                            else out["confidence"].map(lambda c: _classify(c) if pd.notna(c) else "NA"))
    # Affinity columns only exist for small molecules; keep them if present.
    out["affinity_log_ic50"] = (pd.to_numeric(df["affinity_pred_value"], errors="coerce")
                                if "affinity_pred_value" in df.columns else float("nan"))
    out["affinity_prob"] = (pd.to_numeric(df["affinity_probability"], errors="coerce")
                            if "affinity_probability" in df.columns else float("nan"))

    # Collapse to one row per design (libraries often hold multiple model rows per candidate):
    # keep the best-confidence model so the board ranks distinct designs, not duplicates.
    n_before = len(out)
    out = (out.sort_values("confidence", ascending=False, na_position="last")
              .drop_duplicates(subset="name", keep="first"))
    if len(out) < n_before:
        logger.info("Collapsed %d model row(s) -> %d distinct design(s) (kept best confidence each).",
                    n_before, len(out))

    # Primary unified ranking: interface confidence (descending). Common to every row.
    out = out.sort_values("confidence", ascending=False, na_position="last").reset_index(drop=True)
    out.insert(0, "rank", range(1, len(out) + 1))
    # Secondary rank WITHIN each modality (peptides by confidence, ligands by affinity).
    out["rank_in_mode"] = 0
    for mode, grp in out.groupby("mode"):
        if mode == "ligand" and grp["affinity_log_ic50"].notna().any():
            order = grp.sort_values("affinity_log_ic50", ascending=True).index  # lower = stronger
        else:
            order = grp.sort_values("confidence", ascending=False).index
        for i, idx in enumerate(order, start=1):
            out.loc[idx, "rank_in_mode"] = i
    return out


def render(board: pd.DataFrame):
    """Print the leaderboard to the terminal (the novelty beat's closing table)."""
    if board.empty:
        logger.warning("Leaderboard is empty — nothing to rank.")
        return
    n_pep = int((board["mode"] == "peptide").sum())
    n_lig = int((board["mode"] == "ligand").sum())
    logger.info("=== Unified design leaderboard: %d candidate(s) — %d peptide, %d small-molecule ===",
                len(board), n_pep, n_lig)
    logger.info("Ranked by interface confidence (ipTM*100). Affinity shown for ligands "
                "(log-IC50, lower = stronger). Confidence is NOT measured affinity — see METHODS.md.")
    logger.info("%-4s %-26s %-9s %6s  %-15s %s", "#", "design", "mode", "conf", "class", "affinity(logIC50/prob)")
    for _, r in board.iterrows():
        aff = ""
        if pd.notna(r["affinity_log_ic50"]):
            aff = f"{r['affinity_log_ic50']:.2f}"
            if pd.notna(r["affinity_prob"]):
                aff += f" / p={r['affinity_prob']:.2f}"
        conf = f"{r['confidence']:.1f}" if pd.notna(r["confidence"]) else "  NA"
        logger.info("%-4d %-26s %-9s %6s  %-15s %s",
                    r["rank"], str(r["name"])[:26], r["mode"], conf, r["binding_class"], aff)
    top = board.iloc[0]
    logger.info("Top by interface confidence: %s (%s, conf=%.1f)",
                top["name"], top["mode"], top["confidence"] if pd.notna(top["confidence"]) else float("nan"))


def main():
    parser = argparse.ArgumentParser(description="Unified peptide + small-molecule design leaderboard.")
    parser.add_argument("--input", default=None,
                        help="Candidate library CSV (default: first of outputs/reports/candidate_library.csv / _v2.csv).")
    parser.add_argument("--out", default="outputs/reports/leaderboard.csv")
    args = parser.parse_args()

    path = args.input
    if path is None:
        path = next((p for p in DEFAULT_INPUTS if os.path.exists(p)), None)
    if not path or not os.path.exists(path):
        logger.error("No candidate library found (looked for %s). Run boltz_designer.py first.",
                     args.input or " / ".join(DEFAULT_INPUTS))
        raise SystemExit(1)

    logger.info("Loading candidate library: %s", path)
    df = pd.read_csv(path)
    logger.info("Loaded %d candidate(s) with columns: %s", len(df), list(df.columns))

    board = build_leaderboard(df)
    render(board)
    board.to_csv(args.out, index=False)
    logger.info("Wrote leaderboard -> %s", args.out)


if __name__ == "__main__":
    main()
