"""
Multi-oncogene target panel — the headline Flash fan-out showcase.

Folds every candidate against EVERY target (a targets x candidates matrix) so the fleet
scales to |targets| * |candidates| concurrent jobs, then scores each candidate's
SELECTIVITY for an intended target:

    selectivity_margin = ipTM(intended target) - max(ipTM over all off-targets)

A high margin = binds the intended oncogene but not the others — a scientifically real
design axis the single-target pipeline can't express. Reuses boltz_designer's fan-out
(timeouts, drop-and-flag, metrics) per target, so reliability/observability come for free.

METHODS / HONESTY NOTE: `selectivity_margin` is an interface-CONFIDENCE proxy (a
difference of Boltz-2 ipTM*100 scores), NOT a difference of measured binding affinities.
It ranks which candidate the model is most confident docks the intended target over the
off-targets. For small-molecule candidates that carry the affinity head's prediction, the
panel ALSO emits `affinity_selectivity_margin` (a difference of predicted log-IC50 values,
lower = stronger), which is the affinity-grounded selectivity to prefer for ligands.

Usage:
    python target_panel.py --intended EGFR                       # fan out on Flash
    python target_panel.py --intended EGFR --local               # fold in-process
    python target_panel.py --targets data/sample_data/targets.csv \
        --candidates data/sample_data/candidates.csv --intended EGFR
"""

import os
import asyncio
import argparse
import logging

import pandas as pd

import boltz_designer as bd

logging.basicConfig(
    level=getattr(logging, os.environ.get("GENOTHERMAL_LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("target_panel.log"), logging.StreamHandler()],
)
logger = logging.getLogger("TargetPanel")


def _load(path, kind):
    logger.info("Loading %s from %s", kind, path)
    df = pd.read_csv(path)
    if "name" not in df.columns:
        raise ValueError(f"{kind} CSV needs a 'name' column.")
    logger.info("Loaded %d %s rows.", len(df), kind)
    return df


def run_panel(targets, candidates, use_msa_server, local):
    """Fold candidates against each target; return a flat list of result records tagged by target."""
    logger.info("run_panel: %d targets x %d candidates, mode=%s, use_msa=%s",
                len(targets), len(candidates), "LOCAL" if local else "FLASH", use_msa_server)
    rows = []
    for t in targets:
        logger.info("=== Folding %d candidates against target %s ===", len(candidates), t["name"])
        if local:
            recs = bd._fold_local(t["seq"], candidates, use_msa_server)
        else:
            recs = asyncio.run(bd._fold_remote(t["seq"], candidates, use_msa_server))
        logger.info("Target %s: %d fold results returned.", t["name"], len(recs))
        for r in recs:
            r["target"] = t["name"]
            rows.append(r)
    logger.info("run_panel complete: %d total rows.", len(rows))
    return rows


def selectivity_table(rows, intended):
    """Pivot ipTM by (candidate, target) and rank candidates by selectivity for `intended`."""
    logger.info("Building selectivity table from %d result rows (intended=%s).", len(rows), intended)
    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("No rows to pivot — returning empty selectivity table.")
        return df, df
    matrix = df.pivot_table(index="job_name", columns="target", values="plddt_score", aggfunc="max")
    logger.info("Pivot matrix shape: %s", matrix.shape)

    if intended not in matrix.columns:
        raise ValueError(f"Intended target '{intended}' not among folded targets {list(matrix.columns)}.")

    off_targets = [c for c in matrix.columns if c != intended]
    logger.info("Off-targets: %s", off_targets)
    intended_score = matrix[intended]
    best_off = matrix[off_targets].max(axis=1) if off_targets else 0.0

    ranked = pd.DataFrame({
        "intended_iptm": intended_score,
        "best_offtarget_iptm": best_off,
        "selectivity_margin": intended_score - best_off,
    }).sort_values("selectivity_margin", ascending=False)
    logger.info("Ranked %d candidates by selectivity margin.", len(ranked))

    # --- ADDITIVE: affinity-based selectivity for small molecules (Boltz-2 affinity head) ---
    # affinity_pred_value is predicted log(IC50): LOWER = stronger binder. A candidate is
    # selective when it binds the intended target MUCH more strongly (lower) than any off-target.
    # The most concerning off-target is therefore the MIN log-IC50 over off-targets, and
    #   affinity_selectivity_margin = strongest_offtarget_logIC50 - intended_logIC50   (higher = better).
    # This is measured-affinity-grounded selectivity, complementing the confidence margin above.
    # Left empty (NaN) for peptide-only panels, so the confidence path is completely unchanged.
    if "affinity_pred_value" in df.columns and df["affinity_pred_value"].notna().any():
        aff = df.pivot_table(index="job_name", columns="target",
                             values="affinity_pred_value", aggfunc="min")
        if intended in aff.columns:
            aff_off = [c for c in aff.columns if c != intended]
            intended_aff = aff[intended]
            strongest_off = aff[aff_off].min(axis=1) if aff_off else float("nan")
            ranked = ranked.join(pd.DataFrame({
                "intended_affinity_logIC50": intended_aff,
                "strongest_offtarget_logIC50": strongest_off,
                "affinity_selectivity_margin": strongest_off - intended_aff,
            }))
            logger.info("Added affinity-based selectivity for %d candidate(s) with affinity data.",
                        int(intended_aff.notna().sum()))
    return matrix, ranked


def render_heatmap(matrix, ranked, intended, out):
    """Render the candidates x targets confidence matrix as a heatmap PNG — the picture
    for the novelty beat. Rows are sorted by selectivity for `intended` (best at top) and
    the intended target column is outlined in red."""
    logger.info("Rendering selectivity heatmap -> %s", out)
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in ranked.index if r in matrix.index]
    cols = [intended] + [c for c in matrix.columns if c != intended]  # intended first
    m = matrix.reindex(index=rows, columns=cols)
    data = m.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(2.0 + 1.3 * len(cols), 1.5 + 0.55 * max(1, len(rows))))
    im = ax.imshow(data, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    ax.set_xlabel("target")
    ax.set_ylabel(f"candidate (sorted by selectivity for {intended})")
    ax.set_title(f"Selectivity panel — interface confidence (ipTM×100)\n"
                 f"rows sorted by margin · red column = intended ({intended})")
    vmax = np.nanmax(data) if data.size and not np.all(np.isnan(data)) else 1.0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if v < vmax * 0.6 else "black")
    ax.add_patch(plt.Rectangle((-0.5, -0.5), 1, len(rows), fill=False, edgecolor="red", lw=2.5))
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("interface confidence (ipTM×100) — NOT measured affinity")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    logger.info("Wrote %s", out)


def main():
    parser = argparse.ArgumentParser(description="Multi-oncogene target x candidate selectivity panel.")
    parser.add_argument("--targets", default="data/sample_data/targets.csv")
    parser.add_argument("--candidates", default="data/sample_data/candidates.csv")
    parser.add_argument("--intended", default="EGFR", help="Target the candidate should be selective FOR.")
    parser.add_argument("--local", action="store_true", help="Fold in-process instead of on Flash.")
    parser.add_argument("--no_msa_server", action="store_true")
    parser.add_argument("--out", default="outputs/reports/panel_selectivity.csv")
    parser.add_argument("--heatmap", default="outputs/figures/panel_selectivity_heatmap.png",
                        help="Render the selectivity matrix as a heatmap PNG (set '' to skip).")
    args = parser.parse_args()

    targets = _load(args.targets, "targets").to_dict("records")
    cand_df = _load(args.candidates, "candidates")
    if not ({"seq", "smiles"} & set(cand_df.columns)):
        raise ValueError("candidates CSV needs 'seq' (peptide) or 'smiles' (small molecule).")
    candidates = cand_df.to_dict("records")

    logger.info("Panel: %d targets x %d candidates = up to %d concurrent fold jobs",
                len(targets), len(candidates), len(targets) * len(candidates))

    rows = run_panel(targets, candidates, not args.no_msa_server, args.local)
    matrix, ranked = selectivity_table(rows, args.intended)

    if ranked.empty:
        logger.warning("No fold results — nothing to rank.")
        return

    matrix.to_csv(args.out.replace(".csv", "_matrix.csv"))
    ranked.to_csv(args.out)
    logger.info("Wrote %s and %s", args.out, args.out.replace(".csv", "_matrix.csv"))

    if args.heatmap:
        try:
            render_heatmap(matrix, ranked, args.intended, args.heatmap)
        except Exception as e:  # a plotting hiccup must never sink the panel phase
            logger.warning("Heatmap render skipped (%s).", e)
    top = ranked.index[0]
    logger.info("Most %s-selective by CONFIDENCE: %s (margin=%.1f: %s ipTM=%.1f vs best off-target %.1f)",
                args.intended, top, ranked.loc[top, "selectivity_margin"],
                args.intended, ranked.loc[top, "intended_iptm"], ranked.loc[top, "best_offtarget_iptm"])

    # Affinity-grounded selectivity winner (small molecules) — complements the confidence margin.
    if "affinity_selectivity_margin" in ranked.columns and ranked["affinity_selectivity_margin"].notna().any():
        aff_ranked = (ranked.dropna(subset=["affinity_selectivity_margin"])
                            .sort_values("affinity_selectivity_margin", ascending=False))
        atop = aff_ranked.index[0]
        logger.info("Most %s-selective by AFFINITY: %s (margin=%.2f log-IC50 units: "
                    "intended=%.2f vs strongest off-target=%.2f; lower log-IC50 = stronger)",
                    args.intended, atop, aff_ranked.loc[atop, "affinity_selectivity_margin"],
                    aff_ranked.loc[atop, "intended_affinity_logIC50"],
                    aff_ranked.loc[atop, "strongest_offtarget_logIC50"])


if __name__ == "__main__":
    main()
