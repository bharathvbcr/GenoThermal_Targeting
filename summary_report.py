"""Phase 12 — terminal summary AND a single self-contained HTML rollup.

Two outputs from one pass over the artifacts in outputs/:
  1. The original projector-friendly text log (unchanged behaviour).
  2. outputs/reports/summary_report.html — ONE portable file that embeds every figure
     in outputs/figures/ (base64, so it can be emailed/shared as a single file), links the
     interactive 3D structure viewers and the Flash scaling chart, and tabulates the target,
     candidate library, leaderboard, evolution log, and Flash fan-out metrics.

Every section is independently guarded: a missing or malformed artifact degrades to a
"NOT FOUND / not run yet" note instead of crashing the report. Any figure that is not
claimed by a named section still appears in a catch-all gallery, so no figure is ever
orphaned from the report (previously panel_selectivity_heatmap.png and others were).
"""

import base64
import glob
import html
import json
import logging
import os

import pandas as pd

logging.basicConfig(
    level=getattr(logging, os.environ.get("GENOTHERMAL_LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("summary_report.log"), logging.StreamHandler()],
)
logger = logging.getLogger("SummaryReport")

LOGO_PATH = os.path.join("docs", "assets", "geno-thermal-logo.png")
REPORTS_DIR = os.path.join("outputs", "reports")
FIGURES_DIR = os.path.join("outputs", "figures")
HTML_OUT = os.path.join(REPORTS_DIR, "summary_report.html")

# Friendly captions for the figures we know about. Anything not listed still gets shown
# in the gallery with its filename as the caption, so new figures surface automatically.
FIGURE_CAPTIONS = {
    "target_expression.png": "Phase 1 — AlphaGenome expression: normal vs mutated locus",
    "docking_comparison.png": "Phase 2 — Boltz-2 docking: pLDDT / PAE / binder classification",
    "promoter_convergence.png": "Phase 4 — GA promoter fitness convergence",
    "promoter_composition.png": "Phase 4 — Evolved promoter motif / GC composition",
    "evolution_trajectory.png": "Phase 4 — Evolution trajectory (tumor vs normal vs heat)",
    "thermo_profile.png": "Phase 5 — Thermo-switch melting curve",
    "nano_surface_coverage.png": "Phase 6 — Nanoparticle surface coverage (Monte Carlo)",
    "nano_surface.png": "Phase 6 — Nanoparticle surface topology",
    "circuit_heatmap_narrative.png": "Phase 7 — Biological AND-gate kill-switch response",
    "circuit_heatmap.png": "Phase 7 — Biological circuit kill-switch heatmap",
    "therapeutic_window.png": "Phase 9 — Therapeutic window: tumor vs normal kill curves",
    "panel_selectivity_heatmap.png": "Target panel — cross-target selectivity heatmap",
    "flash_scaling.png": "RunPod Flash — 0→N→0 autoscaling concurrency & cost",
}


# --------------------------------------------------------------------------- helpers

def _read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.debug("could not read %s (%s)", path, e)
        return None


def _read_csv(path):
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        logger.debug("could not read %s (%s)", path, e)
        return None


def _first_existing(*paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def _img_data_uri(path):
    """Base64 data URI so the report is one self-contained, shareable file."""
    try:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        return f"data:image/png;base64,{b64}"
    except OSError:
        return None


def _figure_block(path, caption):
    uri = _img_data_uri(path)
    if not uri:
        return ""
    return (
        f'<figure><img src="{uri}" alt="{html.escape(caption)}"/>'
        f'<figcaption>{html.escape(caption)}</figcaption></figure>'
    )


def _df_to_html_table(df, max_rows=25):
    if df is None or df.empty:
        return "<p class='muted'>No data.</p>"
    return df.head(max_rows).to_html(index=False, border=0, classes="data", escape=True)


# --------------------------------------------------------------------------- text log

def log_text_summary():
    """The original terminal summary (unchanged) — still the live-demo narration."""
    logger.info("--- Geno-Thermal Targeting: Project Summary ---")
    logger.info("3D logo asset: %s", LOGO_PATH)

    target = _read_json(os.path.join(REPORTS_DIR, "target_report.json"))
    if target:
        logger.info("Phase 1 Target: %s (Conf: %s) [source=%s]",
                    target.get("gene_id"),
                    target.get("predictions", {}).get("confidence"),
                    target.get("data_source", "unknown"))
        logger.info("Classification: %s", target.get("predictions", {}).get("classification"))
    else:
        logger.warning("Phase 1: outputs/reports/target_report.json NOT FOUND.")

    cand_path = _first_existing(
        os.path.join(REPORTS_DIR, "candidate_library_v2.csv"),
        os.path.join(REPORTS_DIR, "candidate_library.csv"),
    )
    df = _read_csv(cand_path) if cand_path else None
    if df is not None and "plddt_score" in df.columns:
        logger.info("Phase 2 Candidates Generated: %d (from %s)", len(df), os.path.basename(cand_path))
        best = df.loc[df["plddt_score"].idxmax()]
        name = best.get("job_name") or best.get("name") or best.get("structure_path", "unknown")
        logger.info("Best Candidate: %s (pLDDT %.2f)", name, best["plddt_score"])
        if best["plddt_score"] > 80:
            logger.info("SUCCESS: Identified High-Confidence Binder!")
        else:
            logger.warning("No high-confidence binder found (threshold 80).")
    else:
        logger.warning("Phase 2: candidate library NOT FOUND or missing plddt_score.")

    phases = _read_json("flash_metrics.json")
    if phases:
        peak = max((p.get("peak_inflight", 0) for p in phases), default=0)
        cost = sum(p.get("est_cost_usd", 0.0) for p in phases)
        best_speedup = max((p.get("speedup_vs_serial", 0.0) for p in phases), default=0.0)
        logger.info("RunPod Flash fan-out: %d phase(s), peak %d workers, best %.1fx, est $%.4f",
                    len(phases), peak, best_speedup, cost)
    else:
        logger.info("Flash: no flash_metrics.json yet (run with --flash to record fan-out metrics).")

    viewers = glob.glob(os.path.join(FIGURES_DIR, "*_3d.html"))
    logger.info("Interactive 3D viewers: %d", len(viewers))


# --------------------------------------------------------------------------- HTML report

def build_html_report():
    used = set()  # figure basenames already shown in a named section
    sections = []

    def fig(name):
        """Render a named figure and mark it used so it won't duplicate into the gallery."""
        path = os.path.join(FIGURES_DIR, name)
        if os.path.exists(path):
            used.add(name)
            return _figure_block(path, FIGURE_CAPTIONS.get(name, name))
        return ""

    # ---- provenance banner: real APIs vs synthetic fallback, read from the artifacts ----
    target = _read_json(os.path.join(REPORTS_DIR, "target_report.json")) or {}
    intel = _read_json(os.path.join(REPORTS_DIR, "target_intel.json")) or []
    flash = _read_json("flash_metrics.json") or []
    ag_src = target.get("data_source", "unknown")
    bd_src = (intel[0].get("source") if isinstance(intel, list) and intel else "not run")
    flash_ran = bool(flash) and any(p.get("n_ok", 0) > 0 for p in flash)

    def badge(real, label_real, label_fallback):
        cls = "ok" if real else "warn"
        txt = label_real if real else label_fallback
        return f'<span class="badge {cls}">{html.escape(txt)}</span>'

    provenance = (
        '<div class="prov">'
        + badge(ag_src == "API", "AlphaGenome: LIVE API", f"AlphaGenome: {ag_src} (synthetic)")
        + badge(bd_src == "brightdata", "Bright Data: LIVE SERP", f"Bright Data: {bd_src}")
        + badge(flash_ran, "RunPod Flash: REAL fan-out", "RunPod Flash: not recorded")
        + '</div>'
    )

    # ---- Phase 1: target ----
    preds = target.get("predictions", {})
    if target:
        epi = preds.get("epigenetic_profile", {})
        epi_rows = "".join(f"<tr><td>{html.escape(k)}</td><td>{html.escape(str(v))}</td></tr>"
                           for k, v in epi.items())
        body = (
            f"<table class='data'><tr><th>Gene</th><td>{html.escape(str(target.get('gene_id')))}</td></tr>"
            f"<tr><th>Classification</th><td>{html.escape(str(preds.get('classification')))}</td></tr>"
            f"<tr><th>Normal score</th><td>{preds.get('normal_score')}</td></tr>"
            f"<tr><th>Mutated score</th><td>{preds.get('mutated_score')}</td></tr>"
            f"<tr><th>Confidence</th><td>{preds.get('confidence')}</td></tr>"
            f"<tr><th>Data source</th><td>{html.escape(str(ag_src))}</td></tr></table>"
            + (f"<table class='data'><tr><th colspan=2>Epigenetic profile</th></tr>{epi_rows}</table>" if epi_rows else "")
            + fig("target_expression.png")
        )
    else:
        body = "<p class='muted'>target_report.json NOT FOUND — run Phase 1 (genomic_discovery.py).</p>"
    sections.append(("Phase 1 — Genomic Discovery (AlphaGenome)", body))

    # ---- Phase 1.5: target intelligence ----
    if isinstance(intel, list) and intel:
        rows = ""
        for it in intel:
            links = "".join(f"<li><a href='{html.escape(r.get('url',''))}'>{html.escape(r.get('title', r.get('url','')))}</a></li>"
                            for r in it.get("results", []))
            rows += (
                f"<div class='intel'><b>{html.escape(str(it.get('target')))}</b> "
                f"<span class='badge {'ok' if it.get('source')=='brightdata' else 'warn'}'>{html.escape(str(it.get('source')))}</span>"
                f"<p>{html.escape(str(it.get('headline','')))}</p>"
                + (f"<ul>{links}</ul>" if links else "")
                + (f"<p class='muted'>{html.escape(str(it.get('note','')))}</p>" if it.get('note') else "")
                + "</div>"
            )
        sections.append(("Phase 1.5 — Target Intelligence (Bright Data)", rows))

    # ---- Phase 2: candidate library + leaderboard + 3D viewers ----
    cand_path = _first_existing(
        os.path.join(REPORTS_DIR, "candidate_library_v2.csv"),
        os.path.join(REPORTS_DIR, "candidate_library.csv"),
    )
    cand_df = _read_csv(cand_path) if cand_path else None
    lb_df = _read_csv(os.path.join(REPORTS_DIR, "leaderboard.csv"))
    viewers = sorted(glob.glob(os.path.join(FIGURES_DIR, "*_3d.html")))
    viewer_links = "".join(
        f"<li><a href='../figures/{html.escape(os.path.basename(v))}'>{html.escape(os.path.basename(v))}</a></li>"
        for v in viewers
    )
    body = ""
    if cand_df is not None:
        body += f"<h3>Candidate library <span class='muted'>({os.path.basename(cand_path)})</span></h3>" + _df_to_html_table(cand_df)
    if lb_df is not None:
        body += "<h3>Leaderboard</h3>" + _df_to_html_table(lb_df)
    body += fig("docking_comparison.png")
    if viewer_links:
        body += f"<h3>Interactive 3D structure viewers</h3><ul>{viewer_links}</ul>"
    if not body:
        body = "<p class='muted'>No candidate library found — run Phase 2 (boltz_designer.py).</p>"
    sections.append(("Phase 2 — Ligand Engineering (Boltz-2)", body))

    # ---- Phase 4: evolution ----
    evo_df = _read_csv(os.path.join(REPORTS_DIR, "evolution_log.csv"))
    body = (_df_to_html_table(evo_df) if evo_df is not None else "") + fig("promoter_convergence.png") + fig("promoter_composition.png") + fig("evolution_trajectory.png")
    if body:
        sections.append(("Phase 4 — Evolutionary Promoter Design", body))

    # ---- Phases 5-7: thermo / nano / circuit ----
    body = fig("thermo_profile.png") + fig("nano_surface_coverage.png") + fig("nano_surface.png") + fig("circuit_heatmap_narrative.png") + fig("circuit_heatmap.png")
    if body:
        sections.append(("Phases 5–7 — Thermo-switch · Nanoparticle · Bio-circuit", body))

    # ---- Phase 9: therapeutic window ----
    body = fig("therapeutic_window.png")
    if body:
        sections.append(("Phase 9 — Therapeutic Window", body))

    # ---- Flash fan-out ----
    if flash:
        head = ["phase", "resource", "n_jobs", "n_ok", "n_failed", "peak_inflight",
                "speedup_vs_serial", "wall_s", "est_cost_usd"]
        rows = "".join(
            "<tr>" + "".join(f"<td>{html.escape(str(p.get(c, '')))}</td>" for c in head) + "</tr>"
            for p in flash
        )
        table = "<table class='data'><tr>" + "".join(f"<th>{c}</th>" for c in head) + f"</tr>{rows}</table>"
        sections.append(("RunPod Flash — Serverless GPU Fan-out", table + fig("flash_scaling.png")))

    # ---- gallery: every remaining figure, so nothing is orphaned ----
    remaining = sorted(os.path.basename(p) for p in glob.glob(os.path.join(FIGURES_DIR, "*.png"))
                       if os.path.basename(p) not in used)
    if remaining:
        gallery = "".join(_figure_block(os.path.join(FIGURES_DIR, n), FIGURE_CAPTIONS.get(n, n)) for n in remaining)
        sections.append(("Additional Figures", gallery))

    # ---- assemble ----
    logo_uri = _img_data_uri(LOGO_PATH)
    logo_html = f'<img class="logo" src="{logo_uri}" alt="logo"/>' if logo_uri else ""
    body_html = "".join(
        f"<section><h2>{html.escape(title)}</h2>{content}</section>" for title, content in sections
    )
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Geno-Thermal Targeting — Run Report</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 1100px;
         margin: 0 auto; padding: 2rem; background:#0e1116; color:#e6edf3; }}
  header {{ display:flex; align-items:center; gap:1rem; border-bottom:1px solid #30363d; padding-bottom:1rem; }}
  .logo {{ height:64px; }}
  h1 {{ font-size:1.6rem; margin:0; }}
  h2 {{ font-size:1.25rem; border-bottom:1px solid #30363d; padding-bottom:.3rem; margin-top:2.2rem; }}
  h3 {{ font-size:1rem; color:#9db4d8; margin:1.2rem 0 .4rem; }}
  section {{ margin-bottom:1.5rem; }}
  figure {{ margin:1rem 0; }}
  img {{ max-width:100%; border:1px solid #30363d; border-radius:8px; background:#fff; }}
  figcaption {{ color:#8b949e; font-size:.85rem; margin-top:.3rem; }}
  table.data {{ border-collapse:collapse; margin:.6rem 0; font-size:.85rem; width:100%; }}
  table.data th, table.data td {{ border:1px solid #30363d; padding:.35rem .6rem; text-align:left; }}
  table.data th {{ background:#161b22; }}
  .muted {{ color:#8b949e; }}
  .prov {{ display:flex; flex-wrap:wrap; gap:.5rem; margin:1rem 0; }}
  .badge {{ padding:.25rem .6rem; border-radius:999px; font-size:.8rem; font-weight:600; }}
  .badge.ok {{ background:#1a7f37; color:#fff; }}
  .badge.warn {{ background:#9e6a03; color:#fff; }}
  .intel {{ border:1px solid #30363d; border-radius:8px; padding:.8rem; margin:.6rem 0; }}
  a {{ color:#58a6ff; }}
  footer {{ margin-top:3rem; color:#8b949e; font-size:.8rem; border-top:1px solid #30363d; padding-top:1rem; }}
</style></head>
<body>
<header>{logo_html}<div><h1>Geno-Thermal Targeting — Run Report</h1>
<div class="muted">Discover → Design → Verify → Visualize</div></div></header>
{provenance}
{body_html}
<footer>Self-contained report — all figures embedded. Generated by summary_report.py.
Badges reflect what actually ran (real API vs synthetic fallback), read from the artifacts on disk.</footer>
</body></html>"""

    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(HTML_OUT, "w") as f:
        f.write(doc)
    logger.info("HTML report written: %s (%d sections, %d figures embedded)",
                HTML_OUT, len(sections), len(used) + len(remaining))
    return HTML_OUT


def main():
    log_text_summary()
    try:
        build_html_report()
    except Exception as e:  # a report is best-effort; never fail the pipeline over it
        logger.warning("HTML report generation failed (%s) — text summary above is unaffected.", e)


if __name__ == "__main__":
    main()
