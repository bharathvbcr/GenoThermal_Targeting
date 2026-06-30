"""
Render the Flash fan-out story from flash_metrics.json into outputs/figures/flash_scaling.png:
  * top panel    — jobs in-flight over time (step chart, one line per phase) = autoscaling
  * bottom panel — estimated cost per phase (bar) with total = cost awareness

Usage:
    python flash_dashboard.py                       # reads flash_metrics.json
    python flash_dashboard.py --metrics other.json --out fig.png
"""

import os
import sys
import json
import argparse
import logging

# Color is OPT-IN to a real terminal only: never emit ANSI to a pipe/file/CI (would corrupt
# the text), and honor the NO_COLOR convention. Disabled output falls back to plain ASCII.
_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def _c(text, code):
    return f"\033[{code}m{text}\033[0m" if _COLOR else text


_LEVEL = getattr(logging, os.environ.get("GENOTHERMAL_LOG_LEVEL", "INFO").upper(), logging.INFO)
logging.basicConfig(
    level=_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("flash_dashboard.log"), logging.StreamHandler()],
)
logger = logging.getLogger("FlashDashboard")


def _inflight_series(jobs):
    """Build (time, count) step points from job intervals via a sweep line."""
    logger.debug("_inflight_series: %d total jobs, %d with end time",
                 len(jobs), sum(1 for j in jobs if j["end"] is not None))
    events = []
    for j in jobs:
        if j["end"] is None:
            continue
        events.append((j["start"], 1))
        events.append((j["end"], -1))
    events.sort()
    times, counts, cur = [0.0], [0], 0
    for t, delta in events:
        times.append(t)
        counts.append(cur)
        cur += delta
        times.append(t)
        counts.append(cur)
    logger.debug("_inflight_series: produced %d time-points, peak=%d",
                 len(times), max(counts) if counts else 0)
    return times, counts


def main():
    parser = argparse.ArgumentParser(description="Render Flash fan-out dashboard.")
    parser.add_argument("--metrics", default="flash_metrics.json")
    parser.add_argument("--out", default="outputs/figures/flash_scaling.png")
    parser.add_argument("--project", type=int, default=0, metavar="N",
                        help="Extrapolate the candidate-screen phase to N candidates "
                             "(prints an ESTIMATED full-run cost + wall-clock).")
    args = parser.parse_args()

    logger.info("Loading metrics from %s", args.metrics)
    if not os.path.exists(args.metrics):
        logger.error("No metrics file '%s'. Run a --flash pipeline first.", args.metrics)
        raise SystemExit(f"No metrics file '{args.metrics}'. Run a --flash pipeline first.")

    with open(args.metrics) as f:
        phases = json.load(f)
    if not phases:
        logger.error("Metrics file '%s' is empty.", args.metrics)
        raise SystemExit("Metrics file is empty.")
    logger.info("Loaded %d phase(s) from %s", len(phases), args.metrics)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_conc, ax_cost) = plt.subplots(2, 1, figsize=(10, 8.5))

    def _unit(resource):
        # CPU endpoints (the GA fan-out, cpu5c-*) bill CPU-seconds, not GPU-seconds.
        return "CPU-s" if "cpu" in str(resource).lower() else "GPU-s"

    # --- concurrency over time ---
    peak_overall = 0
    for ph in phases:
        times, counts = _inflight_series(ph.get("jobs", []))
        if len(times) > 1:
            label = f"{ph['phase']} (peak {ph['peak_inflight']}"
            if ph.get("n_failed"):
                label += f", {ph['n_failed']} failed→local"
            label += ")"
            ax_conc.step(times, counts, where="post", label=label)
            peak_overall = max(peak_overall, ph["peak_inflight"])
    ax_conc.set_title(f"RunPod Flash: jobs in-flight over time (autoscaling 0 → {peak_overall} → 0)")
    ax_conc.set_xlabel("seconds since phase start")
    ax_conc.set_ylabel("jobs in-flight")
    ax_conc.set_ylim(0, max(1, peak_overall + 1))
    ax_conc.grid(True, alpha=0.3)
    ax_conc.legend(loc="upper right", fontsize=8)
    ax_conc.annotate(f"peak {peak_overall} workers — no Dockerfile",
                     xy=(0.5, 0.92), xycoords="axes fraction", ha="center", fontsize=9,
                     bbox=dict(boxstyle="round", fc="#FFF3CD", ec="#E0A800", alpha=0.9))

    # --- estimated cost per phase ---
    names = [ph["phase"] for ph in phases]
    costs = [ph["est_cost_usd"] for ph in phases]
    gpu_secs = [ph["gpu_seconds"] for ph in phases]
    units = [_unit(ph["resource"]) for ph in phases]
    bars = ax_cost.bar(names, costs, color="#4C9F70")
    total = sum(costs)
    ax_cost.set_title(f"Estimated cost per phase  (total ≈ ${total:.4f})")
    ax_cost.set_ylabel("est. USD")
    ax_cost.grid(True, axis="y", alpha=0.3)
    for bar, gs, unit, cost in zip(bars, gpu_secs, units, costs):
        # Value label = $ + compute-seconds with the RIGHT unit, so the CPU GA bar reads
        # clearly even when it's a sliver next to a GPU phase.
        ax_cost.annotate(f"${cost:.4f}\n{gs:.0f} {unit}",
                         (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                         ha="center", va="bottom", fontsize=8)
    if costs and max(costs) > 0:
        ax_cost.set_ylim(0, max(costs) * 1.25)  # headroom for the two-line labels
    plt.setp(ax_cost.get_xticklabels(), rotation=15, ha="right")

    # --- headline banner: the one number judges remember ---
    total_compute = sum(ph.get("gpu_seconds", 0.0) for ph in phases)
    best_speedup = max((ph.get("speedup_vs_serial", 0.0) for ph in phases), default=0.0)
    total_ok = sum(ph.get("n_ok", 0) for ph in phases)
    total_failed = sum(ph.get("n_failed", 0) for ph in phases)
    fig.suptitle(f"0 → {peak_overall} workers   ·   {total_compute:.0f} compute-s   ·   "
                 f"~${total:.4f}   ·   up to {best_speedup:.1f}× vs single box   ·   no Dockerfile",
                 fontsize=12, fontweight="bold")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.out, dpi=120)
    logger.info("Wrote %s", args.out)

    # Boxed ASCII banner to the terminal (the closing beat of the live demo).
    banner = [
        f"peak concurrency : 0 -> {peak_overall} -> 0 workers  (no Dockerfile)",
        f"total compute    : {total_compute:.0f} compute-seconds across {len(phases)} phase(s)",
        f"best speedup     : {best_speedup:.1f}x vs one-at-a-time on a single box",
        f"jobs             : {total_ok} ok"
        + (f", {total_failed} failed -> finished locally" if total_failed else ""),
        f"est cost         : ${total:.4f}  (ESTIMATE — verify COST_PER_SEC vs RunPod rates)",
    ]
    # --- optional projection: extrapolate the observed screen to a full N-candidate run ---
    if args.project and args.project > 0:
        # Pick the candidate-screen phase (a "fold" phase) if present, else the costliest one.
        scaleable = [p for p in phases if p.get("n_jobs")]
        basis = next((p for p in scaleable if "fold" in p["phase"].lower()),
                     max(scaleable, key=lambda p: p.get("est_cost_usd", 0.0), default=None)) \
            if scaleable else None
        if basis:
            n_obs = basis["n_jobs"]
            per_job_s = basis["gpu_seconds"] / n_obs
            per_job_cost = basis["est_cost_usd"] / n_obs
            peak = max(1, basis.get("peak_inflight", 1))
            N = args.project
            proj_compute = per_job_s * N
            proj_cost = per_job_cost * N
            # waves of `peak` concurrent jobs; round up the partial final wave
            import math as _math
            proj_wall = _math.ceil(N / peak) * per_job_s
            banner.append("")
            banner.append(f"PROJECTION ({N} candidates, basis '{basis['phase']}' @ peak {peak}):")
            banner.append(f"  ~{proj_compute:,.0f} compute-s · ~${proj_cost:,.2f} · "
                          f"~{proj_wall / 60:.1f} min wall  (ESTIMATE, linear extrapolation)")

    # ljust BEFORE coloring so ANSI codes never throw off the box alignment.
    w = max(len(b) for b in banner)
    edge = "+" + "-" * (w + 2) + "+"
    print("\n" + _c(edge, "32;1"))
    for b in banner:
        body = b.ljust(w)
        if "failed" in b:
            body = _c(body, "33")           # reliability line in yellow
        elif b and not b.startswith(" ") and ":" not in b[:4]:
            body = _c(body, "32;1")          # headline lines in bold green
        print(f"{_c('|', '32;1')} {body} {_c('|', '32;1')}")
    print(_c(edge, "32;1") + "\n")
    logger.info("Total estimated cost across %d phase(s): $%.4f "
                "(ESTIMATE — verify COST_PER_SEC against RunPod rates)", len(phases), total)


if __name__ == "__main__":
    main()
