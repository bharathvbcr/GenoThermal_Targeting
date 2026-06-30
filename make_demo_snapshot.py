"""
Build an ILLUSTRATIVE Flash fan-out snapshot (outputs/reports/demo_metrics.json) and render it
to outputs/figures/flash_scaling.png — the on-screen FALLBACK if a live `--flash` call stalls
during the demo.

⚠️ HONESTY: the numbers here are a SYNTHETIC, representative shape of the four fan-out
phases (GA fitness, Boltz-2 fold, PPO sweep, MD verify) — NOT a measured run. Treat the
committed outputs/figures/flash_scaling.png as a layout/fallback placeholder and REGENERATE it
from a real captured `flash_metrics.json` at the 3:30 freeze:

    GENOTHERMAL_FLASH=1 python run_pipeline.py --demo      # captures a real flash_metrics.json
    python flash_dashboard.py                              # renders the real flash_scaling.png

Reuses FanoutMetrics.summary() so every derived field (speedup, peak, cost) is computed by
the exact same code the live path uses — only the raw job intervals are synthesized.

    python make_demo_snapshot.py        # writes outputs/reports/demo_metrics.json + outputs/figures/flash_scaling.png
"""

import os
import subprocess
import sys

from flash_metrics import FanoutMetrics

SNAPSHOT = "outputs/reports/demo_metrics.json"

# (phase, resource, n_jobs, start_spread_s, base_dur_s, n_failed) — shaped to tell the story:
# a wide CPU GA fan-out, a mid GPU fold burst, a PPO seed sweep, a single MD verify.
PHASES = [
    ("ga-fitness",  "cpu5c-4-8",  50, 1.5,  4.0, 0),
    ("fold-boltz2", "A100_80GB",   5, 4.0, 42.0, 0),
    ("ppo-sweep",   "RTX_4090",    8, 3.0, 31.0, 1),  # 1 failed -> finished locally (reliability beat)
    ("md-gpu",      "RTX_4090",    1, 0.0, 58.0, 0),
]


def _build():
    if os.path.exists(SNAPSHOT):
        os.remove(SNAPSHOT)  # FanoutMetrics.save() appends; start clean
    for name, resource, n, spread, dur, n_fail in PHASES:
        m = FanoutMetrics(phase=name, resource=resource)
        for i in range(n):
            start = round((i / max(1, n - 1)) * spread, 2)
            # deterministic per-index jitter so the step chart isn't a flat block
            end = round(start + dur + (i % 5) * 0.4, 2)
            m.jobs.append({"start": start, "end": end, "ok": i >= n_fail})
        s = m.save(SNAPSHOT)
        print(f"  {name:<12} peak={s['peak_inflight']:>2}  {s['gpu_seconds']:>6.0f} compute-s  "
              f"{s['speedup_vs_serial']:>5.1f}x  ${s['est_cost_usd']:.4f}"
              + (f"  ({s['n_failed']} failed)" if s["n_failed"] else ""))
    print(f"Wrote illustrative snapshot -> {SNAPSHOT}")


if __name__ == "__main__":
    print("Building ILLUSTRATIVE Flash snapshot (replace with a real run before submit):")
    _build()
    subprocess.run([sys.executable, "flash_dashboard.py",
                    "--metrics", SNAPSHOT, "--out", "outputs/figures/flash_scaling.png"], check=True)
