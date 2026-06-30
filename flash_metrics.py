"""
Observability + cost accounting for RunPod Flash fan-out (maps to the hackathon's
named judging criteria: observability and cost awareness).

FanoutMetrics records each job's in-flight interval and the GPU/CPU flavor it ran on,
then derives latency percentiles, throughput, peak concurrency, GPU-seconds, and an
ESTIMATED dollar cost. Results append to flash_metrics.json (one entry per phase);
flash_dashboard.py renders them.

Honesty notes (state these to judges):
  * "peak_inflight" is the number of jobs in flight, a LOWER BOUND on workers used —
    not a queried fleet size (the SDK isn't asked for worker count here).
  * "est_cost_usd" is GPU-seconds x a hardcoded per-second rate; it is an ESTIMATE.
    Update COST_PER_SEC with RunPod's actual published rates before quoting figures.
    It bills only in-flight compute time: it EXCLUDES keep-warm / cold-start / idle-timeout
    seconds, and the per-job interval may include some queue/scheduling wait.
  * "speedup" = total compute-seconds / wall-clock seconds for the phase: a throughput
    speedup vs running the same jobs one-at-a-time on a single box. It is NOT a $ saving —
    the GPU-seconds (and therefore the cost) are the same either way; you pay for the
    same compute, just in parallel and finished sooner.
"""

import os
import json
import time
import logging

logger = logging.getLogger("FlashMetrics")

# Rough on-demand $/second ESTIMATES per resource key. Adjust to RunPod's real rates.
COST_PER_SEC = {
    "A100_80GB": 0.00050,    # ~$1.80/hr
    "RTX_4090": 0.00019,     # ~$0.69/hr
    "cpu5c-4-8": 0.00002,    # ~$0.07/hr (CpuInstanceType.CPU5C_4_8)
}

METRICS_FILE = "flash_metrics.json"


class FanoutMetrics:
    """Lightweight, dependency-free recorder for one phase's fan-out."""

    # Set GENOTHERMAL_LIVE=1 to print a one-line in-flight ticker on every start/done —
    # lights up the "0 -> N -> 0" fan-out beat in real time during the live demo.
    _LIVE = bool(os.environ.get("GENOTHERMAL_LIVE"))

    def __init__(self, phase, resource):
        self.phase = phase
        self.resource = resource          # key into COST_PER_SEC
        self.t0 = time.time()
        self.jobs = []                    # [{start, end, ok}] times relative to t0
        self._inflight = 0                # live counter (driver-side asyncio -> race-free)
        self._done = 0
        self._failed = 0

    def _tick(self):
        if not self._LIVE:
            return
        # \r overwrite keeps it to a single climbing line on the projector.
        print(f"\r[{self.phase}] in-flight: {self._inflight:>3} | done {self._done}/"
              f"{len(self.jobs)}" + (f" | {self._failed} failed" if self._failed else ""),
              end="", flush=True)

    def start(self):
        rec = {"start": time.time() - self.t0, "end": None, "ok": None}
        self.jobs.append(rec)
        self._inflight += 1
        self._tick()
        return rec

    def done(self, rec, ok=True):
        rec["end"] = time.time() - self.t0
        rec["ok"] = ok
        self._inflight = max(0, self._inflight - 1)
        if ok:
            self._done += 1
        else:
            self._failed += 1
        self._tick()
        if self._LIVE and self._inflight == 0:
            print(flush=True)  # newline once the fleet drains back to zero

    def summary(self):
        done = [j for j in self.jobs if j["end"] is not None]
        durs = sorted(j["end"] - j["start"] for j in done)

        def pct(p):
            if not durs:
                return 0.0
            k = min(len(durs) - 1, int(p * len(durs)))
            return durs[k]

        # peak concurrency via a sweep line over (time, +1 start / -1 end) events
        events = []
        for j in done:
            events.append((j["start"], 1))
            events.append((j["end"], -1))
        events.sort()
        cur = peak = 0
        for _, delta in events:
            cur += delta
            peak = max(peak, cur)

        wall = max((j["end"] for j in done), default=0.0)
        gpu_sec = sum(j["end"] - j["start"] for j in done)
        # Split compute time by outcome so a killed/timed-out job's seconds don't silently
        # inflate the "productive" cost figure (a timed-out job still consumed compute, so it
        # stays in the total — but the dashboard can show the failed share separately).
        gpu_sec_ok = sum(j["end"] - j["start"] for j in done if j["ok"])
        gpu_sec_failed = sum(j["end"] - j["start"] for j in done if j["ok"] is False)
        cost = gpu_sec * COST_PER_SEC.get(self.resource, 0.0)
        n_ok = sum(1 for j in done if j["ok"])
        # Throughput counts only SUCCESSFUL jobs (failed jobs produced no result).
        # Speedup = total compute-seconds / wall-clock: how much faster the fan-out finished
        # the same work vs one-at-a-time on a single box (a TIME speedup, not a $ saving).
        speedup = (gpu_sec / wall) if wall else 0.0
        s = {
            "phase": self.phase,
            "resource": self.resource,
            "n_jobs": len(self.jobs),
            "n_ok": n_ok,
            "n_failed": sum(1 for j in self.jobs if j["ok"] is False),
            "wall_s": round(wall, 2),
            "gpu_seconds": round(gpu_sec, 2),
            "gpu_seconds_ok": round(gpu_sec_ok, 2),
            "gpu_seconds_failed": round(gpu_sec_failed, 2),
            "peak_inflight": peak,
            "latency_p50_s": round(pct(0.5), 2),
            "latency_p95_s": round(pct(0.95), 2),
            "throughput_jobs_per_s": round(n_ok / wall, 3) if wall else 0.0,
            "speedup_vs_serial": round(speedup, 2),
            "time_saved_s": round(max(0.0, gpu_sec - wall), 2),
            "est_cost_usd": round(cost, 4),
            "jobs": self.jobs,
        }
        logger.debug("summary: phase=%s, peak=%d, wall=%.2fs, gpu_sec=%.2f, cost=$%.4f",
                     s["phase"], s["peak_inflight"], s["wall_s"], s["gpu_seconds"], s["est_cost_usd"])
        return s

    def save(self, path=METRICS_FILE):
        """Append this phase's summary to the metrics file (creates/updates a list)."""
        logger.info("Saving metrics for phase '%s' to %s", self.phase, path)
        data = []
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)
                if not isinstance(data, list):
                    data = []
            except (json.JSONDecodeError, OSError):
                logger.warning("Could not parse existing metrics file %s; starting fresh.", path)
                data = []
        data.append(self.summary())
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        s = data[-1]
        logger.info("[metrics] %s: %d/%d ok, peak in-flight=%d, wall=%.2fs, "
                    "GPU-s=%.2f, est $%.4f",
                    s["phase"], s["n_ok"], s["n_jobs"], s["peak_inflight"],
                    s["wall_s"], s["gpu_seconds"], s["est_cost_usd"])
        return s
