"""
pipeline_monitor.py — a zero-dependency live progress "pop-up" for the e2e run.

When `run_pipeline.py` runs with --monitor (implied by --demo), it instantiates a
PipelineMonitor. The monitor:

  * pre-seeds the full phase flow as "pending" so the whole pipeline is visible up front,
  * serves a single self-contained HTML page on http://127.0.0.1:<port> (stdlib
    http.server only — no Flask/extra deps; the venue laptop already has Python),
  * auto-opens that page in the default browser (the "pop-up"),
  * and is updated by the orchestrator before/after every phase, so the page shows
    each phase light up running -> success/failed/skipped in real time, with live
    elapsed timers, the discover->design->verify->report flow, and the Flash
    0->N->0 fan-out metrics streamed from flash_metrics.json.

It deliberately reads the SAME artifacts the rest of the pipeline already writes
(`flash_metrics.json`, `pipeline_master.log`) rather than inventing a new data path,
and it writes a `outputs/reports/pipeline_status.json` snapshot that is also a useful
standalone artifact (and lets a second browser/tab attach after the fact).

The whole thing runs in a daemon thread; if anything in here fails it must NEVER take
the pipeline down — every public method swallows its own errors and logs at debug.
"""

import base64
import glob
import json
import logging
import os
import re
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

logger = logging.getLogger("PipelineMonitor")

# Canonical phase flow. Labels MUST match the descriptions run_pipeline.py passes to
# step(); the orchestrator looks phases up by exact label. `stage` groups them into the
# discover -> design -> verify -> report lanes the flow diagram draws.
DEFAULT_PHASES = [
    ("Phase 1: Genomic Discovery", "discover"),
    ("Phase 1.5: Target Intelligence (Bright Data fan-out)", "discover"),
    ("Phase 2: Ligand Engineering (Boltz-2)", "design"),
    ("Phase 5: Thermo-Switch Protein Design", "design"),
    ("Phase 6: Nanoparticle Surface Topology", "design"),
    ("Phase 7: Biological Circuit Integration", "design"),
    ("Phase 4: Evolutionary Promoter Design", "design"),
    ("Phase 8: RL-Driven Sequence Design", "design"),
    ("Phase 9: Physics Verification (OpenMM)", "verify"),
    ("Phase 10: Visualization", "report"),
    ("Phase 11: Flash Fan-out Dashboard", "report"),
    ("Phase 12: Summary Report", "report"),
]

STATUS_PATH = os.path.join("outputs", "reports", "pipeline_status.json")
METRICS_PATH = "flash_metrics.json"
MASTER_LOG_PATH = "pipeline_master.log"

# The monitor serves result files (figures/reports) back to the browser so figures show up
# as thumbnails the instant a phase produces them. File serving is sandboxed to this root.
ARTIFACT_ROOT = os.path.abspath("outputs")
# Non-recursive scan globs (keeps the gallery to top-level results, not the thousands of
# per-model AlphaFold JSONs in outputs/alphafold_results/*/).
ARTIFACT_GLOBS = [
    "outputs/figures/*.png",
    "outputs/figures/*.svg",
    "outputs/reports/*.html",
    "outputs/reports/*.csv",
    "outputs/reports/*.json",
]
# pipeline_status.json is our own bookkeeping, not a result — don't list it in the gallery.
# pipeline_report.html is the static export we write ourselves — don't list it either.
ARTIFACT_SKIP = {"pipeline_status.json", "pipeline_report.html"}

STATIC_REPORT_PATH = os.path.join("outputs", "reports", "pipeline_report.html")


# ---------------------------------------------------------------------------
# Sub-progress parser. The orchestrator tees each child phase's stdout/stderr
# through here so a RUNNING card can show "gen 23/50" or "MC step 40,000/50,000".
# Driven entirely off the phases' existing log lines — no changes to the phases.
# ---------------------------------------------------------------------------
_RE_TOTAL = re.compile(r"\b(?:gens?|Gen)\s*=\s*(\d+)")
_RE_STEP = re.compile(r"Step\s+([\d,]+)\s*/\s*([\d,]+)\s*\((\d+)%\)")
_RE_GEN = re.compile(r"\bGen\s+(\d+)\b")
_RE_BEST = re.compile(r"(Best (?:Fitness|Score)\s*=\s*[-\d.]+)")
_RE_TM = re.compile(r"(Tm\s*=\s*[\d.]+\s*°?C)")
_RE_JOBS = re.compile(r"(\d+)\s*/\s*(\d+)\s+(?:jobs?\s+succeeded|ok\b|target)")


class ProgressTracker:
    """Stateful per-phase parser: .feed(line) -> (frac|None, detail|None)."""

    def __init__(self):
        self.total = None

    def feed(self, line):
        m = _RE_TOTAL.search(line)
        if m:
            try:
                self.total = int(m.group(1))
            except ValueError:
                pass
        m = _RE_STEP.search(line)
        if m:
            cur = int(m.group(1).replace(",", ""))
            tot = int(m.group(2).replace(",", ""))
            return int(m.group(3)) / 100.0, f"step {cur:,}/{tot:,}"
        m = _RE_GEN.search(line)
        if m:
            g = int(m.group(1))
            frac = (g / self.total) if self.total else None
            extra = ""
            b = _RE_BEST.search(line)
            t = _RE_TM.search(line)
            if b:
                extra = " · " + b.group(1)
            elif t:
                extra = " · " + t.group(1)
            label = f"gen {g}" + (f"/{self.total}" if self.total else "")
            return frac, label + extra
        m = _RE_JOBS.search(line)
        if m:
            cur, tot = int(m.group(1)), int(m.group(2))
            if tot:
                return cur / tot, f"{cur}/{tot} jobs done"
        return None, None


class PipelineMonitor:
    def __init__(self, phases=None, port=8765, mode=None, auto_open=True,
                 status_path=STATUS_PATH):
        phases = phases or DEFAULT_PHASES
        self._lock = threading.Lock()
        self._status_path = status_path
        self._mode = mode or {}
        self._started_at = time.time()
        self._ended_at = None
        self._overall = "running"
        self._failures = []
        self._phases = [
            {
                "label": label,
                "stage": stage,
                # short label for the card: drop the "Phase N: " prefix if present
                "short": label.split(":", 1)[1].strip() if ":" in label else label,
                "phase_no": label.split(":", 1)[0].replace("Phase", "").strip()
                if label.lower().startswith("phase") else "",
                "status": "pending",
                "started_at": None,
                "ended_at": None,
                "elapsed": None,
                "detail": None,
                "frac": None,
            }
            for label, stage in phases
        ]
        self._by_label = {p["label"]: p for p in self._phases}

        self._httpd = None
        self._thread = None
        self.port = port
        self._auto_open = auto_open

    # ----- lifecycle -------------------------------------------------------
    def start(self):
        """Start the HTTP server thread and pop the browser open. Best-effort."""
        try:
            self._write_snapshot()
            handler = _make_handler(self)
            # Try the requested port; fall back to an ephemeral port if it's taken.
            try:
                self._httpd = ThreadingHTTPServer(("127.0.0.1", self.port), handler)
            except OSError:
                self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
            self.port = self._httpd.server_address[1]
            self._thread = threading.Thread(
                target=self._httpd.serve_forever, name="pipeline-monitor", daemon=True
            )
            self._thread.start()
            url = f"http://127.0.0.1:{self.port}"
            logger.info("Live progress monitor: %s  (a browser pop-up should open)", url)
            if self._auto_open:
                # Open from a tiny timer so serve_forever is definitely up first.
                threading.Timer(0.4, self._open_browser, args=(url,)).start()
            return url
        except Exception as e:  # never let the monitor break the pipeline
            logger.debug("PipelineMonitor.start failed: %s", e)
            return None

    def _open_browser(self, url):
        try:
            webbrowser.open(url)
        except Exception as e:
            logger.debug("Could not auto-open monitor: %s", e)

    def stop(self):
        try:
            if self._httpd is not None:
                self._httpd.shutdown()
                self._httpd = None
        except Exception as e:
            logger.debug("PipelineMonitor.stop failed: %s", e)

    def linger(self):
        """Keep the page alive after the run so the final flow can be studied.

        Interactive terminal -> block until the user presses Enter, then shut down.
        Non-interactive (CI / MCP / headless) -> return immediately so nothing hangs;
        the daemon server just dies with the process.
        """
        if self._httpd is None:
            return
        try:
            interactive = bool(getattr(sys, "stdin", None)) and sys.stdin.isatty()
        except Exception:
            interactive = False
        if not interactive:
            return
        try:
            logger.info("Live monitor still up at http://127.0.0.1:%d — "
                        "study the final flow, then press Enter here to close it.", self.port)
            input()
        except (EOFError, KeyboardInterrupt):
            pass
        except Exception as e:
            logger.debug("PipelineMonitor.linger failed: %s", e)
        finally:
            self.stop()

    # ----- phase updates (called by run_pipeline) --------------------------
    def start_phase(self, label):
        with self._lock:
            p = self._ensure(label)
            p["status"] = "running"
            p["started_at"] = time.time()
            p["ended_at"] = None
            p["elapsed"] = None
            p["detail"] = None
            p["frac"] = None
            self._write_snapshot_locked()

    def update_progress(self, label, detail=None, frac=None):
        """Live sub-progress for a running phase (e.g. GA generation, MC step %).
        In-memory only — NO file write — because this is called once per child log
        line; the browser polls snapshot() (memory) at 800ms, which is plenty."""
        with self._lock:
            p = self._by_label.get(label)
            if p is None or p["status"] != "running":
                return
            if detail is not None:
                p["detail"] = detail
            if frac is not None:
                p["frac"] = max(0.0, min(1.0, frac))

    def end_phase(self, label, ok, elapsed=None, optional=False):
        with self._lock:
            p = self._ensure(label)
            now = time.time()
            p["ended_at"] = now
            if elapsed is None and p["started_at"] is not None:
                elapsed = now - p["started_at"]
            p["elapsed"] = elapsed
            p["detail"] = None
            p["frac"] = None
            if ok:
                p["status"] = "success"
            else:
                p["status"] = "optional-failed" if optional else "failed"
                self._failures.append(label)
            self._write_snapshot_locked()

    def skip_phase(self, label, reason=""):
        with self._lock:
            p = self._ensure(label)
            p["status"] = "skipped"
            p["reason"] = reason
            self._write_snapshot_locked()

    def skip_remaining(self, reason="aborted"):
        """Mark every still-queued phase as skipped (used when a required phase aborts
        the run, so the flow doesn't leave later phases stuck on 'queued')."""
        with self._lock:
            for p in self._phases:
                if p["status"] == "pending":
                    p["status"] = "skipped"
                    p["reason"] = reason
            self._write_snapshot_locked()

    def finish(self, failures=None):
        with self._lock:
            self._overall = "complete_with_failures" if (failures or self._failures) else "complete"
            if failures:
                self._failures = failures
            self._ended_at = time.time()
            self._write_snapshot_locked()
        # Write the shareable, server-free static report (outside the lock — it reads
        # via snapshot()/helpers which take the lock themselves).
        path = self.write_static_report()
        if path:
            logger.info("Shareable static report: %s  (self-contained — open or send anywhere)", path)

    # ----- internals -------------------------------------------------------
    def _ensure(self, label):
        p = self._by_label.get(label)
        if p is None:  # unknown phase -> append it so nothing is lost
            p = {
                "label": label, "stage": "design",
                "short": label.split(":", 1)[1].strip() if ":" in label else label,
                "phase_no": "", "status": "pending",
                "started_at": None, "ended_at": None, "elapsed": None,
                "detail": None, "frac": None,
            }
            self._phases.append(p)
            self._by_label[label] = p
        return p

    def write_static_report(self, path=STATIC_REPORT_PATH):
        """Dump a single self-contained HTML file with the final state + metrics +
        artifacts (images base64-inlined) so it renders with no server. Best-effort."""
        try:
            snap = self.snapshot()
            try:
                metrics = json.loads(_read_metrics() or "[]")
            except Exception:
                metrics = []
            embedded = []
            for a in _list_artifacts(self._started_at):
                try:
                    with open(a["path"], "rb") as f:
                        raw = f.read()
                    ctype = _content_type(a["path"]).split(";")[0]
                    a2 = dict(a)
                    a2["url"] = "data:%s;base64,%s" % (ctype, base64.b64encode(raw).decode())
                    embedded.append(a2)
                except Exception:
                    continue
            payload = {
                "status": snap, "metrics": metrics,
                "artifacts": embedded, "log": _tail_log(), "static": True,
            }
            # Escape "</" so a stray "</script>" inside any embedded string (log line,
            # phase label) can't prematurely close the bootstrap <script> tag.
            data = json.dumps(payload).replace("</", "<\\/")
            boot = "<script>window.__GENOTHERMAL__=" + data + ";</script>"
            html = _PAGE_HTML.replace("</head>", boot + "\n</head>")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(html)
            return path
        except Exception as e:
            logger.debug("write_static_report failed: %s", e)
            return None

    def snapshot(self):
        with self._lock:
            return {
                "overall": self._overall,
                "started_at": self._started_at,
                "ended_at": self._ended_at,
                "now": time.time(),
                "mode": self._mode,
                "failures": list(self._failures),
                "phases": [dict(p) for p in self._phases],
            }

    def _write_snapshot_locked(self):
        try:
            os.makedirs(os.path.dirname(self._status_path), exist_ok=True)
            with open(self._status_path, "w") as f:
                json.dump({
                    "overall": self._overall,
                    "started_at": self._started_at,
                    "ended_at": self._ended_at,
                    "now": time.time(),
                    "mode": self._mode,
                    "failures": list(self._failures),
                    "phases": self._phases,
                }, f, indent=2)
        except Exception as e:
            logger.debug("Could not write status snapshot: %s", e)

    def _write_snapshot(self):
        with self._lock:
            self._write_snapshot_locked()


# ---------------------------------------------------------------------------
# HTTP handler factory — routes:
#   GET /            -> the live HTML page (embedded below)
#   GET /status.json -> current pipeline status snapshot
#   GET /metrics.json-> raw flash_metrics.json (Flash fan-out), or [] if absent
#   GET /log         -> tail of pipeline_master.log
# ---------------------------------------------------------------------------
def _make_handler(monitor):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # silence per-request console spam
            pass

        def _send(self, body, content_type="application/json"):
            if isinstance(body, str):
                body = body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(body)
            except BrokenPipeError:
                pass

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/" or path == "/index.html":
                self._send(_PAGE_HTML, "text/html; charset=utf-8")
            elif path == "/status.json":
                self._send(json.dumps(monitor.snapshot()))
            elif path == "/metrics.json":
                self._send(_read_metrics())
            elif path == "/artifacts.json":
                self._send(json.dumps(_list_artifacts(monitor._started_at)))
            elif path == "/file":
                self._send_file()
            elif path == "/log":
                self._send(_tail_log(), "text/plain; charset=utf-8")
            else:
                self.send_response(404)
                self.end_headers()

        def _send_file(self):
            """Serve a result file, sandboxed to ARTIFACT_ROOT (outputs/)."""
            from urllib.parse import urlparse, parse_qs
            q = parse_qs(urlparse(self.path).query)
            rel = (q.get("path") or [""])[0]
            real = os.path.realpath(os.path.join(os.getcwd(), rel))
            if not real.startswith(ARTIFACT_ROOT + os.sep) or not os.path.isfile(real):
                self.send_response(404)
                self.end_headers()
                return
            ctype = _content_type(real)
            try:
                with open(real, "rb") as f:
                    body = f.read()
            except Exception:
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(body)
            except BrokenPipeError:
                pass

    return Handler


_CTYPES = {
    ".png": "image/png", ".svg": "image/svg+xml", ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg", ".gif": "image/gif", ".html": "text/html; charset=utf-8",
    ".json": "application/json", ".csv": "text/csv; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
}


def _content_type(path):
    return _CTYPES.get(os.path.splitext(path)[1].lower(), "application/octet-stream")


def _list_artifacts(since):
    """Curated, non-recursive list of result files for the gallery, newest first.
    `fresh` marks files (re)written during this run so the page can highlight them."""
    seen, items = set(), []
    for pattern in ARTIFACT_GLOBS:
        for p in glob.glob(pattern):
            name = os.path.basename(p)
            if name in ARTIFACT_SKIP or p in seen or not os.path.isfile(p):
                continue
            seen.add(p)
            try:
                mtime = os.path.getmtime(p)
                size = os.path.getsize(p)
            except OSError:
                continue
            ext = os.path.splitext(p)[1].lower()
            kind = "image" if ext in (".png", ".svg", ".jpg", ".jpeg", ".gif") else ext.lstrip(".")
            items.append({
                "path": p, "name": name, "kind": kind,
                "mtime": mtime, "size": size,
                "fresh": mtime >= (since - 1.0),
            })
    items.sort(key=lambda d: d["mtime"], reverse=True)
    return items[:30]


def _read_metrics():
    try:
        with open(METRICS_PATH) as f:
            return f.read()
    except Exception:
        return "[]"


def _tail_log(max_lines=160):
    try:
        with open(MASTER_LOG_PATH, "r", errors="replace") as f:
            lines = f.readlines()
        return "".join(lines[-max_lines:])
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# The page. One self-contained file: no external CSS/JS, polls /status.json.
# ---------------------------------------------------------------------------
_PAGE_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Geno-Thermal · Live Pipeline</title>
<style>
  :root{
    --bg:#0b1020; --panel:#121a31; --panel2:#0e1528; --ink:#e7ecff; --muted:#8a97bd;
    --line:#26314f; --pending:#3a4566; --running:#ffce4d; --ok:#36d399; --fail:#f87272;
    --skip:#5a6688; --accent:#6ea8ff;
  }
  *{box-sizing:border-box}
  body{margin:0;background:radial-gradient(1200px 600px at 70% -10%,#16213f 0,#0b1020 60%);
       color:var(--ink);font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
  header{display:flex;align-items:center;gap:16px;padding:16px 22px;border-bottom:1px solid var(--line);
         position:sticky;top:0;background:rgba(11,16,32,.85);backdrop-filter:blur(6px);z-index:5}
  h1{font-size:16px;margin:0;letter-spacing:.3px;font-weight:650}
  h1 .dot{display:inline-block;width:9px;height:9px;border-radius:50%;background:var(--running);
          margin-right:9px;box-shadow:0 0 0 0 rgba(255,206,77,.7);animation:pulse 1.4s infinite}
  .badges{display:flex;gap:8px;margin-left:auto;flex-wrap:wrap}
  .badge{font-size:11px;padding:4px 10px;border:1px solid var(--line);border-radius:999px;color:var(--muted)}
  .badge.on{color:#0b1020;background:var(--accent);border-color:var(--accent);font-weight:600}
  .clock{font-variant-numeric:tabular-nums;font-weight:600;color:var(--accent)}
  main{padding:20px 22px;display:grid;grid-template-columns:1fr 340px;gap:20px;max-width:1280px;margin:0 auto}
  @media(max-width:980px){main{grid-template-columns:1fr}}
  .lane{margin-bottom:18px}
  .lane-h{font-size:11px;text-transform:uppercase;letter-spacing:1.5px;color:var(--muted);margin:0 0 9px 2px}
  .cards{display:flex;flex-wrap:wrap;gap:10px}
  .card{position:relative;min-width:200px;flex:1 1 200px;background:var(--panel);border:1px solid var(--line);
        border-radius:12px;padding:12px 14px;transition:border-color .2s,transform .12s,box-shadow .2s}
  .card .no{position:absolute;top:10px;right:12px;font-size:10px;color:var(--muted);font-variant-numeric:tabular-nums}
  .card .name{font-weight:600;font-size:13.5px;padding-right:26px}
  .card .meta{margin-top:8px;display:flex;align-items:center;gap:8px;color:var(--muted);font-size:12px}
  .card .ico{width:18px;height:18px;border-radius:50%;flex:0 0 auto;display:inline-grid;place-items:center;
             font-size:11px;font-weight:700}
  .pending .ico{background:#22304f;color:#7e8cb5}
  .running{border-color:var(--running);box-shadow:0 0 0 1px rgba(255,206,77,.25),0 8px 30px -12px rgba(255,206,77,.5)}
  .running .ico{background:var(--running);color:#3a2c00;animation:spin 1.1s linear infinite}
  .success{border-color:rgba(54,211,153,.5)}
  .success .ico{background:var(--ok);color:#053827}
  .failed .ico,.optional-failed .ico{background:var(--fail);color:#3a0808}
  .failed{border-color:rgba(248,114,114,.55)}
  .optional-failed{border-color:rgba(248,114,114,.3)}
  .skipped{opacity:.55}
  .skipped .ico{background:#33405f;color:#9fb0d8}
  .card .et{font-variant-numeric:tabular-nums}
  .side{display:flex;flex-direction:column;gap:16px}
  .panel{background:var(--panel2);border:1px solid var(--line);border-radius:12px;padding:14px 15px}
  .panel h2{margin:0 0 12px;font-size:12px;letter-spacing:1px;text-transform:uppercase;color:var(--muted)}
  .stat{display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px dashed var(--line);font-size:13px}
  .stat:last-child{border-bottom:0}
  .stat b{font-variant-numeric:tabular-nums;color:var(--ink)}
  .bar{height:8px;border-radius:6px;background:#16203b;overflow:hidden;margin-top:6px}
  .bar>i{display:block;height:100%;background:linear-gradient(90deg,var(--accent),var(--ok));width:0;transition:width .4s}
  pre.log{margin:0;max-height:280px;overflow:auto;font:11.5px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;
          color:#aab6dd;white-space:pre-wrap;word-break:break-word}
  .overall-ok{--running:var(--ok)} .overall-ok h1 .dot{animation:none;background:var(--ok)}
  .overall-fail h1 .dot{animation:none;background:var(--fail)}
  @keyframes spin{to{transform:rotate(360deg)}}
  @keyframes pulse{0%{box-shadow:0 0 0 0 rgba(255,206,77,.6)}70%{box-shadow:0 0 0 8px rgba(255,206,77,0)}100%{box-shadow:0 0 0 0 rgba(255,206,77,0)}}
  .prog{height:5px;background:#16203b;border-radius:3px;margin:10px 22px 0;overflow:hidden}
  .prog>i{display:block;height:100%;background:linear-gradient(90deg,var(--accent),var(--ok));width:0;transition:width .5s}
  .banner{margin:0 0 18px;padding:13px 16px;border-radius:12px;border:1px solid var(--line);
          display:flex;align-items:center;gap:14px;font-size:13.5px}
  .banner.ok{background:rgba(54,211,153,.08);border-color:rgba(54,211,153,.45)}
  .banner.fail{background:rgba(248,114,114,.08);border-color:rgba(248,114,114,.45)}
  .banner .big{font-size:18px;font-weight:700}
  .banner .pill{font-size:11px;padding:3px 9px;border-radius:999px;border:1px solid var(--line);color:var(--muted)}
  .gallery{display:grid;grid-template-columns:1fr 1fr;gap:9px}
  .art{border:1px solid var(--line);border-radius:9px;overflow:hidden;background:#0c1426;text-decoration:none;
       color:var(--ink);display:flex;flex-direction:column;transition:border-color .2s,transform .1s}
  .art:hover{border-color:var(--accent);transform:translateY(-1px)}
  .art.fresh{border-color:rgba(54,211,153,.6);animation:flash 1.2s ease-out}
  .art img{width:100%;height:78px;object-fit:cover;background:#fff;display:block}
  .art .doc{height:78px;display:grid;place-items:center;font-size:26px;color:var(--accent)}
  .art .cap{padding:6px 8px;font-size:10.5px;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .art.wide{grid-column:1 / -1}
  .gallery .empty{grid-column:1 / -1;color:var(--muted);font-size:12.5px;padding:4px 2px}
  @keyframes flash{0%{box-shadow:0 0 0 0 rgba(54,211,153,.6)}100%{box-shadow:0 0 0 10px rgba(54,211,153,0)}}
  /* per-phase live sub-progress inside a running card */
  .sub{margin-top:9px}
  .sub .sbar{height:5px;border-radius:4px;background:#1a2747;overflow:hidden}
  .sub .sbar>i{display:block;height:100%;background:linear-gradient(90deg,var(--running),var(--ok));width:0;transition:width .4s}
  .sub .sdet{margin-top:5px;font-size:11px;color:var(--running);font-variant-numeric:tabular-nums;
             white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  /* "where the wall-time went" durations chart */
  .durow{display:grid;grid-template-columns:150px 1fr 56px;align-items:center;gap:10px;margin:6px 0;font-size:12px}
  .durow .dlbl{color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .durow .dbar{height:10px;background:#16203b;border-radius:5px;overflow:hidden}
  .durow .dbar>i{display:block;height:100%;border-radius:5px;background:var(--accent)}
  .durow .dbar>i.success{background:linear-gradient(90deg,var(--accent),var(--ok))}
  .durow .dbar>i.failed,.durow .dbar>i.optional-failed{background:var(--fail)}
  .durow .dbar>i.skipped{background:#33405f}
  .durow .dval{text-align:right;font-variant-numeric:tabular-nums;color:var(--ink)}
</style>
</head>
<body>
<header>
  <h1><span class="dot"></span>Geno-Thermal · Live Pipeline</h1>
  <span id="phaseProg" class="badge">0 / 0 phases</span>
  <div class="badges" id="badges"></div>
  <span class="clock" id="clock">00:00</span>
</header>
<div class="prog"><i id="progbar"></i></div>
<main>
  <div>
    <div id="banner"></div>
    <div id="lanes"></div>
    <div id="durations"></div>
  </div>
  <div class="side">
    <div class="panel">
      <h2>⚡ Flash fan-out</h2>
      <div id="flash">
        <div class="stat"><span>Waiting for first GPU fan-out…</span><b></b></div>
      </div>
    </div>
    <div class="panel">
      <h2>🖼 Artifacts</h2>
      <div class="gallery" id="gallery"><div class="empty">Figures and reports appear here as each phase produces them…</div></div>
    </div>
    <div class="panel">
      <h2>Live log</h2>
      <pre class="log" id="log">…</pre>
    </div>
  </div>
</main>
<script>
const STAGES = [["discover","① Discover"],["design","② Design"],["verify","③ Verify"],["report","④ Report"]];
const ICON = {success:"✓",failed:"✕","optional-failed":"!",skipped:"–",running:"",pending:""};

function fmt(s){ if(s==null||isNaN(s)) return "";
  s=Math.max(0,Math.floor(s)); const m=Math.floor(s/60), ss=s%60;
  return (m<10?"0":"")+m+":"+(ss<10?"0":"")+ss; }

function render(st){
  // overall body class
  document.body.className = st.overall==="complete" ? "overall-ok"
    : st.overall==="complete_with_failures" ? "overall-fail" : "";

  // header badges
  const m = st.mode||{};
  document.getElementById("badges").innerHTML =
    `<span class="badge ${m.flash?'on':''}">⚡ Flash ${m.flash?'ON':'off'}</span>`+
    `<span class="badge ${m.smoke?'on':''}">smoke ${m.smoke?'ON':'off'}</span>`+
    `<span class="badge">${st.overall.replace(/_/g,' ')}</span>`;

  // global clock + progress
  const terminal = ["success","failed","optional-failed","skipped"];
  const done = st.phases.filter(p=>terminal.includes(p.status)).length;
  const total = st.phases.length;
  const running = st.phases.find(p=>p.status==="running");
  document.getElementById("phaseProg").textContent = done+" / "+total+" phases";
  document.getElementById("progbar").style.width = total? (100*done/total)+"%" : "0%";
  // freeze the clock at the final wall-time once the run ends
  const elapsed = ((st.ended_at || st.now) - st.started_at);
  document.getElementById("clock").textContent = fmt(elapsed);

  // dynamic tab title — track progress even from a backgrounded tab
  const pct = total? Math.round(100*done/total):0;
  document.title = st.overall==="complete" ? "✓ Pipeline complete · Geno-Thermal"
    : st.overall==="complete_with_failures" ? "✕ Pipeline finished (failures) · Geno-Thermal"
    : (running ? `▶ ${pct}% · ${running.short.slice(0,28)}` : `▶ ${pct}% · Geno-Thermal`);

  // completion banner
  const okN = st.phases.filter(p=>p.status==="success").length;
  const failN = st.phases.filter(p=>["failed","optional-failed"].includes(p.status)).length;
  const skipN = st.phases.filter(p=>p.status==="skipped").length;
  const bn = document.getElementById("banner");
  if(st.overall==="running"){ bn.innerHTML=""; }
  else {
    const cls = st.overall==="complete" ? "ok":"fail";
    const icon = st.overall==="complete" ? "✓":"✕";
    bn.innerHTML = `<div class="banner ${cls}"><span class="big">${icon}</span>
      <div><b>Pipeline ${st.overall==="complete"?"complete":"finished with failures"}</b> in ${fmt(elapsed)}
      &nbsp;<span class="pill">${okN} ok</span> <span class="pill">${failN} failed</span>
      <span class="pill">${skipN} skipped</span>${
        st.failures && st.failures.length ? `<br><span style="color:var(--fail)">${st.failures.join(" · ")}</span>`:""}</div></div>`;
  }

  // lanes
  let html="";
  for(const [stage,title] of STAGES){
    const ph = st.phases.filter(p=>p.stage===stage);
    if(!ph.length) continue;
    html += `<div class="lane"><div class="lane-h">${title}</div><div class="cards">`;
    for(const p of ph){
      let et="";
      if(p.status==="running" && p.started_at) et = fmt(st.now-p.started_at)+" elapsed";
      else if(p.elapsed!=null) et = p.elapsed.toFixed(2)+"s";
      else if(p.status==="skipped") et = "skipped";
      else et = "queued";
      let sub="";
      if(p.status==="running" && (p.detail || p.frac!=null)){
        const w = p.frac!=null ? Math.round(100*p.frac) : null;
        sub = `<div class="sub">${w!=null?`<div class="sbar"><i style="width:${w}%"></i></div>`:""}
          ${p.detail?`<div class="sdet">${p.detail}${w!=null?` · ${w}%`:""}</div>`:""}</div>`;
      }
      html += `<div class="card ${p.status}">
        <div class="no">${p.phase_no||""}</div>
        <div class="name">${p.short}</div>
        <div class="meta"><span class="ico">${ICON[p.status]||""}</span>
        <span class="et">${et}</span></div>${sub}</div>`;
    }
    html += `</div></div>`;
  }
  document.getElementById("lanes").innerHTML = html;
  renderDurations(st);
}

function renderDurations(st){
  const el = document.getElementById("durations");
  const ph = st.phases.filter(p=>p.elapsed!=null && p.elapsed>0).sort((a,b)=>b.elapsed-a.elapsed);
  if(!ph.length){ el.innerHTML=""; return; }
  const max = Math.max.apply(null, ph.map(p=>p.elapsed));
  const totalT = ph.reduce((s,p)=>s+p.elapsed,0);
  let h = `<div class="panel" style="margin-top:18px"><h2>⏱ Where the wall-time went · ${totalT.toFixed(1)}s of phase time</h2>`;
  for(const p of ph){
    const w = max? Math.max(2,100*p.elapsed/max):0;
    h += `<div class="durow"><span class="dlbl" title="${p.short}">${p.short}</span>
      <span class="dbar"><i class="${p.status}" style="width:${w}%"></i></span>
      <span class="dval">${p.elapsed.toFixed(2)}s</span></div>`;
  }
  el.innerHTML = h + `</div>`;
}

function renderFlash(metrics){
  if(!Array.isArray(metrics) || !metrics.length) return;
  let jobs=0, ok=0, gpu=0, cost=0, peak=0, saved=0, sp=0, n=0;
  for(const m of metrics){
    jobs+=m.n_jobs||0; ok+=m.n_ok||0; gpu+=m.gpu_seconds||0; cost+=m.est_cost_usd||0;
    peak=Math.max(peak,m.peak_inflight||0); saved+=m.time_saved_s||0;
    if(m.speedup_vs_serial){ sp+=m.speedup_vs_serial; n++; }
  }
  const okPct = jobs? Math.round(100*ok/jobs):0;
  document.getElementById("flash").innerHTML =
    `<div class="stat"><span>GPU jobs OK</span><b>${ok} / ${jobs} (${okPct}%)</b></div>
     <div class="bar"><i style="width:${okPct}%"></i></div>
     <div class="stat"><span>Peak workers in-flight</span><b>${peak}</b></div>
     <div class="stat"><span>GPU-seconds</span><b>${gpu.toFixed(1)}s</b></div>
     <div class="stat"><span>Avg speed-up vs serial</span><b>${n?(sp/n).toFixed(1):'—'}×</b></div>
     <div class="stat"><span>Wall-time saved</span><b>${saved.toFixed(0)}s</b></div>
     <div class="stat"><span>Est. cost</span><b>$${cost.toFixed(4)}</b></div>`;
}

const DOCICON = {html:"📄",csv:"📊",json:"🧬"};
function renderGallery(arts){
  const el = document.getElementById("gallery");
  if(!Array.isArray(arts) || !arts.length){
    el.innerHTML = `<div class="empty">Figures and reports appear here as each phase produces them…</div>`;
    return;
  }
  let html="";
  for(const a of arts){
    const href = a.url ? a.url : "/file?path="+encodeURIComponent(a.path);
    const fresh = a.fresh ? " fresh":"";
    const wide = a.name==="summary_report.html" ? " wide":"";
    if(a.kind==="image"){
      html += `<a class="art${fresh}${wide}" href="${href}" target="_blank" title="${a.name}">
        <img loading="lazy" src="${href}"/><div class="cap">${a.name}</div></a>`;
    } else {
      const tag = a.name==="summary_report.html" ? "▶ Open unified report" : a.name;
      html += `<a class="art${fresh}${wide}" href="${href}" target="_blank" title="${a.name}">
        <div class="doc">${DOCICON[a.kind]||"📁"}</div><div class="cap">${tag}</div></a>`;
    }
  }
  el.innerHTML = html;
}

function setLog(t){
  const el = document.getElementById("log");
  const atBottom = el.scrollTop+el.clientHeight >= el.scrollHeight-20;
  el.textContent = t || "(no log yet)";
  if(atBottom) el.scrollTop = el.scrollHeight;
}

async function tick(){
  try{
    const st = await (await fetch("/status.json",{cache:"no-store"})).json();
    render(st);
    if(st.overall!=="running"){ document.querySelector("h1 .dot").style.animation="none"; }
  }catch(e){}
  try{ renderFlash(await (await fetch("/metrics.json",{cache:"no-store"})).json()); }catch(e){}
  try{ renderGallery(await (await fetch("/artifacts.json",{cache:"no-store"})).json()); }catch(e){}
  try{ setLog(await (await fetch("/log",{cache:"no-store"})).text()); }catch(e){}
}

// Static export (pipeline_report.html) inlines everything in window.__GENOTHERMAL__ and
// renders once with no server / no fetch. The live page (no EMBED) polls instead.
const EMBED = window.__GENOTHERMAL__ || null;
if(EMBED){
  render(EMBED.status);
  renderFlash(EMBED.metrics||[]);
  renderGallery(EMBED.artifacts||[]);
  setLog(EMBED.log||"");
  document.querySelector("h1 .dot").style.animation="none";
  // mark the page as a saved snapshot
  const b=document.getElementById("phaseProg"); if(b) b.textContent += " · saved snapshot";
} else {
  tick(); setInterval(tick, 800);
}
</script>
</body>
</html>
"""
