import subprocess
import sys
import os
import time
import argparse
import logging

from env_utils import load_dotenv

# Load .env ONCE here, before any phase is spawned. Child phases are subprocesses that
# inherit this process's os.environ, so loading the keys here is what finally lets the
# CLI pipeline reach the real AlphaGenome / NVIDIA / Bright Data / RunPod APIs instead
# of silently falling back to synthetic data.
_ENV_LOADED = load_dotenv()

# Setup logging for the master pipeline. The console gets a clean, projector-friendly
# ">> message" format so the orchestrator's own STEP/SUCCESS lines stand out among the
# child phases' full-format logs; the file keeps the full timestamped format for debugging.
_LEVEL = getattr(logging, os.environ.get("GENOTHERMAL_LOG_LEVEL", "INFO").upper(), logging.INFO)
_file_handler = logging.FileHandler("pipeline_master.log")
_file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(logging.Formatter(">> %(message)s"))
logging.basicConfig(level=_LEVEL, handlers=[_file_handler, _console_handler])
logger = logging.getLogger("PipelineMaster")

def run_step(command, description):
    # Use the SAME interpreter for child phases (the venue's `python` may lack our deps).
    if command.startswith("python "):
        command = f'"{sys.executable}" ' + command[len("python "):]
    logger.info(f"--- STEP: {description} ---")
    logger.info(f"Executing: {command}")
    
    start_time = time.time()
    try:
        # Inherit the terminal (NO capture_output) so each phase's live logs scroll on screen
        # as they happen — the GA fan-out's per-generation progress and the
        # "[metrics] peak in-flight=N" lines ARE the demo's headline beat. Capturing them
        # buffered them until the phase ended and hid them at the default INFO level.
        # The child writes its own *.log file too, and stderr now goes straight to the
        # console, so the failure path no longer needs e.stdout/e.stderr.
        subprocess.run(command, shell=True, check=True)
        elapsed = time.time() - start_time
        logger.info(f"SUCCESS: {description} (Time: {elapsed:.2f}s)")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        logger.error(f"FAILED: {description} after {elapsed:.2f}s (exit {e.returncode}) "
                     f"— see this phase's output above and its *.log file.")
        return False

def check_openmm():
    try:
        import openmm
        logger.debug("check_openmm: openmm available (version %s)", getattr(openmm, "__version__", "?"))
        return True
    except ImportError:
        logger.debug("check_openmm: openmm not installed.")
        return False

def main():
    parser = argparse.ArgumentParser(description="Geno-Thermal Targeting master pipeline.")
    parser.add_argument("--smoke", action="store_true",
                        help="Tiny GA/PPO/folding workload for a fast end-to-end demo (<1 min).")
    parser.add_argument("--flash", action="store_true",
                        help="Fan compute out on RunPod Flash (same as GENOTHERMAL_FLASH=1).")
    parser.add_argument("--demo", action="store_true",
                        help="One-shot judge demo: implies --smoke --flash --keep-going --monitor and renders the dashboard.")
    parser.add_argument("--keep-going", action="store_true",
                        help="Warn and continue past a failed phase instead of aborting (protects a live demo).")
    parser.add_argument("--monitor", action="store_true",
                        help="Pop open a live browser dashboard showing each phase + the Flash fan-out in real time.")
    parser.add_argument("--monitor-port", type=int, default=8765,
                        help="Port for the live monitor (default 8765; falls back to a free port if taken).")
    args = parser.parse_args()

    if args.demo:
        args.smoke = args.flash = args.keep_going = args.monitor = True

    # Propagate to child phases via the environment (subprocesses inherit os.environ).
    if args.flash:
        os.environ["GENOTHERMAL_FLASH"] = "1"
        logger.info("Set GENOTHERMAL_FLASH=1 in environment for all child phases.")
    if args.smoke:
        os.environ["GENOTHERMAL_SMOKE"] = "1"
        logger.info("Set GENOTHERMAL_SMOKE=1 in environment for all child phases.")

    keep_going = args.keep_going
    if _ENV_LOADED:
        logger.info("Loaded .env — API keys are available to all child phases.")
    else:
        logger.info("No .env found — phases run in their synthetic-fallback paths.")
    flash_mode = bool(os.environ.get("GENOTHERMAL_FLASH"))
    smoke_mode = bool(os.environ.get("GENOTHERMAL_SMOKE"))
    logger.info("Geno-Thermal Targeting: Master Pipeline Starting")
    logger.info("Compute mode: %s", "RunPod Flash (serverless GPU fan-out)" if flash_mode
                else "LOCAL (set GENOTHERMAL_FLASH=1 to fan out on Flash)")
    if smoke_mode:
        logger.info("SMOKE MODE: reduced GA/PPO/folding workload for a fast demo run.")
    if keep_going:
        logger.info("KEEP-GOING: a failed phase warns and continues instead of aborting.")

    failures = []

    # Optional live progress "pop-up": a self-contained browser dashboard that shows every
    # phase light up running -> success/failed/skipped in real time, plus the Flash fan-out
    # metrics. Runs in a daemon thread; if it can't start it must never break the pipeline.
    monitor = None
    if args.monitor:
        try:
            from pipeline_monitor import PipelineMonitor
            monitor = PipelineMonitor(
                port=args.monitor_port,
                mode={"flash": flash_mode, "smoke": smoke_mode},
            )
            monitor.start()
        except Exception as e:
            logger.debug("Could not start live monitor: %s", e)
            monitor = None

    def step(command, description, optional=False):
        """Run a phase. On failure: warn-and-continue if keep_going/optional, else abort."""
        if monitor:
            monitor.start_phase(description)
        start = time.time()
        ok = run_step(command, description)
        if monitor:
            monitor.end_phase(description, ok, elapsed=time.time() - start, optional=optional)
        if ok:
            return True
        failures.append(description)
        if keep_going or optional:
            logger.warning("Continuing past failed phase: %s", description)
            return False
        if monitor:
            monitor.finish(failures)
        sys.exit(1)

    # 1. Genomic Discovery
    step("python genomic_discovery.py --target_gene EGFR", "Phase 1: Genomic Discovery")

    # 1.5 Target intelligence — live web data via Bright Data, fanned out on Flash. Optional and
    # self-healing: degrades to a flagged local stub without a token, so it never blocks a demo.
    step("python bright_data_intel.py" + ("" if flash_mode else " --local"),
         "Phase 1.5: Target Intelligence (Bright Data fan-out)", optional=True)

    # 2. Ligand Design — Boltz-2 folding + affinity on Flash (replaces AlphaFold Server)
    step("python boltz_designer.py --output_csv outputs/reports/candidate_library_v2.csv"
         + ("" if flash_mode else " --local"),
         "Phase 2: Ligand Engineering (Boltz-2)")

    # 3. Thermo-Switch Design
    step("python hard_mode/thermo_fold.py", "Phase 5: Thermo-Switch Protein Design")

    # 4. Nano Topology
    step("python hard_mode/nano_topology.py", "Phase 6: Nanoparticle Surface Topology")

    # 5. Bio Circuit
    step("python hard_mode/bio_circuit.py", "Phase 7: Biological Circuit Integration")

    # 6. Evolutionary Design (Hard Mode)
    step("python hard_mode/evolver.py", "Phase 4: Evolutionary Promoter Design")

    # 7. RL Training (Hard Mode) — PPO trains on a Flash GPU worker when enabled
    step("python flash_gpu_jobs.py ppo", "Phase 8: RL-Driven Sequence Design")

    # 8. Physics Verification — runs on a Flash CUDA worker; locally it needs OpenMM
    if flash_mode or check_openmm():
        step("python flash_gpu_jobs.py md", "Phase 9: Physics Verification (OpenMM)", optional=True)
    else:
        logger.info("SKIPPING Phase 9: Physics Verification ('openmm' not found; set GENOTHERMAL_FLASH=1 to run on Flash)")
        if monitor:
            monitor.skip_phase("Phase 9: Physics Verification (OpenMM)", reason="openmm not installed")

    # 9. Visualization
    step("python visualize_results.py", "Phase 10: Visualization", optional=True)

    # 10. Flash observability/cost dashboard (only if a Flash run recorded metrics)
    if os.path.exists("flash_metrics.json"):
        step("python flash_dashboard.py", "Phase 11: Flash Fan-out Dashboard", optional=True)
    elif monitor:
        monitor.skip_phase("Phase 11: Flash Fan-out Dashboard", reason="no flash_metrics.json")

    # 11. Terminal summary report
    step("python summary_report.py", "Phase 12: Summary Report", optional=True)

    if monitor:
        monitor.finish(failures)

    logger.info("=" * 60)
    if failures:
        logger.warning("PIPELINE COMPLETE WITH %d FAILED PHASE(S): %s", len(failures), "; ".join(failures))
    else:
        logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info("Artifacts generated. See 'pipeline_master.log' for details.")
    if os.path.exists("outputs/figures/flash_scaling.png"):
        logger.info("Flash fan-out chart: outputs/figures/flash_scaling.png  (open it to show the 0->N->0 autoscaling story)")

    # The single unified rollup: one self-contained HTML file with every figure embedded,
    # the 3D viewers linked, and a real-vs-synthetic provenance banner. This is the
    # "show me everything" artifact. Auto-open it on --demo so a judge sees it immediately.
    report = os.path.join("outputs", "reports", "summary_report.html")
    if os.path.exists(report):
        logger.info("UNIFIED REPORT: %s  (open this to see all results + visualizations in one page)", report)
        if args.demo:
            try:
                import webbrowser
                webbrowser.open("file://" + os.path.abspath(report))
                logger.info("Opened the unified report in your browser (--demo).")
            except Exception as e:
                logger.debug("Could not auto-open report: %s", e)

    # Keep the live monitor page up after the run so the final flow + artifacts can be
    # studied. Blocks for Enter in an interactive terminal; returns at once otherwise.
    if monitor:
        monitor.linger()

if __name__ == "__main__":
    main()
