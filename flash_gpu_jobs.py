"""
Phases 8 & 9 on RunPod Flash GPU workers — SELF-CONTAINED (no sibling repo imports).

To close the #1 bundling risk, the worker logic is inlined here (mirrors hard_mode/ppo_agent.py,
hard_mode/rl_gene_designer.py, and hard_mode/physics_verify.py). Heavy libs (gymnasium,
stable-baselines3, torch, openmm, pdbfixer, numpy) are imported INSIDE the worker functions,
so this module stays importable on a laptop without them and ships nothing but itself to Flash.
The originals remain the source of truth for local runs; keep this in sync if they change.

  * Phase 8 — PPO promoter design. Supports a SEED SWEEP: fan out N seeds across the fleet and
    keep the highest-fitness design (`python flash_gpu_jobs.py ppo --sweep 8`).
  * Phase 9 — OpenMM thermal-switch verification on a CUDA worker.

Both scale 0 -> N -> 0 and record metrics for the dashboard.
"""

import os
import time
import logging
import threading

# Honor GENOTHERMAL_LOG_LEVEL (e.g. DEBUG) so the per-line debug logs below are reachable
# at the venue without editing source. Defaults to INFO.
_LEVEL = getattr(logging, os.environ.get("GENOTHERMAL_LOG_LEVEL", "INFO").upper(), logging.INFO)
logging.basicConfig(
    level=_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("FlashGPUJobs")


def _log_gpu(stage: str):
    """Best-effort GPU telemetry (device + memory) for the RunPod worker. Self-contained:
    uses torch (a worker dep for the PPO/CUDA paths) and is a no-op off-CUDA."""
    try:
        import torch
        if not getattr(torch, "cuda", None) or not torch.cuda.is_available():
            logger.info("[gpu:%s] CUDA not available (CPU worker).", stage)
            return
        i = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(i)
        total = props.total_memory / 1e9
        peak = torch.cuda.max_memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        logger.info("[gpu:%s] %s (cc %d.%d) | reserved=%.2fGB peak=%.2fGB / total=%.1fGB (%.0f%% peak)",
                    stage, props.name, props.major, props.minor,
                    reserved, peak, total, 100.0 * peak / total if total else 0.0)
    except Exception as e:  # torch missing (e.g. OpenMM-only worker) — never load-bearing
        logger.debug("[gpu:%s] telemetry unavailable: %s", stage, e)


class _GpuSampler:
    """Background GPU-UTILIZATION sampler (SM occupancy % + memory via NVML). Runs on a daemon
    thread for the lifetime of a `with` block, then logs mean/peak utilization. Self-contained
    and fully guarded: a no-op when pynvml or a GPU is absent. Works even on the OpenMM-only MD
    worker (no torch) because it talks to NVML directly. Tells you whether the GPU is actually
    BUSY — the signal that distinguishes 'right-sized' from 'idle/over-provisioned'."""

    def __init__(self, stage, interval=2.0):
        self.stage, self.interval = stage, interval
        self._stop = threading.Event()
        self._thread = None
        self._nvml = None
        self._handle = None
        self.samples = []  # (util_pct, mem_used_gb)

    def __enter__(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            logger.debug("[gpu-util:%s] NVML sampler started (interval=%.1fs)", self.stage, self.interval)
        except Exception as e:  # no pynvml / no GPU / no driver — sampling is best-effort
            logger.debug("[gpu-util:%s] sampler unavailable (%s); skipping.", self.stage, e)
        return self

    def _run(self):
        while not self._stop.wait(self.interval):
            try:
                u = self._nvml.nvmlDeviceGetUtilizationRates(self._handle)
                m = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
                self.samples.append((u.gpu, m.used / 1e9))
            except Exception:
                break

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._nvml:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
        if self.samples:
            us = [u for u, _ in self.samples]
            ms = [m for _, m in self.samples]
            logger.info("[gpu-util:%s] %d samples | SM util mean=%.0f%% peak=%d%% | mem used mean=%.2fGB peak=%.2fGB",
                        self.stage, len(self.samples), sum(us) / len(us), max(us),
                        sum(ms) / len(ms), max(ms))
        else:
            logger.debug("[gpu-util:%s] no utilization samples collected.", self.stage)
        return False  # never suppress exceptions from the wrapped block


# --- Phase 8: PPO promoter design (self-contained) ------------------------
def _local_reward(dna):
    """Local heuristic reward (inlined from SequenceJudge._local_score). No network/siblings."""
    import random as _random
    score = 0.0
    has_tata = "TATA" in dna
    has_hse = "GAA" in dna and "TTC" in dna
    if has_tata:
        score += 5.0
    if has_hse:
        score += 8.0
    gc = (dna.count("G") + dna.count("C")) / max(len(dna), 1)
    if gc > 0.7:
        score -= 5.0
    score += _random.uniform(-1, 1)
    logger.debug("_local_reward: TATA=%s, HSE=%s, gc=%.3f -> score=%.3f", has_tata, has_hse, gc, score)
    return score


def _make_env_class():
    """Build the PromoterDesignEnv class (inlined from rl_gene_designer) once gymnasium is importable."""
    import numpy as np
    import gymnasium as gym
    from gymnasium import spaces

    class PromoterDesignEnv(gym.Env):
        def __init__(self, target_length=200):
            super().__init__()
            self.target_length = target_length
            self.current_step = 0
            self.sequence = []
            self.action_space = spaces.Discrete(4)
            self.observation_space = spaces.Box(low=-1, high=3, shape=(target_length,), dtype=np.int32)

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.current_step = 0
            self.sequence = [-1] * self.target_length
            return np.array(self.sequence, dtype=np.int32), {}

        def step(self, action):
            self.sequence[self.current_step] = int(action)
            self.current_step += 1
            terminated = self.current_step >= self.target_length
            reward = _local_reward(self.to_str()) if terminated else 0.0
            return np.array(self.sequence, dtype=np.int32), reward, terminated, False, {}

        def to_str(self):
            m = {0: "A", 1: "C", 2: "G", 3: "T"}
            return "".join(m.get(i, "N") for i in self.sequence if i != -1)

    return PromoterDesignEnv


def _train_and_generate(timesteps=10000, length=200, seed=None):
    """Train the PPO promoter agent and emit one design. Runs on the GPU worker."""
    logger.info("_train_and_generate: timesteps=%d, length=%d, seed=%s", timesteps, length, seed)
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor

    Env = _make_env_class()
    # Wrap in Monitor so SB3 populates ep_info_buffer — without it ep_rew_mean is always nan
    # (both in the live [ppo] telemetry and in the reward_history we plot).
    env = DummyVecEnv([lambda: Monitor(Env(length))])
    n_steps = min(2048, timesteps)
    logger.info("Building PPO model (n_steps=%d, batch_size=%d)", n_steps, min(64, n_steps))
    model = PPO("MlpPolicy", env, verbose=0, learning_rate=3e-4,
                n_steps=n_steps, batch_size=min(64, n_steps), gamma=0.99, seed=seed)
    try:
        import torch as _torch
        if _torch.cuda.is_available():
            _torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass
    _log_gpu("ppo-start")

    # Algorithm-level PPO telemetry. SB3's own verbose=0 is silent on the worker, so we attach
    # a callback that logs REAL learning progress each rollout (mirrors hard_mode/ppo_agent's
    # ProgressCallback but with actual metrics, inlined to keep this module self-contained).
    from stable_baselines3.common.callbacks import BaseCallback

    reward_history = []  # (timesteps, ep_rew_mean) per rollout -> returned for the learning-curve plot

    class _PPOProgress(BaseCallback):
        def _on_step(self) -> bool:
            return True

        def _on_rollout_end(self) -> None:
            buf = getattr(self.model, "ep_info_buffer", None)
            if buf:
                mean_r = sum(ep["r"] for ep in buf) / len(buf)
                mean_l = sum(ep["l"] for ep in buf) / len(buf)
                reward_history.append((int(self.num_timesteps), float(mean_r)))
            else:
                mean_r = mean_l = float("nan")
            stats = getattr(self.model.logger, "name_to_value", {}) or {}  # train/* lags one update
            logger.info("[ppo] steps=%d/%d | ep_rew_mean=%.3f ep_len_mean=%.1f | "
                        "loss=%.4f entropy=%.4f expl_var=%.3f",
                        self.num_timesteps, timesteps, mean_r, mean_l,
                        stats.get("train/loss", float("nan")),
                        stats.get("train/entropy_loss", float("nan")),
                        stats.get("train/explained_variance", float("nan")))

    logger.info("Starting PPO.learn (%d timesteps)...", timesteps)
    _t_learn = time.time()
    with _GpuSampler("ppo-learn"):  # sample SM utilization during training
        model.learn(total_timesteps=timesteps, callback=_PPOProgress())
    _learn_s = time.time() - _t_learn
    _log_gpu("ppo-done")
    logger.info("PPO training complete in %.1fs (%.0f timesteps/s). Generating one sequence via deterministic rollout.",
                _learn_s, timesteps / _learn_s if _learn_s else 0.0)

    gen = Env(length)
    obs, _ = gen.reset(seed=seed)
    done, reward = False, 0.0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action.item()) if hasattr(action, "item") else int(action)
        obs, reward, terminated, truncated, _ = gen.step(action)
        done = terminated or truncated
    seq = gen.to_str()
    logger.info("Generated sequence (first 30bp): %s... | fitness=%.4f", seq[:30], reward)
    # reward_history travels back in the payload so the LOCAL driver can plot the learning
    # curve even when training ran on a remote Flash worker.
    return {"sequence": seq, "predicted_fitness": float(reward), "seed": seed,
            "reward_history": reward_history}


# --- Phase 9: OpenMM thermal-switch verification (self-contained) ---------
def _verify_physics(pdb_text):
    """Run the OpenMM thermal-switch MD check on a CUDA worker (inlined from physics_verify)."""
    logger.info("_verify_physics: PDB text length=%d chars", len(pdb_text))
    import tempfile
    import numpy as np
    import openmm as mm
    from openmm import app, unit
    try:
        from pdbfixer import PDBFixer
        have_fixer = True
        logger.info("pdbfixer available — will fix topology before simulation.")
    except ImportError:
        have_fixer = False
        logger.warning("pdbfixer not available — skipping PDB fix step.")

    FF, WATER, STEPS = "amber14-all.xml", "amber14/tip3p.xml", 5000

    with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False) as f:
        f.write(pdb_text)
        pdb_path = f.name

    def fix(p):
        if not have_fixer:
            logger.debug("fix: pdbfixer absent, returning original %s", p)
            return p
        logger.debug("fix: running pdbfixer on %s", p)
        fx = PDBFixer(filename=p)
        fx.findMissingResidues()
        fx.findNonstandardResidues()
        fx.replaceNonstandardResidues()
        fx.findMissingAtoms()
        fx.addMissingAtoms()
        # NOTE: deliberately do NOT addMissingHydrogens here. pdbfixer's H naming for terminal
        # residues conflicts with amber14's terminal templates (residue 0 ALA gets read as NSER).
        # Let OpenMM's Modeller.addHydrogens(ff) add them in setup() — it knows the FF's terminal
        # variants and builds a template-consistent topology.
        out = p.replace(".pdb", "_fixed.pdb")
        with open(out, "w") as fh:
            app.PDBFile.writeFile(fx.topology, fx.positions, fh)
        logger.debug("fix: fixed PDB written to %s", out)
        return out

    def setup(p, temp_k):
        logger.debug("setup: T=%.2f K, PDB=%s", temp_k, p)
        if have_fixer:
            p = fix(p)
        logger.debug("setup: loading PDB...")
        pdb = app.PDBFile(p)
        logger.debug("setup: creating ForceField (%s, %s)...", FF, WATER)
        ff = app.ForceField(FF, WATER)
        mod = app.Modeller(pdb.topology, pdb.positions)
        # Always let the forcefield add hydrogens (template-consistent terminals); pdbfixer only
        # repairs missing heavy atoms above.
        logger.debug("setup: addHydrogens via forcefield...")
        mod.addHydrogens(ff)
        logger.debug("setup: addSolvent (padding=1.0 nm)...")
        mod.addSolvent(ff, padding=1.0 * unit.nanometers)
        logger.debug("setup: createSystem (PME, HBonds)...")
        system = ff.createSystem(mod.topology, nonbondedMethod=app.PME,
                                 nonbondedCutoff=1.0 * unit.nanometers, constraints=app.HBonds)
        logger.debug("setup: LangevinMiddleIntegrator at %.2f K...", temp_k)
        integ = mm.LangevinMiddleIntegrator(temp_k * unit.kelvin, 1.0 / unit.picosecond,
                                            0.002 * unit.picoseconds)
        platform, props = None, {}
        for name in ("CUDA", "OpenCL"):
            try:
                platform = mm.Platform.getPlatformByName(name)
                props = {"Precision": "mixed"} if name == "CUDA" else {}
                logger.info("GPU platform selected: %s", name)
                break
            except Exception as _plat_err:
                logger.debug("Platform %s not available (%s), trying next.", name, _plat_err)
                continue
        if platform is None:
            logger.warning("No GPU platform found (CUDA/OpenCL); falling back to CPU.")
            platform = mm.Platform.getPlatformByName("CPU")
        sim = app.Simulation(mod.topology, system, integ, platform, props)
        sim.context.setPositions(mod.positions)
        return sim

    def rmsd(a, b):
        p1 = np.array(a.value_in_unit(unit.nanometers))
        p2 = np.array(b.value_in_unit(unit.nanometers))
        d = p1 - p2
        return float(np.sqrt((d * d).sum() / len(p1)))

    def run(sim):
        logger.debug("run: minimizing energy...")
        sim.minimizeEnergy()
        logger.debug("run: equilibrating (100 steps)...")
        sim.step(100)
        start = sim.context.getState(getPositions=True).getPositions()
        logger.debug("run: production MD (%d steps)...", STEPS)
        sim.step(STEPS)
        end = sim.context.getState(getPositions=True).getPositions()
        r = rmsd(start, end)
        logger.debug("run: RMSD=%.4f nm", r)
        return r

    try:
        _t_md = time.time()
        with _GpuSampler("md"):  # sample SM utilization across both MD runs
            logger.info("Running MD at 37°C (310.15 K)...")
            r37 = run(setup(pdb_path, 310.15))
            logger.info("MD at 37°C complete in %.1fs. RMSD=%.4f nm", time.time() - _t_md, r37)
            _t_md43 = time.time()
            logger.info("Running MD at 43°C (316.15 K)...")
            r43 = run(setup(pdb_path, 316.15))
            logger.info("MD at 43°C complete in %.1fs. RMSD=%.4f nm", time.time() - _t_md43, r43)
        _log_gpu("md-done")
        logger.info("MD total compute: %.1fs (%d steps x2 temps).", time.time() - _t_md, STEPS)
    except Exception as e:
        logger.error("MD simulation failed: %s", e)
        os.unlink(pdb_path)
        return {"thermal_switch_verified": False, "error": str(e)}
    os.unlink(pdb_path)
    passed = (r37 < 0.5) and (r43 > r37 * 1.2)
    logger.info("Thermal switch verification: %s (RMSD 37C=%.4f, 43C=%.4f)",
                "PASS" if passed else "FAIL", r37, r43)
    return {"thermal_switch_verified": bool(passed),
            "rmsd_37C": round(r37, 4), "rmsd_43C": round(r43, 4)}


# --- Flash endpoints ------------------------------------------------------
try:
    from runpod_flash import Endpoint, GpuType

    @Endpoint(
        name="genothermal-ppo",
        gpu=GpuType.NVIDIA_GEFORCE_RTX_4090,
        workers=(0, 8),                 # raised from 1 -> 8 so a seed sweep fans out
        dependencies=["stable-baselines3", "gymnasium", "torch", "numpy", "pynvml"],
        idle_timeout=30,
    )
    async def train_ppo_endpoint(payload: dict) -> dict:
        return _train_and_generate(
            payload.get("timesteps", 10000),
            payload.get("length", 200),
            payload.get("seed"),
        )

    @Endpoint(
        name="genothermal-md",
        gpu=GpuType.NVIDIA_GEFORCE_RTX_4090,   # tiny single-peptide MD; 4090 is plenty (cost win)
        workers=(0, 1),
        # pdbfixer is NOT on PyPI (conda/git only) — a bare "pdbfixer" fails the whole
        # build-time pip install. Pin the git URL so it installs on the worker; the code
        # below already degrades gracefully (have_fixer=False) if it's still absent.
        dependencies=["openmm", "pdbfixer @ git+https://github.com/openmm/pdbfixer.git", "numpy", "pynvml"],
        idle_timeout=20,
    )
    async def verify_physics_endpoint(payload: dict) -> dict:
        return _verify_physics(payload["pdb_text"])

    FLASH_AVAILABLE = True
except (ImportError, AttributeError, ValueError):  # bad GpuType/cpu flavor degrades to local, not crash
    train_ppo_endpoint = None
    verify_physics_endpoint = None
    FLASH_AVAILABLE = False
    logger.warning("runpod_flash unavailable — GPU-job endpoints disabled.")


def _run_remote(endpoint, payload, timeout_s=1800):
    logger.info("_run_remote: endpoint=%s, timeout=%ds, payload_keys=%s",
                getattr(endpoint, "name", str(endpoint)), timeout_s, list(payload.keys()))
    import asyncio

    async def go():
        # Decorator endpoint -> await it directly (returns the result dict), not .run()/job.wait().
        out = await asyncio.wait_for(endpoint(payload), timeout=timeout_s)
        logger.info("_run_remote: job complete.")
        return out

    return asyncio.run(go())


def _ppo_sweep(n_seeds, timesteps, length, timeout_s=1800):
    """Fan N seeds across the PPO fleet concurrently; keep the highest-fitness design."""
    import asyncio
    from flash_metrics import FanoutMetrics

    logger.info("_ppo_sweep: n_seeds=%d, timesteps=%d, length=%d, timeout=%ds",
                n_seeds, timesteps, length, timeout_s)
    metrics = FanoutMetrics(phase="ppo-sweep", resource="RTX_4090")

    async def _run():
        logger.debug("_ppo_sweep._run: dispatching %d seed jobs...", n_seeds)

        async def _await(seed):
            rec = metrics.start()
            try:
                # Decorator endpoint -> await directly (returns the result dict), not .run()/job.wait().
                result = await asyncio.wait_for(
                    train_ppo_endpoint({"timesteps": timesteps, "length": length, "seed": seed}),
                    timeout=timeout_s)
                metrics.done(rec, ok=True)
                logger.debug("_ppo_sweep._await: seed job done, fitness=%.4f",
                             result.get("predicted_fitness", float("nan")))
                return result
            except Exception as e:
                logger.warning("PPO seed job failed (%s); dropping.", e)
                metrics.done(rec, ok=False)
                return None

        return await asyncio.gather(*(_await(s) for s in range(n_seeds)))

    results = [r for r in asyncio.run(_run()) if r]
    metrics.save()
    if not results:
        return {"error": "all PPO seed jobs failed"}
    best = max(results, key=lambda r: r["predicted_fitness"])
    logger.info("Sweep of %d seeds -> best fitness %.2f (seed %s)",
                n_seeds, best["predicted_fitness"], best.get("seed"))
    return {"best": best, "all": results}


def _skip(reason: str) -> dict:
    """Clean, honest skip for a local run when the heavy GPU dep lives only on the Flash worker."""
    logger.warning("SKIPPING — %s. Set GENOTHERMAL_FLASH=1 to run this on a Flash GPU worker "
                   "(deps ship with the @Endpoint).", reason)
    return {"skipped": True, "reason": reason}


def _module_missing(mod: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(mod) is None


# The toy idealised stub PDBs (atoms on an integer grid) have non-physical geometry that the
# amber forcefield can't template. Real Boltz-2 folds do. So for MD we prefer a real fold.
_STUB_PDB_DEFAULT = os.path.join("outputs", "simulated_pdbs", "unknown_complex.pdb")


def _resolve_md_pdb_text(requested_path: str):
    """Return PDB text for the MD phase, preferring a real folded structure.

    - If the caller passed a real, non-stub .pdb, use it verbatim.
    - Otherwise pick a real Boltz-2 fold from outputs/predicted_structures (skipping the
      poly-alanine negative control) and convert CIF->PDB via OpenMM.
    - Falls back to the requested file's raw text if no fold / no OpenMM is available.
    """
    import glob

    def _read(p):
        with open(p) as fh:
            return fh.read()

    explicit = requested_path and os.path.exists(requested_path) and \
        os.path.abspath(requested_path) != os.path.abspath(_STUB_PDB_DEFAULT)
    if explicit and requested_path.endswith(".pdb"):
        logger.info("MD: using caller-provided structure %s", requested_path)
        return _read(requested_path)

    folds = sorted(glob.glob(os.path.join("outputs", "predicted_structures", "*model_0.cif")))
    folds = [f for f in folds if "poly_alanine" not in f.lower() and "neg_control" not in f.lower()] or folds
    if folds and not _module_missing("openmm"):
        cif = folds[0]
        try:
            from openmm import app
            import io
            obj = app.PDBxFile(cif)
            buf = io.StringIO()
            app.PDBFile.writeFile(obj.topology, obj.positions, buf)
            logger.info("MD: using real Boltz-2 fold %s (converted CIF->PDB)", os.path.basename(cif))
            return buf.getvalue()
        except Exception as e:
            logger.warning("MD: CIF->PDB conversion failed (%s); falling back to requested path.", e)

    if requested_path and os.path.exists(requested_path):
        logger.info("MD: falling back to %s", requested_path)
        return _read(requested_path)
    return None


# --- Figures (so Phases 8 & 9 are no longer chart-less in the rollup) ------
FIGURES_DIR = os.path.join("outputs", "figures")
REPORTS_DIR = os.path.join("outputs", "reports")


def _render_ppo_curve(out: dict):
    """Plot the PPO learning curve(s) from the reward_history carried in the result payload.
    Handles a single run ({reward_history}) and a sweep ({best, all})."""
    runs = out.get("all") if isinstance(out.get("all"), list) else [out]
    series = [(r.get("seed"), r.get("reward_history") or []) for r in runs if isinstance(r, dict)]
    series = [(s, h) for s, h in series if h]
    if not series:
        logger.info("PPO: no reward_history to plot (skipping learning-curve figure).")
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import csv as _csv
    except ImportError:
        logger.warning("matplotlib not available — skipping PPO learning-curve figure.")
        return None

    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for seed, hist in series:
        xs = [t for t, _ in hist]
        ys = [r for _, r in hist]
        ax.plot(xs, ys, marker="o", ms=3, lw=1.5, label=f"seed {seed}" if seed is not None else "PPO")
    ax.set_xlabel("Training timesteps")
    ax.set_ylabel("Mean episode reward (promoter fitness)")
    ax.set_title("Phase 8 — PPO RL promoter design: learning curve")
    ax.grid(alpha=0.3)
    if len(series) > 1:
        ax.legend(fontsize=8, ncol=2)
    png = os.path.join(FIGURES_DIR, "ppo_reward_curve.png")
    fig.tight_layout()
    fig.savefig(png, dpi=150)
    plt.close(fig)

    # Also persist the best run's curve as CSV (version-friendly; tensorboard dir is gitignored).
    best = out.get("best") or out
    csv_path = os.path.join(REPORTS_DIR, "ppo_reward_log.csv")
    with open(csv_path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["timesteps", "ep_rew_mean"])
        w.writerows(best.get("reward_history") or [])
    logger.info("PPO learning curve saved to %s (and %s)", png, csv_path)
    return png


def _render_rmsd(out: dict):
    """Bar chart of the OpenMM thermal-switch RMSD at 37°C vs 43°C with the pass threshold."""
    if "rmsd_37C" not in out or "rmsd_43C" not in out:
        logger.info("MD: no RMSD values to plot (skipping physics figure).")
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping physics RMSD figure.")
        return None

    os.makedirs(FIGURES_DIR, exist_ok=True)
    r37, r43 = out["rmsd_37C"], out["rmsd_43C"]
    passed = out.get("thermal_switch_verified")
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(["37°C (body)", "43°C (hyperthermia)"], [r37, r43],
                  color=["#1f77b4", "#d62728"])
    ax.axhline(0.5, ls="--", color="gray", lw=1, label="stability threshold (0.5 nm)")
    ax.set_ylabel("Backbone RMSD vs start (nm)")
    ax.set_title(f"Phase 9 — Thermal-switch MD: {'PASS' if passed else 'FAIL'} "
                 f"(unfolds on heating)")
    ax.bar_label(bars, fmt="%.3f")
    ax.legend(fontsize=8)
    png = os.path.join(FIGURES_DIR, "physics_rmsd.png")
    fig.tight_layout()
    fig.savefig(png, dpi=150)
    plt.close(fig)
    logger.info("Physics RMSD chart saved to %s", png)
    return png


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Geno-Thermal GPU phases (Flash or local).")
    parser.add_argument("phase", choices=["ppo", "md"], help="ppo = Phase 8, md = Phase 9")
    parser.add_argument("--pdb", default="outputs/simulated_pdbs/unknown_complex.pdb")
    parser.add_argument("--sweep", type=int, default=1, help="PPO only: number of seeds to fan out (Flash).")
    args = parser.parse_args()

    use_flash = bool(os.environ.get("GENOTHERMAL_FLASH")) and FLASH_AVAILABLE
    smoke = bool(os.environ.get("GENOTHERMAL_SMOKE"))
    logger.info("GENOTHERMAL_FLASH=%s, GENOTHERMAL_SMOKE=%s, FLASH_AVAILABLE=%s",
                bool(os.environ.get("GENOTHERMAL_FLASH")), smoke, FLASH_AVAILABLE)
    logger.info("%s phase '%s'%s", "FLASH" if use_flash else "LOCAL",
                args.phase, " (smoke)" if smoke else "")

    def _ppo_local(timesteps):
        """Local PPO run, or a clean skip if sb3 isn't installed here."""
        if _module_missing("stable_baselines3"):
            return _skip("stable_baselines3 is not installed locally")
        logger.info("PPO: Local training (no Flash).")
        return _train_and_generate(timesteps)

    def _md_local():
        """Local MD run, or a clean skip if openmm / a usable structure is missing."""
        if _module_missing("openmm"):
            return _skip("openmm is not installed locally")
        pdb_text = _resolve_md_pdb_text(args.pdb)
        if not pdb_text:
            return _skip(f"no usable MD structure (requested {args.pdb})")
        return _verify_physics(pdb_text)

    if args.phase == "ppo":
        timesteps = 200 if smoke else 10000
        logger.info("PPO: timesteps=%d, sweep=%d, use_flash=%s", timesteps, args.sweep, use_flash)
        if use_flash and args.sweep > 1:
            logger.info("PPO: Fan-out sweep of %d seeds on Flash.", args.sweep)
            try:
                out = _ppo_sweep(args.sweep, timesteps, 200)
            except Exception as e:
                logger.warning("PPO Flash sweep failed (%s) — falling back to local.", e)
                out = _ppo_local(timesteps)
        elif use_flash:
            logger.info("PPO: Single Flash GPU job.")
            from flash_metrics import FanoutMetrics
            metrics_single = FanoutMetrics(phase="ppo-gpu", resource="RTX_4090")
            rec = metrics_single.start()
            try:
                out = _run_remote(train_ppo_endpoint, {"timesteps": timesteps})
                metrics_single.done(rec, ok=True)
                metrics_single.save()
            except Exception as e:
                # The genothermal-ppo endpoint may not be deployed (`flash deploy`). Rather than
                # crash the phase, record the miss and fall back to local training/skip.
                metrics_single.done(rec, ok=False)
                metrics_single.save()
                logger.warning("PPO Flash dispatch failed (%s) — falling back to local.", e)
                out = _ppo_local(timesteps)
        else:
            out = _ppo_local(timesteps)
    else:  # md
        if use_flash:
            logger.info("MD: dispatching to Flash GPU worker.")
            from flash_metrics import FanoutMetrics
            m = FanoutMetrics(phase="md-gpu", resource="RTX_4090")
            rec = m.start()
            try:
                pdb_text = _resolve_md_pdb_text(args.pdb)
                if not pdb_text:
                    raise FileNotFoundError(f"no usable MD structure (requested {args.pdb})")
                out = _run_remote(verify_physics_endpoint, {"pdb_text": pdb_text})
                m.done(rec, ok=True)
                m.save()
            except Exception as e:
                # genothermal-md endpoint may not be deployed; fall back to local OpenMM/skip.
                m.done(rec, ok=False)
                m.save()
                logger.warning("MD Flash dispatch failed (%s) — falling back to local.", e)
                out = _md_local()
        else:
            out = _md_local()

    # Render the phase's figure so Phases 8 & 9 are represented in the unified report.
    try:
        if args.phase == "ppo":
            _render_ppo_curve(out)
        else:
            _render_rmsd(out)
    except Exception as e:  # a figure is best-effort; never fail the phase over it
        logger.warning("Figure rendering failed (%s) — phase output is unaffected.", e)

    # Don't dump the (potentially long) reward_history into the terminal JSON.
    _printable = {k: v for k, v in out.items() if k != "reward_history"} if isinstance(out, dict) else out
    logger.info("Output:\n%s", __import__("json").dumps(_printable, indent=2))
