# METHODS — metrics, quantities, and honest caveats

One auditable place defining every number the Geno-Thermal × RunPod Flash pipeline reports,
and exactly what it does and does NOT mean. If a judge asks "what is that number?", it's here.

> TL;DR honesty stance: every score below is a **model prediction or an estimate**, not a wet-lab
> measurement. Confidence scores are not binding affinities; cost is an estimate; the speedup is a
> time/throughput win, not a dollar saving.

---

## 1. Structure & binding (Boltz-2, `flash_boltz.py`)

| Quantity | Definition | Caveat |
|----------|------------|--------|
| `plddt_score` | Interface confidence = **ipTM × 100** for the predicted complex (falls back to ptm/confidence). | Named `plddt_score` only for drop-in schema compatibility with the old AlphaFold path. It is an **interface-confidence** score, **NOT** a per-residue pLDDT and **NOT** measured binding affinity. |
| `binding_class` | Confidence bucket from `plddt_score`: ≥80 STRONG, ≥60 MODERATE, ≥40 WEAK, else NON. | "STRONG_BINDER" = **high-confidence interface**, not high measured affinity (Kd/IC50). |
| `complex_pde` | Boltz-2 predicted distance error for the complex. | Honestly labeled — it is Boltz **PDE**, not AlphaFold PAE. |
| `affinity_pred_value` | Boltz-2 affinity head: predicted **log(IC50)** in µM for small-molecule ligands. **LOWER = stronger** binder. | Only present for `mode == ligand` (SMILES candidates). A model prediction, not an assay. |
| `affinity_probability` | Boltz-2 affinity head binary binder probability (0–1). | Model prediction. |
| `fold_seconds` | Worker compute time for one fold (cold-start excluded). | For optimization tracking only. |

`mode` distinguishes `peptide` (second protein chain, ranked by interface confidence) from
`ligand` (SMILES → ligand chain + affinity head).

## 2. Selectivity panel (`target_panel.py`)

| Quantity | Definition | Caveat |
|----------|------------|--------|
| `selectivity_margin` | `ipTM*100(intended target) − max ipTM*100(off-targets)`. | A difference of **interface-confidence** scores, **NOT** a difference of measured affinities. Ranks which candidate the model is most confident docks the intended oncogene over the others. For small molecules prefer the affinity margin below. |
| `affinity_selectivity_margin` | `min log-IC50(off-targets) − log-IC50(intended)`, from the Boltz-2 affinity head (lower log-IC50 = stronger). **Higher margin = more selective.** | Emitted only for small-molecule candidates that carry affinity data; NaN for peptide-only panels. The affinity-grounded selectivity to prefer for ligands. Still a model prediction, not an assay. |
| heatmap colorbar | Interface confidence (ipTM×100) per (candidate, target). | Same caveat; the colorbar is labeled accordingly. |

Targets are EXACT UniProt sequences trimmed to the binding-relevant domain (`fetch_targets.py`):
EGFR (ectodomain fragment), KRAS (full), HER2_ECD (P04626 res 23–652), BRAF_KD (P15056 res 457–717).
BRAF is intracellular → most meaningful against the small-molecule screen, not ectodomain peptide binders.

## 3. GA fitness (`flash_fitness.py`, `hard_mode/evolver.py`)

The `mode='Local'` worker fitness is a fast **regex motif-count heuristic with a small random
jitter** (`±1`) — **not deterministic and not a learned model**. It rewards tumor/heat-shock
motifs, penalizes normal-tissue motifs and GC deviation. It is a stand-in for a real expression
oracle so the fan-out is demonstrable without burning the AlphaGenome API across 50 workers.

## 4. Flash fan-out metrics (`flash_metrics.py`)

| Quantity | Definition | Caveat |
|----------|------------|--------|
| `peak_inflight` | Max jobs simultaneously in flight (sweep-line over start/end events). | A **lower bound** on workers used, not a queried fleet size — the SDK is not asked for worker count. |
| `gpu_seconds` | Σ (end − start) over all jobs that ran (ok **and** failed). | A timed-out job genuinely consumed compute, so its time stays in the total. `gpu_seconds_ok` / `gpu_seconds_failed` split it. |
| `throughput_jobs_per_s` | **n_ok** / wall_s. | Counts only successful jobs (failed jobs produced no result). |
| `speedup_vs_serial` | `gpu_seconds / wall_s`. | A **time/throughput** speedup vs running the same jobs one-at-a-time on a single box. It is **NOT a $ saving** — the GPU-seconds (and thus cost) are identical; you pay for the same compute, just finished sooner. Reported **per phase** (phases run sequentially). |
| `time_saved_s` | `max(0, gpu_seconds − wall_s)`. | Wall-clock saved by parallelism. |
| `est_cost_usd` | `gpu_seconds × COST_PER_SEC[resource]`. | An **ESTIMATE**. `COST_PER_SEC` is hardcoded — update to RunPod's published rates before quoting. **Excludes** keep-warm / cold-start / idle-timeout seconds; per-job interval may include some queue/scheduling wait. |
| `--project N` (dashboard) | Linear extrapolation of the screen phase to N candidates. | A rough **ESTIMATE** (linear, assumes same peak concurrency and per-job time). |

`COST_PER_SEC` (estimates): A100_80GB ≈ $0.00050/s (~$1.80/hr), RTX_4090 ≈ $0.00019/s (~$0.69/hr),
cpu5c-4-16 ≈ $0.00002/s (~$0.07/hr). CPU endpoints bill **CPU-seconds**, labeled "CPU-s" (not GPU-s).

## 4b. Target intelligence (`bright_data_intel.py`, Phase 1.5)

Live web intel per target via **Bright Data** (SERP API / Web Unlocker), fanned out on RunPod
Flash exactly like the other phases (records a `bright-intel` phase in `flash_metrics.json`).

| Quantity | Definition | Caveat |
|----------|------------|--------|
| `source` | `brightdata` (live SERP results) or `local-stub` (deterministic canned note). | A run with no `BRIGHTDATA_API_TOKEN`, no network, or any error degrades to `local-stub` — clearly flagged, never silent. Only `source=brightdata` rows are live data. |
| `results` | Top organic SERP results (`title`, `url`, `snippet`) from Bright Data's parsed JSON (`brd_json=1`). | A search-engine snapshot at query time, not a curated database. |

Uses **both** hackathon sponsors in one pipeline: Flash (the 0→N→0 fan-out) + Bright Data (the
web data). The Bright Data **MCP server** (`.mcp.json`) is the complementary interactive-agent
path — same token (`BRIGHTDATA_API_TOKEN`), different entry point.

## 5. Physics verification (`flash_gpu_jobs.py`, Phase 9)

Two short OpenMM MD runs (5000 steps each) at 37 °C and 43 °C. `thermal_switch_verified` is
`True` iff `rmsd_37C < 0.5 nm` **and** `rmsd_43C > 1.2 × rmsd_37C` (folded at body temp, unfolds on
heating). A small single-peptide MD on a 4090 — a verification heuristic, not a converged free-energy result.

## 6. Reliability & demo artifacts

- **Drop-and-flag**: failed/timed-out folds are dropped (not crashed); the GA falls back to the local
  threadpool if a Flash generation errors ("kill an endpoint, it finishes locally").
- **CSV guard**: `boltz_designer.py` refuses to overwrite a committed library with an empty/partial
  fan-out (partial results go to a `*.partial.csv` sidecar).
- **`flash_scaling.png` / `demo_metrics.json`**: the committed copies are a **SYNTHETIC, illustrative**
  fallback shape (see `make_demo_snapshot.py`). **Regenerate from a real `--flash` run before submission**:
  `GENOTHERMAL_FLASH=1 python run_pipeline.py --demo && python flash_dashboard.py`.

## 7. Reproduce

```bash
make preflight        # 11 local sanity checks (no GPU/Flash)
make story            # 3-min demo path: GA fan-out -> panel -> dashboard -> summary
make board            # unified peptide + small-molecule leaderboard
python flash_dashboard.py --project 5000   # estimated cost/time of a full 5,000-candidate screen
```
