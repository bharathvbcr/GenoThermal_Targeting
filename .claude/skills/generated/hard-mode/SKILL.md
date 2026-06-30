---
name: hard-mode
description: "Skill for the Hard_mode area of GenoThermal_Targeting. 52 symbols across 10 files."
---

# Hard_mode

52 symbols | 10 files | Cohesion: 92%

## When to Use

- Working with code in `hard_mode/`
- Understanding how pct, valid, calculate_fitness work
- Modifying hard_mode-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `hard_mode/evolver.py` | AlphaGenomeOracle, GeneticOptimizer, _evaluate_population, _score_via_threads, _score_via_flash (+14) |
| `hard_mode/thermo_fold.py` | _calculate_thermodynamics, predict_melting_temp, predict_folded_fraction, predict_plddt, mutate (+2) |
| `hard_mode/physics_verify.py` | fix_pdb, setup_simulation, run_md_protocol, calculate_rmsd, verify_thermal_switch |
| `hard_mode/rl_gene_designer.py` | reset, step, _indices_to_string, score, _local_score |
| `hard_mode/nano_topology.py` | __init__, _initialize_grid, calculate_energy, run_annealing |
| `flash_gpu_jobs.py` | _local_reward, step, to_str |
| `flash_metrics.py` | summary, pct, save |
| `hard_mode/bio_circuit.py` | get_promoter_activity, get_switch_state, run_simulation |
| `preflight.py` | _, valid |
| `hard_mode/ppo_agent.py` | generate_sequence |

## Entry Points

Start here when exploring this area:

- **`pct`** (Function) — `flash_metrics.py:55`
- **`valid`** (Function) — `preflight.py:195`
- **`calculate_fitness`** (Function) — `hard_mode/evolver.py:126`
- **`fix_pdb`** (Function) — `hard_mode/physics_verify.py:48`
- **`setup_simulation`** (Function) — `hard_mode/physics_verify.py:73`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `AlphaGenomeOracle` | Class | `hard_mode/evolver.py` | 65 |
| `GeneticOptimizer` | Class | `hard_mode/evolver.py` | 134 |
| `pct` | Function | `flash_metrics.py` | 55 |
| `valid` | Function | `preflight.py` | 195 |
| `calculate_fitness` | Function | `hard_mode/evolver.py` | 126 |
| `fix_pdb` | Function | `hard_mode/physics_verify.py` | 48 |
| `setup_simulation` | Function | `hard_mode/physics_verify.py` | 73 |
| `run_md_protocol` | Function | `hard_mode/physics_verify.py` | 131 |
| `calculate_rmsd` | Function | `hard_mode/physics_verify.py` | 155 |
| `verify_thermal_switch` | Function | `hard_mode/physics_verify.py` | 163 |
| `generate_sequence` | Function | `hard_mode/ppo_agent.py` | 84 |
| `step` | Method | `flash_gpu_jobs.py` | 90 |
| `to_str` | Method | `flash_gpu_jobs.py` | 97 |
| `summary` | Method | `flash_metrics.py` | 51 |
| `save` | Method | `flash_metrics.py` | 94 |
| `mutate` | Method | `hard_mode/evolver.py` | 253 |
| `crossover` | Method | `hard_mode/evolver.py` | 263 |
| `check_convergence` | Method | `hard_mode/evolver.py` | 271 |
| `run` | Method | `hard_mode/evolver.py` | 290 |
| `evaluate_sequence_properties` | Method | `hard_mode/evolver.py` | 91 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `_ → _scan_motifs` | cross_community | 6 |
| `Main → Pct` | cross_community | 6 |
| `Score_endpoint → _scan_motifs` | cross_community | 6 |
| `Main → Pct` | cross_community | 5 |
| `Calculate_fitness → _scan_motifs` | intra_community | 5 |
| `Run → _await` | intra_community | 5 |
| `_ → _count` | cross_community | 4 |
| `Run → _score_via_threads` | intra_community | 3 |
| `Generate_sequence → _indices_to_string` | intra_community | 3 |

## Connected Areas

| Area | Connections |
|------|-------------|
| Cluster_7 | 9 calls |
| Cluster_5 | 2 calls |
| Cluster_9 | 1 calls |
| Cluster_10 | 1 calls |
| Cluster_4 | 1 calls |

## How to Explore

1. `context({name: "pct"})` — see callers and callees
2. `query({search_query: "hard_mode"})` — find related execution flows
3. Read key files listed above for implementation details
4. `explain({target: "<file or symbol>"})` — persisted taint findings (source→sink data flows), when indexed with `--pdg`
