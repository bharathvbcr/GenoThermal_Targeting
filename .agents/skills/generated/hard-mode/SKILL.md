---
name: hard-mode
description: "Skill for the Hard_mode area of Geno-Thermal_Targeting. 49 symbols across 9 files."
---

# Hard_mode

49 symbols | 9 files | Cohesion: 97%

## When to Use

- Working with code in `hard_mode/`
- Understanding how score, reset, step work
- Modifying hard_mode-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `hard_mode/evolver.py` | __init__, _scan_motifs, evaluate_sequence_properties, score, _local_score (+7) |
| `hard_mode/thermo_fold.py` | _calculate_thermodynamics, predict_melting_temp, predict_folded_fraction, predict_plddt, mutate (+5) |
| `hard_mode/rl_gene_designer.py` | score, _local_score, PromoterDesignEnv, reset, step (+4) |
| `hard_mode/physics_verify.py` | fix_pdb, setup_simulation, run_md_protocol, calculate_rmsd, verify_thermal_switch |
| `hard_mode/nano_topology.py` | __init__, _initialize_grid, calculate_energy, run_annealing |
| `hard_mode/ppo_agent.py` | ProgressCallback, train_agent, generate_sequence |
| `hard_mode/bio_circuit.py` | get_promoter_activity, get_switch_state, run_simulation |
| `alphagenome_utils.py` | AlphaGenomeClient, parse_fasta |
| `genomic_discovery.py` | main |

## Entry Points

Start here when exploring this area:

- **`score`** (Function) — `hard_mode/rl_gene_designer.py:76`
- **`reset`** (Function) — `hard_mode/rl_gene_designer.py:129`
- **`step`** (Function) — `hard_mode/rl_gene_designer.py:135`
- **`train_agent`** (Function) — `hard_mode/ppo_agent.py:51`
- **`generate_sequence`** (Function) — `hard_mode/ppo_agent.py:88`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `PromoterDesignEnv` | Class | `hard_mode/rl_gene_designer.py` | 103 |
| `ProgressCallback` | Class | `hard_mode/ppo_agent.py` | 33 |
| `AlphaGenomeClient` | Class | `alphagenome_utils.py` | 28 |
| `ProteinPhysicsOracle` | Class | `hard_mode/thermo_fold.py` | 30 |
| `SequenceJudge` | Class | `hard_mode/rl_gene_designer.py` | 48 |
| `score` | Function | `hard_mode/rl_gene_designer.py` | 76 |
| `reset` | Function | `hard_mode/rl_gene_designer.py` | 129 |
| `step` | Function | `hard_mode/rl_gene_designer.py` | 135 |
| `train_agent` | Function | `hard_mode/ppo_agent.py` | 51 |
| `generate_sequence` | Function | `hard_mode/ppo_agent.py` | 88 |
| `predict_melting_temp` | Function | `hard_mode/thermo_fold.py` | 58 |
| `predict_folded_fraction` | Function | `hard_mode/thermo_fold.py` | 64 |
| `predict_plddt` | Function | `hard_mode/thermo_fold.py` | 75 |
| `mutate` | Function | `hard_mode/thermo_fold.py` | 89 |
| `fitness` | Function | `hard_mode/thermo_fold.py` | 97 |
| `run` | Function | `hard_mode/thermo_fold.py` | 106 |
| `plot_melting_curve` | Function | `hard_mode/thermo_fold.py` | 131 |
| `main` | Function | `genomic_discovery.py` | 17 |
| `parse_fasta` | Function | `alphagenome_utils.py` | 56 |
| `fix_pdb` | Function | `hard_mode/physics_verify.py` | 48 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `Calculate_fitness → _scan_motifs` | intra_community | 5 |
| `Run → _calculate_thermodynamics` | intra_community | 5 |
| `Verify_thermal_switch → _local_score` | cross_community | 5 |
| `Main → _pad` | cross_community | 4 |
| `Main → _cage_score` | cross_community | 4 |
| `Main → _histone_level` | cross_community | 4 |
| `Generate_sequence → _local_score` | intra_community | 4 |
| `Plot_melting_curve → _calculate_thermodynamics` | intra_community | 4 |
| `Verify_thermal_switch → _indices_to_string` | cross_community | 4 |
| `Main → _local_expression` | cross_community | 3 |

## Connected Areas

| Area | Connections |
|------|-------------|
| Cluster_5 | 1 calls |

## How to Explore

1. `gitnexus_context({name: "score"})` — see callers and callees
2. `gitnexus_query({query: "hard_mode"})` — find related execution flows
3. Read key files listed above for implementation details
