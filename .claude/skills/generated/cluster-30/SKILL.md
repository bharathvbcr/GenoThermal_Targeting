---
name: cluster-30
description: "Skill for the Cluster_30 area of GenoThermal_Targeting. 4 symbols across 1 files."
---

# Cluster_30

4 symbols | 1 files | Cohesion: 100%

## When to Use

- Understanding how run_step, check_openmm, main work
- Modifying cluster_30-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `run_pipeline.py` | run_step, check_openmm, main, step |

## Entry Points

Start here when exploring this area:

- **`run_step`** (Function) — `run_pipeline.py:19`
- **`check_openmm`** (Function) — `run_pipeline.py:45`
- **`main`** (Function) — `run_pipeline.py:54`
- **`step`** (Function) — `run_pipeline.py:90`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `run_step` | Function | `run_pipeline.py` | 19 |
| `check_openmm` | Function | `run_pipeline.py` | 45 |
| `main` | Function | `run_pipeline.py` | 54 |
| `step` | Function | `run_pipeline.py` | 90 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `Main → Run_step` | intra_community | 3 |

## How to Explore

1. `context({name: "run_step"})` — see callers and callees
2. `query({search_query: "cluster_30"})` — find related execution flows
3. Read key files listed above for implementation details
4. `explain({target: "<file or symbol>"})` — persisted taint findings (source→sink data flows), when indexed with `--pdg`
