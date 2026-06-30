---
name: cluster-4
description: "Skill for the Cluster_4 area of GenoThermal_Targeting. 8 symbols across 2 files."
---

# Cluster_4

8 symbols | 2 files | Cohesion: 80%

## When to Use

- Understanding how main, run_panel, selectivity_table work
- Modifying cluster_4-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `boltz_designer.py` | _save_structure, _fold_remote, _fold_local, main |
| `target_panel.py` | _load, run_panel, selectivity_table, main |

## Entry Points

Start here when exploring this area:

- **`main`** (Function) — `boltz_designer.py:110`
- **`run_panel`** (Function) — `target_panel.py:46`
- **`selectivity_table`** (Function) — `target_panel.py:65`
- **`main`** (Function) — `target_panel.py:92`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `main` | Function | `boltz_designer.py` | 110 |
| `run_panel` | Function | `target_panel.py` | 46 |
| `selectivity_table` | Function | `target_panel.py` | 65 |
| `main` | Function | `target_panel.py` | 92 |
| `_save_structure` | Function | `boltz_designer.py` | 54 |
| `_fold_remote` | Function | `boltz_designer.py` | 68 |
| `_fold_local` | Function | `boltz_designer.py` | 104 |
| `_load` | Function | `target_panel.py` | 37 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `Main → _ligand_smiles` | cross_community | 6 |
| `Main → Pct` | cross_community | 6 |
| `Main → _ligand_smiles` | cross_community | 5 |
| `Main → Pct` | cross_community | 5 |
| `Main → _log_gpu` | cross_community | 5 |
| `Main → _sanitize` | cross_community | 5 |
| `Main → Classify_binding` | cross_community | 5 |
| `Main → Start` | cross_community | 5 |
| `Main → Done` | cross_community | 5 |
| `Main → _log_gpu` | cross_community | 4 |

## Connected Areas

| Area | Connections |
|------|-------------|
| Hard_mode | 1 calls |
| Cluster_7 | 1 calls |
| Cluster_5 | 1 calls |

## How to Explore

1. `context({name: "main"})` — see callers and callees
2. `query({search_query: "cluster_4"})` — find related execution flows
3. Read key files listed above for implementation details
4. `explain({target: "<file or symbol>"})` — persisted taint findings (source→sink data flows), when indexed with `--pdg`
