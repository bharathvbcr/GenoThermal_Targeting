---
name: cluster-7
description: "Skill for the Cluster_7 area of GenoThermal_Targeting. 7 symbols across 1 files."
---

# Cluster_7

7 symbols | 1 files | Cohesion: 78%

## When to Use

- Understanding how classify_binding, fold_complex, fold_endpoint work
- Modifying cluster_7-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `flash_boltz.py` | _log_gpu, classify_binding, _sanitize, _ligand_smiles, _build_yaml (+2) |

## Entry Points

Start here when exploring this area:

- **`classify_binding`** (Function) — `flash_boltz.py:63`
- **`fold_complex`** (Function) — `flash_boltz.py:101`
- **`fold_endpoint`** (Function) — `flash_boltz.py:212`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `classify_binding` | Function | `flash_boltz.py` | 63 |
| `fold_complex` | Function | `flash_boltz.py` | 101 |
| `fold_endpoint` | Function | `flash_boltz.py` | 212 |
| `_log_gpu` | Function | `flash_boltz.py` | 41 |
| `_sanitize` | Function | `flash_boltz.py` | 73 |
| `_ligand_smiles` | Function | `flash_boltz.py` | 78 |
| `_build_yaml` | Function | `flash_boltz.py` | 87 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `Main → _ligand_smiles` | cross_community | 6 |
| `Main → _ligand_smiles` | cross_community | 5 |
| `Main → _log_gpu` | cross_community | 5 |
| `Main → _sanitize` | cross_community | 5 |
| `Main → Classify_binding` | cross_community | 5 |
| `Main → _log_gpu` | cross_community | 4 |
| `Main → _sanitize` | cross_community | 4 |
| `Main → Classify_binding` | cross_community | 4 |
| `Fold_endpoint → _ligand_smiles` | intra_community | 4 |
| `Fold_endpoint → _log_gpu` | intra_community | 3 |

## How to Explore

1. `context({name: "classify_binding"})` — see callers and callees
2. `query({search_query: "cluster_7"})` — find related execution flows
3. Read key files listed above for implementation details
4. `explain({target: "<file or symbol>"})` — persisted taint findings (source→sink data flows), when indexed with `--pdg`
