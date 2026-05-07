---
name: cluster-3
description: "Skill for the Cluster_3 area of Geno-Thermal_Targeting. 8 symbols across 2 files."
---

# Cluster_3

8 symbols | 2 files | Cohesion: 100%

## When to Use

- Understanding how main, create_docking_job, create_batch_jobs work
- Modifying cluster_3-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `alphafold_utils.py` | AlphaFoldClient, _sanitize_name, create_docking_job, create_batch_jobs, _parse_result_dir (+2) |
| `ligand_designer.py` | main |

## Entry Points

Start here when exploring this area:

- **`main`** (Function) — `ligand_designer.py:17`
- **`create_docking_job`** (Function) — `alphafold_utils.py:40`
- **`create_batch_jobs`** (Function) — `alphafold_utils.py:59`
- **`parse_result_zip`** (Function) — `alphafold_utils.py:137`
- **`parse_all_results`** (Function) — `alphafold_utils.py:144`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `AlphaFoldClient` | Class | `alphafold_utils.py` | 20 |
| `main` | Function | `ligand_designer.py` | 17 |
| `create_docking_job` | Function | `alphafold_utils.py` | 40 |
| `create_batch_jobs` | Function | `alphafold_utils.py` | 59 |
| `parse_result_zip` | Function | `alphafold_utils.py` | 137 |
| `parse_all_results` | Function | `alphafold_utils.py` | 144 |
| `_sanitize_name` | Function | `alphafold_utils.py` | 34 |
| `_parse_result_dir` | Function | `alphafold_utils.py` | 86 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `Main → _parse_result_dir` | intra_community | 4 |
| `Main → _sanitize_name` | intra_community | 3 |

## How to Explore

1. `gitnexus_context({name: "main"})` — see callers and callees
2. `gitnexus_query({query: "cluster_3"})` — find related execution flows
3. Read key files listed above for implementation details
