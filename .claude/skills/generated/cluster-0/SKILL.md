---
name: cluster-0
description: "Skill for the Cluster_0 area of GenoThermal_Targeting. 7 symbols across 2 files."
---

# Cluster_0

7 symbols | 2 files | Cohesion: 100%

## When to Use

- Understanding how main, create_docking_job, create_batch_jobs work
- Modifying cluster_0-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `alphafold_utils.py` | _sanitize_name, create_docking_job, create_batch_jobs, _parse_result_dir, parse_result_zip (+1) |
| `ligand_designer.py` | main |

## Entry Points

Start here when exploring this area:

- **`main`** (Function) — `ligand_designer.py:17`
- **`create_docking_job`** (Method) — `alphafold_utils.py:40`
- **`create_batch_jobs`** (Method) — `alphafold_utils.py:62`
- **`parse_result_zip`** (Method) — `alphafold_utils.py:151`
- **`parse_all_results`** (Method) — `alphafold_utils.py:160`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `main` | Function | `ligand_designer.py` | 17 |
| `create_docking_job` | Method | `alphafold_utils.py` | 40 |
| `create_batch_jobs` | Method | `alphafold_utils.py` | 62 |
| `parse_result_zip` | Method | `alphafold_utils.py` | 151 |
| `parse_all_results` | Method | `alphafold_utils.py` | 160 |
| `_sanitize_name` | Method | `alphafold_utils.py` | 34 |
| `_parse_result_dir` | Method | `alphafold_utils.py` | 91 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `Main → _parse_result_dir` | intra_community | 4 |
| `Main → _sanitize_name` | intra_community | 3 |

## How to Explore

1. `context({name: "main"})` — see callers and callees
2. `query({search_query: "cluster_0"})` — find related execution flows
3. Read key files listed above for implementation details
4. `explain({target: "<file or symbol>"})` — persisted taint findings (source→sink data flows), when indexed with `--pdg`
