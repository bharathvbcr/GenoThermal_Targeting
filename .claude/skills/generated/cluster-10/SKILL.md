---
name: cluster-10
description: "Skill for the Cluster_10 area of GenoThermal_Targeting. 10 symbols across 1 files."
---

# Cluster_10

10 symbols | 1 files | Cohesion: 95%

## When to Use

- Understanding how fix, setup, rmsd work
- Modifying cluster_10-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `flash_gpu_jobs.py` | _log_gpu, _make_env_class, _train_and_generate, _verify_physics, fix (+5) |

## Entry Points

Start here when exploring this area:

- **`fix`** (Function) — `flash_gpu_jobs.py:166`
- **`setup`** (Function) — `flash_gpu_jobs.py:184`
- **`rmsd`** (Function) — `flash_gpu_jobs.py:221`
- **`run`** (Function) — `flash_gpu_jobs.py:227`
- **`train_ppo_endpoint`** (Function) — `flash_gpu_jobs.py:274`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `fix` | Function | `flash_gpu_jobs.py` | 166 |
| `setup` | Function | `flash_gpu_jobs.py` | 184 |
| `rmsd` | Function | `flash_gpu_jobs.py` | 221 |
| `run` | Function | `flash_gpu_jobs.py` | 227 |
| `train_ppo_endpoint` | Function | `flash_gpu_jobs.py` | 274 |
| `verify_physics_endpoint` | Function | `flash_gpu_jobs.py` | 288 |
| `_log_gpu` | Function | `flash_gpu_jobs.py` | 30 |
| `_make_env_class` | Function | `flash_gpu_jobs.py` | 69 |
| `_train_and_generate` | Function | `flash_gpu_jobs.py` | 104 |
| `_verify_physics` | Function | `flash_gpu_jobs.py` | 145 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `Verify_physics_endpoint → Rmsd` | intra_community | 4 |
| `Verify_physics_endpoint → Fix` | intra_community | 4 |

## How to Explore

1. `context({name: "fix"})` — see callers and callees
2. `query({search_query: "cluster_10"})` — find related execution flows
3. Read key files listed above for implementation details
4. `explain({target: "<file or symbol>"})` — persisted taint findings (source→sink data flows), when indexed with `--pdg`
