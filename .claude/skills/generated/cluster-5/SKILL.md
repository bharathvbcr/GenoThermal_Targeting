---
name: cluster-5
description: "Skill for the Cluster_5 area of GenoThermal_Targeting. 6 symbols across 3 files."
---

# Cluster_5

6 symbols | 3 files | Cohesion: 80%

## When to Use

- Understanding how start, done work
- Modifying cluster_5-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `flash_gpu_jobs.py` | _ppo_sweep, _run, _await |
| `flash_metrics.py` | start, done |
| `boltz_designer.py` | _await |

## Entry Points

Start here when exploring this area:

- **`start`** (Method) — `flash_metrics.py:42`
- **`done`** (Method) — `flash_metrics.py:47`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `start` | Method | `flash_metrics.py` | 42 |
| `done` | Method | `flash_metrics.py` | 47 |
| `_await` | Function | `boltz_designer.py` | 84 |
| `_ppo_sweep` | Function | `flash_gpu_jobs.py` | 314 |
| `_run` | Function | `flash_gpu_jobs.py` | 323 |
| `_await` | Function | `flash_gpu_jobs.py` | 330 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `Main → Start` | cross_community | 5 |
| `Main → Done` | cross_community | 5 |
| `Main → Start` | cross_community | 4 |
| `Main → Done` | cross_community | 4 |

## Connected Areas

| Area | Connections |
|------|-------------|
| Hard_mode | 1 calls |

## How to Explore

1. `context({name: "start"})` — see callers and callees
2. `query({search_query: "cluster_5"})` — find related execution flows
3. Read key files listed above for implementation details
4. `explain({target: "<file or symbol>"})` — persisted taint findings (source→sink data flows), when indexed with `--pdg`
