---
name: cluster-2
description: "Skill for the Cluster_2 area of GenoThermal_Targeting. 4 symbols across 1 files."
---

# Cluster_2

4 symbols | 1 files | Cohesion: 86%

## When to Use

- Understanding how _api_expression, _pad, _cage_score work
- Modifying cluster_2-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `alphagenome_utils.py` | _api_expression, _pad, _cage_score, _histone_level |

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `_pad` | Function | `alphagenome_utils.py` | 81 |
| `_cage_score` | Function | `alphagenome_utils.py` | 102 |
| `_histone_level` | Function | `alphagenome_utils.py` | 116 |
| `_api_expression` | Method | `alphagenome_utils.py` | 78 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `Main → _pad` | cross_community | 4 |
| `Main → _cage_score` | cross_community | 4 |
| `Main → _histone_level` | cross_community | 4 |

## How to Explore

1. `context({name: "_api_expression"})` — see callers and callees
2. `query({search_query: "cluster_2"})` — find related execution flows
3. Read key files listed above for implementation details
4. `explain({target: "<file or symbol>"})` — persisted taint findings (source→sink data flows), when indexed with `--pdg`
