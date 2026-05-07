---
name: cluster-5
description: "Skill for the Cluster_5 area of Geno-Thermal_Targeting. 6 symbols across 1 files."
---

# Cluster_5

6 symbols | 1 files | Cohesion: 91%

## When to Use

- Understanding how get_expression_score work
- Modifying cluster_5-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `alphagenome_utils.py` | get_expression_score, _api_expression, _pad, _cage_score, _histone_level (+1) |

## Entry Points

Start here when exploring this area:

- **`get_expression_score`** (Function) — `alphagenome_utils.py:67`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `get_expression_score` | Function | `alphagenome_utils.py` | 67 |
| `_api_expression` | Function | `alphagenome_utils.py` | 72 |
| `_pad` | Function | `alphagenome_utils.py` | 75 |
| `_cage_score` | Function | `alphagenome_utils.py` | 96 |
| `_histone_level` | Function | `alphagenome_utils.py` | 108 |
| `_local_expression` | Function | `alphagenome_utils.py` | 142 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `Main → _pad` | cross_community | 4 |
| `Main → _cage_score` | cross_community | 4 |
| `Main → _histone_level` | cross_community | 4 |
| `Main → _local_expression` | cross_community | 3 |

## How to Explore

1. `gitnexus_context({name: "get_expression_score"})` — see callers and callees
2. `gitnexus_query({query: "cluster_5"})` — find related execution flows
3. Read key files listed above for implementation details
