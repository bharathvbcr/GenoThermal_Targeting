---
name: cluster-9
description: "Skill for the Cluster_9 area of GenoThermal_Targeting. 4 symbols across 1 files."
---

# Cluster_9

4 symbols | 1 files | Cohesion: 75%

## When to Use

- Understanding how score_sequences, score_endpoint work
- Modifying cluster_9-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `flash_fitness.py` | _count, _local_score, score_sequences, score_endpoint |

## Entry Points

Start here when exploring this area:

- **`score_sequences`** (Function) — `flash_fitness.py:62`
- **`score_endpoint`** (Function) — `flash_fitness.py:101`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `score_sequences` | Function | `flash_fitness.py` | 62 |
| `score_endpoint` | Function | `flash_fitness.py` | 101 |
| `_count` | Function | `flash_fitness.py` | 44 |
| `_local_score` | Function | `flash_fitness.py` | 48 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `_ → _scan_motifs` | cross_community | 6 |
| `Score_endpoint → _scan_motifs` | cross_community | 6 |
| `_ → _count` | cross_community | 4 |
| `Score_endpoint → _count` | intra_community | 4 |

## Connected Areas

| Area | Connections |
|------|-------------|
| Hard_mode | 1 calls |

## How to Explore

1. `context({name: "score_sequences"})` — see callers and callees
2. `query({search_query: "cluster_9"})` — find related execution flows
3. Read key files listed above for implementation details
4. `explain({target: "<file or symbol>"})` — persisted taint findings (source→sink data flows), when indexed with `--pdg`
