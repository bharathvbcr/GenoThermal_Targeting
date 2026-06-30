---
name: cluster-1
description: "Skill for the Cluster_1 area of GenoThermal_Targeting. 4 symbols across 2 files."
---

# Cluster_1

4 symbols | 2 files | Cohesion: 86%

## When to Use

- Understanding how main, parse_fasta, get_expression_score work
- Modifying cluster_1-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `alphagenome_utils.py` | parse_fasta, get_expression_score, _local_expression |
| `genomic_discovery.py` | main |

## Entry Points

Start here when exploring this area:

- **`main`** (Function) — `genomic_discovery.py:17`
- **`parse_fasta`** (Method) — `alphagenome_utils.py:57`
- **`get_expression_score`** (Method) — `alphagenome_utils.py:71`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `main` | Function | `genomic_discovery.py` | 17 |
| `parse_fasta` | Method | `alphagenome_utils.py` | 57 |
| `get_expression_score` | Method | `alphagenome_utils.py` | 71 |
| `_local_expression` | Method | `alphagenome_utils.py` | 152 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `Main → _pad` | cross_community | 4 |
| `Main → _cage_score` | cross_community | 4 |
| `Main → _histone_level` | cross_community | 4 |
| `Main → _local_expression` | intra_community | 3 |

## Connected Areas

| Area | Connections |
|------|-------------|
| Cluster_2 | 1 calls |

## How to Explore

1. `context({name: "main"})` — see callers and callees
2. `query({search_query: "cluster_1"})` — find related execution flows
3. Read key files listed above for implementation details
4. `explain({target: "<file or symbol>"})` — persisted taint findings (source→sink data flows), when indexed with `--pdg`
