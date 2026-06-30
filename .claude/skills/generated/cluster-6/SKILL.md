---
name: cluster-6
description: "Skill for the Cluster_6 area of GenoThermal_Targeting. 3 symbols across 1 files."
---

# Cluster_6

3 symbols | 1 files | Cohesion: 100%

## When to Use

- Understanding how fetch, trim, main work
- Modifying cluster_6-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `fetch_targets.py` | fetch, trim, main |

## Entry Points

Start here when exploring this area:

- **`fetch`** (Function) — `fetch_targets.py:39`
- **`trim`** (Function) — `fetch_targets.py:51`
- **`main`** (Function) — `fetch_targets.py:63`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `fetch` | Function | `fetch_targets.py` | 39 |
| `trim` | Function | `fetch_targets.py` | 51 |
| `main` | Function | `fetch_targets.py` | 63 |

## How to Explore

1. `context({name: "fetch"})` — see callers and callees
2. `query({search_query: "cluster_6"})` — find related execution flows
3. Read key files listed above for implementation details
4. `explain({target: "<file or symbol>"})` — persisted taint findings (source→sink data flows), when indexed with `--pdg`
