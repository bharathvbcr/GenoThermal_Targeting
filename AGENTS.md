<!-- repo-local:start -->
# Repo-local Agent Rules

## graphify

- **graphify** (`~/.Codex/skills/graphify/SKILL.md`) - any input to knowledge graph. Trigger: `/graphify`
- When the user types `/graphify`, invoke the Skill tool with `skill: "graphify"` before doing anything else.

## GitNexus Runtime Notes

- The GitNexus repo name is `Geno-Thermal_Targeting`; always pass this repo explicitly because this machine has multiple GitNexus indexes.
- CLI examples: `gitnexus context AlphaGenomeClient -r Geno-Thermal_Targeting`, `gitnexus impact AlphaGenomeClient -r Geno-Thermal_Targeting`, `gitnexus cypher -r Geno-Thermal_Targeting "<query>"`.
- MCP tool calls should include `repo: "Geno-Thermal_Targeting"` when the tool supports a repo parameter.
- On this Windows install, LadybugDB `fts` and `VECTOR` extensions are disabled because they segfault natively. Prefer `gitnexus_context`, `gitnexus_impact`, `gitnexus_cypher`, resources, and generated skills over `gitnexus_query`.
- For concept lookup, use Cypher name/path search if full-text query is empty: `MATCH (n) WHERE toLower(n.name) CONTAINS 'term' OR toLower(n.filePath) CONTAINS 'term' RETURN n.name, n.filePath LIMIT 25`.
- Refresh the map after meaningful source or agent-file changes with `gitnexus analyze --force --skills`.

<!-- repo-local:end -->

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **GenoThermal_Targeting** (428 symbols, 796 relationships, 35 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> Index stale? Run `node .gitnexus/run.cjs analyze` from the project root — it auto-selects an available runner. No `.gitnexus/run.cjs` yet? `npx gitnexus analyze` (npm 11 crash → `npm i -g gitnexus`; #1939).

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows. For regression review, compare against the default branch: `detect_changes({scope: "compare", base_ref: "main"})`.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `query({search_query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `context({name: "symbolName"})`.
- For security review, `explain({target: "fileOrSymbol"})` lists taint findings (source→sink flows; needs `analyze --pdg`).

## Never Do

- NEVER edit a function, class, or method without first running `impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `rename` which understands the call graph.
- NEVER commit changes without running `detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/GenoThermal_Targeting/context` | Codebase overview, check index freshness |
| `gitnexus://repo/GenoThermal_Targeting/clusters` | All functional areas |
| `gitnexus://repo/GenoThermal_Targeting/processes` | All execution flows |
| `gitnexus://repo/GenoThermal_Targeting/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |
| Work in the Hard_mode area (52 symbols) | `.claude/skills/generated/hard-mode/SKILL.md` |
| Work in the Cluster_10 area (10 symbols) | `.claude/skills/generated/cluster-10/SKILL.md` |
| Work in the Cluster_4 area (8 symbols) | `.claude/skills/generated/cluster-4/SKILL.md` |
| Work in the Cluster_0 area (7 symbols) | `.claude/skills/generated/cluster-0/SKILL.md` |
| Work in the Cluster_7 area (7 symbols) | `.claude/skills/generated/cluster-7/SKILL.md` |
| Work in the Cluster_5 area (6 symbols) | `.claude/skills/generated/cluster-5/SKILL.md` |
| Work in the Cluster_1 area (4 symbols) | `.claude/skills/generated/cluster-1/SKILL.md` |
| Work in the Cluster_2 area (4 symbols) | `.claude/skills/generated/cluster-2/SKILL.md` |
| Work in the Cluster_9 area (4 symbols) | `.claude/skills/generated/cluster-9/SKILL.md` |
| Work in the Cluster_30 area (4 symbols) | `.claude/skills/generated/cluster-30/SKILL.md` |
| Work in the Cluster_6 area (3 symbols) | `.claude/skills/generated/cluster-6/SKILL.md` |

<!-- gitnexus:end -->

## Claude Science MCP Server

- The pipeline is exposed to Claude as an MCP server: `mcp_geno_thermal.py`, registered in `.mcp.json` under `geno-thermal-targeting`.
- It must run under `.venv-flash/bin/python` — that is the only environment with `numpy`/`pandas`/`alphagenome` and `mcp` installed together; the plain `venv/bin/python` lacks `numpy` and will fail.
- Tools: `discover_target`, `design_ligands`, `design_thermal_switch`, `verify_with_bionemo`, `run_full_pipeline`, `design_promoter_flash`, `screen_and_verify` — each wraps existing project code rather than reimplementing it (see `mcp_geno_thermal.py` module docstring for the mapping). `design_promoter_flash` runs the GA on the RunPod Flash fleet and returns the live 0→N→0 autoscaling metrics; `screen_and_verify` includes them when `use_flash=True`.
- The companion skill `.claude/skills/geno-thermal-targeting/SKILL.md` documents the discover -> design -> verify loop and the adversarial-verification rule (only report a candidate validated if BioNeMo corroborates the project's own confidence).
- Smoke-test changes to the server with `.venv-flash/bin/python mcp_geno_thermal.py --selftest` before relying on it through an MCP client.
