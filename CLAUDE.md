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

This project is indexed by GitNexus as **Geno-Thermal_Targeting** (487 symbols, 719 relationships, 15 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, prefer `gitnexus_context`, `gitnexus_cypher`, resources, and generated skills per the repo-local runtime notes above. Use `gitnexus_query` only if the local FTS extension is available and returning results.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/Geno-Thermal_Targeting/context` | Codebase overview, check index freshness |
| `gitnexus://repo/Geno-Thermal_Targeting/clusters` | All functional areas |
| `gitnexus://repo/Geno-Thermal_Targeting/processes` | All execution flows |
| `gitnexus://repo/Geno-Thermal_Targeting/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |
| Work in the Hard_mode area (49 symbols) | `.claude/skills/generated/hard-mode/SKILL.md` |
| Work in the Cluster_3 area (8 symbols) | `.claude/skills/generated/cluster-3/SKILL.md` |
| Work in the Cluster_5 area (6 symbols) | `.claude/skills/generated/cluster-5/SKILL.md` |
| Work in the Cluster_2 area (3 symbols) | `.claude/skills/generated/cluster-2/SKILL.md` |

<!-- gitnexus:end -->
