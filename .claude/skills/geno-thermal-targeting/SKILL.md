---
name: geno-thermal-targeting
description: >-
  End-to-end design loop for temperature-gated, tumor-selective therapeutics:
  genomic target discovery (AlphaGenome) -> binder fold/dock (Boltz-2) ->
  thermal-switch protein design -> independent BioNeMo verification. Use this
  skill whenever the user wants to discover a genomic target, design a binder or
  thermo-switch protein, screen a candidate library, or run the Geno-Thermal
  pipeline. Triggers: "design a thermal switch", "score this target", "dock these
  peptides", "run the geno-thermal pipeline", "verify this binder", "Geno-Thermal".
license: MIT
---

# Geno-Thermal Targeting

A reproducible, auditable pipeline for designing therapeutics that switch ON only
inside a heated tumor (≈43°C) and stay OFF in healthy tissue (37°C). Every step is
backed by project code and exposed as MCP tools via the `geno-thermal-targeting`
server (see `.mcp.json`).

## When to use this skill

Use it for any of: scoring a genomic locus, designing/screening binders against a
receptor, evolving a temperature-gated ("thermal switch") protein, or running the
full multi-phase pipeline — then **independently verifying** the result.

## The design loop (run in order)

1. **Discover** — call `discover_target(target_gene, mutated_seq?)`.
   Returns a SUPER_ENHANCER vs NORMAL call + epigenetic profile + confidence.
   Gate: proceed only if `classification == "SUPER_ENHANCER"` (a tumor-active locus).

2. **Design ligands** — call `design_ligands(candidates_file, target_seq?, use_flash?)`.
   Folds/docks each candidate with Boltz-2 (peptides via `seq`, small molecules via
   `smiles`) and returns a ranked library with interface confidence (`plddt_score` =
   ipTM*100), `binding_class`, and predicted affinity. `use_flash` defaults to True
   (fans out across the serverless GPU fleet); pass `use_flash=False` to force an
   in-process local fold instead.

3. **Design the thermal switch** — call `design_thermal_switch(scaffold, generations)`.
   Evolves a protein with a sharp folded(37°C)→unfolded(43°C) transition. The key
   output is `switch_delta` (pLDDT@37 − pLDDT@43): bigger = sharper gating.

4. **Verify independently** — for each top binder, call
   `verify_with_bionemo(target_seq, binder_seq, project_plddt)`.
   Re-folds the complex on NVIDIA's **BioNeMo Boltz-2 NIM** (a model the project did
   NOT design) and returns a verdict: `CORROBORATED`, `WEAK_AGREEMENT`, or
   `DIVERGENT_FLAG_FOR_REVIEW`. **Only present a candidate as validated if the
   independent verifier corroborates the project's own confidence.** This is the
   adversarial-reviewer pattern: a second, independent model must agree.

5. **(Optional) Evolve a promoter on RunPod Flash** — call
   `design_promoter_flash(use_flash=True, smoke=True)`. Runs the GA promoter
   optimizer and, when Flash genuinely engages, fans the per-individual fitness
   scoring out across the serverless GPU/CPU fleet (0→N→0). The response's `mode`
   field reports what actually happened (`FLASH` vs `LOCAL`) rather than echoing the
   request — check it before claiming a fan-out demo occurred.

6. **(Optional) Full pipeline** — `run_full_pipeline(smoke=True)` runs all 12 phases
   (discovery → intel → fold → thermo → nano-topology → bio-circuit → evolver → RL →
   physics → viz → summary) and writes artifacts under `outputs/`.

**One-call alternative** — `screen_and_verify(target_gene, candidates_file, use_flash)`
chains steps 1, 2, (5 when `use_flash=True`) and 4 into a single ranked report; it is
the headline demo artifact. It enforces the discovery gate (refuses to report
validated candidates if the gate failed) and flags `stale_data_warning: true` if step 2
returned a previously-committed library instead of a fresh fold.

## Reporting

Always report, per candidate: the AlphaGenome target call, the Boltz-2 interface
confidence (and whether it came from a fresh fold or a stale committed library — check
`stale`/`data_freshness`), the thermal `switch_delta`, and the BioNeMo verdict. Treat a
`DIVERGENT_FLAG_FOR_REVIEW` as a blocker — say so plainly rather than overclaiming. If
`design_promoter_flash` was called, report its `mode` honestly — never describe a run as
"on RunPod Flash" when `mode == "LOCAL"`.

## Provenance & limits

- No `ALPHAGENOME_API_KEY` → discovery uses the project's deterministic local model
  (clearly flagged via `mode`). No `NVIDIA_API_KEY` → verification returns a
  labelled local second opinion instead of the real NIM. State which mode produced
  each number.
- Compute: `use_flash`/`flash` default to True (RunPod Flash GPUs); pass `False` to
  force local folding, which self-skips if the local GPU toolchain is absent. Check the
  returned `mode`/`stale` fields — they report what actually happened, not just what
  was requested, since a Flash call can silently fall back to local.
- Artifacts (CSVs, structures, figures) land in `outputs/` and are the auditable
  record of a run.
