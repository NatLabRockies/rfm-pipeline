# MEMORY.md

## Project

`bsm-public-rf`

## Core rule

Always audit the exact live repo on disk first.
Do not trust prior generated bundles, summaries, or assumptions over the local repo state.
Make only cumulative fixes.

## Current objective

Transition `bsm-public-rf` from a tested public workflow package into a **complete manuscript
reproduction package** for the JDS BSM manuscript.

## Manuscript-reproduction priority shift

The highest-priority work is no longer general package hardening. The repo now needs a stage-for-stage
reconstruction of the workflow described in the manuscript, with:

- a deterministic toy pipeline for automated validation;
- real-data notebooks that execute the full workflow once placeholder paths are updated;
- regenerated manuscript tables and figures;
- executable regression checks for the manuscript's reported counts and holdout metrics.

## Current active phase

**Phase 2 is complete.**

The manuscript contract is now frozen in `docs/manuscript_contract.md` and
`configs/manuscript_case_study.yml`. Manuscript-explicit values remain authoritative. Manuscript
ambiguities are now represented as explicit repo-frozen reconstruction decisions that later phases
must implement and that the manuscript must be revised to match.

## Phase 1 completion notes

Phase 1 added the executable manifests and placeholder paths for:

- the real case-study input matrix;
- the real case-study output matrix;
- the released feature catalog defining the exact 26,560-term candidate library;
- the notebook/runtime configuration needed to run the manuscript workflow end to end.

## Phase 2 completion notes

Phase 2 added:

- tracked notebook skeletons at every frozen manuscript stage path;
- a runtime/path-resolution layer in `bsm_rfm.manuscript_runtime`;
- deterministic demo-data fallback so the notebook gate can execute without the private case-study files;
- data-intake validation helpers that freeze the required artifact columns.

## Immediate next phase

Proceed to **Phase 3 — implement the manuscript-stage computations behind the notebook entrypoints**.

Phase 3 must implement, in source form and then wire into the notebooks:

- exact output conditioning and reduced-response representation;
- exact empirical-null screening;
- exact interaction discovery;
- exact nonlinear transformation discovery;
- exact sparse selection and stability filtering;
- exact manuscript tables and figures regeneration.

## Contract rules for subsequent phases

- manuscript-explicit case-study values outrank current package defaults;
- repo-frozen reconstruction decisions in the case-study YAML must be implemented literally unless
  the manuscript is revised and the contract is updated in lockstep;
- later code, scripts, and notebooks must read the manuscript case-study config rather than
  retyping constants;
- if executable truth differs from the manuscript, the repo must either preserve the manuscript
  contract and explain the discrepancy or update the manuscript to match the executable truth.

## User preferences for this repo

- strict live-repo audit first
- focus on root-cause fixes
- minimal discussion when errors are provided
- provide exact updated files/scripts
- provide zipped bundles in repo-relative folder structure
- provide an `rsync` command from `~/Downloads/{bundle}/` to `~/src/bsm-public-rf/`
- provide separate `git add` / `git rm` commands
- provide local Pixi validation commands
- provide git commit messages separately
- do not claim success without actual validation
- no compatibility layers unless explicitly requested
