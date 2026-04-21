# MEMORY.md

## Project

`bsm-public-rf`

## Core rule

Always audit the exact live repo on disk first.
Do not trust prior generated bundles, summaries, or assumptions over the local repo state.
Make only cumulative fixes.

## Current objective

Harden and finalize `bsm-public-rf` into a fully tested, documented, and reproducible public package for the reduced-form modeling workflow applied to the Biomass Scenario Model (BSM).

## Confirmed live repo state from this session

This memory update is based on the live repo copy that was edited and revalidated in this session.

### Tracked modified files currently present in the live repo copy

- `README.md`
- `docs/MEMORY.md`
- `docs/api.rst`
- `pixi.toml`
- `src/bsm_rfm/__init__.py`
- `src/bsm_rfm/final_ols.py`
- `src/bsm_rfm/regularized_screening.py`
- `src/bsm_rfm/workflow.py`
- `test_repo.sh`
- `tests/test_ci_contract.py`
- `tests/test_final_ols.py`
- `tests/test_gate_contract.py`
- `tests/test_markdown_formatting_contract.py`
- `tests/test_regularized_screening.py`
- `tests/test_tools.py`
- `tests/test_workflow.py`
- `tools/check_markdown.py`
- `tools/format_markdown.py`
- `tools/notebook_hygiene.py`
- untracked new file: `tools/markdown_files.py`

## Confirmed engineering state

The repo-engineering layer has been materially hardened relative to the earlier scaffold state.

### Gate / tooling changes now present in the live repo

- `tools/markdown_files.py` exists and is the shared Markdown discovery contract.
- `tools/format_markdown.py` and `tools/check_markdown.py` both use the shared Markdown discovery logic.
- `pixi.toml` was updated so Markdown tooling runs as Python modules rather than assuming `tools` is importable as a package when scripts are executed directly.
- `tools/notebook_hygiene.py` now repairs the narrow malformed-code-cell corruption mode where literal escaped control sequences such as `\n` appear in notebook code instead of real newlines.
- notebook hygiene now also validates code-cell syntax rather than only stripping outputs and execution counts.
- `test_repo.sh` was hardened so missing Pixi fails with an explicit error instead of exiting silently via `set -e` during command substitution.
- `test_repo.sh --clean` was hardened to re-enter through the actual script path rather than relying on `$0` semantics.
- repo-engineering tests were shifted away from brittle shell-literal overfitting and toward behavior-oriented contract checks.

## Confirmed scientific/workflow state

The repo is no longer only a scaffold. It now contains executable scientific workflow foundations and a canonical orchestration layer.

### Implemented scientific foundations now present in the live repo

#### `src/bsm_rfm/regularized_screening.py`

- `ScreeningSelectionResult`
- `fit_multitask_elastic_net_screen(...)`
- `screening_selection_table(...)`

#### `src/bsm_rfm/final_ols.py`

- `FinalOLSFitResult`
- `fit_final_ols(...)`
- `predict_final_ols(...)`
- `make_holdout_nrmse_summary(...)`
- `build_postfit_artifacts(...)`

#### `src/bsm_rfm/workflow.py`

- canonical orchestration entrypoint `run_canonical_workflow(...)`
- export-writing support via `write_postfit_bundle(...)`
- workflow result container `CanonicalWorkflowRun`
- workflow provenance status was updated to reflect the implemented state rather than the earlier scaffold-only state

### What the canonical workflow now covers

- training/holdout split assumptions are already handled upstream by the existing data layer
- screening on training data
- final OLS fit on retained features
- holdout bootstrap nRMSE summary generation
- canonical post-fit artifact assembly
- export writing to disk
- manifest-aware bundle writing and reload path through `bsm_rfm.viz_io.load_postfit_bundle`

## Confirmed test state from this session

The following were actually run in this environment and passed:

- `python -m pytest -q`
- `python tools/compile_check.py`
- `python tools/import_smoke.py`
- `python tools/clean_transients.py`
- `python tools/check_repo.py`

Important nuance:

- `python tools/check_repo.py` failed immediately after running tests because tests created `__pycache__` directories.
- after running `python tools/clean_transients.py`, `python tools/check_repo.py` passed.
- this means transient cleanup remains part of the real repo hygiene contract.

### Not run / not claimable from this environment

Do **not** claim local Pixi or CI success from this session.
These were not runnable here because the container does not have Pixi installed:

- `pixi run test`
- `./test_repo.sh --fix`
- `./test_repo.sh --check`

Also do not claim docs build success from this session unless re-run locally in the Pixi environment.
Earlier attempts in this container failed because Sphinx was unavailable here.

## Current testing-regime assessment

The testing regime is in better shape than before, but it still needs final local-copy validation.

### Current strengths

- meaningful behavioral tests exist for the data, metrics, screening, final OLS, workflow, artifact, and repo-tooling layers
- the repo now has real end-to-end workflow tests rather than only isolated helper tests
- repo-engineering tests are less brittle than before
- notebook hygiene and Markdown discovery are now closer to real gate behavior

### Remaining testing/finalization questions to answer from the local copy

- do `pixi run test`, `./test_repo.sh --fix`, and `./test_repo.sh --check` all pass on the local repo exactly as modified?
- does docs build pass locally inside the Pixi environment?
- do pre-commit, local gate, and CI remain aligned after the cumulative scientific changes?
- are parquet/export semantics, manifest contents, and public API expectations fully consistent with the intended public package surface?

## Recommended next phase from the local repo

Start from the local copy and do a strict finalization audit.

### Priority order

1. **Local gate validation and repair**
   - run the actual Pixi/gate commands locally
   - fix any remaining root-cause failures until `pixi run test`, `./test_repo.sh --fix`, and `./test_repo.sh --check` all pass
1. **Docs/API finalization**
   - rebuild docs locally
   - verify the API docs reflect the now-implemented workflow surface
   - remove any stale scaffold language still contradicted by the live code
1. **Workflow/public-package reconciliation**
   - verify `README.md`, `docs/api.rst`, exported symbols, artifact schema, and workflow semantics are mutually consistent
1. **Case-study and feature-expansion reconciliation**
   - audit what parts of the original BSM notebook workflow are still missing from the canonical module path
   - especially verify feature-expansion provenance and any remaining notebook-derived drift

## Engineering rules

- make only cumulative changes
- fix root causes
- no hacks, no shims, no compatibility layers unless explicitly requested
- keep local and CI behavior aligned
- do not claim anything passes unless it was actually run
- prefer behavior-oriented tests over brittle implementation-string assertions unless the literal string is itself an intentional contract

## Validation expectations for future changes

If repo-engineering or docs files change, run at minimum:

- `pixi run test`
- `./test_repo.sh --fix`
- `./test_repo.sh --check`
- `git status --short`

If scientific/modeling code changes:

- run targeted tests first
- then the broader gate as appropriate

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

## Subsequent live-repo hardening phase

This follow-on update is also based on the exact live repo copy on disk.

### Additional cumulative fixes now present in the live repo copy

- `docs/conf.py` now derives the Sphinx `release` and `version` from `pyproject.toml` instead of duplicating a hardcoded version string.
- `docs/index.md` and `docs/overview.md` now describe the repo as a workflow package rather than a refactor scaffold.
- `src/bsm_rfm/__init__.py` now exports the canonical visualization-side bundle reload helpers:
  - `load_postfit_bundle`
  - `load_pipeline_outputs`
- new tests now guard:
  - docs/package version consistency
  - package-root export of the bundle reload helpers
  - API-doc coverage for `bsm_rfm.feature_expansion` and `bsm_rfm.viz_io`

### Actual validation run in this subsequent phase

The following were actually run in this environment and passed after the cumulative edits above:

- `python -m pytest -q tests/test_public_api.py tests/test_build_docs.py`
- `python -m pytest -q`
- `python tools/import_smoke.py`
- `python tools/compile_check.py`
- `python tools/clean_transients.py`
- `python tools/check_repo.py`

### Recommended next phase after this one

Once the local Pixi environment is available, verify:

- `pixi run docs`
- `pixi run package-build`
- `./test_repo.sh --fix`
- `./test_repo.sh --check`

Then fix any remaining release-facing issues exposed by those true local gate runs.

## Subsequent live-repo docs-finalization phase

This update is based on the exact live repo copy after the local Pixi path was reported
passing by the user.

### Additional cumulative fixes now present in the live repo copy

- Added `docs/quickstart.md` with a direct user-facing guide for
  `run_canonical_workflow(...)`, `write_postfit_bundle(...)`, and
  `load_postfit_bundle(...)`.
- Added `docs/export_bundle.md` documenting the canonical bundle layout,
  `manifest.json`, and the `postfit_diagnostics/` table contract.
- Updated `docs/index.md` so the built docs tree includes the new quickstart and bundle
  contract guides.
- Rewrote `docs/module_plan.md` from an outdated proposed layout into a current module map
  that matches the implemented package.
- Rewrote `docs/workflow_audit.md` and `docs/manuscript_summary_log.md` to remove stale
  initial/refactor framing while preserving the recovered provenance details.
- Updated package docstrings in `src/bsm_rfm.workflow` and `src/bsm_rfm.null_screening`
  to describe the live package rather than a refactor scaffold.
- Added docs regression tests guarding the new user guides and guarding against obsolete
  module names in `docs/module_plan.md`.

### Actual validation run in this docs-finalization phase

The following were actually run in this environment and passed after the cumulative edits:

- `python -m pytest -q tests/test_build_docs.py tests/test_public_api.py tests/test_workflow.py`
- `python -m pytest -q`
- `python tools/import_smoke.py`
- `python tools/compile_check.py`
- `python tools/clean_transients.py`
- `python tools/check_repo.py`

### Recommended next phase after this one

The highest-value remaining work is release and publication polish:

- tighten package metadata for public release if needed
- verify final docs titles and user-facing language are exactly what should ship
- decide whether any remaining notebook-derived helper behavior should be promoted into the
  canonical workflow or remain explicitly out of scope
