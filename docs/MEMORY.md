# MEMORY.md

## Project

`bsm-public-rf`

## Core engineering rules

- Never patch from assumed state.
- Always audit the exact live repo before making changes.
- Make every patch cumulative.
- Fix root causes, not symptoms.
- No hacks, no shims, no compatibility layers unless explicitly requested.
- Do not suppress warnings instead of fixing them correctly.
- Do not claim anything passes unless it was actually run against the live repo.
- `./test_repo.sh --fix` must fully prepare the repo for a clean commit.
- `./test_repo.sh --check` and `./test_repo.sh --ci` must enforce the locked, already-prepared state.

## Scientific provenance that should be treated as canonical

- Original simulator sample size: 300,000
- Modeling subset size: 20,000
- Canonical subset path:
  - stratify by AFSC/UAEORO boolean scenario combination
  - 4 strata
  - sample 5,000 run IDs independently within each stratum
  - recombine into balanced 20k sample
- `null_distribution.py` is canonical for null-model / null-screening calculations
- notebooks are not canonical scientific truth and may contain drift

## What was actually stabilized in the live repo

### 1) Pixi/package build environment

The package-build path is now aligned and verified:

- `pixi.toml` uses:
  - `[pypi-dependencies]`
  - `build = ">=1.2"`
- `pixi.toml` task:
  - `package-build = "python -m build --no-isolation --sdist --wheel"`
- `pixi.lock` contains the PyPI `build` wheel entries
- live Pixi version observed during debugging: `0.59.0` locally and `0.67.0` on GitHub Actions

### 2) Repository gate behavior

`test_repo.sh` was corrected so that it now:

- anchors to repo root
- resolves and uses a pinned Pixi executable path
- prints repo root and pixi binary for traceability
- performs an inline Python smoke check for `build`
- runs `build-import-smoke` immediately after environment sync
- uses:
  - `pixi install` for `--fix`
  - `pixi install --locked` for `--check` / `--ci`
- `--clean` removes `.pixi` and rebuilds through `--fix`

This fixed the earlier mismatch where manual commands succeeded but the script failed.

### 3) Cleaner / transient handling

The cleaner was corrected so it:

- removes `.DS_Store`
- preserves `.pixi`
- preserves installed environment contents
- still removes repo build/cache transients such as `build`, `dist`, caches, and `docs/_build`

Regression coverage was added so the cleaner does not delete Pixi environment contents.

### 4) Markdown formatting contract

A CI/local mismatch was found and fixed:

- local formatting originally missed untracked Markdown files
- CI later failed on those files once they were committed
- `tools/format_markdown.py` was updated to operate on both tracked and untracked non-ignored Markdown files using:
  - `git ls-files --cached --others --exclude-standard`
- regression coverage was added for untracked Markdown formatting discovery

### 5) Docs build path and artifact handling

A docs artifact path mismatch was resolved:

- local/docs task had been building to a temp directory
- GitHub Actions tried to upload `docs/_build/html`
- `tools/build_docs.py` was changed to build deterministically to:
  - `docs/_build/html`
- CI artifact upload now targets that same deterministic path

A second CI mismatch was then fixed:

- the gate correctly cleans transients, including `docs/_build`
- this removed docs before the artifact upload step
- GitHub Actions was updated to:
  - run `./test_repo.sh --ci`
  - then rebuild docs with `pixi run docs`
  - then upload `docs/_build/html`

This preserves a strict clean gate while still publishing docs artifacts.

### 6) CI contract tests

The CI tests were updated to reflect the actual current contract:

- CI invokes `./test_repo.sh --ci`
- the locked install requirement is enforced in `test_repo.sh`
- docs are rebuilt before upload
- docs artifact upload targets `docs/_build/html`

## Latest verified repo status

At the end of this debugging sequence:

- local `./test_repo.sh --fix` passes
- local `./test_repo.sh --check` passes
- GitHub CI passes
- docs artifacts upload successfully after an explicit post-gate docs rebuild

This means the engineering layer is now in a substantially better state:

- local and CI are aligned
- package build is exercised
- markdown/doc formatting mismatches are caught locally
- docs upload path is deterministic

## Important design contracts that now exist

- `./test_repo.sh --fix` is the authoritative local prep path before commit
- `./test_repo.sh --check` is the locked local validation path
- `./test_repo.sh --ci` is the locked CI validation path
- docs build validation is part of the gate
- docs artifact publication is a separate post-gate CI step
- transient cleanup is allowed to remove docs build outputs because the artifact step rebuilds them explicitly

## Current likely next step

The engineering-layer stabilization for packaging/gate/docs/CI is now mostly complete.

The next chat should begin by auditing the live repo and deciding the next highest-value scientific/repo step. Plausible next-step categories to audit before choosing:

1. documentation cleanup and publication readiness
1. scientific workflow/module implementation status versus canonical report
1. notebook drift versus canonical module/script truth
1. missing tests around the actual reduced-form modeling pipeline
1. package/API cleanup for public release readiness

Do not assume which is next. Audit the live repo first and choose based on actual current state.

## Required audit files for the next chat

Before making changes, inspect at least:

- `git status --short`
- `MEMORY.md`
- `pixi.toml`
- `pyproject.toml`
- `test_repo.sh`
- `.github/workflows/*`
- `docs/`
- `src/bsm_rfm/`
- `tests/`
- `tools/`
- any module/script associated with null screening, subset generation, modeling workflow, and scientific provenance

## Scientific workflow reconciliation progress

- `bsm_rfm.null_screening` is the canonical adapter boundary for the recovered `null_distribution.py` source workflow.
- `bsm_rfm.regularized_screening` now encodes the recovered downstream screening divergence explicitly:
  - archived script path: `MultiTaskElasticNetCV`, 5% holdout, direct standardized multi-output response, tuning via `tune_vars.csv`
  - notebook-derived path: PCA-compressed response plus de-biased LASSO and final OLS handoff, 10% holdout, `9782` outputs retained after culling, `346` selected features
- The next unresolved scientific boundary after this stage is the final OLS handoff/export contract.

## Validation requirements before claiming success in future chats

Run and report actual results for all relevant checks touched by the next change. At minimum, if repo-engineering files are changed:

- `./test_repo.sh --fix`
- `git status --short`
- `./test_repo.sh --check`

If CI-relevant paths are changed, ensure the local contract still mirrors CI.
