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
- `./test_repo.sh --check` and CI must enforce the locked, already-prepared state.

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

## What was actually fixed in the live repo

### 1) Pixi/package build environment

The `build` import failure was eventually fixed in the **live repo workflow**, not by changing scientific code.

Current correct design:

- `pixi.toml` uses:
  - `[pypi-dependencies]`
  - `build = ">=1.2"`
- `pixi.toml` task:
  - `package-build = "python -m build --no-isolation --sdist --wheel"`
- `pixi.lock` contains the PyPI `build` wheel entries
- Pixi version on live machine: `0.59.0`

### 2) test_repo.sh gate behavior

The script was updated so it now:

- anchors to repo root
- resolves and uses a pinned Pixi executable path
- prints repo root and pixi binary for traceability
- runs an inline Python smoke check for `build`
- runs `build-import-smoke` immediately after environment sync
- uses:
  - `pixi install` for `--fix`
  - `pixi install --locked` for `--check` / `--ci`
- `--clean` removes `.pixi` and rebuilds through `--fix`

This fixed the previous situation where manual commands worked but the script failed.

### 3) Transient cleaner / gate tests

The cleaner/tests were updated so that:

- `.DS_Store` is removed as a transient artifact
- `.pixi` is preserved
- regression coverage exists for preserving Pixi env contents
- stale tests that still expected `python-build` were updated to the current `build`-via-PyPI design

## Latest verified gate status

`./test_repo.sh --fix` now progresses through:

- pixi install
- build import smoke
- clean-transients
- format-python
- format-markdown
- notebook hygiene
- repo hygiene
- lint
- format-check
- markdown-check
- notebook-check
- compile-check
- unit-tests
- workflow-tests
- notebook-tests

All of the above passed in the live repo run. The current failure is now in the docs step, which is good because it means the earlier gate issues were resolved. The failing docs output showed that `docs/MEMORY.md` is being scanned by Sphinx but is not included in any toctree, and warnings are treated as errors. :contentReference[oaicite:0]{index=0}

## Current blocker

Sphinx docs build fails with:

- `docs/MEMORY.md: WARNING: document isn't included in any toctree [toc.not_included]`
- warnings are treated as errors in the docs build
- this currently stops `./test_repo.sh --fix` at the docs step :contentReference[oaicite:1]{index=1}

## Most likely correct next step

Audit the docs configuration and fix the doc-structure issue correctly.

Likely valid repair directions to verify against the live repo:

1. If `docs/MEMORY.md` is intended to be published documentation, include it in a toctree.
1. If `docs/MEMORY.md` is only an internal engineering artifact, exclude it from Sphinx input via `docs/conf.py`.

Do not guess. Audit the live docs tree and choose the correct fix based on actual repo intent.

## Required audit files for the next chat

Before changing anything, inspect at least:

- `git status --short`
- `docs/conf.py`
- `docs/index.md` and/or `docs/index.rst`
- all files under `docs/`
- `tools/build_docs.py`
- `test_repo.sh`
- relevant tests covering docs / repo gate / CI
- `.github/workflows/*`

## Validation requirements before claiming success

After the next patch, run and report actual results for:

- `pixi run docs`
- `./test_repo.sh --fix`
- `git status --short`
- `./test_repo.sh --check`

Do not claim success unless the docs step passes and the repo is clean afterward.
