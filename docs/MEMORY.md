# BSM_RFM_MEMORY_NEXT_CHAT.md

## Project
BSM reduced-form modeling workflow refactor in repo `bsm-public-rf`.

## Core objective
Rebuild the notebook-driven workflow into a clean, tested, documented package structure based on the **actual canonical source scripts**, not notebook drift, while hardening the repo engineering layer so `./test_repo.sh --fix` truly prepares the repo for a clean commit and CI pass.

## Non-negotiable engineering rules from user
- Audit the **actual live repo on disk first**
- Do **not** trust prior bundles, summaries, or claims without verifying on-disk state
- No hacks
- No shims
- No compatibility layers unless explicitly requested
- No warning suppression as a substitute for source fixes
- Fix root causes, not symptoms
- Use rigorous tests
- Keep local and CI behavior aligned
- Report only what is actually verified on disk

## Canonical scientific provenance already established
Treat these as canonical unless the audited live source code contradicts them:
- Original simulator sample size: **300,000**
- Modeling subset size: **20,000**
- Canonical subset-generation path:
  - stratify by AFSC/UAEORO boolean scenario combination
  - there are 4 scenario strata
  - sample **5,000** run IDs independently within each stratum
  - recombine into a balanced **20,000-row** modeling sample
- `null_distribution.py` is canonical for the null-model / null-screening calculations
- Current notebooks are not canonical scientific truth; they contain fused notebook drift and later extensions

## Repo requirements
The repo is intended to use:
- Pixi
- Python 3.12
- formatting/linting for Python, Markdown, and notebooks
- NumPy docstrings
- Sphinx package-built docs
- GitHub CI
- a robust `test_repo.sh`

## Important known engineering history
These issues already occurred and should not be repeated:

### 1. Sphinx duplicate object description failure
Earlier docs build failed with warnings-as-errors for duplicate object descriptions on fields of `bsm_rfm.data.StandardizationBundle`.

Root cause already diagnosed from the uploaded archive:
- not duplicate page inclusion
- not autosummary duplication
- instead: interaction between Napoleon rendering of NumPy `Attributes` plus autodoc/dataclass field member documentation

A fix using:
- `napoleon_use_ivar = True`
in `docs/conf.py` was identified as the correct source-level docs fix.

This must still be verified against the actual live repo on disk in the next chat before assuming it is present.

### 2. Markdown formatting on Python 3.12
`.mdformat.toml` previously used unsupported `exclude = [...]` behavior requiring Python 3.13.
Correct direction:
- keep repo on Python 3.12
- do path exclusion in wrapper logic, not unsupported mdformat config

This also must be audited in the actual live repo.

### 3. Package-build gate went off the rails
The package-build path became confused across multiple contradictory fixes.

What is now known:

- A custom `tools/package_build.py` helper was introduced and later recognized as unnecessary over-engineering.
- The user explicitly wants the **simplest best-practice** build test.
- The correct simple contract should be:
  - no custom build helper
  - Pixi task directly runs `python -m build --no-isolation --sdist --wheel`
  - `test_repo.sh` syncs the env first, then runs the task
- The repo was changed toward:
  - deleting `tools/package_build.py`
  - deleting `tests/test_package_build.py`
  - putting `package-build = "python -m build --no-isolation --sdist --wheel"` directly in `pixi.toml`

### 4. Current blocking problem is still unresolved
Despite the simplification above, the user still reports:

```text
>>> pixi run package-build
✨ Pixi task (package-build): python -m build --no-isolation --sdist --wheel
/Users/dhetting/src/bsm-public-rf/.pixi/envs/default/bin/python: No module named build
```

This is the current blocker.

### 5. Important contradictory build-dependency attempts already tried
Several mutually inconsistent approaches were tried. The next chat must not assume any of them are correct without auditing the live repo:

Attempt history:
- `python-build` as Pixi conda dependency
- `build` under `[pypi-dependencies]`
- custom `tools/package_build.py`
- tests that asserted `python-build`
- later tests updated to assert `[pypi-dependencies] build`

Because of this churn, the next chat must audit:
- the actual current `pixi.toml`
- whether `pixi.lock` is in sync
- whether `tools/package_build.py` still exists
- whether `tests/test_package_build.py` still exists
- whether `test_repo.sh` is aligned with current tasks
- whether CI config mirrors the same contract

Do not assume consistency.

## User’s desired final engineering outcome
The user wants something simple and standard:
- `./test_repo.sh --fix` should fully prepare the repo for commit
- after it succeeds, pre-commit and CI should also pass
- no custom over-engineered wrapper logic unless truly necessary
- use Pixi best practices
- keep the build test basic and straightforward

## Most likely next priority
The next chat should start with a strict repo audit and determine the **actual live root cause** of the persistent:
- `No module named build`
failure during `pixi run package-build`

Possible sources to verify from live repo and live environment:
- `pixi.toml` dependency placement
- `pixi.lock` not synced with `pixi.toml`
- CI environment/task mismatch
- stale deleted files still referenced
- task definitions not matching manifest structure expected by current Pixi version
- build dependency not actually installed because of incorrect dependency section or manifest syntax
- some other live-repo inconsistency

Do not guess. Audit and prove.

## Required first-pass audit for new chat
Before proposing changes, inspect and report findings from at least:
- `git status --short`
- `pixi.toml`
- `pixi.lock`
- `pyproject.toml`
- `test_repo.sh`
- `.github/workflows/...`
- `docs/conf.py`
- `src/bsm_rfm/data.py`
- docs pages referencing `StandardizationBundle`
- whether `tools/package_build.py` exists
- whether `tests/test_package_build.py` exists
- `tests/test_gate_contract.py`
- relevant repo tooling under `tools/`

Also verify from the live environment:
- `pixi list`
- `pixi run python -c "import build; print(build.__file__)"`
- `pixi run package-build`

## Required approach for next chat
1. Audit actual live repo on disk
2. Reproduce current build failure
3. Determine actual root cause of the missing `build` module in the Pixi task
4. Fix it using the **simplest Pixi best-practice** approach
5. Remove stale/contradictory tests or helpers
6. Re-run the full gate
7. Confirm repo cleanliness after the gate
8. Only then proceed to the next repo-hardening or workflow step

## Validation expectations before claiming success
Must actually run and report results for relevant checks, including as applicable:
- `pixi install`
- `pixi run python -c "import build; print(build.__file__)"`
- `pixi run package-build`
- `./test_repo.sh --fix`
- `./test_repo.sh --check`
- unit tests
- docs build
- notebook checks / execution checks
- repo cleanliness check after gate

## Deliverable style preference
If file downloads fail, provide raw markdown or raw file contents directly in chat.
