# MEMORY.md

## Project

`bsm-public-rf`

## Current objective

Build a fully tested, documented, and reproducible public package and workflow for the reduced-form modeling process applied to the Biomass Scenario Model (BSM), using the audited live repo as the only source of truth.

## Current confirmed repo state

As of the latest completed step:

- local `./test_repo.sh --fix` passes
- local `./test_repo.sh --check` passes
- local `pixi run test` passes
- GitHub CI passes
- docs artifacts upload successfully after the CI workflow rebuilds docs post-gate
- the local engineering layer has been hardened enough that formatting, docs, package build, and CI/gate contracts are now aligned

## Important recent engineering changes

The following repo-engineering state was established and should be treated as current unless the live repo proves otherwise:

- `pixi.toml` now includes explicit task names for:
  - `format-python`
  - `format-markdown`
  - `fix-notebooks`
  - `markdown-check`
  - `notebook-check`
  - `lint`
  - `format-check`
  - `unit-tests`
  - `workflow-tests`
  - `compile-check`
  - plus broader gate/build/docs tasks
- `pixi.toml` includes the necessary tooling dependencies for the gate, including:
  - `ruff`
  - `mdformat`
  - `mdformat-gfm`
  - `nbformat`
  - `sphinx`
  - `myst-parser`
  - `build`
  - `pre-commit`
- `tools/check_markdown.py` was added because the gate referenced it but it did not exist
- `.pre-commit-config.yaml` and `test_repo.sh` were aligned to the actual Pixi task contract
- `test_repo.sh` was adjusted to satisfy explicit repo-engineering tests that check for exact strings and ordering, including:
  - `echo ">>> $PIXI_BIN install"`
  - `echo ">>> $PIXI_BIN install --locked"`
  - `run_python_smoke`
  - `run_task build-import-smoke` before `run_task clean-transients`

## Important caution about the testing regime

We now need to reassess the testing strategy and repo best practices from the live repo itself.

Working hypothesis to audit:

- `test_repo.sh` should prepare the repo for a successful commit and push
- but the test suite should not be overly dependent on asserting exact `test_repo.sh` implementation details unless those contracts are truly intentional and valuable
- repo-engineering tests may currently be too coupled to shell-script literals and ordering
- we should audit whether those tests are enforcing real behavior or just freezing incidental implementation

Do not assume this hypothesis is correct until the live repo is audited.

## Scientific/workflow state

The repo started from a stabilized packaging/gate/docs/CI layer and then moved into scientific workflow reconciliation.

Previously identified scientific gaps included:

- workflow provenance boundary
- feature expansion boundary
- regularized screening boundary
- final OLS handoff/export boundary
- evaluation/export reconciliation
- notebook-to-module migration and notebook drift
- canonical provenance of actual modeling workflow steps and artifacts

However, do **not** trust prior summaries or generated bundles over the live repo. Re-audit the live repo to determine what is actually present now.

## Core workflow rule

Always audit the exact live repo first.
Do not patch from assumptions.
Do not trust prior patch bundles, summaries, or claimed repo state over the uploaded live repo.

## Engineering rules

- Make only cumulative changes
- Fix root causes
- No hacks, no shims, no compatibility layers unless explicitly requested
- Keep local and CI behavior aligned
- Do not claim anything passes unless it was actually run
- Prefer tests that validate behavior and contracts over brittle implementation-string assertions, unless the string-level contract is intentional and justified

## Next-session starting point

Start with a strict repo audit focused on:

1. engineering best practices
1. whether the current testing regime is appropriately designed
1. what scientific/modeling functionality is actually implemented vs only documented
1. what remains to reach a fully tested, documented, reproducible reduced-form modeling workflow package

## Validation expectations for future changes

If repo-engineering files are changed, run at minimum:

- `pixi run test`
- `./test_repo.sh --fix`
- `./test_repo.sh --check`
- inspect `git status --short`

If scientific/modeling code changes:

- run targeted tests first
- then run the broader relevant gate

## User preferences for this repo

- strict live-repo audit first
- focus on root-cause fixes
- minimal discussion when errors are provided
- provide exact updated files/scripts
- do not claim success without actual validation
- no compatibility layers unless explicitly requested
