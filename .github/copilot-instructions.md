# Copilot Instructions for `bsm-public-rf`

## Role

You are working inside a scientific/reproducibility repository, not a generic software project. Treat code, tests, docs, notebooks, configuration, and manuscript-facing artifacts as one coupled system.

Your job is to make rigorous, minimal, validated changes that preserve scientific meaning and improve reproducibility.

## Core repository goals

This repository implements and documents the public reduced-form modeling workflow for the BSM case study. The long-term goal is a fully tested, documented, reproducible workflow from manuscript input artifacts through screening, sparse selection, final OLS, manuscript tables/figures, and reproduction audit outputs.

The repository must distinguish clearly between:

- exact manuscript-recovered implementations;
- source-backed public surrogates;
- contract-only stages;
- not-yet-validated approximations.

Never upgrade a scientific exactness claim unless tests and provenance support it.

## Non-negotiable workflow

Before editing:

1. inspect the live repo state;
1. inspect relevant files and tests;
1. identify the root cause;
1. add or strengthen tests first;
1. implement the smallest correct fix;
1. run targeted tests;
1. run the full gate.

Do not assume prior chat summaries, old patches, or branch descriptions are accurate. The live repo is the source of truth.

## Required commands

Start every slice from a clean branch:

```bash
cd ~/src/bsm-public-rf
git switch main
git fetch origin
git pull --ff-only origin main
git status --short
git switch -C <short-branch-name>
```

Run targeted tests during development, then the full gate:

```bash
./test_repo.sh --fix
./test_repo.sh --check
```

Do not claim work is complete until `./test_repo.sh --check` passes locally.

## CI/local parity

GitHub CI and local behavior must match. If GitHub fails but local passes, that is a repo defect. Add a local guard that reproduces or prevents the CI failure.

Do not dismiss CI-only failures as platform noise. Treat them as missing deterministic tests, missing environment parity checks, or platform-sensitive numerical behavior.

## Current sensitive area

The manuscript demo sparse-selection chain has shown platform-sensitive behavior. GitHub Linux CI produced empty `final_stable_support` while local macOS tests passed. Downstream final-artifact and reproduction tests then failed because final artifact generation correctly refuses empty support.

Do not weaken this guard:

```text
Final manuscript artifacts require a non-empty final support.
```

Fix the upstream cause instead.

Before changing demo fixture values or sparse-selection thresholds, inspect sparse-selection diagnostics:

- candidate terms
- full support terms
- final stable support terms
- selected support per retained component
- stability selection frequency
- mean resample Jaccard
- mean resample Spearman
- pass/fail flags for stability thresholds

Any demo fixture must satisfy all feature-domain invariants, including `log1p_x2` requiring `x2 > -1`.

## De-biased-LASSO status

The de-biased-LASSO implementation is not complete. The repo currently has a contract-only slice that records recovered notebook facts.

Relevant files:

- `src/bsm_rfm/debiased_lasso_contract.py`
- `docs/debiased_lasso_contract.md`
- `tests/test_debiased_lasso_contract.py`

Do not implement de-biased LASSO blindly. First preserve and extend the implementation contract. The exact source-notebook de-biasing estimator must be recovered or explicitly specified before executable implementation.

The current public sparse-selection implementation is still an EBIC/L1 component-union surrogate with stability diagnostics. Do not rename it or document it as exact de-biased LASSO.

## Documentation expectations

Documentation must be updated with code changes. At minimum, consider:

- `README.md`
- `docs/MEMORY.md`
- `docs/manuscript_alignment_audit.md`
- `docs/manuscript_contract.md`
- `docs/debiased_lasso_contract.md`
- `docs/index.md`
- `configs/manuscript_case_study.yml`

Every new documentation page must be included in the Sphinx toctree. Sphinx warnings are treated as errors.

## Testing expectations

Prefer direct, meaningful tests over broad downstream failures.

When a downstream final-artifact test fails because an upstream artifact is invalid, add an upstream test that catches the problem earlier.

Do not loosen tests to match broken behavior. Fix the implementation, fixture, or contract.

Tests should verify:

- artifact schema;
- deterministic outputs;
- feature-domain validity;
- non-empty required handoff artifacts;
- provenance status;
- local/CI parity assumptions;
- documentation build integration.

## Style and quality

Use existing style. Keep code simple and explicit.

Do not add compatibility shims, wrappers, aliases, or hidden fallback behavior unless explicitly requested.

Do not hide errors that indicate invalid scientific state. Guards should remain strong.

Do not make speculative changes to numerical thresholds or fixtures without diagnostics.

## Commit discipline

Use small commits with precise messages.

Good commit examples:

```text
Freeze de-biased LASSO implementation contract
Add sparse-selection demo support invariant test
Stabilize manuscript demo runtime fixture
Document current manuscript-stage exactness gaps
```

Bad commit examples:

```text
Fix stuff
Make tests pass
Update files
Temporary workaround
```

## Handoff expectations

When presenting results, include:

- what changed;
- why it changed;
- tests run;
- whether the full gate passed;
- whether CI still needs confirmation;
- any remaining risks or blockers.

Never claim CI passes unless the GitHub check actually passed.
