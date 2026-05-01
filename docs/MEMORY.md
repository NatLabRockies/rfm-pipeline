# BSM Public Reduced-Form Repo Memory

## Repository purpose

This repository, `bsm-public-rf`, is the public, reproducible reduced-form modeling workflow package for the BSM case study and related manuscript/release artifacts. Its purpose is to provide a tested, documented, reproducible implementation scaffold for the workflow described in the Journal of Data Science manuscript.

The repository is not merely a utility package. It is a scientific workflow artifact. Code, tests, notebooks, documentation, configuration, provenance records, and generated manuscript-facing artifacts must remain internally consistent.

## Current scientific workflow goals

The target workflow is:

1. deterministic manuscript/demo runtime resolution;
1. output conditioning and PCA-reduced response representation;
1. empirical-null screening;
1. interaction discovery;
1. nonlinear discovery;
1. sparse selection/stability;
1. deterministic de-biased-LASSO stage, not yet fully implemented;
1. final OLS refit and HC3 inferential filtering;
1. manuscript table/figure regeneration;
1. reproduction audit with artifact manifests and metric checks.

The current public implementation includes deterministic public surrogates for several manuscript stages. These surrogates must not be described as manuscript-exact unless equivalence has been validated against the recovered/source workflow.

## De-biased-LASSO status

A contract-only de-biased-LASSO slice has been added.

Known recovered notebook facts are frozen in:

- `src/bsm_rfm/debiased_lasso_contract.py`
- `docs/debiased_lasso_contract.md`
- `tests/test_debiased_lasso_contract.py`

The target source artifact is `LASSO_to_OLS_v9.ipynb`.

Frozen recovered facts include:

- 352 candidate inputs
- 23,495 outputs before culling
- 9,782 outputs after culling
- 39 retained PCA components
- 18,000 train rows / 2,000 holdout rows
- holdout fraction 0.10
- `l1_ratio = 1.00`
- EBIC gamma 0.5
- alpha fraction grid: `0.75, 0.50, 0.25, 0.10, 0.05, 0.02, 0.01`
- recovered selected alpha fraction 0.10
- approximate selected absolute alpha 2.8541
- recovered notebook selected-feature count 346
- final downstream support count 340

Important: the exact de-biasing estimator definition is not yet frozen. The current public sparse-selection method remains an EBIC/L1 component-union surrogate with stability diagnostics. It must not be renamed or treated as exact de-biased LASSO until the source notebook cells have been audited and the emitted outputs validated.

## CI/local parity history

A past CI/local mismatch occurred in the demo manuscript sparse-selection/final-artifact chain. GitHub Linux CI produced an empty `final_stable_support`, causing downstream final-artifact, reproduction-chain, audit, and reproducibility-example tests to fail.

That issue is **resolved** as of commit `10e2542` ("Stabilize demo manuscript sparse-selection fixture"). GitHub Actions CI runs 43–47 on `main` all succeeded. The fix stabilized the demo fixture so the EBIC/L1 selection path produces a non-empty support on both macOS and Linux.

Bad speculative fixes were attempted during that debugging period and must not be repeated:

- changing demo `x2` values to include values `<= -1`, which broke the existing `log1p_x2` feature-domain invariant;
- changing the number of demo sample IDs without updating holdout split lengths;
- perturbing demo fixture values without first diagnosing which sparse/stability filter eliminates the support.

Any future fixture change must preserve: equal column lengths, valid `log1p_x2` domain (`x2 > -1`), sufficient rows for stability resampling, non-collinear predictors, and signal strong enough to survive EBIC/L1 and subsample-stability filters on both macOS and Linux.

## Engineering rules

Treat the live repo as the only source of truth. Do not trust prior patch descriptions, chat summaries, or assumed local state.

Before changing files:

1. inspect `git status --short`;
1. inspect recent commits;
1. inspect relevant implementation and tests;
1. run targeted failing tests;
1. identify the root cause;
1. write or strengthen tests first;
1. implement the smallest root-cause fix;
1. run targeted tests;
1. run the full gate.

Do not patch from memory. Do not patch from partial snippets. Do not make speculative fixture changes. Do not weaken guards merely to make tests pass.

The repository gate is authoritative:

```bash
./test_repo.sh --fix
./test_repo.sh --check
```

CI must match the local non-mutating gate. The local gate must catch failures before GitHub Actions catches them.

## Required validation discipline

For every development slice:

1. Start from clean `main`.
1. Create a short-lived branch.
1. Add failing/guardrail tests first.
1. Make a minimal implementation change.
1. Run targeted tests.
1. Run the full gate.
1. Commit only after the full gate passes.
1. Merge and delete the branch.
1. Confirm CI passes.

Recommended branch workflow:

```bash
cd ~/src/bsm-public-rf
git switch main
git fetch origin
git pull --ff-only origin main
git status --short
git switch -C <branch-name>
```

Recommended validation before commit:

```bash
./test_repo.sh --fix
./test_repo.sh --check
```

Recommended merge/cleanup after local pass:

```bash
cd ~/src/bsm-public-rf && \
git status --short && \
git switch main && \
git pull --ff-only origin main && \
git merge --ff-only <branch-name> && \
git push origin main && \
git branch -D <branch-name> && \
git push origin --delete <branch-name> 2>/dev/null || true && \
git fetch --prune
```

## Platform/handoff limitations learned

ChatGPT artifact downloads failed repeatedly with upload-status errors. Do not rely on zipped artifact handoff from ChatGPT for this repo.

The ChatGPT GitHub connector was inconsistent. It successfully created one branch/PR and created new files, but later existing-file updates and even a branch-creation probe were blocked by the platform safety layer. Do not rely on ChatGPT direct GitHub writes for urgent repo fixes.

The durable workflow should be:

- ChatGPT/Claude: research, design review, audit prompts, engineering plans, documentation/manuscript writing, external-current method review.
- Copilot/local agent: live repo audit, file edits, tests, local validation, commits, pushes, PR/merge cleanup.
- Human: runs local gate and reviews/approves scientific claims.

## Documentation expectations

Documentation is part of the tested artifact surface.

When repo behavior changes, update:

- `README.md`
- `docs/MEMORY.md`
- `docs/manuscript_alignment_audit.md`
- `docs/manuscript_contract.md` if the scientific contract changes
- `docs/debiased_lasso_contract.md` if de-biased-LASSO assumptions change
- `configs/manuscript_case_study.yml` only when the frozen workflow contract changes
- Sphinx toctree files whenever new docs are added

Sphinx warnings are errors. Every new documentation page must be included in the toctree.

## Testing expectations

Tests must guard scientific invariants, not merely implementation details.

Critical invariants include:

- demo manuscript artifacts validate and load;
- all demo artifact table columns have equal lengths;
- transformed features such as `log1p_x2` are defined on the demo domain;
- demo sparse-selection produces non-empty final stable support;
- final manuscript artifacts refuse empty support;
- reproduction chain writes all expected artifact families;
- audit manifest and metric checks are produced;
- docs build without warnings;
- local gate mirrors CI.

## Immediate next development priority

The CI/local parity issue is resolved. The repository gate and CI both pass on the current `main`.

Next development priorities in order:

1. Add a direct sparse-selection guardrail test that asserts non-empty `final_stable_support` on the demo fixture before downstream final-artifact tests can observe the failure.
1. Begin the deterministic de-biased-LASSO implementation test scaffold: freeze ALO/KKT diagnostics, EBIC path construction, and per-component coefficient artifact schema in tests before adding executable code.
1. Validate empirical-null screening equivalence against the recovered Delta-null script or replace the public surrogate with a released private retained-term artifact.
1. Replace or externalize the interaction stage (tree-SHAP) and nonlinear stage (GAM EDF/p-value) surrogates, or require released private artifact tables as the ground truth.
