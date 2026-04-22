# Scope boundary

The public package exposes a tested canonical workflow, but it does not claim that every
historical notebook stage has been fully ported into the live module implementation.

## Implemented directly in the package

These stages are part of the canonical package path today:

- regularized screening
- final OLS fitting
- holdout nRMSE summarization
- manifest-aware post-fit artifact assembly
- bundle writing and bundle reload for downstream visualization

## Explicitly bounded stages

Use `bsm_rfm.workflow_scope_boundary_table()` to inspect the current non-foundation stages.
At the time of this release boundary, the canonical workflow still reports two important
limits:

- `upstream_null_screening` is represented through the recovered source adapter in
  `bsm_rfm.null_screening`; the package does not yet reimplement the upstream
  permutation-null Delta workflow natively.
- `feature_expansion` has an explicit tested contract in `bsm_rfm.feature_expansion`, but the
  notebook-specific recovered default specification is still only partially promoted into the
  canonical package path.

## Why this is documented explicitly

The package is intended to be honest about provenance. Users can run the implemented
screening/final-fit/evaluation/export workflow directly, while still seeing where the
remaining source-derived and notebook-derived boundaries sit.
