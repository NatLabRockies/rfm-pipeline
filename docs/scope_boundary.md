# Workflow scope boundary

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

Use `workflow_scope_boundary_table()` as the machine-readable source of truth for the current
non-foundation stages. At the current release boundary, the canonical workflow still reports two
important limits:

- `upstream_null_screening` is represented through the recovered source adapter in
  `rfm_pipeline.null_screening`; the package does not yet reimplement the upstream
  permutation-null Delta workflow natively.
- `feature_expansion` has an explicit tested contract in `rfm_pipeline.feature_expansion`, but the
  notebook-specific recovered default specification is still only partially promoted into the
  canonical package path.

These names and statuses are frozen by tests so that the package scope cannot drift silently.
