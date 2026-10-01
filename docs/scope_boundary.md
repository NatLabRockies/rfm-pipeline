# Workflow scope boundary

The canonical API intentionally covers a smaller surface than the advanced
staged runner.

## Implemented directly in the package

`run_canonical_workflow(...)` implements:

- regularized screening
- final OLS fitting
- holdout nRMSE summarization
- manifest-aware post-fit artifact assembly
- bundle writing and bundle reload for downstream visualization

## Explicitly bounded stages

Use `workflow_scope_boundary_table()` as the machine-readable statement of the
remaining canonical-API limits:

- `upstream_null_screening` is available through the source adapter in
  `rfm_pipeline.null_screening`, but is not called by the canonical workflow.
- `feature_expansion` has a tested contract in
  `rfm_pipeline.feature_expansion`, but no default expansion is called by the
  canonical workflow.

Prepare those features before calling the canonical API, or use the staged
workflow. The names and statuses are frozen by tests so the boundary cannot
drift silently.
