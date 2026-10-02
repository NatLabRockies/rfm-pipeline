# Export bundle contract

The canonical workflow produces a manifest-aware post-fit bundle for
prediction and inspection.

## Writer entrypoint

Use `rfm_pipeline.write_postfit_bundle(...)` on the `artifacts` payload returned by
`rfm_pipeline.run_canonical_workflow(...)` or `rfm_pipeline.final_ols.build_postfit_artifacts(...)`.

## Bundle layout

The bundle root contains:

- `manifest.json`
- `postfit_diagnostics/`

The `postfit_diagnostics/` subtree contains these canonical tables.
These logical table names are frozen by `rfm_pipeline.canonical_postfit_artifact_names()` and
`rfm_pipeline.canonical_bundle_loader_keys()`.

- `all_input_metadata`
- `selected_input_metadata`
- `output_metadata`
- `coef_matrix_standardized`
- `coef_matrix_raw_scale`
- `x_standardization`
- `y_standardization`
- `nrmse_summary`

Each table is written as `.parquet` when a parquet engine is available. If the requested
output suffix is `.parquet` but no parquet engine is importable, the writer falls back to `.csv`
and records the actual written path in `manifest.json`.

## Manifest payload

The manifest top-level keys are frozen by `rfm_pipeline.canonical_manifest_top_level_keys()`.
The position-map payload keys are frozen by `rfm_pipeline.canonical_manifest_position_map_keys()`.

They are:

- `dataset_tag`
- `n_all_input_features`
- `n_selected_features`
- `n_retained_features`
- `n_outputs`
- `all_input_features`
- `selected_features`
- `retained_features`
- `output_names`
- `files`
- `metrics`
- `evaluation`
- `upstream_provenance`
- `all_input_position_map`
- `selected_input_position_map`
- `retained_input_position_map`
- `output_position_map`

The `files` mapping is expected to contain one relative path for each canonical artifact table.
The position-map payload preserves original ordering so downstream code can reconstruct
feature/output provenance without recomputing modeling steps.

## Prediction equation

For raw-scale coefficient matrix `B`, retained input row `x`, training input
means `x_mean`, and training output means `y_mean`:

```text
y_hat = y_mean + (x - x_mean) @ B.T
```

Use `retained_features` and `output_names` from `manifest.json` to preserve
column and row order. During the fitting process, prefer
`predict_final_ols(run.final_ols_result, X_new)` over reconstructing this
equation manually.

## Reader entrypoint

Use `rfm_pipeline.load_postfit_bundle(...)` to load the canonical tables. The
loader follows the exact relative paths in `manifest.json` and rejects missing,
absolute, or bundle-escaping paths.

Use `rfm_pipeline.predict_from_postfit_bundle(...)` to predict from raw retained
features without recreating the in-memory estimator.
