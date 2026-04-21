# Export bundle contract

The canonical workflow produces a manifest-aware post-fit bundle for downstream inspection and
visualization.

## Writer entrypoint

Use `bsm_rfm.write_postfit_bundle(...)` on the `artifacts` payload returned by
`bsm_rfm.run_canonical_workflow(...)` or `bsm_rfm.final_ols.build_postfit_artifacts(...)`.

## Bundle layout

The bundle root contains:

- `manifest.json`
- `postfit_diagnostics/`

The `postfit_diagnostics/` subtree contains these canonical tables:

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

The manifest records:

- dataset tag
- counts for all input features, selected features, retained features, and outputs
- ordered feature and output names
- file map for each canonical artifact
- optional metrics, evaluation metadata, and upstream provenance metadata

## Reader entrypoint

Use `bsm_rfm.load_postfit_bundle(...)` to reload the canonical tables used by downstream
visualization code. `bsm_rfm.load_pipeline_outputs(...)` remains as a package-level alias for the
same canonical loader.
