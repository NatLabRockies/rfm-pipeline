# Artifact reference

`write_postfit_bundle(...)` creates this portable output layout:

```text
<bundle>/
├── manifest.json
└── postfit_diagnostics/
    ├── all_input_metadata.{parquet,csv}
    ├── selected_input_metadata.{parquet,csv}
    ├── output_metadata.{parquet,csv}
    ├── coef_matrix_standardized.{parquet,csv}
    ├── coef_matrix_raw_scale.{parquet,csv}
    ├── x_standardization.{parquet,csv}
    ├── y_standardization.{parquet,csv}
    └── nrmse_summary.{parquet,csv}
```

| Logical table              | Use                                                       |
| -------------------------- | --------------------------------------------------------- |
| `all_input_metadata`       | Original candidate feature order                          |
| `selected_input_metadata`  | Features retained by screening                            |
| `output_metadata`          | Output names and positions                                |
| `coef_matrix_standardized` | Coefficients on standardized feature/output scales        |
| `coef_matrix_raw_scale`    | Coefficients for raw-scale outputs with centered features |
| `x_standardization`        | Feature means, scales, and positions                      |
| `y_standardization`        | Output means, scales, and positions                       |
| `nrmse_summary`            | Holdout macro nRMSE and bootstrap interval                |

`manifest.json` records the dataset tag, feature/output order, logical file
map, metrics, evaluation settings, and supplied upstream provenance. Use
`load_postfit_bundle(...)` instead of constructing filenames yourself.
Use `predict_from_postfit_bundle(...)` for inference from the written bundle.

See [Export bundle contract](export_bundle.md) for the exact manifest keys and
prediction equation.
