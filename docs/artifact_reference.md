# Artifact reference

The repository has two output layouts. Choose the section that matches the
entry point you ran.

## Canonical Python API bundle

`write_postfit_bundle(...)` writes:

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

See [Export bundle contract](export_bundle.md) for the exact manifest keys.

## Staged workflow outputs

The staged runner writes one directory per stage. These are the most useful
files to inspect first:

| Directory                     | Key artifacts                                                                                                      |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `output_conditioning/`        | `output_filter_diagnostics.csv`, `pca_explained_variance.csv`, `pca_scores.csv`, `output_conditioning_summary.csv` |
| `empirical_null_screen/`      | `retained_terms.csv`, `feature_screening_statistics.csv`, `empirical_null_screen_summary.csv`                      |
| `interaction_discovery/`      | `retained_interaction_pairs.csv`, `interaction_pair_scores.csv`, `interaction_discovery_summary.csv`               |
| `nonlinear_discovery/`        | `retained_transformations.csv`, `transformation_scores.csv`, `nonlinear_discovery_summary.csv`                     |
| `sparse_selection/`           | `final_stable_support.csv`, `stability_feature_summary.csv`, `sparse_selection_summary.csv`                        |
| `final_manuscript_artifacts/` | final model, performance tables, figures, and summary files                                                        |
| `runtime_diagnostics/`        | stage timing and progress telemetry                                                                                |

The `final_manuscript_artifacts` directory name is retained for compatibility;
it is the terminal output directory for any staged run.

## Run markers

The staged runner writes explicit state files at the output root:

| Marker                 | Meaning                                                    |
| ---------------------- | ---------------------------------------------------------- |
| `run_started.json`     | Run began and records its process/config identity          |
| `run_complete.json`    | Requested stage window completed                           |
| `run_failed.json`      | An exception stopped execution                             |
| `run_interrupted.json` | A termination signal stopped execution                     |
| `run_abandoned.json`   | A prior run had no terminal marker and its process is gone |

Treat `run_complete.json` as the primary completion signal. Summary CSVs
describe the scientific results; a directory's presence alone does not prove
that its stage completed.

## Audit artifacts

The deterministic reproduction example and BSM case study can additionally
write `reproduction_audit/`:

| File                    | Use                               |
| ----------------------- | --------------------------------- |
| `artifact_manifest.csv` | Paths, sizes, and SHA-256 digests |
| `metric_checks.csv`     | Named pass/fail checks            |
| `audit_summary.csv`     | One-row overall QA status         |

Artifact schemas evolve with their producing stage. For programmatic use,
read the column headers from the generated file and the matching writer in
`rfm_pipeline.manuscript_stages`; do not infer success from historical BSM
row counts.
