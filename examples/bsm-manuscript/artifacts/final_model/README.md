# `artifacts/final_model/` — final OLS export bundle

This directory ships the artifacts needed to evaluate the manuscript v22
reduced-form surrogate without re-running the pipeline.

## Files

| File | Purpose |
| --- | --- |
| `coefficient_matrix_raw_scale.csv` | OLS coefficients on the raw output scale (23,495 outputs × 245 features). |
| `coefficient_matrix_standardized.csv` | OLS coefficients on the train-standardized scale (23,495 × 245). |
| `x_standardization.csv` | Per-feature `mean` and `scale` used to standardize candidate inputs. |
| `y_standardization.csv` | Per-output `mean` and `scale` (scale = 1.0; mean is the per-output training mean used as the intercept). |
| `per_output_intercepts.csv` | Explicit intercept vector (identical to `y_standardization.mean`); see `README_intercept.md`. |
| `final_support_features.csv` | The 245 selected feature names with type tags (main / interaction / transformation). |
| `prefilter_support_features.csv` | The 360 enriched features that entered HC3 filtering and pruning. |
| `hc3_inferential_filter_summary.csv` | HC3 Wald-filter pass/fail summary per candidate feature. |
| `hc3_wald_intervals.csv` | HC3 Wald 95% intervals per feature × output. |
| `feature_pruning_impact.csv` | Per-feature impact of removal on macro nRMSE (used by delta-threshold diagnostic). |
| `final_ols_summary.csv` | One-row summary of the final OLS export — see column glossary below. |

## `final_ols_summary.csv` column glossary

| Column | Meaning | Manuscript reference |
| --- | --- | --- |
| `stage` | Stage identifier (`final_ols_and_manuscript_artifacts`). | — |
| `n_training_rows` | 28,500 — training rows after the 5% holdout split. | Table 1 |
| `n_holdout_rows` | 1,500 — holdout rows. | Table 1 |
| `n_prefilter_features` | 360 — enriched features after discovery, before HC3 + pruning. | Table 2 |
| `n_hc3_features` | 360 — features surviving the HC3 Wald inferential filter. | Table 2 |
| `n_final_features` | 245 — final OLS support after delta-threshold pruning. | Table 2 |
| `n_hc3_removed_features` | 0 — HC3 retained all enriched features for the publication run. | Table 2 |
| `n_pruning_removed_features` | 115 — features dropped by delta-threshold diagnostic. | Table 2 |
| `n_retained_outputs` | 23,495 — total scalar outputs the final OLS covers. | Table 2 |
| `n_variance_filtered_outputs` | 9,954 — outputs retained by train-only variance / dynamic-range filtering for PCA. | Table 2 |
| `final_ols_holdout_nrmse` | 0.0679 — macro holdout nRMSE on 1,500 holdout runs. | Table 2 |
| `final_ols_holdout_nrmse_ci_lower` | 0.0663 — lower 95% bootstrap bound from 100 holdout-row resamples. | §5.2 |
| `final_ols_holdout_nrmse_ci_upper` | 0.0690 — upper 95% bootstrap bound. | §5.2 |
| `null_mean_holdout_nrmse` | 0.1653 — null-mean baseline (predicting the training mean per output). | Appendix Table |
| `bootstrap_count` | 100 — number of holdout-row resamples for the macro-nRMSE CI. | Table 1 |
| `bootstrap_alpha` | 0.05 — two-sided bootstrap CI level (95% interval). | Table 1 |
| `inferential_filter_interval_method` | `hc3_wald_95_percent_drop_if_zero_compatible_for_all_outputs` — HC3 Wald-based per-feature filter. | Table 1 |
| `inferential_filter_alpha` | 0.05 — HC3 filter level (95% Wald interval). Distinct from `bootstrap_alpha` despite the shared name. | Table 1 |
| `nrmse_denominator_definition` | `macro_average_rmse_divided_by_training_response_range` — definition of macro nRMSE used throughout. | §5.1 |
| `nrmse_reference_matrix` | `Y_train` — the matrix from which per-output response ranges are computed for nRMSE normalization. | §5.1 |
| `nrmse_min_range` | 1e-6 — exclusion floor for zero-range outputs in macro nRMSE. | Table 1 |
| `manuscript_final_predictor_count_reference` | 245 — manuscript Table 2 reference value used for tripwire assertions (corrected 30k run). | Table 2 |
| `manuscript_final_ols_holdout_nrmse_reference` | 0.0679 — manuscript Table 2 reference value used for tripwire assertions (corrected 30k run). | Table 2 |

## Prediction equation

For output *i* and input row *r*:

```
y_hat[i, r] = intercept[i]
            + sum_j  coef_raw_scale[i, j] * (X_raw[r, j] - mean_x[j])
```

where `intercept[i] = per_output_intercepts.csv[intercept]` (equivalently
`y_standardization.csv[mean]`), `coef_raw_scale` comes from
`coefficient_matrix_raw_scale.csv`, and `mean_x` comes from
`x_standardization.csv`. The `coef_raw_scale` matrix already absorbs both the
per-feature `scale_x` and any per-output `scale_y` standardization, so the
prediction is on the raw response scale with only mean-centring of `X` needed.
If you instead use `coefficient_matrix_standardized.csv` (the
train-standardized coefficients), apply

```
y_hat[i, r] = intercept[i]
            + scale_y[i] * sum_j coef_std[i, j] * (X_raw[r, j] - mean_x[j]) / scale_x[j]
```

with `mean_x`, `scale_x` from `x_standardization.csv` and `scale_y` from
`y_standardization.csv`.
