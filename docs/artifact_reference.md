# Artifact Reference

This document describes every artifact (CSV, SVG, metadata file) produced by each stage of the workflow, including column names, data types, and interpretation guidance.

## Table of Contents

1. Output Conditioning
1. Empirical-Null Screening
1. Interaction Discovery
1. Nonlinear Discovery
1. Sparse Selection & Stability
1. Final OLS & Bundle Export
1. Final Manuscript Tables & Figures
1. QA Audit Artifacts

______________________________________________________________________

## Stage 2: Output Conditioning

**Purpose:** Filter outputs by variance, compute PCA representation of responses.

**Location:** `{artifact_root}/output_conditioning/`

### `output_filter_diagnostics.csv`

Summary of outputs before/after variance filtering.

| Column                  | Type   | Description                              |
| ----------------------- | ------ | ---------------------------------------- |
| `output_name`           | string | Output identifier                        |
| `train_variance`        | float  | Training set variance                    |
| `retained_after_filter` | bool   | Whether output passed variance threshold |

**Example:**

```
output_name,train_variance,retained_after_filter
y1,125.4,true
y2,0.0001,false
```

**Interpretation:**

- Outputs with `train_variance < threshold` are marked `retained = false`
- These outputs are excluded from all downstream stages
- Expected: Most outputs retained (>90%)

### `pca_scores.csv`

PCA representation of the retained outputs in training set.

| Column      | Type   | Description                          |
| ----------- | ------ | ------------------------------------ |
| `sample_id` | string | Training sample identifier           |
| `pc1`       | float  | First principal component score      |
| `pc2`       | float  | Second principal component score     |
| ...         | ...    | Additional PCs up to number retained |

**Dimensions:** (n_train_samples, n_pca_components)

**Example:**

```
sample_id,pc1,pc2,pc3
sample_001,-0.456,0.123,0.789
sample_002,0.234,-0.567,0.012
```

**Interpretation:**

- Used as the "screening response" in subsequent stages
- Columns: typically 30–50 principal components (see `pca_explained_variance.csv`)
- Values: standardized (mean ~0, std ~1)

### `pca_loadings.csv`

Relationship between original outputs and principal components.

| Column        | Type   | Description              |
| ------------- | ------ | ------------------------ |
| `output_name` | string | Original output variable |
| `pc1`         | float  | Loading (weight) on PC1  |
| `pc2`         | float  | Loading (weight) on PC2  |
| ...           | ...    | Additional loadings      |

**Interpretation:**

- Loadings: correlation-like values between [-1, 1]
- High |loading| → that output contributes strongly to that PC
- Used to interpret what each PC represents in terms of original outputs

### `pca_explained_variance.csv`

Variance explained by each principal component.

| Column                      | Type  | Description                |
| --------------------------- | ----- | -------------------------- |
| `component`                 | int   | PC index (1, 2, 3, ...)    |
| `explained_variance_ratio`  | float | Fraction of variance (0–1) |
| `cumulative_variance_ratio` | float | Cumulative fraction (0–1)  |
| `retained_for_screening`    | bool  | Whether this PC is kept    |

**Example:**

```
component,explained_variance_ratio,cumulative_variance_ratio,retained_for_screening
1,0.425,0.425,true
2,0.156,0.581,true
3,0.089,0.670,true
...
42,0.0001,0.990,true
43,0.0001,0.991,false
```

**Interpretation:**

- Cumulative up to last retained PC ≈ 0.99 (99% variance retained)
- Fewer PCs → faster computation but less information

### `output_conditioning_summary.csv`

Summary statistics for the stage.

| Column                  | Type   | Description                      |
| ----------------------- | ------ | -------------------------------- |
| `stage`                 | string | "output_conditioning"            |
| `n_outputs_input`       | int    | Total outputs before filtering   |
| `n_outputs_retained`    | int    | Outputs with sufficient variance |
| `n_pca_components`      | int    | Number of PCs kept               |
| `pca_variance_retained` | float  | Cumulative variance (≈0.99)      |

______________________________________________________________________

## Stage 3: Empirical-Null Screening

**Purpose:** Identify features significantly associated with PCA-reduced response.

**Location:** `{artifact_root}/empirical_null_screen/`

### `feature_screening_statistics.csv`

Screening statistic (coefficient norm) for each feature vs. each PCA component.

| Column            | Type   | Description                        |
| ----------------- | ------ | ---------------------------------- |
| `feature_name`    | string | Feature identifier                 |
| `pc1_coefficient` | float  | Univariate regression coeff on PC1 |
| `pc2_coefficient` | float  | Coeff on PC2                       |
| ...               | ...    | Coefficients for all PCs           |

**Dimensions:** (n_features, n_pca_components)

**Interpretation:**

- Larger |coefficient| → feature more important for that PC
- These are univariate OLS coefficients (one feature per PC)
- Used to compute the L2 norm (coefficient-row-norm) per feature

### `component_coefficients.csv`

Per-feature summary: coefficient norm across all PCs.

| Column                     | Type   | Description                        |
| -------------------------- | ------ | ---------------------------------- |
| `feature_name`             | string | Feature identifier                 |
| `coefficient_norm_l2`      | float  | √(coeff_pc1² + coeff_pc2² + ...)   |
| `retained_after_screening` | bool   | Whether feature survived BH filter |

**Example:**

```
feature_name,coefficient_norm_l2,retained_after_screening
x1,2.345,true
x2,0.001,false
x3,1.567,true
```

**Interpretation:**

- Norm aggregates signal across all PCs
- Larger norm → feature more strongly associated overall
- ~30–50% of features retained (depends on α)

### `permutation_null_summary.csv`

Null distribution: coefficient norms from response-permuted resamples.

| Column                       | Type  | Description         |
| ---------------------------- | ----- | ------------------- |
| `percentile`                 | int   | Percentile (1–100)  |
| `coefficient_norm_threshold` | float | Null quantile value |

**Example:**

```
percentile,coefficient_norm_threshold
1,0.012
5,0.045
10,0.067
...
90,0.234
95,0.289
99,0.345
```

**Interpretation:**

- These thresholds define the null distribution from data permutations
- BH threshold (α=0.10) selects features with norm > ~90th percentile
- Higher norm = more significant (exceeds null)

### `retained_terms.csv`

Final list of features that passed the BH filter.

| Column                | Type   | Description                               |
| --------------------- | ------ | ----------------------------------------- |
| `feature_name`        | string | Feature identifier                        |
| `coefficient_norm_l2` | float  | Screening statistic                       |
| `q_value`             | float  | BH-adjusted p-value                       |
| `significant`         | bool   | q-value < α (should be true for all rows) |

**Interpretation:**

- These features feed into downstream stages (interaction, nonlinear, sparse selection)
- Typically 50–150 features retained
- q_value: adjusted p-value accounting for multiple testing

### `empirical_null_provenance.csv`

Metadata about this stage's implementation.

| Column  | Type   | Description    |
| ------- | ------ | -------------- |
| `key`   | string | Metadata field |
| `value` | string | Value          |

**Example:**

```
key,value
stage_name,empirical_null_screening
method,public_surrogate
implementation,coefficient_row_norm_with_permutation_null
note,Not exact Delta-null workflow; see manuscript_alignment_audit.md
```

**Interpretation:**

- Indicates whether implementation is exact or approximate
- Links to `docs/manuscript_alignment_audit.md` for scientific equivalence status

### `empirical_null_screen_summary.csv`

Summary statistics.

| Column                | Type   | Description                 |
| --------------------- | ------ | --------------------------- |
| `stage`               | string | "empirical_null_screening"  |
| `n_features_input`    | int    | Input features from catalog |
| `n_features_retained` | int    | After BH filter             |
| `bh_alpha`            | float  | BH FDR threshold            |
| `n_permutations`      | int    | Permutation samples         |

______________________________________________________________________

## Stage 4: Interaction Discovery

**Purpose:** Score and retain interaction pairs (2-factor products).

**Location:** `{artifact_root}/interaction_discovery/`

### `interaction_pair_scores.csv`

Residualized interaction term scores for each candidate pair.

| Column                       | Type   | Description                           |
| ---------------------------- | ------ | ------------------------------------- |
| `feature1_name`              | string | First feature                         |
| `feature2_name`              | string | Second feature                        |
| `residualized_product_score` | float  | Incremental effect of product term    |
| `max_component_score`        | float  | Largest score across all PCs          |
| `retained_after_threshold`   | bool   | Whether pair exceeds permutation null |

**Dimensions:** (n_candidate_pairs, 5)

**Interpretation:**

- Candidate pairs: all combinations of retained (screening) features
- Score: how much additional variance the product explains beyond main effects
- Larger score → more important interaction
- **Note:** This is a public surrogate; see `interaction_discovery_provenance.csv`

### `component_interaction_scores.csv`

Per-pair, per-PC score breakdown.

| Column          | Type   | Description                      |
| --------------- | ------ | -------------------------------- |
| `feature1_name` | string | First feature                    |
| `feature2_name` | string | Second feature                   |
| `pc1_score`     | float  | Product term contribution to PC1 |
| `pc2_score`     | float  | Contribution to PC2              |
| ...             | ...    | Scores for all PCs               |

**Interpretation:**

- Allows investigating which PCs drive each interaction
- Most pairs will have non-negligible scores for 2–3 PCs

### `retained_interaction_pairs.csv`

Final list of retained interaction pairs.

| Column                       | Type   | Description                                  |
| ---------------------------- | ------ | -------------------------------------------- |
| `feature1_name`              | string | First feature                                |
| `feature2_name`              | string | Second feature                               |
| `residualized_product_score` | float  | Score                                        |
| `percentile_vs_null`         | float  | Percentile rank vs. permutation null (0–100) |

**Example:**

```
feature1_name,feature2_name,residualized_product_score,percentile_vs_null
x1,x3,0.456,95.2
x2,x7,0.389,92.1
x4,x5,0.234,87.3
```

**Interpretation:**

- Pairs with high percentile are most significant
- Typically 10–50 interaction pairs retained

### `interaction_discovery_provenance.csv`

Stage implementation metadata.

**Key field:**

```
note,Public residualized-product surrogate; tree-SHAP not yet ported
```

This indicates the public implementation differs from the manuscript method. See `docs/manuscript_alignment_audit.md`.

### `interaction_discovery_summary.csv`

Summary statistics.

| Column              | Type   | Description                            |
| ------------------- | ------ | -------------------------------------- |
| `stage`             | string | "interaction_discovery"                |
| `n_candidate_pairs` | int    | Possible pairs from screening features |
| `n_pairs_retained`  | int    | After permutation null threshold       |

______________________________________________________________________

## Stage 5: Nonlinear Discovery

**Purpose:** Score and retain nonlinear transformations (e.g., log, squared).

**Location:** `{artifact_root}/nonlinear_discovery/`

### `transformation_scores.csv`

Residualized nonlinear transformation scores.

| Column                         | Type   | Description                                 |
| ------------------------------ | ------ | ------------------------------------------- |
| `feature_name`                 | string | Input feature                               |
| `transformation_name`          | string | Transformation type (e.g., "log", "square") |
| `residualized_nonlinear_score` | float  | Incremental effect beyond linear term       |
| `retained_after_threshold`     | bool   | Above permutation null                      |

**Example:**

```
feature_name,transformation_name,residualized_nonlinear_score,retained_after_threshold
x1,log,0.234,true
x1,square,0.056,false
x2,log,0.189,true
```

**Interpretation:**

- Score: marginal effect of nonlinear term after removing linear relationship
- Larger score → transformation captures important nonlinearity
- **Note:** Public surrogate for GAM EDF/p-value workflow

### `retained_transformations.csv`

Final list of retained transformations.

| Column                         | Type   | Description           |
| ------------------------------ | ------ | --------------------- |
| `feature_name`                 | string | Input feature         |
| `transformation_name`          | string | Transformation type   |
| `residualized_nonlinear_score` | float  | Score                 |
| `percentile_vs_null`           | float  | Rank vs. null (0–100) |

**Interpretation:**

- Typically 20–60 transformations retained
- Each feature may have multiple transformations

### `nonlinear_discovery_provenance.csv`

**Key field:**

```
note,Public residualized-parametric surrogate; GAM EDF/p-value not yet ported
```

### `nonlinear_discovery_summary.csv`

| Column                        | Type   | Description              |
| ----------------------------- | ------ | ------------------------ |
| `stage`                       | string | "nonlinear_discovery"    |
| `n_transformations_candidate` | int    | Possible transformations |
| `n_transformations_retained`  | int    | After threshold          |

______________________________________________________________________

## Stage 6: Sparse Selection & Stability

**Purpose:** Build union of screened + interaction + nonlinear features; fit EBIC-selected L1 models; evaluate subsample stability.

**Location:** `{artifact_root}/sparse_selection_stability/`

### `support_candidates.csv`

Union of all candidate features going into EBIC/L1 selection.

| Column         | Type   | Description                                |
| -------------- | ------ | ------------------------------------------ |
| `feature_name` | string | Feature identifier                         |
| `source`       | string | "screening", "interaction", or "nonlinear" |

**Example:**

```
feature_name,source
x1,screening
x1_x3,interaction
log_x2,nonlinear
```

**Interpretation:**

- Combination of all upstream stages
- Typically 100–300 candidates
- Interaction pairs and transformations create new "features"

### `component_model_selection.csv`

EBIC model selection results per PCA component.

| Column                    | Type  | Description                     |
| ------------------------- | ----- | ------------------------------- |
| `component`               | int   | PCA component index             |
| `selected_alpha_fraction` | float | Selected L1 penalty (0–1 scale) |
| `selected_n_features`     | int   | Nonzero features in model       |
| `ebic_value`              | float | Extended BIC criterion          |

**Interpretation:**

- Models fit per-component for interpretability
- Smaller EBIC → better fit-complexity trade-off
- Alpha fraction: L1 penalty strength (higher = sparser)

### `component_coefficients.csv`

Fitted coefficients for retained features in each component.

| Column         | Type   | Description                      |
| -------------- | ------ | -------------------------------- |
| `component`    | int    | PCA component                    |
| `feature_name` | string | Feature                          |
| `coefficient`  | float  | Fitted L1 regression coefficient |

**Interpretation:**

- Only nonzero coefficients included
- One row per (component, feature) pair with nonzero coeff

### `stability_resample_summary.csv`

Subsample stability evaluation: frequency of feature selection across resamples.

| Column                | Type   | Description                                             |
| --------------------- | ------ | ------------------------------------------------------- |
| `feature_name`        | string | Feature identifier                                      |
| `selection_frequency` | float  | Fraction of resamples where feature selected (0–1)      |
| `stable_support`      | bool   | True if frequency ≥ stability threshold (typically 0.5) |

**Example:**

```
feature_name,selection_frequency,stable_support
x1,0.95,true
x2,0.42,false
x3,0.87,true
```

**Interpretation:**

- High frequency → robust, reliable feature
- Low frequency → unstable, included only by chance
- Stability filtering removes low-frequency features

### `final_stable_support.csv`

Final set of features for downstream stages.

| Column                | Type   | Description                                |
| --------------------- | ------ | ------------------------------------------ |
| `feature_name`        | string | Feature identifier                         |
| `selection_frequency` | float  | Stability score                            |
| `source`              | string | "screening", "interaction", or "nonlinear" |

**Interpretation:**

- These feed into final OLS stage (Stage 7)
- Typically 50–150 features
- All have `selection_frequency ≥ 0.5`

### `sparse_selection_summary.csv`

Summary statistics.

| Column                            | Type   | Description                    |
| --------------------------------- | ------ | ------------------------------ |
| `stage`                           | string | "sparse_selection_stability"   |
| `n_candidates_input`              | int    | Inputs from prior stages       |
| `n_features_final_stable_support` | int    | After EBIC + stability filters |
| `n_pca_components`                | int    | Number of per-component models |

______________________________________________________________________

## Stage 7: Final OLS & Bundle Export

**Purpose:** Fit final OLS model on retained features; compute holdout predictions and nRMSE; create post-fit bundle.

**Location:** `{artifact_root}/` and `{artifact_root}/bundle_export/`

### `final_support_features.csv`

Final set of features used in OLS model.

| Column          | Type   | Description                    |
| --------------- | ------ | ------------------------------ |
| `feature_name`  | string | Feature identifier             |
| `feature_index` | int    | Column index in final X matrix |

### `coefficient_matrix_raw_scale.csv`

OLS coefficients in original data scale.

| Index                | Column       | Description                  |
| -------------------- | ------------ | ---------------------------- |
| (implicit row index) | feature_name | Feature identifier           |
| (implicit row index) | y1, y2, ...  | Coefficients for each output |

**Shape:** (n_final_features, n_outputs)

**Interpretation:**

- One coefficient per (feature, output) pair
- Multiply feature by coefficient, sum across features to predict
- Raw scale: same units as original data

### `coefficient_matrix_standardized.csv`

OLS coefficients in standardized (zero-mean, unit-variance) scale.

**Interpretation:**

- Standardized: feature values subtracted mean, divided by std
- Easier to compare relative importance across features
- Must apply standardization transform (see `x_standardization.csv`) before predicting

### `x_standardization.csv`

Standardization parameters for features.

| Column         | Type   | Description                     |
| -------------- | ------ | ------------------------------- |
| `feature_name` | string | Feature identifier              |
| `mean`         | float  | Training set mean               |
| `std`          | float  | Training set standard deviation |

**Use for prediction:**

```
x_standardized = (x_raw - mean) / std
y_pred = coeff_standardized @ x_standardized
```

### `y_standardization.csv`

Standardization parameters for outputs.

| Column        | Type   | Description                                             |
| ------------- | ------ | ------------------------------------------------------- |
| `output_name` | string | Output identifier                                       |
| `mean`        | float  | Training set mean                                       |
| `std`         | float  | Training set standard deviation (for nRMSE denominator) |

### `holdout_nrmse_summary.csv`

Holdout-set prediction accuracy.

| Column                | Type  | Description                     |
| --------------------- | ----- | ------------------------------- |
| `nrmse_point`         | float | Point estimate of holdout nRMSE |
| `ci_lower`            | float | 95% bootstrap lower bound       |
| `ci_upper`            | float | 95% bootstrap upper bound       |
| `n_bootstrap_samples` | int   | Bootstrap resamples             |

**Interpretation:**

- nRMSE ≈ RMSE / Y_train_range (normalized by training output range)
- Lower is better (max = 1.0 for null model)
- CI: uncertainty in point estimate

### `postfit_bundle/`

Tabular artifacts for visualization/publication.

______________________________________________________________________

## Stage 8: Final Manuscript Tables & Figures

**Purpose:** Compute HC3 Wald inferential filter; refine model; generate manuscript tables and figures.

**Location:** `{artifact_root}/final_manuscript_artifacts/`

### `final_model/prefilter_support_features.csv`

Features before HC3 filtering.

| Column                  | Type   | Description               |
| ----------------------- | ------ | ------------------------- |
| `feature_name`          | string | Feature identifier        |
| `coefficient_raw_scale` | float  | Raw-scale OLS coefficient |

### `final_model/final_support_features.csv`

Features after HC3 Wald 95% inferential filter.

| Column                      | Type   | Description                    |
| --------------------------- | ------ | ------------------------------ |
| `feature_name`              | string | Feature identifier             |
| `coefficient_raw_scale`     | float  | Coefficient (after HC3 filter) |
| `hc3_retained_after_filter` | bool   | Whether retained by HC3 filter |

**Interpretation:**

- Features with 95% CI excluding zero are retained
- Fewer features → more conservative, more interpretable

### `final_model/hc3_wald_intervals.csv`

HC3 heteroskedasticity-consistent 95% confidence intervals.

| Column          | Type   | Description                         |
| --------------- | ------ | ----------------------------------- |
| `feature_name`  | string | Feature identifier                  |
| `output_name`   | string | Output variable                     |
| `coefficient`   | float  | OLS point estimate                  |
| `ci_lower`      | float  | 95% CI lower bound                  |
| `ci_upper`      | float  | 95% CI upper bound                  |
| `excludes_zero` | bool   | CI excludes 0 (significant at 0.05) |

**Example:**

```
feature_name,output_name,coefficient,ci_lower,ci_upper,excludes_zero
x1,y1,0.123,-0.045,0.291,false
x2,y1,0.567,0.234,0.900,true
```

**Interpretation:**

- CIs account for heteroskedasticity (HC3 robust standard errors)
- Only features with `excludes_zero = true` survive the filter

### `final_model/hc3_inferential_filter_summary.csv`

Summary of HC3 filtering.

| Column                 | Type   | Description                       |
| ---------------------- | ------ | --------------------------------- |
| `stage`                | string | "hc3_wald_inferential_filter"     |
| `n_prefilter_features` | int    | Features before HC3               |
| `n_final_features`     | int    | Features after HC3 (significant)  |
| `hc3_alpha`            | float  | Significance level (0.05 for 95%) |

### `final_model/final_ols_summary.csv`

Final model summary metrics.

| Column                 | Type   | Description                |
| ---------------------- | ------ | -------------------------- |
| `stage`                | string | "final_ols_and_hc3_filter" |
| `n_prefilter_features` | int    | Features entering OLS      |
| `n_final_features`     | int    | After HC3 filter           |
| `holdout_nrmse`        | float  | Holdout set nRMSE          |
| `holdout_ci_lower`     | float  | 95% bootstrap CI lower     |
| `holdout_ci_upper`     | float  | 95% bootstrap CI upper     |

### `tables/workflow_stage_summary.csv`

**Table 1** in the manuscript.

| Column       | Type   | Description             |
| ------------ | ------ | ----------------------- |
| `stage`      | string | Pipeline stage name     |
| `n_inputs`   | int    | Features entering stage |
| `n_retained` | int    | Features exiting stage  |

**Interpretation:**

- Shows feature attrition through pipeline
- Documents how many features remain at each step

### `tables/model_performance.csv`

**Table 2** in the manuscript.

| Column       | Type   | Description                                 |
| ------------ | ------ | ------------------------------------------- |
| `model_name` | string | Model type (e.g., "final_ols", "null_mean") |
| `nrmse`      | float  | nRMSE on holdout set                        |
| `ci_lower`   | float  | 95% CI lower                                |
| `ci_upper`   | float  | 95% CI upper                                |

**Interpretation:**

- Final model vs. null baseline
- Lower nRMSE better
- CI should exclude null-model nRMSE

### `figures/figure_model_performance.svg`

**Figure 1**: Model performance plot (SVG, publication-ready).

**Contents:**

- Holdout predictions vs. true values
- nRMSE with bootstrap uncertainty
- Null baseline for reference

### `figures/figure_support_composition.svg`

**Figure 2**: Final support composition plot (SVG).

**Contents:**

- Distribution of features by source (screening, interaction, nonlinear)
- Feature importance or coefficient magnitudes

### `figures/figure_model_performance_data.csv`

Plotting data for Figure 1.

| Column                 | Type   | Description          |
| ---------------------- | ------ | -------------------- |
| `output_name`          | string | Output variable      |
| `y_true_holdout`       | float  | Actual holdout value |
| `y_predicted_mean`     | float  | Point prediction     |
| `y_predicted_ci_lower` | float  | 95% CI lower         |
| `y_predicted_ci_upper` | float  | 95% CI upper         |

### `figures/figure_support_composition_data.csv`

Plotting data for Figure 2.

| Column                  | Type   | Description                             |
| ----------------------- | ------ | --------------------------------------- |
| `feature_name`          | string | Feature identifier                      |
| `coefficient_magnitude` | float  |                                         |
| `source_type`           | string | "screening", "interaction", "nonlinear" |

______________________________________________________________________

## QA Audit Artifacts

**Location:** `{artifact_root}/reproduction_audit/`

### `artifact_manifest.csv`

Inventory of all generated outputs.

| Column            | Type   | Description                     |
| ----------------- | ------ | ------------------------------- |
| `artifact_name`   | string | File path (relative)            |
| `file_suffix`     | string | File extension (csv, svg, etc.) |
| `file_size_bytes` | int    | File size                       |
| `file_exists`     | bool   | File present                    |
| `file_nonempty`   | bool   | File has content                |
| `sha256_digest`   | string | File hash (for validation)      |

**Interpretation:**

- Complete record of all outputs
- Hashes enable validation that files haven't been corrupted/modified
- Expected: all rows with `file_exists = true`, `file_nonempty = true`

### `metric_checks.csv`

Pass/fail checks for expected properties.

| Column         | Type   | Description                 |
| -------------- | ------ | --------------------------- |
| `check_name`   | string | Check description           |
| `check_passed` | bool   | Pass (true) or fail (false) |
| `details`      | string | Additional info if failed   |

**Example checks:**

```
check_name,check_passed,details
output_conditioning_files_exist,true,
empirical_null_screens_created,true,
final_support_nonempty,true,
holdout_nrmse_positive,true,
all_figures_svg_valid,true,
artifact_manifest_complete,true,
```

**Interpretation:**

- All checks should be true
- If any fail, investigate `details` column
- Catches common QA failures (empty files, missing artifacts, invalid metrics)

### `audit_summary.csv`

Overall audit status.

| Column                      | Type   | Description                          |
| --------------------------- | ------ | ------------------------------------ |
| `total_artifacts_generated` | int    | Count of all outputs                 |
| `artifacts_missing`         | int    | Count of missing files (should be 0) |
| `metric_checks_total`       | int    | Total checks run                     |
| `metric_checks_passed`      | int    | Checks that passed                   |
| `qa_status`                 | string | "passed" or "failed"                 |

**Example:**

```
total_artifacts_generated,artifacts_missing,metric_checks_total,metric_checks_passed,qa_status
127,0,18,18,passed
```

**Interpretation:**

- `qa_status = "passed"` → all outputs valid, workflow succeeded
- `qa_status = "failed"` → investigation needed; check `metric_checks.csv`

______________________________________________________________________

## Workflow Feature Flow

```
Data Input
  ↓
Stage 2: Output Conditioning
  → pca_scores.csv (response representation)
  ↓
Stage 3: Empirical-Null Screening
  → retained_terms.csv (~50–150 features)
  ↓
Stage 4: Interaction Discovery
  → retained_interaction_pairs.csv (~10–50 pairs)
  ↓
Stage 5: Nonlinear Discovery
  → retained_transformations.csv (~20–60 transformations)
  ↓
Stage 6: Sparse Selection & Stability
  → final_stable_support.csv (~50–150 features)
  ↓
Stage 7: Final OLS & Bundle
  → coefficient_matrix_*.csv
  → postfit_bundle/
  ↓
Stage 8: HC3 Inferential Filter & Tables/Figures
  → final_support_features.csv (HC3-filtered, most conservative)
  → workflow_stage_summary.csv (Table 1)
  → model_performance.csv (Table 2)
  → figure_*.svg (Figures 1–2)
  ↓
QA Audit
  → reproduction_audit/
     - artifact_manifest.csv (all files)
     - metric_checks.csv (QA pass/fail)
     - audit_summary.csv (overall status)
```

______________________________________________________________________

## Common Data Validation Checks

### Feature Count Attrition

Expected pattern:

```
Stage → Features
Input → ~350
Empirical-null screen → ~100 (70% filtered)
Sparse selection → ~80 (20% filtered)
HC3 filter → ~60 (25% filtered)
Final → ~60
```

If fewer than ~50 final features, investigate:

- Data quality (too much noise?)
- PCA variance (is PCA capturing signal?)
- Screening threshold too strict?

### nRMSE Reasonableness

- Null-model baseline: typically ~1.0
- Final model: typically 0.04–0.15 (4–15% of Y_train range)
- If final nRMSE > null baseline: model not learning

### Stability Frequencies

Expected:

- Most features: 0.7–1.0 (stable)
- Few features: \<0.6 (unstable, filtered)

If most features \<0.6: model unstable (increase regularization α, more bootstrap resamples)

### Artifact Sizes

Rough expectations:

- `retained_terms.csv`: ~50KB (hundreds of features)
- `coefficient_matrix_*.csv`: ~10–50KB
- `figure_*.svg`: ~50–500KB (SVG scales with data)

If any artifact is 0 bytes: failed computation (check error logs)
