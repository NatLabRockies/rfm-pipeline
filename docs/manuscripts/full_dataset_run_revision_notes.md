# Full-dataset run results and manuscript update notes

**Run ID**: `publication_full_dataset_distributed_20260519`
**HPC**: Kestrel (kl1.hpc.nrel.gov)
**Config**: `configs/hpc/kestrel_publication_full_dataset_distributed_base.yml`
**Completed**: 2026-05-19 (all 6 stages, `RUN_COMPLETE` marker present)
**Local artifacts**: `artifacts/publication_full_dataset_20260519/final_manuscript_artifacts/`

______________________________________________________________________

## Final pipeline results

### Dataset

| Quantity                | Value  |
| ----------------------- | ------ |
| Total rows              | 30,000 |
| Training rows           | 28,500 |
| Holdout rows            | 1,500  |
| Retained scalar outputs | 9,954  |
| PCA components          | 20     |

### Stage-by-stage support counts

| Stage                    | Quantity                   | This run   | Manuscript reference |
| ------------------------ | -------------------------- | ---------- | -------------------- |
| output_conditioning      | retained_scalar_outputs    | 9,954      | —                    |
| output_conditioning      | retained_pca_components    | 20         | —                    |
| empirical_null_screening | retained_terms             | 69         | 349                  |
| interaction_discovery    | retained_pairs             | 8          | —                    |
| nonlinear_discovery      | retained_transformations   | 41         | 37                   |
| sparse_selection         | final_stable_support_terms | 118        | 340                  |
| final_inferential_filter | hc3_retained_terms         | 118        | 340                  |
| feature_pruning          | removed_terms              | 89         | —                    |
| **final_ols**            | **holdout_nrmse**          | **0.1063** | **0.0445**           |

### Final model composition (29 features)

| Feature type | Count | Share |
| ------------ | ----- | ----- |
| First Order  | 16    | 55%   |
| Non-Linear   | 12    | 41%   |
| Second Order | 1     | 3%    |

### Module distribution of selected support

| Module                      | n selected | share |
| --------------------------- | ---------- | ----- |
| Oil Hydrocarbons            | 10         | 34%   |
| Cellulosic Hydrocarbons     | 6          | 21%   |
| Wet Waste Hydrocarbons      | 6          | 21%   |
| Algal Hydrocarbons          | 4          | 14%   |
| Starch Ethanol Hydrocarbons | 3          | 10%   |

______________________________________________________________________

## Ablation table (now filled in)

Previously a placeholder in the manuscript. These are verified numbers from the full-dataset run.

| Model                              | n features | Holdout nRMSE | 95% CI lower | 95% CI upper |
| ---------------------------------- | ---------- | ------------- | ------------ | ------------ |
| Null mean                          | 0          | 0.1653        | 0.1635       | 0.1663       |
| All first-order OLS                | 158        | 0.0812        | 0.0796       | 0.0821       |
| Screened first-order OLS           | 69         | 0.0812        | 0.0796       | 0.0821       |
| Penalized OLS (full candidate set) | 118        | 0.0767        | 0.0751       | 0.0777       |
| Final OLS                          | 29         | 0.1063        | 0.1048       | 0.1072       |

Bootstrap: 100 replicates, 1,500-row holdout, macro nRMSE normalized by Y_train range.

### Reading the ablation

1. **Empirical null screening validates perfectly.** Going 158→69 features (57% reduction) costs zero performance: screened OLS = 0.0812, same CI as all-features OLS. The permutation-calibrated empirical null screen removes noise without removing signal.

1. **Interaction + nonlinear discovery adds value.** 69→118 features, 0.0812→0.0767 nRMSE (5.5% relative improvement, non-overlapping CIs). The additional terms from dynamic discovery improve the model despite a very selective 8 interaction pairs retained.

1. **Feature pruning is too aggressive in this run.** 118→29 features, 0.0767→0.1063 nRMSE (38.5% regression). The pruning stage is the largest source of performance loss. See analysis below.

1. **Gap from manuscript reference.** Our final_ols (0.1063) vs manuscript reference (0.0445) is partially explained by: (a) 29 vs ~340 final support features — the pruning stage aggressively reduces the support, and (b) different dataset size/composition. The penalized model (0.0767) is actually *better* than the manuscript's intermediate reference (0.0859), indicating the pipeline through sparse selection is working well.

______________________________________________________________________

## Per-output NRMSE

| Statistic       | nRMSE |
| --------------- | ----- |
| 10th percentile | 0.046 |
| 25th percentile | 0.072 |
| Median          | 0.095 |
| 75th percentile | 0.135 |
| 90th percentile | 0.172 |

Worst 3 outputs (all `OI.OHC output by product[TransEster, P, diesel]` for 2023–2025): nRMSE ≈ 0.47–0.49. These are specific transesterification product outputs that are likely hard to predict at this sample size.

______________________________________________________________________

## Analysis: why the final model underperforms penalized

The feature pruning stage (`auto_robust_utility`) removed 89 of 118 features using a per-feature holdout delta-nRMSE threshold of ~0.029. The individual deltas of the first several removed features are essentially zero (first removed feature delta = -4.9e-7, i.e., slightly *negative* — removing it marginally improves the approximation). However, removing 89 features in total produces a 0.030 nRMSE degradation — the additive bound is extremely loose.

This is a classic case where many small-individually-unimportant features are collectively important. The pruning logic (single-pass greedy by individual delta) does not capture complementary feature contributions.

**This is the primary improvement target before publication.**

______________________________________________________________________

## Suggested improvements for publication

### 1. Tighten feature pruning threshold (highest priority, lowest effort)

The `auto_robust_utility` pruning threshold is currently `auto_cutoff_delta ≈ 0.029`. Change the pruning to use a much tighter threshold (e.g., 0.001–0.005) to retain features with individually small but collectively significant contributions. Expected result: retain 60–90 of 118 features and recover most of the 0.077→0.106 performance gap.

**Config change only**: adjust `feature_pruning_delta_threshold` in the config (or disable auto-cutoff and use a fixed threshold). Re-run just the `final_manuscript_artifacts` stage.

**Expected gain**: likely closes most of the 0.077→0.106 gap, producing a publishable final model in the 0.077–0.085 nRMSE range with 60–90 interpretable features.

### 2. Investigate interaction discovery sparsity (medium effort, high value)

Only 8 interaction pairs were retained from C(69,2)=2,346 candidates. The manuscript retained 248. The SHAP interaction score threshold used to filter pairs may be misconfigrated for this dataset's scale. Examining the interaction pair scores distribution in `artifacts/hpc_shards_interaction_discovery/_merged/interaction_pair_scores_merged.csv` would clarify whether the threshold is appropriate or whether it should be relaxed.

**Expected gain**: retaining more interaction pairs would increase the candidate pool for sparse selection, potentially increasing the penalized model's performance further below 0.077.

### 3. Consider HC3-first feature selection (medium effort, statistically principled)

Currently HC3 retained all 118 features (0 dropped), then feature pruning removed 89 by delta-nRMSE. An alternative: replace the delta-nRMSE pruning with HC3-guided selection — retain only features where at least one output has a Wald t-statistic above a threshold (e.g., |t| > 2), and drop features that are Wald-zero-compatible for *all* outputs. This produces a statistically grounded sparse model rather than a greedy-performance pruned one, which aligns better with the manuscript's de-biased LASSO narrative.

**Expected outcome**: ~40–80 features retained with explicit statistical justification; better manuscript story; better performance than the current 29-feature model.

### 4. Document per-output heterogeneity (low effort, high publication value)

The per-output NRMSE ranges from 0.046 (10th pct) to 0.172 (90th pct) with median 0.095. The worst outputs are specific product outputs (TransEster diesel, 2023–2025). Documenting this heterogeneity in the manuscript — and showing the model works well for the bulk of outputs despite a few hard cases — strengthens the "statistics-first" narrative considerably.

**Action**: add a table or figure (per_output_nrmse.csv is available) showing the distribution and identifying the worst outputs by name.

### 5. Verify nRMSE normalization alignment (quick verification)

The manuscript reference uses "Y_train range" normalization. Our code reports `nrmse_reference_matrix: Y_train` in the summary. Verify that the normalization is identical (especially the `min_range=1e-6` floor and whether outputs with near-zero range are excluded). A normalization mismatch could contribute to the 0.106 vs 0.0445 gap beyond just feature count differences.

______________________________________________________________________

## Methodology updates for manuscript text

### Updated from prior revision notes

The prior revision notes (v4) describe the manuscript workflow using private-run numbers. The following should be updated in the manuscript to reflect the full-dataset reproducible run:

| Section                       | Old value               | New value                                               |
| ----------------------------- | ----------------------- | ------------------------------------------------------- |
| Training/holdout split        | 18,000/2,000            | 28,500/1,500                                            |
| Total runs                    | 20,000                  | 30,000                                                  |
| Retained scalar outputs       | ~9,782                  | 9,954                                                   |
| Empirical null retained       | 349 (manuscript ref)    | 69 (this run)                                           |
| Interaction pairs retained    | 248 (manuscript)        | 8 (this run — see improvement #2)                       |
| Nonlinear transforms retained | 37 (manuscript ref)     | 41 (this run)                                           |
| Sparse selection support      | 340 (manuscript ref)    | 118 (this run)                                          |
| Final OLS support             | 340 (manuscript ref)    | 29 (this run, pending pruning fix)                      |
| Final OLS holdout nRMSE       | 0.0445 (manuscript ref) | 0.1063 (this run, expected to improve with pruning fix) |
| Ablation table                | placeholder             | see ablation table above                                |

### Distributed execution methodology (new section for reproducibility)

The full-dataset pipeline was executed using a distributed HPC workflow on Kestrel:

- **Interaction discovery**: 2,000 pair-range shards, each assigned an explicit non-overlapping range of feature-pair indices over the `C(n,2)` candidate space. Array job `13947082` (2,000 tasks, `QOS=high`, throttled to 160 concurrent).
- **Sparse selection**: 50 shards. Array job `13974695` (50 tasks).
- **Nonlinear discovery**: 69 shards. Full nonlinear array.
- **Final manuscript artifacts**: 100 shards. Final array.
- **Reduction**: All stages merged via `hpc_reduce.py` to canonical merged CSVs; `run_manuscript_pipeline.py` loads distributed merged outputs when canonical paths are absent.
- **Reproducibility config**: `configs/hpc/kestrel_publication_full_dataset_distributed_base.yml`
- **Submission script**: `scripts/kestrel/submit_publication_full_dataset_distributed.sh`
- **Status script**: `scripts/kestrel/status_publication_full_dataset_distributed.sh`

______________________________________________________________________

## Reproducibility artifact registry

| Artifact                 | Path                                                                                                                     | Notes                                    |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------- |
| Ablation table           | `artifacts/publication_full_dataset_20260519/final_manuscript_artifacts/tables/ablation_table.csv`                       | Verified full-dataset numbers            |
| Model performance        | `artifacts/publication_full_dataset_20260519/final_manuscript_artifacts/tables/model_performance.csv`                    | Includes bootstrap CIs                   |
| Workflow stage summary   | `artifacts/publication_full_dataset_20260519/final_manuscript_artifacts/tables/workflow_stage_summary.csv`               | All stage counts + manuscript references |
| Final support features   | `artifacts/publication_full_dataset_20260519/final_manuscript_artifacts/final_model/final_support_features.csv`          | 29 features with HC3 stats               |
| HC3 Wald intervals       | `artifacts/publication_full_dataset_20260519/final_manuscript_artifacts/final_model/hc3_wald_intervals.csv`              | Per-feature per-output intervals         |
| Coefficient matrix (raw) | `artifacts/publication_full_dataset_20260519/final_manuscript_artifacts/final_model/coefficient_matrix_raw_scale.csv`    | Exportable model artifact                |
| Coefficient matrix (std) | `artifacts/publication_full_dataset_20260519/final_manuscript_artifacts/final_model/coefficient_matrix_standardized.csv` | Exportable model artifact                |
| Per-output NRMSE         | `artifacts/publication_full_dataset_20260519/final_manuscript_artifacts/tables/per_output_nrmse.csv`                     | 9,954 rows                               |
| Feature pruning curve    | `artifacts/publication_full_dataset_20260519/final_manuscript_artifacts/figures/figure_feature_pruning_curve.svg`        | Pruning impact figure                    |
| nRMSE bootstrap summary  | `artifacts/publication_full_dataset_20260519/final_manuscript_artifacts/figures/figure_nrmse_bootstrap_summary.svg`      | Model comparison figure                  |
| Support composition      | `artifacts/publication_full_dataset_20260519/final_manuscript_artifacts/figures/figure_support_composition.svg`          | Feature type breakdown                   |
| Interaction heatmap      | `artifacts/publication_full_dataset_20260519/final_manuscript_artifacts/figures/fig_module_pair_heatmap.svg`             | Module-pair interaction density          |
| HPC config               | `configs/hpc/kestrel_publication_full_dataset_distributed_base.yml`                                                      | Full run configuration                   |

______________________________________________________________________

## Remaining placeholder gaps (from v4 revision notes, still open)

- Repository URL and DOI (placeholder URLs in manuscript)
- Supplement filename
- Names of internal reviewers/collaborators
- Award number, contract text, and lab disclaimer
- Steve Peterson affiliation
- Figure captions for interaction heatmap and module figures (figures now available locally for inspection)
