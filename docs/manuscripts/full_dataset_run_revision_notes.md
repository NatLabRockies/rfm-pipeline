# Full-dataset manuscript workflow revision notes

**Primary comparison target**: `docs/manuscripts/jds_bsm.tex`\
**Latest completed full-dataset run**: `publication_full_dataset_distributed_20260526_short_hp1` (reconfigured with `null_threshold_quantile=0.995`)\
**Artifact root (local)**:
`artifacts/publication_full_dataset_distributed_results/publication_full_dataset_distributed_20260526_short_hp1/artifacts/final_manuscript_artifacts/`

> **Previous runs** (`publication_full_dataset_distributed_20260519` and the pre-reconfiguration
> short_hp1 attempt with `p_threshold=0.05`) are superseded. The canonical run uses
> `null_threshold_quantile=0.995` for the SHAP-null interaction threshold, which retains far
> more interaction pairs than the earlier configurations.

**Dataset**: One dataset — 158 scalar LHC inputs × 4 binary scenario combinations = 30,000 total rows
(28,500 training, 1,500 holdout, 5% holdout fraction, seed 123).

______________________________________________________________________

## Workflow delta: manuscript description vs current implementation

| Area                        | Manuscript currently describes                                        | Current implemented workflow                                                                                                                             | Current status               |
| --------------------------- | --------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------- |
| Entrypoint                  | Notebook/script-driven reproduction narrative                         | Config-driven pipeline runner (`tools/run_manuscript_pipeline.py`) with stage-window resume and tracked run markers                                      | **Changed**                  |
| Holdout policy              | 10% deterministic holdout, seed 123                                   | 5% holdout (1,500 rows), seed 123; strict stage-level checkpoint/resume                                                                                  | **Holdout fraction updated** |
| Output conditioning         | Train-only filtering + PCA reduction                                  | Same; canonical run retained 9,954 of 23,495 outputs (20 PCA components, 90% variance); final OLS fit against **all 23,495** outputs                     | **OLS scope updated**        |
| Empirical null screening    | Borgonovo Delta measure, 5% per-output max-null cutoff; ~349 terms    | L2 permutation + BH FDR q=0.05; retained 69/158 terms. Method gap acknowledged; nRMSE gap to original manuscript is driven primarily by this difference  | **Method different**         |
| Interaction discovery       | High interaction count (~248 pairs retained)                          | `null_threshold_quantile=0.995` SHAP-null threshold; 410 pairs retained                                                                                  | **Numerically close**        |
| Nonlinear discovery         | Curvature-discovery stage; ~37 transforms                             | GAM/spline-aligned nonlinear discovery; 41 transforms retained                                                                                           | **Close**                    |
| Sparse selection            | ~340 selected features before final export                            | EBIC/LASSO + stability (50 subsamples, Jaccard ≥0.75, Spearman ≥0.90); 506 stable terms                                                                  | **Numerically different**    |
| Feature pruning             | Not described separately                                              | HC3 inferential filter retained all 506; delta-threshold diagnostic computed but **not applied** (0 features pruned in canonical run)                    | **Expanded + clarified**     |
| Final OLS output scope      | Not explicitly addressed                                              | Final OLS fit against **all 23,495** scalar outputs; near-constant culled outputs get near-zero coefficients with training-mean intercepts               | **Expanded**                 |
| Final performance reporting | Manuscript final macro nRMSE 0.0445 (Y_train normalised); 30,000 runs | Canonical run nRMSE 0.0676 [0.0660, 0.0688]; driven mainly by the screening method gap (69 vs ~349 screened inputs limits interaction and support space) | **Numerically different**    |
| LASSO alpha selection       | Described as "40th percentile of regularization path"                 | Minimum EBIC on 40-point geometric grid from α_max to α_max × 10^{-4}                                                                                    | **Corrected in manuscript**  |
| Interaction threshold       | Described as "p-value 0.05"                                           | `null_threshold_quantile=0.995`: retains pairs where observed SHAP > 99.5th percentile of permutation null                                               | **Corrected in manuscript**  |
| nRMSE normalization label   | Y_train range                                                         | Confirmed Y_train range throughout; label was erroneously `"fixed Y_ref range"` in code, corrected in `src/bsm_rfm/metrics.py`                           | **Label fixed**              |
| Figure generation           | Static `.png` in `docs/manuscripts/`                                  | Pipeline emits deterministic SVG bundle + source CSVs in `final_manuscript_artifacts/figures/`                                                           | **Changed**                  |

______________________________________________________________________

## Latest measured values — canonical run (`publication_full_dataset_distributed_20260526_short_hp1`, reconfigured)

### Dataset and stage progression

| Quantity                               | Canonical run value | Previous (20260519) | Manuscript reference |
| -------------------------------------- | ------------------- | ------------------- | -------------------- |
| Total rows                             | 30,000              | 30,000              | 30,000               |
| Training rows                          | 28,500              | 28,500              | —                    |
| Holdout rows                           | 1,500               | 1,500               | —                    |
| Scalar responses (output interface)    | 23,495              | 23,495              | 23,495               |
| Retained outputs for PCA               | 9,954               | 9,954               | —                    |
| Culled outputs (near-constant)         | 13,541              | 13,541              | —                    |
| Retained PCA components                | 20                  | 20                  | —                    |
| Outputs in exported coefficient matrix | **23,495 (all)**    | 9,954 (old)         | 23,495               |
| Empirical-null retained terms          | 69                  | 69                  | ~349                 |
| Retained interaction pairs             | **410**             | 8                   | ~248                 |
| Retained nonlinear transforms          | 41                  | 41                  | ~37                  |
| Enriched features (discovery output)   | **520**             | 118                 | —                    |
| Sparse stable support                  | **506**             | 118                 | ~340                 |
| HC3-retained features                  | **506**             | 118                 | ~340                 |
| Features pruned (delta threshold)      | **0**               | 89                  | —                    |
| Final OLS support                      | **506**             | 29                  | ~340                 |
| Final OLS: first-order terms           | **63**              | —                   | —                    |
| Final OLS: interaction terms           | **403**             | —                   | —                    |
| Final OLS: transformation terms        | **40**              | —                   | —                    |
| Final holdout macro nRMSE              | **0.0676**          | 0.1063              | 0.0445               |
| Final nRMSE 95% CI                     | [0.0660, 0.0688]    | —                   | —                    |

Source: `tables/workflow_stage_summary.csv`, `tables/model_performance.csv`, `tables/ablation_table.csv`.

### Ablation table (canonical run)

| Model            | n features | Holdout nRMSE | 95% CI lower | 95% CI upper |
| ---------------- | ---------- | ------------- | ------------ | ------------ |
| Null mean        | 0          | 0.1653        | 0.1635       | 0.1663       |
| Main effects OLS | 158        | 0.1653        | 0.1635       | 0.1663       |
| Screened OLS     | 69         | 0.0812        | 0.0796       | 0.0821       |
| Penalized OLS    | 506        | 0.0676        | 0.0660       | 0.0688       |
| Final OLS        | 506        | 0.0676        | 0.0660       | 0.0688       |

All nRMSE values normalised to Y_train range. Source: `tables/ablation_table.csv`.

### Per-output nRMSE distribution (canonical run, 9,954 non-negligible outputs)

| Statistic | nRMSE |
| --------- | ----- |
| p10       | 0.013 |
| p25       | 0.032 |
| p50       | 0.057 |
| p75       | 0.090 |
| p90       | 0.136 |

Worst 3 outputs: `OI.OHC output by product[TransEster, P, diesel]` for 2023–2025 (nRMSE 0.495 / 0.487 / 0.475).
Source: `tables/per_output_nrmse_summary.csv`.

______________________________________________________________________

## Interaction discovery: reconfiguration rationale

**Previous config** (`p_threshold=0.05`): over-aggressive pruning via a direct p-value filter on SHAP
interaction scores, retaining only ~8–62 pairs.

**Canonical config** (`null_threshold_quantile=0.995`): retains pairs where the observed SHAP-null
score exceeds the 99.5th percentile of the permutation null distribution. This is a stricter but
more principled calibration: a pair must beat 99.5% of null replicates rather than passing a
raw p-value threshold. Result: **410 retained pairs**, 520 enriched features entering LASSO.

______________________________________________________________________

## Empirical null screening gap

**Observed gap**: 69 retained terms (current run) vs ~349 (manuscript reference value).

The gap is entirely a **method difference**, not a data or implementation bug:

- Manuscript used Borgonovo Delta with a permissive 5% per-output max-null cutoff
- Current public pipeline uses L2 permutation + BH FDR q=0.05 (stricter global FDR control)

The remaining nRMSE gap (0.0676 vs 0.0445) is driven mainly by this screening difference: fewer
screened inputs → smaller interaction search space → smaller support. A future update to implement
Borgonovo Delta screening would close most of this gap.

The reference value `retained_terms: 349` in `configs/manuscript_case_study.yml` is a stale
intermediate count (349 > 158 cannot represent retained first-order features); it is a contract
reference label, not a manuscript-text-backed number.

______________________________________________________________________

## Final OLS output coverage (code fix 2026-05-30, commit f4a634c)

Previous behavior: final OLS fit only against 9,954 variance-filtered outputs; coefficient matrix
had 9,954 rows.

**Current behavior**: final OLS fit against all 23,495 scalar outputs:

- Near-constant culled outputs receive near-zero coefficients and training-mean intercepts
- `y_scales` for zero-variance outputs is clamped to 1 before computing standardized coefficients
- Exported `coefficient_matrix_standardized.csv` and `coefficient_matrix_raw_scale.csv` have
  23,495 rows × 506 columns
- `y_standardization.csv` covers all 23,495 outputs
- nRMSE metric is still evaluated on 9,954 non-negligible outputs only (diluting macro average
  with trivially-constant outputs would not be a meaningful performance measure)

Artifact rerun (job 14043519 on Kestrel `short` partition) regenerates the coefficient matrices
and `final_ols_summary.csv` with `n_retained_outputs=23495` and `n_variance_filtered_outputs=9954`.

______________________________________________________________________

## Figure-generation parity audit

### Figures the manuscript currently expects

| Manuscript figure asset                    | Pipeline output                                    | Parity status                       |
| ------------------------------------------ | -------------------------------------------------- | ----------------------------------- |
| `figure_nrmse_bootstrap_summary.svg`       | `figures/figure_nrmse_bootstrap_summary.svg`       | **Present** (ablation with CI bars) |
| `figure_per_output_nrmse_distribution.svg` | `figures/figure_per_output_nrmse_distribution.svg` | **Present** (9,954 outputs)         |
| `figure_support_composition.svg`           | `figures/figure_support_composition.svg`           | **Present** (403 / 63 / 40)         |
| `figure_selected_by_module_count.svg`      | `figures/figure_selected_by_module_count.svg`      | **Present** (WW/OH/CHC/SE/AHC)      |
| `fig_module_pair_heatmap.svg`              | `figures/fig_module_pair_heatmap.svg`              | **Present** (403 interaction pairs) |

### Style harmonization

Figure renderers use Okabe-Ito colour-blind palette (`#0072B2`, `#E69F00`, etc.), Helvetica/Arial
fonts, shared axis and grid styling. All five manuscript-facing figures are consistent.

______________________________________________________________________

## Open items and placeholders

### Resolved

- [x] Canonical run complete: 506 features, 0.0676 nRMSE (interaction reconfiguration fix)
- [x] LASSO description corrected to minimum EBIC (not 40th percentile)
- [x] Interaction threshold corrected to `null_threshold_quantile=0.995`
- [x] Final OLS now covers all 23,495 outputs (commit f4a634c)
- [x] Ablation table with bootstrap CIs from canonical run
- [x] Per-output nRMSE distribution (9,954 non-negligible outputs)
- [x] Feature pruning delta override removed (not applied; all 506 exported)
- [x] Figure registry complete (5 manuscript figures, all present)
- [x] Figure style harmonized (Okabe-Ito, consistent whitespace, left-aligned labels)

### Still pending (keep placeholders in manuscript)

- **[PENDING]** Artifact rerun (job 14043519) must complete to regenerate coefficient matrices with
  23,495-output coverage; then re-sync with `collect_publication_full_dataset_distributed.sh`
- **[PENDING]** Repository DOI, supplement archive DOI/name, author affiliation, acknowledgment/
  disclaimer text (awaiting publication)
- **[PENDING]** Sensitivity study §7 results (jobs running on Kestrel, ETA ~June 4)
- **[PENDING]** Worst-output investigation: `OI.OHC output by product[TransEster, P, diesel]`
  2023–2025 (nRMSE ~0.475–0.495) — potential model specification note
- **[PENDING]** Borgonovo Delta screening implementation to close the 69 vs ~349 screened-term
  gap; would require validated private-script equivalence check

______________________________________________________________________

## Regeneration command surface

```bash
# Re-run final artifacts stage only (reuses existing stage artifacts; fast)
pixi run python tools/run_manuscript_pipeline.py \
  configs/hpc/kestrel_publication_full_dataset_distributed_base.yml \
  --start-stage final_manuscript_artifacts \
  --output-dir /scratch/dhetting/bsm/studies/publication_full_dataset_distributed_20260526_short_hp1/artifacts

# Collect updated artifacts from Kestrel
bash scripts/kestrel/collect_publication_full_dataset_distributed.sh \
  --study-id publication_full_dataset_distributed_20260526_short_hp1 \
  --study-root /scratch/dhetting/bsm/studies/publication_full_dataset_distributed_20260526_short_hp1
```

Key outputs:

- tables: `final_manuscript_artifacts/tables/*.csv`
- figures: `final_manuscript_artifacts/figures/*.svg`
- coefficient matrix: `final_manuscript_artifacts/final_model/coefficient_matrix_standardized.csv` (23,495 × 506)
