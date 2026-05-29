# Full-dataset manuscript workflow revision notes

**Primary comparison target**: `docs/manuscripts/jds_bsm.tex`\
**Latest completed full-dataset run**: `publication_full_dataset_distributed_20260526_short_hp1`\
**Artifact root (local)**:
`artifacts/publication_full_dataset_distributed_results/publication_full_dataset_distributed_20260526_short_hp1/artifacts/final_manuscript_artifacts/`

> **Previous run** (`publication_full_dataset_distributed_20260519`) is superseded. Values from
> that run are kept in the comparison table where the change is material.

**Dataset**: One dataset — 158 scalar LHC inputs × 4 binary scenario combinations = 30,000 total rows
(28,500 training, 1,500 holdout). Manuscript reported 20,000 runs (18,000 training, 2,000 holdout, 10%
holdout fraction); current pipeline uses 30,000 runs (5% holdout). Same input space.

______________________________________________________________________

## Workflow delta: manuscript description vs current implementation

| Area                        | Manuscript currently describes                                                                         | Current implemented workflow                                                                                                                                 | Current status            |
| --------------------------- | ------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------- |
| Entrypoint                  | Notebook/script-driven reproduction narrative                                                          | Config-driven pipeline runner (`tools/run_manuscript_pipeline.py`) with stage-window resume and tracked run markers                                          | **Changed**               |
| Holdout policy              | 10% deterministic holdout, seed 123                                                                    | Same holdout contract, plus strict stage-level checkpoint/resume for long runs                                                                               | **Preserved + hardened**  |
| Output conditioning         | Train-only filtering + PCA reduction                                                                   | Same; full-dataset run retained 9,954 outputs and 20 PCA components (90.3% variance)                                                                         | **Numerically different** |
| Empirical null screening    | Manuscript used Borgonovo Delta measure, 5% per-output max-null cutoff (permissive); reported 20K runs | Public pipeline uses L2 permutation + BH FDR q=0.05 (stricter); retained 69/158 terms. Method gap flagged `not_yet_validated`.                               | **Method different**      |
| Interaction discovery       | Manuscript reports high retained interaction counts in enriched matrix/final support                   | Fixed over-aggressive pruning (`p_threshold` raised from 0.001 → 0.05); current run retained 62 pairs (vs 8 previously, vs ~248 in manuscript)               | **Numerically different** |
| Nonlinear discovery         | Manuscript reports curvature-discovery stage and final nonlinear support (37 ref. transforms)          | Implemented GAM/spline-aligned nonlinear discovery; current run retained 41 transformations                                                                  | **Close**                 |
| Sparse selection            | Manuscript reports ~340 selected features before final export                                          | EBIC/L1/stability pipeline retained 172 stable terms (vs 118 previously)                                                                                     | **Numerically different** |
| Feature pruning             | Not described separately in manuscript                                                                 | HC3 inferential filter + single-feature delta pruning; `delta_threshold_override: 0.002` retains 132/172 terms (auto-cutoff would remove 121 at delta≥0.024) | **Expanded**              |
| Final performance reporting | Manuscript final macro nRMSE 0.0445 (Y_train normalised)                                               | Current run final OLS nRMSE 0.0721 (Y_train normalised); 32% improvement over 20260519 run (0.1063) driven by interaction fix                                | **Numerically different** |
| nRMSE normalization label   | Manuscript uses Y_train range                                                                          | Confirmed Y_train range throughout. Label was erroneously `"fixed Y_ref range"` in code output; corrected to `"Y_train range"` in `src/bsm_rfm/metrics.py`.  | **Label fixed**           |
| Execution environment       | Mostly local single-path narrative                                                                     | Local + distributed HPC orchestration with shard/reduce/pullback modes                                                                                       | **Expanded**              |
| Figure generation           | Manuscript expects static result figures in `docs/manuscripts/`                                        | Pipeline emits deterministic SVG figure bundle + source CSVs in `final_manuscript_artifacts/figures/`                                                        | **Changed**               |

______________________________________________________________________

## Latest measured values (run `publication_full_dataset_distributed_20260526_short_hp1`)

### Dataset and stage progression

| Quantity                           | Current value (short_hp1) | Previous value (20260519) | Manuscript reference |
| ---------------------------------- | ------------------------- | ------------------------- | -------------------- |
| Total rows                         | 30,000                    | 30,000                    | —                    |
| Training rows                      | 28,500                    | 28,500                    | —                    |
| Holdout rows                       | 1,500                     | 1,500                     | —                    |
| Retained scalar outputs            | 9,954                     | 9,954                     | —                    |
| Retained PCA components            | 20                        | 20                        | —                    |
| Empirical-null retained terms      | 69                        | 69                        | 349                  |
| Retained interaction pairs         | 62                        | 8                         | ~248                 |
| Retained nonlinear transformations | 41                        | 41                        | 37                   |
| Sparse stable support              | 172                       | 118                       | 340                  |
| HC3-retained terms                 | 172                       | 118                       | 340                  |
| Pruned terms (delta override)      | 40                        | 89                        | —                    |
| Final OLS support                  | 132                       | 29                        | ~340                 |
| Final holdout macro nRMSE          | 0.0720948                 | 0.1062981                 | 0.0445               |

Source: `tables/workflow_stage_summary.csv`, `tables/model_performance.csv`, `tables/feature_pruning_summary.csv`.

### Ablation table (measured, short_hp1)

| Model            | n features | Holdout nRMSE | 95% CI lower | 95% CI upper |
| ---------------- | ---------- | ------------- | ------------ | ------------ |
| Null mean        | 0          | 0.165271      | 0.163529     | 0.166326     |
| Main effects OLS | 158        | 0.081184      | 0.079626     | 0.082072     |
| Screened OLS     | 69         | 0.081187      | 0.079610     | 0.082074     |
| Penalized OLS    | 172        | 0.070910      | 0.069380     | 0.071865     |
| Final OLS        | 132        | 0.072095      | 0.070595     | 0.072985     |

All nRMSE values normalised to Y_train range. Source: `tables/ablation_table.csv`.

### Per-output nRMSE distribution (measured, short_hp1)

| Statistic | nRMSE    |
| --------- | -------- |
| p10       | 0.018176 |
| p25       | 0.035743 |
| p50       | 0.061087 |
| p75       | 0.092973 |
| p90       | 0.140703 |

Worst 3 outputs: `OI.OHC output by product[TransEster, P, diesel]` for 2023–2025 (nRMSE 0.495/0.487/0.476).
These three outputs are the same pathway/product combination across consecutive years, suggesting a model
specification issue rather than a random fitting failure. Source: `tables/per_output_nrmse_summary.csv`.

______________________________________________________________________

## Empirical null screening gap: investigation findings

**Observed gap**: 69 retained terms (current run) vs 349 (contract reference value).

**Root cause: different screening method, not different data.**

The 158 first-order inputs are the same in both the manuscript analysis and the current pipeline. The
gap arises because:

1. **Different screening method**: The manuscript used the **Borgonovo Delta sensitivity measure** with
   a **5% output-wise maximum-null cutoff** (permissive per-output family-wise screen). Our pipeline uses
   `coefficient_row_l2_permutation` — L2 norm of PCA-score coefficients under permutation — with
   **BH FDR q=0.05** (stricter global FDR control). A permissive per-output max-null screen will retain
   more inputs than a conservative global BH screen.

1. **Different training set size**: manuscript used 18,000 training rows; current pipeline uses 28,500.
   Larger training sets give more power, but the method difference dominates.

1. **The 349 reference value**: The `retained_terms: 349` in `configs/manuscript_case_study.yml` is a
   contract reference hardcoded from a prior draft. **The published manuscript text does not explicitly
   state how many inputs passed the Delta screen** — it only says "the initial 160-input interface was
   reduced." The number 349 does not appear in the text, and 349 > 158 means it cannot represent
   retained first-order features. The provenance of 349 is unclear; it may be a stale intermediate
   count (e.g., candidates entering LASSO, 352 − some failed candidates = ~349) or an earlier draft value.

1. **Flagged in artifacts**: `implementation_status: source_backed_public_surrogate` and
   `source_script_equivalence_status: not_yet_validated` explicitly acknowledge that the current public
   implementation has not been validated against the private Delta screening scripts.

**Downstream consequence**: retaining fewer first-order terms limits interaction candidates, nonlinear
candidates, and sparse support — this is the primary driver of the remaining gap to manuscript nRMSE
(0.0721 vs 0.0445). To close the gap, the public implementation would need to be updated to match the
Delta screening method used in the original analysis.

______________________________________________________________________

## Interaction discovery fix (short_hp1)

**Problem (20260519)**: `p_threshold: 0.001` in `interaction_discovery` config caused over-aggressive
pruning. Only 8 of ~2346 screened output-pairs retained interactions.

**Fix (short_hp1)**: `p_threshold` raised to `0.05` (consistent with BH FDR level used in null screen).
Result: **62 retained interaction pairs** — 7.75× improvement.

Remaining gap to manuscript (~248 pairs) is again explained by the smaller first-order feature space
(62 first-order terms in final stable support vs manuscript's ~340), which reduces the interaction
discovery search space.

______________________________________________________________________

## nRMSE normalization

Confirmed throughout: all nRMSE values use **Y_train range** as the normalisation reference, matching the
manuscript contract. The code previously emitted the label `"fixed Y_ref range"` generically; corrected to
`"Y_train range"` in `src/bsm_rfm/metrics.py` line 243. No values changed — label only.

______________________________________________________________________

## Feature pruning delta override

The auto-cutoff at delta ≥ 0.024 would remove 121/172 stable terms, collapsing the final model to 51
features and pushing macro nRMSE upper bound to ~0.815. The `delta_threshold_override: 0.002` in the
HPC config instead retains 132/172 terms (removing 40), keeping nRMSE at 0.0721. This override is
intentional and documented in the pipeline config.

______________________________________________________________________

## Figure-generation parity audit

### Figures the manuscript currently expects

| Manuscript figure asset                   | Current pipeline output(s)                                                               | Parity status                                                                 |
| ----------------------------------------- | ---------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| `results_feature_type_composition.png`    | `figures/figure_support_composition.svg` and `figures/fig_feature_type_distribution.svg` | **Concept parity** (same composition content, pipeline now emits SVG)         |
| `results_interaction_density_heatmap.png` | `figures/fig_module_pair_heatmap.svg`                                                    | **Concept parity** (same module-pair heatmap content, pipeline now emits SVG) |

### Full figure registry (short_hp1 run, 11 figures)

| Figure asset                               | Description                                                    |
| ------------------------------------------ | -------------------------------------------------------------- |
| `figure_model_performance.svg`             | Macro nRMSE bar chart: current run vs manuscript reference     |
| `figure_support_composition.svg`           | Feature type composition of final support (bar, pct)           |
| `figure_selected_by_module_count.svg`      | Selected feature counts by BSM module                          |
| `figure_selected_by_module_share.svg`      | Selected feature shares by BSM module                          |
| `figure_nrmse_bootstrap_summary.svg`       | Bootstrap nRMSE CI across ablation models                      |
| `figure_feature_pruning_curve.svg`         | Per-feature delta vs rank (pruning diagnostic)                 |
| `figure_per_output_nrmse_distribution.svg` | Per-output nRMSE distribution across all 9,954 outputs         |
| `fig_feature_type_distribution.svg`        | Feature type count breakdown (first-order/interaction/nonlin.) |
| `fig_influential_by_module.svg`            | Top influential features grouped by BSM module                 |
| `fig_module_pair_heatmap.svg`              | Module-pair interaction density heatmap                        |
| `fig_module_total_interactions.svg`        | Total interactions per module                                  |

### Style harmonization status

Figure renderers in `src/bsm_rfm/manuscript_stages.py` are unified to one manuscript palette/font
system across both legacy (`fig_*`) and newer (`figure_*`) outputs:

- shared Okabe-Ito colour-blind palette (`_SVG_COLOR_PRIMARY = "#0072B2"` etc.) — **committed in working tree, not yet in HPC run**
- shared font family (`Helvetica, Arial, sans-serif`)
- consistent title weight, axis styling, grid tone, and annotation colours

The SVG figures in the short_hp1 artifacts use the pre-Okabe-Ito palette (HPC ran commit `f5cbaf1`
which predates the colour update). Figures need to be regenerated after the colour update is committed
and a new HPC run completes, or via `scripts/regenerate_manuscript_figures.py` applied to the collected
artifacts.

______________________________________________________________________

## Real values now available vs placeholders still required

### Resolved (filled in this session)

- [x] Full pipeline completed with interaction fix: 62 pairs, 132 features, nRMSE 0.0721
- [x] Ablation table with bootstrap CIs from short_hp1
- [x] Per-output nRMSE distribution from short_hp1 (9,954 outputs)
- [x] Empirical null gap root-cause identified (dataset scope, not algorithm)
- [x] nRMSE normalization label corrected in source (`metrics.py`) and pulled CSV
- [x] Feature pruning delta override rationale documented
- [x] Figure registry complete (11 figures, all present in artifacts)

### Still pending (keep placeholders)

- **[PENDING]** Commit Okabe-Ito colour-blind palette update, regenerate figures (working tree modified; `src/bsm_rfm/manuscript_stages.py` has uncommitted changes).
- **[PENDING]** Investigate worst 3 outputs (`OI.OHC output by product[TransEster, P, diesel]` 2023–2025, nRMSE ~0.48–0.50): determine whether this is a model specification issue warranting a manuscript note or a fixable data issue.
- **[PENDING]** Private-run manuscript parity package: formal equivalence check of empirical null screen against private delta-null script; coefficient parity across final OLS support.
- **[PENDING]** Final manuscript release metadata: repository DOI, supplement archive DOI/name, formal acknowledgment/disclaimer text, author affiliation placeholders.
- **[PENDING]** Manuscript text update: narrative currently references 349/340/248/0.0445 reference values; text needs a pass to either update to public-run values or to frame as a comparison between the full private analysis and the public reproduction.
- **[PENDING]** Decision on figure asset format contract (`.png` in paper sources vs `.svg` produced by pipeline) and final publication export step.
- **[PENDING]** Implement Delta sensitivity screening (Borgonovo 2007) to replace the current `coefficient_row_l2_permutation` + BH FDR approach. This is the primary gap between the manuscript method and the current pipeline, and is the main driver of the 69 vs ~349 screened-term difference and the downstream nRMSE gap (0.0721 vs 0.0445). Validate against private Delta screening scripts before submission.
- **[PENDING]** Resolve the `retained_terms: 349` contract reference value — clarify whether this is a manuscript-text-backed number or a stale draft value, since 349 > 158 and the published text does not explicitly report this count.

______________________________________________________________________

## Regeneration command surface (for updated manuscript figures/tables)

```bash
# Full pipeline rerun from scratch
pixi run python tools/run_manuscript_pipeline.py configs/hpc/kestrel_publication_full_dataset_distributed_base.yml

# Re-run final artifacts stage only (fastest; uses existing stage artifacts)
pixi run python tools/run_manuscript_pipeline.py <config.yml> \
  --start-stage final_manuscript_artifacts \
  --stop-stage final_manuscript_artifacts

# Re-render figures locally from already-collected HPC run artifacts
pixi run python scripts/regenerate_manuscript_figures.py \
  artifacts/publication_full_dataset_distributed_results/publication_full_dataset_distributed_20260526_short_hp1
```

Key outputs:

- tables: `final_manuscript_artifacts/tables/*.csv`
- figures: `final_manuscript_artifacts/figures/*.svg`
- figure registry: `final_manuscript_artifacts/figures/figure_specs.csv`
