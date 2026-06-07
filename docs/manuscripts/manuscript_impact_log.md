# Manuscript Impact Log

Running ledger of code/config/data changes that affect the JDS BSM
manuscript (`jds_bsm_v22.tex`) or that require re-running upstream
artifacts. Update this file every time a change lands that:

1. invalidates a published or staged result,
1. forces a re-fit / re-run / re-export,
1. changes a value cited in the manuscript text or tables, **or**
1. adds a new figure, table, or numeric claim that should be reflected
   in the manuscript.

Entries are newest-first. Each entry records: date, scope, what
changed, manuscript impact, and the re-run / re-export action
required.

______________________________________________________________________

## 2026-06-07 — Round 9 audit: RF tree count, importance metric, BSM panel labels

### 🟡 R4. RandomForest configuration brought into line with manuscript

- **What changed (code).**
  - `n_estimators` bumped 200 → **500** in `scripts/fit_sensitivity_rf.py`
    and the cross-validated RF fit in `scripts/plot_sensitivity_results.py`
    to match the manuscript's "500-tree Random Forest" wording
    (`jds_bsm_v22.tex:640`).
  - Re-refit `QUALITY_IMPORTANCE` / `RUNTIME_IMPORTANCE` arrays at 500
    trees (third-decimal updates only — RF averaging is already
    stabilised at 200; ordering and headline percentages unchanged).
  - `render_bsm_validation` now computes `n_successful`,
    `n_null_screened`, and the median-γ reference line dynamically from
    the supplied CSV. Removed hardcoded `BSM_MEDIAN_SUCCESSFUL = -0.462`
    and inline labels `(n = 2,583)` / `null-screened (n=534)` which were
    pinned to the pre-wave2 run.
- **Manuscript impact (editor must fix in v22.tex before submission).**
  - **§6 / Figure 8 caption — importance metric label is wrong.**
    Manuscript currently says "mean decrease in **accuracy**" (lines
    640 and 685). `sklearn.RandomForestRegressor.feature_importances_`
    returns **mean decrease in impurity (MDI)** — a different quantity
    (variance-reduction split contribution, not permutation-based
    out-of-bag accuracy drop). The plot's own axis label correctly says
    "mean decrease in impurity". **Action: replace both occurrences of
    "mean decrease in accuracy" with "mean decrease in impurity
    (Gini-equivalent variance reduction; sklearn `feature_importances_`)".**
  - **§6 / Figure 8 caption — sample sizes.** Caption currently cites
    `n = 2{,}583` (quality) and `n = 2{,}049` (runtime, "successful").
    Refit on wave12 uses `n = 4{,}027` and `n = 3{,}014`.
  - **§6 / Figure 9 panel ("BSM operating-point validation").** The
    γ-distribution histogram on the left panel is now drawn from
    wave12; new headline numbers:
    - Successful sensitivity runs: **n = 3,014** (was 2,583)
    - Null-screened runs: **n = 1,013** (was 534)
    - Median γ across successful fits: **−0.448** (was −0.462)
    - q10 / q90 of γ: **(−0.655, −0.304)** (was approx. (−0.695, −0.301))
  - **§6.1 BSM validation prose (jds_bsm_v22.tex:696).** The "103 min,
    99–111 min PI, 1.6% of measured mean" runtime claim cites the
    pre-cleanup RF runtime model. After the wave12 refit, the runtime
    point + 80% PI for the BSM operating point are stale. **Until the
    BSM analog selection / feature-vector capture script is rebuilt,
    treat these numbers as TODO; do not republish the validation point
    estimate.**
- **What was NOT changed (deferred — coupled to BSM refit).**
  - `BSM_RF_PRED = 0.0762`, `BSM_RF_P10 = 0.0729`, `BSM_RF_P90 = 0.0774`
    and the `BSM_ANALOG_NRMSE` list remain at the pre-cleanup values.
    The plot script's BSM-validation figure is therefore not republished
    to `docs/manuscripts/` from round 9; only the importance figure was
    copied. Closing this TODO requires (a) the BSM-config feature
    vector and (b) the analog-selection criterion (k-nearest by which
    metric?) — neither captured in any committed script.
- **Files touched.**
  - `scripts/fit_sensitivity_rf.py` (n_estimators 200→500).
  - `scripts/plot_sensitivity_rf_figures.py` (importance arrays + dynamic
    BSM-panel labels; dropped `BSM_MEDIAN_SUCCESSFUL`).
  - `scripts/plot_sensitivity_results.py` (CV RF n_estimators 200→500).
  - `docs/manuscripts/fig_sensitivity_rf_importance.svg` (regenerated
    at 500 trees).
- **Re-run recipe.**
  ```bash
  pixi run python scripts/fit_sensitivity_rf.py \
      --results artifacts/sensitivity/wave12_combined_clean.csv
  pixi run python scripts/plot_sensitivity_rf_figures.py \
      --results artifacts/sensitivity/wave12_combined_clean.csv \
      --output-dir artifacts/sensitivity/figures_wave12_clean
  cp artifacts/sensitivity/figures_wave12_clean/fig_sensitivity_rf_importance.svg \
     docs/manuscripts/
  # NOTE: do NOT copy fig_sensitivity_bsm_validation.svg until BSM_RF_PRED
  # block is refit on cleaned wave12.
  ```

______________________________________________________________________

## 2026-06-07 — Sensitivity Random Forest refit + Figure 8 regeneration (round 8 audit)

### 🟡 R3. RF meta-regression refit on cleaned wave12 (companion to R1)

- **What changed.** Round 7 refit the **linear** sensitivity meta-regression
  on the cleaned wave12 union (4027 rows) but left the **Random Forest**
  meta-regression untouched. Round-8 adversarial audit caught the
  asymmetry: the hardcoded `QUALITY_IMPORTANCE` / `RUNTIME_IMPORTANCE`
  arrays in `scripts/plot_sensitivity_rf_figures.py` still encoded the
  pre-cleanup wave1 fit (n = 2583/2049) that included the dead
  `lasso_alpha_percentile` column.
- **What we did.**
  - Added reproducible fit script `scripts/fit_sensitivity_rf.py`
    (RandomForestRegressor, n_estimators=200, random_state=42).
  - Refit on cleaned wave12 with the canonical 13-feature
    `_RF_FEATURES` list (constant `lasso_alpha_grid_size = 40` retained;
    sklearn assigns it importance ≈ 0).
  - Quality model: all 4027 rows, target = `nrmse_relative` (null-screened
    rows filled to 0).
  - Runtime model: 3014 successful rows only, target =
    `log(total_wall_seconds)` — matches manuscript Figure 8 caption.
  - Replaced the 8-row hardcoded tuples and regenerated
    `fig_sensitivity_rf_importance.{svg,pdf}`.
- **Manuscript impact (§6 prose + Figure 8 caption).**
  - Quality (was → now):
    - BH threshold (q): 33% → **53%** (still #1, larger share)
    - Sparsity (s): 8% → **11%**
    - Screening permutations: 10% → 7%
    - Stability subsamples: 7% → drops out of top-8
    - LASSO α percentile: 5% → **removed (dead knob)**
  - Runtime (was → now):
    - Input count (d): 31% → 35% (now #2, not #1)
    - Sparsity (s): 27% → **39% (now #1)**
    - Run count (n): 10% → 7%
    - LASSO α percentile: 2% → **removed (dead knob)**
    - Stability subsamples & Interaction permutations enter top-8.
  - Sample sizes: caption `n = 2{,}583` / `n = 2{,}049` → `n = 4{,}027`
    / `n = 3{,}014`.
- **Editor note (replace §6 paragraph starting "For quality, q accounts
  for 33% …").**
  - "For quality, q accounts for **53%** of RF importance, followed by
    sparsity (**11%**), input count d (**7%**), and screening
    permutations (**7%**) (Figure~\\ref{fig:sensitivity-rf-importance}).
    q controls how many inputs enter interaction and nonlinearity
    discovery; loosening it forwards more candidates to the enrichment
    stages, increasing both model quality and runtime."
  - "For runtime, sparsity (**39%**) and the number of inputs d
    (**35%**) together account for over **74%** of the variance
    (Figure~\\ref{fig:sensitivity-rf-importance}, right panel),
    consistent with the screening stage's
    $O(N\_\\mathrm{perm}^{(s)} \\times n \\times d)$ complexity and the
    quadratic growth in candidate interaction pairs with the screened
    input count."
  - Update Figure 8 caption sample sizes to `n = 4{,}027` (quality) and
    `n = 3{,}014` (runtime).
- **What was NOT regenerated (deferred — judgment).**
  - `fig_sensitivity_bsm_validation.{svg,pdf}` and the BSM_RF_PRED /
    BSM_RF_P10 / BSM_RF_P90 / BSM_ANALOG_NRMSE constants remain at
    pre-cleanup values. Their reproduction requires the original
    BSM-analog selection criterion, which is not captured in any
    in-repo script. Treat as TODO before final manuscript submission.
- **Files touched.**
  - `scripts/plot_sensitivity_rf_figures.py` (importance arrays).
  - `scripts/fit_sensitivity_rf.py` (new).
  - `docs/manuscripts/fig_sensitivity_rf_importance.svg` (regenerated).
- **Re-run recipe.**
  ```bash
  pixi run python scripts/fit_sensitivity_rf.py \
      --results artifacts/sensitivity/wave12_combined_clean.csv
  pixi run python scripts/plot_sensitivity_rf_figures.py \
      --results artifacts/sensitivity/wave12_combined_clean.csv \
      --output-dir artifacts/sensitivity/figures_wave12_clean
  cp artifacts/sensitivity/figures_wave12_clean/fig_sensitivity_rf_importance.svg \
     docs/manuscripts/
  ```

______________________________________________________________________

## 2026-06-07 — Sensitivity meta-regression refit + Table 4 / Figure 7 regeneration

### 🟡 R1. Dead-knob refit (wave1 + wave2 combined)

- **What changed.** Per impact-log entries B1 (dead `lasso_alpha_percentile`)
  and H7 (disabled-pruning `delta_threshold_override=None`), the
  meta-regression was refit on the **cleaned** wave1+wave2 union:
  - Combined: 5089 rows (wave1=2583, wave2=2506).
  - Dropped 1008 disabled-pruning rows where
    `stages.final_artifacts.delta_threshold_override` was null.
  - Dropped 54 rows with missing `final_ols_nrmse` target.
  - **Dropped the dead `stages.sparse_selection.lasso_alpha_percentile`
    column entirely**, replacing it with a constant
    `lasso_alpha_grid_size = 40` (the manuscript baseline) so the
    polynomial fit collapses the dead dimension cleanly.
  - Final analysis frame: **4027 rows × 11 predictors**, d=2 polynomial,
    136 polynomial terms, in-sample R²=0.840, CV R²=0.831,
    group-blocked CV R²=0.776.
  - Cleaned CSV: `artifacts/sensitivity/wave12_combined_clean.csv`.
  - Refit formula: `artifacts/sensitivity/wave12_formula_d2_clean.csv`.
- **Dead-knob significance check.** The dead-knob main effect in the
  pre-cleanup fit was nominally significant (t = −8.4, p ≈ 0), but
  this is a statistical artifact of LHS non-orthogonality
  (lasso_alpha_percentile is correlated r ≈ 0.28 with `n_stab` and
  `holdout_fraction`, r ≈ 0.19 with `delta_threshold`). The total R²
  contribution from all 12 dead-knob terms was +0.27 %. **Decision
  (recorded by user 2026-06-07):** drop and refit; no wave-1/2
  rerun is required. A future wave that varies `lasso_alpha_grid_size`
  honestly is optional and would fill the Table 4 [TBD] cell.
- **Manuscript impact (Table 4 / Figure 7 / §6.1 text).**
  - **Figures regenerated** (cleaned wave1+wave2 basis):
    - `docs/manuscripts/fig_sensitivity_main_effects.svg`
    - `docs/manuscripts/fig_sensitivity_runtime_breakdown.svg`
    - `docs/manuscripts/fig_sensitivity_sample_size_curve.svg`
    - `docs/manuscripts/fig_sensitivity_rf_validation.svg`
  - Plot legend `LASSO α percentile` → `LASSO α grid size`
    (`scripts/plot_sensitivity_results.py:593`).
  - **Editor note (required action).** Section 6 / §6.1 / Table 4 /
    Table 5 / Figure 7 caption and inline narrative all referenced the
    dead `lasso_alpha_percentile` predictor and quoted per-predictor
    coefficients/importances from the **pre-refit** fit. The numbers
    below have changed (refit on 4027 rows, dead knob removed,
    pruning-disabled rows removed). The manuscript editor MUST:
    1. Re-pull the predictor list in Table 4 to exclude
       `lasso_alpha_percentile`; insert `lasso_alpha_grid_size` row
       marked `[TBD — pending future sensitivity wave]`.
    1. Replace cited Table 5 / §6.1 coefficients and t-stats with the
       refit values from
       `artifacts/sensitivity/wave12_formula_d2_clean.csv`.
    1. Replace Figure 7 cited correlations with the new
       `fig_sensitivity_main_effects.svg` values:
       - Holdout fraction: 0.2553
       - Variance threshold: 0.0948
       - Screening permutations: 0.4035
       - BH threshold (q): 0.5189
       - Interaction permutations: 0.3487
       - Interaction p-threshold: 0.0623
       - Stability subsamples: 0.2022
       - LASSO α grid size: 0.0000 (constant; awaiting future wave)
       - δ threshold override: 0.3836
    1. Disclose in the §6 discussion the LHS non-orthogonality finding
       (dead knob spuriously reached p ≈ 0 via r ≈ 0.28 confound with
       `n_stability_subsamples` and `holdout_fraction`) as a caveat
       on Table 4 / Table 5 t-stat interpretation.
- **Re-run required:** none. Cleaned-CSV refit closes this work.

______________________________________________________________________

### 🟢 No manuscript-value impact

Round-4 fixes were all developer-experience / drift-prevention. No
shipped manuscript number changes; no re-run required.

- **rfm-pipeline (caeed3a).** `transform_families` now **raises**
  instead of silently dropping (root-cause fix for round-3 finding).
  `configs/manuscript_case_study_fast_sparse.yml` renamed to
  `configs/manuscript_case_study.yml` so the canonical name every
  docstring already referenced actually exists on disk. Docs build
  (`pixi run docs`) restored. `paper.md` now discloses the synthetic-DGP
  sensitivity baseline holds `n_tree_estimators` / `max_tree_depth`
  below manuscript values for LHS-budget reasons (BSM case study
  retains full manuscript baselines — no result impact).
- **bsm-public-rf (pending commit).** 8 HPC configs were carrying the
  dead `lasso_alpha_percentile` key (would crash on
  `load_config` against pinned rfm-pipeline a6c18e6) and 8 also
  carried the dead `transform_families` key (would crash against
  caeed3a). All cleaned. `pixi.lock` refreshed; pin bumped to
  caeed3a. New test `tests/test_configs_loadable.py` (parametrized
  over 16 standalone HPC YAMLs) prevents this drift recurring.
- **Prediction-equation docs corrected.**
  `artifacts/final_model/README.md` and `README_intercept.md` had the
  raw-scale prediction rule wrong (double-divided by `scale_x`, and
  falsely claimed `scale_y == 1` for all outputs). Corrected. No
  shipped CSV changed — the formula in the docs disagreed with the
  numbers; numbers were right. **Reviewers who computed predictions
  from the broken formula would have gotten values inflated by
  `scale_x` per term; flag in any released errata.**

**Re-run action:** none.
**Manuscript text edits:** none.

______________________________________________________________________

### 🔴 B1. `lasso_alpha_percentile` was a dead config knob

- **What changed.** The `SparseStageConfig.lasso_alpha_percentile`
  field was declared and swept by the sensitivity study over
  `(10, 20, 40, 60, 80)`, but the EBIC LASSO selector
  (`_select_component_lasso_by_ebic`) hardcoded the grid as
  `np.geomspace(..., num=40)` and never read the config field.
  Renamed to `lasso_alpha_grid_size` and threaded into the selector
  (rfm-pipeline `a6c18e6`).
- **Manuscript impact.**
  - Section 6 / Table 4 / Table 5 / Figure 7: any RF importance or
    polynomial-meta-regression coefficient attributed to
    "LASSO α percentile" in the **existing** sensitivity wave1/2/3
    results is variance on a constant feature. It must be **dropped**
    from the meta-regression design matrix when fitting against the
    legacy CSVs.
  - Table 4 currently lists this row as
    `LASSO EBIC grid size  40  [TBD]`. After the rename, the swept
    values are `(20, 40, 80, 160)` and the manuscript [TBD] cell can
    be filled in once the **next** sensitivity wave runs with the
    fixed code.
- **Re-run required.**
  - **Meta-regression refit.** Re-run `scripts/fit_meta_regression.py`
    and `scripts/plot_sensitivity_results.py` against the existing
    `wave{1,2,3}_results.csv` **after dropping the
    `stages.sparse_selection.lasso_alpha_percentile` column** (now
    `lasso_alpha_grid_size`, but for legacy data the column is a
    constant noise dimension). Re-export `artifacts/sensitivity/ wave1_rf_quality.pkl` and `wave1_rf_runtime.pkl`.
  - **Future sensitivity wave (optional).** To populate the Table 4
    `[TBD]` cell, run a fresh sensitivity wave that actually varies
    the grid size. This is optional — the manuscript currently lists
    `[TBD]` and can remain so until the user is ready.
- **Manuscript text edits required.**
  - Table 5 / Figure 7 caption: re-state the predictor list excluding
    `lasso_alpha_percentile` (or replace with the corrected
    `lasso_alpha_grid_size` once new data exist).
  - Section 6.1 text quoting per-predictor importances must be
    re-derived from the refit.

### 🟢 H7. Sensitivity sweep `delta_threshold_override` `None` removed

- **What changed.** The sensitivity LHS sweep over
  `delta_threshold_override` was `(0.001, 0.002, 0.005, 0.010, None)`;
  the manuscript Table 4 row lists exactly the first four values.
  `None` disabled pruning entirely (a configuration the manuscript
  never reports) and was hit on ≈20% of the sensitivity runs.
  Removed `None` (rfm-pipeline `a6c18e6`).
- **Manuscript impact.**
  - Existing wave1/2/3 CSVs contain ≈20% rows where
    `delta_threshold_override` is null (disable-pruning runs). The
    meta-regression should either (a) drop those rows or
    (b) keep them and document them in Table 4 as a fifth
    "pruning disabled" sweep value.
- **Re-run required.**
  - **Meta-regression refit:** decide between (a) and (b) and re-fit
    accordingly. Recommended (a): drop disable-pruning rows for
    consistency with manuscript Table 4.

### 🟡 H1. Transform library reduced to 4 manuscript families

- **What changed.** `EXPONENTIAL` removed from public `__all__` and
  from `KNOWN_TRANSFORMATIONS`; manuscript §3.5 lists 4 families
  only (quadratic, logarithmic, inverse, square-root). The
  `DEFAULT_TRANSFORM_LIBRARY` was already 4 families since round 2.
- **Manuscript impact.** None — manuscript already specifies 4
  families. Round-2 already corrected the published default. This
  change just cleans up the public API to match.
- **Re-run required.** None.

### 🟡 B3. Jaccard / Spearman thresholds now in typed config

- **What changed.** `jaccard_threshold = 0.75` and
  `spearman_threshold = 0.90` are now first-class fields on
  `SparseStageConfig` (rfm-pipeline `a6c18e6`).
- **Manuscript impact.** None — values match manuscript Table 1
  baselines. Previously only reachable via undocumented
  `case_study` dict path; now reachable via documented YAML schema.
- **Re-run required.** None.

### 🟢 H4. `manuscript_alignment_audit.md` BH q typo fixed

- **What changed.** Audit doc said BH `q = 0.10`; code default and
  manuscript Table 1 specify `q = 0.05`. Doc fixed.
- **Manuscript impact.** None — actual pipeline already used 0.05.
- **Re-run required.** None.

______________________________________________________________________

## 2026-06-05 — Round-2 adversarial audit (kept for traceability)

### 🟡 Round-2 transform library reduction

- **What changed.** `DEFAULT_TRANSFORM_LIBRARY` reduced from 5 to 4
  families (dropped `exp(x)`); this was the round-2 fix that aligned
  the implementation with manuscript §3.5.
- **Manuscript impact.** None — manuscript already specified 4
  families. This change made the implementation match.
- **Re-run required.** None for the published case-study run (which
  used 4 families regardless). Any **future** full-dataset re-run
  will use the same 4 families, so artifact counts continue to match
  the manuscript (172 enriched / 132 final / 41 transforms / 29 in
  final support).

### 🟡 Round-2 `final_ols_summary.csv` reference columns

- **What changed.** The `manuscript_final_predictor_count_reference`
  and `manuscript_final_ols_holdout_nrmse_reference` columns now
  carry the v22 values (132, 0.0721).
- **Manuscript impact.** None — same values as manuscript Table 2.
  This change protects against regression: tripwire assertions in the
  pipeline now compare against the same constants the manuscript
  cites.
- **Re-run required.** None.

______________________________________________________________________

## Open / pending manuscript actions

The following items affect the manuscript but are **waiting on the
user** (not Copilot-completable):

- **paper.md (rfm-pipeline JOSS submission):** real ORCID, real
  submission date, real DOE BETO award identifier, NREL vs NLR
  affiliation consistency between `paper.md`, `pyproject.toml`,
  `CITATION.cff`, and the manuscript front matter.
- **Manuscript Table 4 `[TBD]` cells:**
  - `Interaction null-quantile threshold` — baseline 0.995, swept
    [TBD]. Code currently only uses the baseline; no sweep planned.
  - `LASSO EBIC grid size` — baseline 40, swept [TBD]. Now properly
    threaded; a future sensitivity wave with `(20, 40, 80, 160)` can
    fill the cell.
- **Sensitivity meta-regression refit:** the existing wave1/2/3 CSVs
  have one no-op column (`lasso_alpha_percentile`) and ≈20% disabled-
  pruning rows. A refit is required before any updated Section 6
  tables, figures, or numeric claims can be re-cited in the
  manuscript. See B1 / H7 above.

______________________________________________________________________

## How to use this log

- Add a new entry under a dated heading every time a change lands
  that meets one of the four criteria at the top.
- Use the 🔴 / 🟡 / 🟢 markers to flag severity:
  - 🔴 invalidates a published result; refit / re-export required
  - 🟡 changes the implementation to match the manuscript or adds
    new infrastructure; no re-fit required
  - 🟢 doc / typo / hygiene fix with no scientific impact
- When a referenced refit completes, edit the entry to add a final
  "**Resolved:** <date> — <commit>" line rather than deleting the
  entry.
