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

## 2026-06-07 — Round 14 audit: enumerate §6 prose drift missed by R6

### 🟠 R7. RF importance, sample-size, and runtime-stratification prose drift

- **What changed (code).** None — round-14 audit surfaced numeric claims
  in `jds_bsm_v22.tex` §6 / §7 prose and Figure 7 + Figure 9 captions
  that R6 did not enumerate. Quality / runtime / null-screen percentages
  in prose still reflect pre-cleanup wave1; refits at rounds 8/9/12
  regenerated the figures but never propagated to the prose tier.
  Mechanical hardening: `scripts/plot_sensitivity_results.py` now derives
  the RF pkl prefix from the input CSV stem (mirrors round-13 fix in
  `fit_sensitivity_rf.py`).
- **Manuscript impact (editor must fix in v22.tex before submission).**
  - **§6 line 645 (sample-size sentence).**
    - "2,583 runs (2,486 pure-synthetic / 97 BSM-structure)" → **4,027
      runs (3,914 pure-synthetic / 113 BSM-structure)**
    - "534 (20.7%) null-screened" → **1,013 (25.2%) null-screened**
    - "21.3% pure-synthetic / 5.2% BSM-structure null-screen rate" →
      **25.8% pure-synthetic / 4.4% BSM-structure**
  - **§6 line 647 (CV RMSE).** "CV RMSE = 0.071" → **CV RMSE = 0.066**
    (computed value 0.0657, γ scale).
  - **§6 line 666, 671 (n importance).** "RF importance for n is 2.9%"
    → **1.6%** (wave12 quality model).
  - **§6 line 678 (quality top-5).**
    - q: **33% → 53.2%** (still dominant)
    - empirical_null permutations: 10% → **6.8%**
    - sparsity: 8% → **10.8%**
    - stability subsamples: 7% → **0.6% — drops out of top 5
      entirely** (now ranks 11th)
  - **§6 line 680 (runtime top drivers).**
    - d (n_inputs): **31% → 35.1%**
    - sparsity: **27% → 38.6%** (now dominant)
  - **§6 line 685 (Figure 9 `rf_importance` caption).**
    - "n = 2,583 runs (quality)" → **n = 4,027 runs**
    - "n = 2,049 successful runs (runtime)" → **n = 3,014**
  - **§6 line 694 (runtime BSM vs pure-synthetic).** Old: "BSM-structure
    runs averaged 2,558 s versus 2,201 s for pure-synthetic". Wave12
    refit, **matched on the BSM d range [105, 195]**:
    - BSM-structure (n=108): **mean 3,300 s, median 1,740 s**
    - pure-synthetic (n=1,447): **mean 3,825 s, median 2,712 s**
    - **The direction reverses on wave12 matched-d.** Pure-synthetic is
      *more* expensive than BSM-structure at comparable d. Recommend
      replacing the sentence with the matched-d comparison and dropping
      the "larger output arrays" causal claim, which is no longer
      supported by the data.
  - **§7 line 710 (practitioner guidance).** "stability subsamples is
    the fourth-ranked quality driver" — **no longer true** (ranks 11th
    at 0.6% importance). Either drop the sentence or rewrite around
    q / null-permutation-budget / sparsity as the top three.
  - **§7 / §8 line 740, 751.** Any inline restatement of the §6
    importance percentages inherits the same drift; sweep the file for
    "33%", "27%", "31%", "7%" attached to the quality / runtime RF.
- **Reproducibility.**
  ```bash
  pixi run python scripts/fit_sensitivity_rf.py \
      --results artifacts/sensitivity/wave12_combined_clean.csv \
      --dump-models artifacts/sensitivity/
  pixi run python scripts/plot_sensitivity_results.py \
      --results artifacts/sensitivity/wave12_combined_clean.csv \
      --output-dir artifacts/sensitivity/figures/
  # Importance percentages: read rf_importance.svg + stdout
  # null-screen rates: wave12_combined_clean.csv null_screened groupby dgp_family
  # BSM vs pure-synthetic runtime: matched-d filter on n_inputs ∈ [105, 195]
  ```
- **Files touched.**
  - `scripts/plot_sensitivity_results.py` (input-derived pkl prefix; 4
    LoC, mirrors round-13 `fit_sensitivity_rf.py` hardening).

______________________________________________________________________

## 2026-06-07 — Round 13 audit: stale R² and runtime stats in manuscript §6

### 🟡 R6. RF goodness-of-fit + runtime distribution drift (manuscript edits)

- **What changed (code).** None — this is a manuscript-text drift entry
  surfacing values that round 12's refits left out of the editor note.
  The values cited in `jds_bsm_v22.tex:647,659,694,751` are from the
  pre-cleanup wave1 fit; round-7 (linear meta-regression) and round-9
  (RF) refits never propagated these specific statistics.
  Also: `fit_sensitivity_rf.py` `--dump-models` filename prefix is now
  derived from the input CSV stem (avoids overwriting wave12 pkl when
  fitting a future wave34 CSV).
- **Manuscript impact (editor must fix in v22.tex before submission).**
  - **R² investigation closed (2026-06-07):** manuscript 0.908 is not
    reproducible. Wave1 at current methodology gives R²=0.860 (matches Jun-2
    commit 7b159eb log "0.858"); wave12 gives 0.778. 0.908 has no provenance
    in committed code — likely transient un-pinned variant during drafting,
    or unblocked KFold mislabeled as group-blocked. **Accept 0.778; no code
    repair needed.** Wave2 expansion (n_succ 2049→3014) genuinely added
    harder DGP regions; cite as strengthening generalization claim.
  - **Editor replacement values (group-blocked CV on wave12, n=3,014 succ):**
    R²=0.778, RMSE=0.0657 γ, MAE=0.0279 γ (γ scale: mean −0.472, std 0.139,
    range [−0.859, −0.186]).
  - **§6 line 647 / 751: RF quality $R^2 = 0.908$ → $R^2 = 0.778$.** The
    pre-cleanup wave1 R² of 0.908 is not reproducible on wave12 with
    cleaned data; the current 10-fold group-blocked CV (groups =
    config_idx × dgp_idx) gives R² = 0.778 on 3,014 successful runs.
    Also update the parenthetical "CV RMSE = 0.071" if recomputed.
  - **§6 line 659 (Figure rf_validation caption): "2,049 successful
    fits (CV $R^2 = 0.908$)" → "3,014 successful fits (CV $R^2 =
    0.778$)".**
  - **§6 line 694: RF runtime $R^2 = 0.965$ → $R^2 = 0.950$.** Wave12
    group-blocked CV on log-wall-seconds for the 3,014 successful runs.
    Also update inline runtime distribution stats:
    - "median 18.6 min" → **22.0 min**
    - "90th percentile 141 min" → **137.8 min**
    - "maximum 234 min" → **479.4 min** (substantially larger; reflects
      a few large-d / large-n stability-subsample-heavy runs in wave2)
    - "BSM-structure averaged 2,558s vs 2,201s for pure-synthetic" —
      stale; refit on wave12 strata if reporting this comparison.
  - **§6 line 751: "predicts BSM nRMSE within 5.7%" — STALE.** Coupled
    to the deferred BSM_RF_PRED block (analog-selection methodology
    still missing from any committed script). Leave as TODO until the
    BSM validation panel is refit end-to-end.
  - **§6 line 647: median, q10/q90 already covered in R4** ("median
    −0.448, q10/q90 (−0.655, −0.304)"; was −0.462 / (−0.695, −0.301)).
    Sample size in §6 line 647 ("2,049 successful fits") → 3,014.
- **Reproducibility.**
  ```bash
  pixi run python scripts/fit_sensitivity_rf.py \
      --results artifacts/sensitivity/wave12_combined_clean.csv \
      --dump-models artifacts/sensitivity/
  # quality R²:   read from plot_sensitivity_results.py output
  # runtime R²:   reproduce via GroupKFold(n_splits=10) on log(total_wall_seconds)
  ```
- **Files touched.**
  - `scripts/fit_sensitivity_rf.py` (input-derived pkl filename prefix).
  - `scripts/plot_sensitivity_results.py` (stale `wave1_rf_quality.pkl`
    / `2049 successful rows` docstring fixed).

______________________________________________________________________

## 2026-06-07 — Round 12 audit: PDP population fix + wave12 RF pickles

### 🟡 R5. `fig_sensitivity_sample_size_curve` PDP base population corrected

- **What changed (code).**
  - `_partial_dependence_n_runs` in `scripts/plot_sensitivity_results.py`
    now restricts the PDP base matrix to **successful runs only**
    (`df["null_screened"].isna()`). Previously it marginalised over the
    full frame including the 1,013 null-screened rows (γ = 0 by
    construction), which damped the PDP curve toward 0 by the
    null-screen fraction (~25% on wave12) and contradicted the
    `_render_sample_size_curve` docstring claim that null-screened runs
    are excluded.
  - The RF model loaded for PDP was `wave1_rf_quality.pkl`, a stale
    pre-cleanup model trained on wave1 (n = 2,583) before the dead
    `lasso_alpha_percentile` knob was removed. Updated path to prefer
    `wave12_rf_quality.pkl` and fall back to the wave1 pkl for legacy
    bundles.
  - `scripts/fit_sensitivity_rf.py` now optionally dumps the fitted
    quality + runtime models via `--dump-models DIR` so the wave12
    PDP-source pickle is reproducible from a single command.
- **Manuscript impact.**
  - Figure 7 (`fig_sensitivity_sample_size_curve`, cited at
    `jds_bsm_v22.tex:658`) is regenerated. The PDP curve no longer
    contains the null-screen pull; the headline claim "the RF importance
    for n is 2.9%, the lowest among DGP properties" still holds
    qualitatively (n_runs sits at ~2-3% in both quality importance
    tables), but the visual PDP magnitude will be slightly larger.
  - Figures `fig_sensitivity_main_effects`, `fig_sensitivity_runtime_breakdown`,
    `fig_sensitivity_rf_validation` regenerated incidentally from the
    same script run. None of `main_effects` or `runtime_breakdown` is
    cited in v22.tex — they ride along as supplementary artifacts.
    `fig_sensitivity_rf_validation` is cited and was already wave12-clean
    from round 7; re-regen is byte-for-byte stable apart from the new
    PDP-driven sample_size_curve.
- **Files touched.**
  - `scripts/fit_sensitivity_rf.py` (added `--dump-models` + `joblib` import).
  - `scripts/plot_sensitivity_results.py` (prefer wave12 pkl;
    PDP base restricted to successful runs).
  - `artifacts/sensitivity/wave12_rf_{quality,runtime}.pkl` (new,
    gitignored — local artifact only).
  - `docs/manuscripts/fig_sensitivity_{sample_size_curve,main_effects,runtime_breakdown,rf_validation}.svg`
    (regenerated).
- **Re-run recipe.**
  ```bash
  pixi run python scripts/fit_sensitivity_rf.py \
      --results artifacts/sensitivity/wave12_combined_clean.csv \
      --dump-models artifacts/sensitivity/
  pixi run python scripts/plot_sensitivity_results.py \
      --results artifacts/sensitivity/wave12_combined_clean.csv \
      --output-dir artifacts/sensitivity/figures_wave12_clean
  cp artifacts/sensitivity/figures_wave12_clean/fig_sensitivity_*.svg \
     docs/manuscripts/
  ```

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
    - **Total sensitivity runs in histogram: n = 4,027** (was 2,583;
      this is the figure-title `(n = ...)` value; null-screened runs
      contribute a γ = 0 spike at the right edge.)
    - Successful sensitivity runs (medians + RF fit): **n = 3,014**
      (was 2,049)
    - Null-screened runs (right-edge spike): **n = 1,013** (was 534)
    - Median γ across successful fits: **−0.448** (was −0.462)
    - q10 / q90 of γ across successful fits: **(−0.655, −0.304)**
      (was approx. (−0.695, −0.301))
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
