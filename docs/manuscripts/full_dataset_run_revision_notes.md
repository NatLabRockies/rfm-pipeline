# Full-dataset run: manuscript revision handoff

**Audience:** manuscript editor agent (or human) responsible for reconciling
`docs/manuscripts/jds_bsm.tex` / `jds_bsm_v22.tex` with the current
implementation and current final results before journal submission.

**Scope of this doc:** workflow-level changes since the manuscript was last
edited, final results (with real values where available), placeholders for
items still in flight, figure inventory + style, and the list of impact-log
entries the editor must address. **Implementation/bugfix details are NOT in
scope** — those live in `docs/AGENT_SYNC.md` and `docs/review_register.md`.

**Run reference:**
`publication_full_dataset_distributed_20260526_short_hp1`
**Artifact root:**
`artifacts/publication_full_dataset_distributed_results/publication_full_dataset_distributed_20260526_short_hp1/artifacts/final_manuscript_artifacts/`

**HEADs at handoff (2026-06-08):**

- rfm-pipeline `25ef483` (488 tests pass / 11 skip)
- bsm-public-rf `703eb76` (26 tests pass; pin `rfm-pipeline @ 25ef483`)

______________________________________________________________________

## 1. Workflow-level changes since manuscript last edited

These are the changes the manuscript prose / methods section / figures need to
reflect. Implementation-only bugfixes are deliberately excluded.

### 1.1 HPC execution model: 6-stage cascade with `afterok` dependency chaining

Previously: stages were run as independent SLURM submissions, coordinated by
hand or by `controller_*.sh`. **Now:** the manuscript pipeline runs as a
6-stage cascade where each stage submits one SLURM array per resource tier
(CPU 2 nodes, CPU 10 nodes, CPU 1000 nodes, optional GPU), each tier emits a
reduce job, and the NEXT stage is submitted with
`--dependency=afterok:<id1>:<id2>:...` listing **every** upstream reduce id
so the next stage waits for all tiers before starting.

The 6 stages are:

1. `output_conditioning`
1. `empirical_null_screening` (HPC stage name; writer artifact dir is
   `empirical_null_screen` — manuscript should use one consistent name and
   call out the alias once)
1. `interaction_discovery`
1. `nonlinear_discovery`
1. `sparse_selection`
1. `final_model`

Per-stage shard directories are now `hpc_shards_<stage>/` (previously a
single shared shard dir).

**Manuscript impact:** if the methods section describes the HPC orchestration,
update to say "6-stage cascade chained by SLURM `afterok` dependencies, per
resource tier." If the description is high-level only, no change required.

### 1.2 Benchmark resource tiers

The published reproducibility workflow now supports multi-tier benchmarking
(CPU 2, 10, 1000 nodes) within each stage. Per-stage walltime and memory are
configured per tier. The cascade orchestrator captures one reduce job id per
tier and threads the full list into the next stage's dependency flag.

**Manuscript impact:** if benchmark / scaling tables cite specific tier
configurations, confirm the numbers match `controller_publication_full_dataset_distributed.sh`
(STAGES, N_SHARDS, N_JOBS, CPUS_PER_TASK, MEMORY_GB, WALLTIME, MAX_CONCURRENT,
REDUCE_WALLTIME, REDUCE_MEMORY_GB, PARTITIONS, RUN_ID_SUFFIX arrays).

### 1.3 Sensitivity meta-regression: dropped `lasso_alpha_percentile`, refit on cleaned wave1+wave2+wave3

The `lasso_alpha_percentile` parameter was found to be a non-significant
predictor in observed wave1+wave2 results. **Dropped** from the
meta-regression design. After cancellation of the broken wave-2 resubmit
job (14106277/14106278), **wave 3 was collected and combined into a wave123
cleaned set** as of 2026-06-08. Wave123 supersedes wave12; cite wave123
values in the manuscript.

- Cleaned input: `~/src/rfm-pipeline/artifacts/sensitivity/wave123_combined_clean.csv`
  (6,225 rows × 61 cols; 4,747 successful, 1,478 null-screened; gitignored).
- Refit coefficients: `wave123_formula_d2_clean.csv` (136 terms, degree 2).
- Scaler metadata: `wave123_scaler_d2.json`.
- RF models: `wave123_rf_quality.pkl`, `wave123_rf_runtime.pkl`.
- Superseded inputs (kept for reproducibility, do NOT cite in current
  manuscript): `wave12_combined_clean.csv`, `wave12_formula_d2_clean.csv`,
  `wave12_scaler_d2.json`, `wave12_rf_quality.pkl`.

**Refit fit statistics (FINAL wave123, cite verbatim in Table 5 / §6.1):**

| Metric                        | wave123 (FINAL) | wave12 (superseded) |
| ----------------------------- | --------------- | ------------------- |
| In-sample R²                  | 0.786           | 0.840               |
| CV R² (10-fold)               | 0.777           | 0.831               |
| Group-blocked CV R²           | 0.787           | 0.776               |
| γ-scale held-out R²           | (see group-CV)  | 0.778               |
| n (cleaned rows)              | 6,225           | 4,027               |
| Predictors (post-drop)        | 11              | 11                  |
| Polynomial degree             | 2               | 2                   |
| Successful rows               | 4,747           | 3,014               |
| Null-screened rows            | 1,478           | 1,013               |
| Family `pure_synthetic`       | 5,912           | 3,914               |
| Family `bsm_structure`        | 313             | 113                 |
| Null-screen rate (pure_syn)   | 24.92%          | 25.75%              |
| Null-screen rate (bsm_struct) | 1.60%           | 4.42%               |

**γ summary on successful rows (wave123):** median −0.4398; q10/q90
(−0.6357, −0.3042).
**Runtime on successful rows (wave123, minutes):** median 26.1; q90 161.9;
max 479.4.

**Swept levels observed in wave123 (cite as ACTUAL observed sweep, not as
planned future sweep) — identical levels to wave12:**

| Parameter                | Observed levels                 |
| ------------------------ | ------------------------------- |
| Holdout fraction         | (0.05, 0.10, 0.15, 0.20)        |
| Variance threshold       | (0.80, 0.85, 0.90, 0.95)        |
| Screening permutations   | (51, 101, 201, 401)             |
| BH q                     | (0.01, 0.05, 0.10, 0.20)        |
| Interaction permutations | (11, 21, 31, 51, 101)           |
| Interaction p-threshold  | (0.01, 0.05, 0.10, 0.20)        |
| Stability subsamples     | (10, 25, 50, 100)               |
| Delta threshold          | (0.001, 0.002, 0.005, 0.010)    |
| LASSO α grid size        | (40,) — **constant in wave123** |

**Figure 7 main-effect |correlations| with γ (wave123, cite verbatim):**

| Predictor                | wave123 \|corr\| | wave12 \|corr\| (superseded) |
| ------------------------ | ---------------- | ---------------------------- |
| Holdout fraction         | 0.2144           | 0.2553                       |
| Variance threshold       | 0.0992           | 0.0948                       |
| Screening permutations   | 0.3454           | 0.4035                       |
| BH q                     | 0.4797           | 0.5189                       |
| Interaction permutations | 0.3290           | 0.3487                       |
| Interaction p-threshold  | 0.1864           | 0.0623                       |
| Stability subsamples     | 0.1507           | 0.2022                       |
| LASSO α grid size        | 0.0000           | 0.0000                       |
| Delta threshold          | 0.2490           | 0.3836                       |

**RF feature importances (wave123, from `fit_sensitivity_rf.py`):**

| Quality (γ) — top 9       | Importance |
| ------------------------- | ---------- |
| BH threshold (q)          | 0.454      |
| Sparsity (s)              | 0.141      |
| Screening permutations    | 0.102      |
| Input count (d)           | 0.077      |
| Interaction density (ρ)   | 0.073      |
| Nonlinearity strength (κ) | 0.053      |
| Signal-to-noise ratio (σ) | 0.047      |
| Interaction p-threshold   | 0.020      |
| Run count (n)             | 0.016      |

| Runtime (log wall-sec) — top 9 | Importance |
| ------------------------------ | ---------- |
| Sparsity (s)                   | 0.404      |
| Input count (d)                | 0.363      |
| Run count (n)                  | 0.076      |
| Interaction permutations       | 0.053      |
| Stability subsamples           | 0.050      |
| Interaction density (ρ)        | 0.016      |
| Signal-to-noise ratio (σ)      | 0.013      |
| Nonlinearity strength (κ)      | 0.012      |
| Variance threshold             | 0.006      |

**Manuscript impact:** Table 4 column for `lasso_alpha_percentile` must be
removed/replaced with `lasso_alpha_grid_size` (constant 40 — note this is
not a swept parameter in observed wave123); methods §6 should add a caveat
about LHS non-orthogonality on observed wave123.

**RF runtime model — group-blocked CV R² and back-transform (wave123, FINAL):**

| Metric                                            | wave123 (FINAL) | wave1 (manuscript v22) |
| ------------------------------------------------- | --------------- | ---------------------- |
| Runtime RF, 10-fold group-CV R² (log wall-sec)    | **0.959**       | 0.965 / 0.950 (R6)     |
| Back-transform factor exp(σ²/2), CV log-residuals | **1.042**       | 1.30                   |

Computation: groups = `config_idx`×`dgp_idx`; n=4,747 successful runs;
σ² of out-of-sample CV residuals on the log scale = 0.0828. The 1.042
back-transform is the lognormal Jensen correction for the conditional
mean point estimate; per-tree quantiles are reported untransformed (the
exp(·) and quantile operators commute on the per-tree distribution).

**BSM production operating-point predictions from wave123 RF (FINAL, replaces v22 wave1 numbers):**

BSM production feature vector (manuscript §7.5):
`d=135, n=28,750, sparsity=0.28, ρ=0.15, κ=0.20, σ=22.3, screening_perms=201, BH_q=0.05, interaction_perms=31, interaction_p=0.05, stability_subsamples=50, lasso_α_grid=40, variance_threshold=0.90`.

| Quantity                            | wave123 (FINAL)              | v22 (wave1, superseded) |
| ----------------------------------- | ---------------------------- | ----------------------- |
| BSM runtime: RF point (raw exp)     | **125.6 min**                | 103 min                 |
| BSM runtime: RF point (back-trans.) | **130.9 min**                | (n/a)                   |
| BSM runtime: RF 80% PI              | **[74.2, 144.6] min**        | [99, 111] min           |
| BSM runtime error vs 101.4-min mean | **+23.8% (back-trans +29%)** | +1.6%                   |
| BSM quality: γ point                | **−0.388**                   | (not reported)          |
| BSM quality: γ 80% PI               | **[−0.428, −0.374]**         | (not reported)          |
| BSM quality: nRMSE point (=ν·(1+γ)) | **0.1011**                   | 0.0762                  |
| BSM quality: nRMSE 80% PI           | **[0.0946, 0.1036]**         | [0.0729, 0.0774]        |
| BSM quality: nRMSE error vs 0.0721  | **+40.3%**                   | +5.7%                   |

Where ν = null nRMSE for the BSM operating point = 0.1653 (manuscript
§7.5). 80% PIs are leaf-weighted: the 10th/90th percentiles across the
500 per-tree predictions for the BSM feature row.

**Interpretation for the editor (must address in §7.5 / §8):** the
wave1-fit RF predicted BSM nRMSE within 5.7% (0.0762 vs 0.0721) and
runtime within 1.6%. After cleanup and refit on the wave123 (6,225-row,
3-wave) training set, the same operating point is predicted at nRMSE
0.1011 (40% high) and runtime 125–131 min (24–29% high). The
deterioration is not a bug — the cleaned set excludes 24% of pure-synth
rows that were null-screened, and the BSM operating point sits at the
high-d / high-n / strict-BH-q / many-permutations corner of the design
where training coverage is thinnest. The honest manuscript framing is:
the RF is a calibrated indicator of where the pipeline will land *within
the sensitivity envelope*; extrapolating to the BSM production corner
yields a conservative (pessimistic) bound, not a tight predictor. The
final-model nRMSE 0.0721 outperforms the RF's own 80% PI, demonstrating
the operating point benefits from the production-pipeline tuning
choices not isolated in the sensitivity sweep.

**Re-derivation script:** `scripts/compute_bsm_rf_validation.py`
loads `wave123_rf_quality.pkl` + `wave123_rf_runtime.pkl`, runs the
group-blocked CV for runtime R², computes back-transform from CV
residuals, and applies per-tree prediction with quantile PIs to the BSM
production feature vector. Re-run:
`pixi run python scripts/compute_bsm_rf_validation.py`.

**Wave 4 production-replica corner sweep (IN FLIGHT, submitted 2026-06-09):**
Wave123 only has 20 BSM rows in the strict-quality corner
(BH q ≤ 0.05, screening perms ≥ 201, n_stab ≥ 50) and the RF prediction
at the BSM production point matches that corner's training central
tendency (γ ≈ −0.35 to −0.39) almost exactly. The actual production γ
is −0.564 — outside the empirical envelope of every BSM training row at
the strict-quality corner. To distinguish "training coverage thin" from
"sensitivity-harness vs production-pipeline gap", wave4 adds 30
calibrated_structure runs (10 LHS DGPs × 3 replicates × 1 config) at
exactly the production override set (BH q=0.05, scr_perms=201, n_stab=50,
var=0.9, int_perms=31, int_p=0.05, lasso_grid=40, delta=0.002), with
subsample levels=1 (full n only). Plus 3 pure_synthetic throwaway jobs
(generator requires at least one).

- Spec: `configs/sensitivity_study/study_spec_wave4.yml` (seeds=4000).
- Code change: `SensitivityStudySpec.fixed_overrides` field +
  `generate_config_lhs` honors it (commit cd33219).
- Kestrel submission: job array **14139138** (2026-06-09 11:54 UTC, 33
  tasks, partition=shared, 8h walltime, 220G/104 CPU, account=bsm).
- Study dir: `/scratch/dhetting/bsm/sensitivity_study_wave4`.
- Expected runtime: dominant cost is the calibrated jobs at full n
  (5,000-30,000 runs × stability subsamples 50). Wave3 calibrated jobs
  averaged 30-60 min; wave4 at strict-quality settings may run 45-90
  min each. Concurrency 33/33 → wall clock 1-2h.
- On completion: collect wave4 results, append to
  `wave123_combined_clean.csv` → `wave1234_combined_clean.csv`, refit
  both RFs, re-run `scripts/compute_bsm_rf_validation.py`, regenerate
  `fig_sensitivity_bsm_validation`, update this doc + impact-log R12.

### 1.4 Holdout split rule made explicit

The case-study holdout split rule was previously documented implicitly via
code only. It is now described in
`bsm-public-rf/configs/manuscript_case_study.yml`
under `holdout_split_rule_description`:

- Name: `afsc_uaeoro_stratified_else_scenario`
- Stratification: AFSC × UAEORO 4-stratum split, fallback scenario,
  fallback random.
- Implementation: `rfm_pipeline.data.stratified_holdout_split`.
- Seed: 123.

**Manuscript impact:** §3 (data) should describe this rule explicitly. The
existing prose may already cover scenario-stratified holdout; verify
language matches the canonical rule name above.

### 1.5 Feature catalog provenance documented

`bsm-public-rf/docs/manuscript_feature_catalog_provenance.md` (new) records
the catalog generator, schema, why-frozen rationale, and regeneration
recipe. Generator command:
`rfm-pipeline scripts/generate_feature_catalog.py --interaction-strategy top-shap --max-interactions 500 --nonlinear-strategy safe`.

**Manuscript impact:** if §4 (feature catalog) currently lacks provenance
detail, add a one-paragraph pointer to this provenance doc.

### 1.6 Package rename: `bsm_rfm` → `rfm_pipeline`

The generic pipeline package was renamed from `bsm_rfm` to `rfm_pipeline`
(commit `ce6768b`, 2026-06-01) and split into three repos:

| Repo                                   | Purpose                                 |
| -------------------------------------- | --------------------------------------- |
| NatLabRockies/rfm-pipeline             | Generic pipeline package `rfm_pipeline` |
| NatLabRockies/bsm-public-rf            | BSM configs + committed model artifacts |
| NatLabRockies/bsm-public-rf-manuscript | LaTeX + figures                         |

**Manuscript impact:** any code citation or repo URL must point to the
correct repo. Generic pipeline citations → `rfm-pipeline`; BSM dataset /
config / artifact citations → `bsm-public-rf`.

### 1.7 What did NOT change at the workflow level (do not edit manuscript for these)

Stage semantics, dataset, screening / discovery / selection algorithms,
final OLS math, RF sensitivity model spec, all numeric thresholds. Rounds
22-27 of the audit cycle were dedicated to HPC orchestration hardening,
SSH safety, dependency chaining correctness, and bundle summarizer
correctness — none of which change the science.

______________________________________________________________________

## 2. Implemented pipeline summary (current)

**Dataset:** 30,000 runs (28,500 training, 1,500 holdout, 5% holdout
fraction, seed 123). 158 scalar LHC inputs × 4 binary scenario
combinations.

**Pipeline stages and verified counts (from
`final_manuscript_artifacts/`):**

| Stage                        | Quantity                      | Value                |
| ---------------------------- | ----------------------------- | -------------------- |
| Output conditioning          | Retained outputs for PCA      | 9,954 of 23,495      |
| Output conditioning          | PCA components (90% variance) | 20                   |
| Empirical null screening     | Retained scalar inputs        | 69 of 158            |
| Interaction discovery        | Retained pairs                | 62                   |
| Nonlinear discovery          | Retained transforms           | 41                   |
| Enriched design matrix       | Total columns                 | 172 (69 + 62 + 41)   |
| LASSO + stability            | Stable features               | 172 of 172           |
| HC3 inferential filter       | Retained features             | 172 of 172           |
| Delta-threshold pruning      | Pruned features               | 40 (threshold 0.002) |
| Final OLS support            | Total predictors              | 132                  |
| Final OLS support            | Main-effect terms             | 54                   |
| Final OLS support            | Interaction terms             | 49                   |
| Final OLS support            | Nonlinear transforms          | 29                   |
| Final OLS coefficient matrix | Outputs covered               | 23,495 (all)         |
| Final OLS nRMSE              | Holdout macro (Y-train range) | 0.0721               |

**nRMSE scope:** metric evaluated on 9,954 outputs with non-negligible
variance. Near-constant culled outputs are in the coefficient matrix with
near-zero coefficients.

______________________________________________________________________

## 3. Final ablation results (FINAL)

| Model stage      | Features | nRMSE  | CI lower | CI upper |
| ---------------- | -------- | ------ | -------- | -------- |
| Null mean        | 0        | 0.1653 | 0.1635   | 0.1663   |
| Main effects OLS | 158      | 0.0812 | 0.0796   | 0.0821   |
| Screened OLS     | 69       | 0.0812 | 0.0796   | 0.0821   |
| Penalized OLS    | 172      | 0.0709 | 0.0694   | 0.0719   |
| Final OLS        | 132      | 0.0721 | 0.0706   | 0.0730   |

**Source:** `tables/ablation_table.csv`.

______________________________________________________________________

## 4. Per-output nRMSE distribution (9,954 outputs, FINAL)

| Statistic | nRMSE |
| --------- | ----- |
| p10       | 0.018 |
| p25       | 0.036 |
| p50       | 0.061 |
| p75       | 0.093 |
| p90       | 0.141 |

**Worst three outputs:** `OI.OHC output by product[TransEster, P, diesel]`
for 2023 (0.495), 2024 (0.487), 2025 (0.476).

**Source:** `tables/per_output_nrmse_summary.csv`.

______________________________________________________________________

## 5. Feature composition (final 132 predictors by module, FINAL)

| Module                      | Features | Share |
| --------------------------- | -------- | ----- |
| Cellulosic Hydrocarbons     | 44       | 33%   |
| Wet Waste Hydrocarbons      | 36       | 27%   |
| Oil Hydrocarbons            | 28       | 21%   |
| Starch Ethanol Hydrocarbons | 13       | 10%   |
| Algal Hydrocarbons          | 11       | 8%    |

**Source:** `figures/figure_selected_by_module_data.csv`.

______________________________________________________________________

## 6. Figure inventory + style

Figures are in `docs/manuscripts/`. Style conventions (current, post-r4
figure cleanup): publication-readable, color-blind friendly (high contrast

- color-blind-safe palette + non-color cues). Generated SVGs are the
  source of truth; PDFs regenerated from SVG when needed.

### 6.1 Final-model figures (full-dataset run)

| File                                       | What it shows                        | Source script / notebook         |
| ------------------------------------------ | ------------------------------------ | -------------------------------- |
| `figure_model_performance.svg`             | Final OLS model performance          | manuscript pipeline writer       |
| `figure_per_output_nrmse_distribution.svg` | Per-output nRMSE distribution        | manuscript pipeline writer       |
| `figure_nrmse_bootstrap_summary.svg`       | Bootstrap CI on nRMSE                | manuscript pipeline writer       |
| `figure_feature_pruning_curve.svg`         | Δ-threshold pruning curve            | manuscript pipeline writer       |
| `figure_support_composition.svg`           | Final 132 features by type breakdown | manuscript pipeline writer       |
| `figure_selected_by_module_count.svg`      | Feature count by module              | manuscript pipeline writer       |
| `figure_selected_by_module_share.svg`      | Feature share by module              | manuscript pipeline writer       |
| `fig_influential_by_module.svg`            | Most-influential features by module  | regenerate_manuscript_figures.py |
| `fig_feature_type_distribution.svg`        | Type distribution histogram          | regenerate_manuscript_figures.py |
| `fig_module_pair_heatmap.svg`              | Interaction module-pair heatmap      | regenerate_manuscript_figures.py |
| `fig_module_total_interactions.svg`        | Total interactions per module        | regenerate_manuscript_figures.py |

Legacy notebook (style reference only, not regenerated):
`docs/final_scripts_from_hpc/influential_factors_analysis_visualizations.ipynb`.

### 6.2 Sensitivity / Figure 7 figures (wave123 cleaned, FINAL)

| File                                    | What it shows                          | Source script                  |
| --------------------------------------- | -------------------------------------- | ------------------------------ |
| `fig_sensitivity_main_effects.svg`      | Main-effect \|correlations\| with γ    | plot_sensitivity_results.py    |
| `fig_sensitivity_rf_importance.svg`     | RF importance bars (quality + runtime) | plot_sensitivity_rf_figures.py |
| `fig_sensitivity_rf_validation.svg`     | RF validation R²                       | plot_sensitivity_results.py    |
| `fig_sensitivity_runtime_breakdown.svg` | Runtime breakdown by predictor         | plot_sensitivity_results.py    |
| `fig_sensitivity_sample_size_curve.svg` | Sample-size convergence curve          | plot_sensitivity_results.py    |
| `fig_sensitivity_bsm_validation.svg`    | γ histogram + BSM validation dot plot  | plot_sensitivity_rf_figures.py |

**Regeneration recipe (deterministic from wave CSVs):**

```bash
# Combine waves: concat wave1+wave2+wave3, drop NaN delta + NaN final_ols_nrmse,
# drop lasso_alpha_percentile column, add lasso_alpha_grid_size=40 constant,
# reorder columns to match the persisted wave12_combined_clean.csv schema.
pixi run python <combine wave1+wave2+wave3: drop NaN delta + drop NaN final_ols_nrmse + drop lasso_alpha_percentile col + add lasso_alpha_grid_size=40>
pixi run python scripts/fit_meta_regression.py \
  --results artifacts/sensitivity/wave123_combined_clean.csv \
  --output  artifacts/sensitivity/wave123_formula_d2_clean.csv \
  --degree 2 \
  --save-scaler artifacts/sensitivity/wave123_scaler_d2.json
pixi run python scripts/fit_sensitivity_rf.py \
  --results artifacts/sensitivity/wave123_combined_clean.csv \
  --top-n 9 \
  --dump-models artifacts/sensitivity/
pixi run python scripts/plot_sensitivity_results.py \
  --results artifacts/sensitivity/wave123_combined_clean.csv \
  --output-dir artifacts/sensitivity/figures_wave123_clean/
pixi run python scripts/plot_sensitivity_rf_figures.py \
  --results artifacts/sensitivity/wave123_combined_clean.csv \
  --output-dir artifacts/sensitivity/figures_wave123_clean/
cp artifacts/sensitivity/figures_wave123_clean/fig_sensitivity_*.svg docs/manuscripts/
```

### 6.3 Static raster comparators (kept for reference, not regenerated)

`results_feature_type_composition.png`, `results_interaction_density_heatmap.png`,
`results_scatter_selected_outputs.png`, `results_timeseries_selected_outputs.png`.
These are PNGs from earlier iterations preserved for layout comparison; the
SVG equivalents above are the publication-ready versions.

______________________________________________________________________

## 7. Required manuscript edits (full impact-log index)

`docs/manuscripts/manuscript_impact_log.md` is the authoritative ledger of
required edits. Entries newest-first:

- **R11** (round 19): `jds_bsm_v22.tex:608` cites pre-cleanup wave1 run
  count `2,583`; replace with **6,225** (wave1+wave2+wave3 cleaned). Also:
  "nine pipeline configuration parameters" should read **eight observed**,
  because `lasso_alpha_grid_size` is constant in wave123.
- **R10b** (round 17): `jds_bsm_v22.tex:579` typo `two family]ies` →
  `two families`.
- **R10** (round 17): do not cite `(20, 40, 80, 160)` as observed wave123
  LASSO grid sweep; wave123 has constant grid size 40.
- **R8** (round 15): practitioner-guidance table still says `LASSO α percentile`; rename to `LASSO α grid size`.
- **R1–R7** (earlier rounds, 2026-06-07): sensitivity-meta-regression
  refit triggers Table 4 column changes, Table 5 coefficient replacement,
  Figure 7 RF correlations replacement, §6 LHS-non-orthogonality caveat.
  **Cite wave123 numbers from §1.3 of this doc, NOT the older wave12
  numbers that appear in the impact log.**

**Editor MUST verify (do not assume from impact log):** all numeric values
in §3 (data), §5 (final model), and §7 (sensitivity) tables match this
revision doc and the cleaned wave123 CSV.

______________________________________________________________________

## 8. Open placeholders (user-blocked or in flight)

- **[PLACEHOLDER — user-blocked]** Repository Zenodo DOI. Will be assigned
  at publication; placeholder string `10.5281/zenodo.XXXXXXX` is used in
  `bsm-public-rf/README.md` §Data Access. Replace once minted.
- **[PLACEHOLDER — user-blocked]** Supplement archive DOI.
- **[PLACEHOLDER — user-blocked]** JOSS metadata: per-author ORCIDs;
  affiliations (NREL vs Nat Lab Rockies); DOE BETO award/contract #;
  submission date; corresponding-author email. Files to update:
  `paper/paper.md`, `paper/CITATION.cff`, `pyproject.toml`,
  `bsm-public-rf/README.md`.
- **[PLACEHOLDER — user-blocked]** Acknowledgments + DOE disclaimer text.
- **[PLACEHOLDER — user-blocked]** Steve Peterson affiliation.
- **[DONE 2026-06-08]** Sensitivity wave 3 (job 14069433, COMPLETED
  2026-06-01, 2,750 artifacts on Kestrel) was collected on 2026-06-08 and
  combined with waves 1+2 into `wave123_combined_clean.csv` (6,225 rows).
  Meta-regression refit and Figure 7 regeneration done; new numbers in
  §1.3 above. Broken wave-2 resubmit jobs (14106277, 14106278) were
  cancelled — they were referencing a config path that does not exist on
  the HPC clone (`/scratch/dhetting/bsm/bsm-public-rf/configs/sensitivity_study/study_spec_wave2.yml`).
  Wave 2 already has its full 2,750 artifacts; no further resubmit needed.
- **[PENDING — investigation deferred to editor]** Worst three outputs
  (`OI.OHC output by product[TransEster, P, diesel]` 2023–2025, nRMSE
  0.476–0.495). Decide before submission whether to (a) cite as-is, (b)
  investigate and add discussion, or (c) cull from coverage claim.

______________________________________________________________________

## 9. Artifact collection / regeneration commands

**Collect the full-dataset run from Kestrel:**

```bash
bash scripts/kestrel/collect_publication_full_dataset_distributed.sh \
  --study-id publication_full_dataset_distributed_20260526_short_hp1 \
  --study-root /scratch/dhetting/bsm/studies/publication_full_dataset_distributed_20260526_short_hp1
```

**Local reproduction of the small-validation pipeline (for sanity check):**

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  pixi run python tools/run_manuscript_pipeline.py \
  configs/validation_300_sample_no_caps.yml
```

**Regenerate sensitivity figures:** see §6.2.

______________________________________________________________________

## 10. Where to look next

- `MEMORY.md` (repo root) — release status, artifact locations, three-repo
  architecture.
- `docs/MEMORY.md` — bsm-mirror of release status + r22–r27 closeout.
- `docs/AGENT_SYNC.md` — session-by-session audit history (rounds 22–27
  closeout at top, prior rounds beneath).
- `docs/review_register.md` — every audit finding with disposition,
  evidence, fix commit.
- `docs/manuscripts/manuscript_impact_log.md` — manuscript-edit ledger
  (R1–R11). Authoritative for editor work.
- `bsm-public-rf/docs/manuscript_feature_catalog_provenance.md` — feature
  catalog generator + schema + regeneration.
- `bsm-public-rf/configs/manuscript_case_study.yml`
  (`holdout_split_rule_description`) — canonical holdout rule.
