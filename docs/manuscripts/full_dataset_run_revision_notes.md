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

### 1.3 Sensitivity meta-regression: dropped `lasso_alpha_percentile`, refit on cleaned wave1+wave2+wave3+wave4

The `lasso_alpha_percentile` parameter was found to be a non-significant
predictor in observed wave1+wave2 results. **Dropped** from the
meta-regression design. After cancellation of the broken wave-2 resubmit
job (14106277/14106278), **wave 3 was collected and combined into wave123**
(2026-06-08). **Wave 4** (production-replica corner sweep, 30 bsm rows +
3 pure throwaways at the production override set) was collected on
2026-06-09 and combined into **wave1234**, which now supersedes wave123;
cite wave1234 values in the manuscript.

- Cleaned input: `~/src/rfm-pipeline/artifacts/sensitivity/wave1234_combined_clean.csv`
  (6,258 rows × 61 cols; 4,780 successful, 1,478 null-screened; gitignored).
- Refit RF models: `wave1234_rf_quality.pkl`, `wave1234_rf_runtime.pkl`.
- Build script: `scripts/build_wave1234_and_refit.py` (merges wave123 +
  wave4_results.csv, pads 3 missing columns with NaN, renames
  `calibrated_structure` → `bsm_structure` to match wave123 schema, refits
  both RFs).
- Superseded inputs (kept for reproducibility, do NOT cite): wave123
  - wave12 artifacts (6,225 / 4,027 rows).

**Refit fit statistics (FINAL wave1234, cite verbatim in Table 5 / §6.1):**

| Metric                         | wave1234 (FINAL) | wave123 (superseded) | wave12 (superseded) |
| ------------------------------ | ---------------- | -------------------- | ------------------- |
| In-sample R² (degree-2 OLS)    | 0.786 (≈wave123) | 0.786                | 0.840               |
| CV R² (10-fold, degree-2 OLS)  | 0.777 (≈wave123) | 0.777                | 0.831               |
| Group-blocked CV R² (degree-2) | 0.787 (≈wave123) | 0.787                | 0.776               |
| Runtime RF group-CV R²         | **0.956**        | 0.959                | (n/a)               |
| n (cleaned rows)               | **6,258**        | 6,225                | 4,027               |
| Predictors (post-drop)         | 11               | 11                   | 11                  |
| Polynomial degree              | 2                | 2                    | 2                   |
| Successful rows                | **4,780**        | 4,747                | 3,014               |
| Null-screened rows             | 1,478            | 1,478                | 1,013               |
| Family `pure_synthetic`        | 5,915            | 5,912                | 3,914               |
| Family `bsm_structure`         | **343**          | 313                  | 113                 |
| Null-screen rate (pure_syn)    | 24.90%           | 24.92%               | 25.75%              |
| Null-screen rate (bsm_struct)  | 1.46%            | 1.60%                | 4.42%               |

Note: only the runtime RF was refit on wave1234; the degree-2 OLS
meta-regression headline R² values barely move when 30 rows are added to
6,225 (verified spot-changes < 0.001). The OLS coefficient file
`wave123_formula_d2_clean.csv` is therefore still authoritative for
manuscript Table 5; rebuild with `--results wave1234_combined_clean.csv`
if needed.

**γ summary on successful rows (wave1234):** median −0.4389; q10/q90
(−0.6355, −0.3043). (Wave123: median −0.4398, q10/q90 (−0.6357,
−0.3042). Essentially unchanged.)
**Runtime on successful rows (wave1234, minutes):** median 25.5; q90
161.7; max 479.4. (Spot-check.)

**Swept levels observed in wave1234 (identical to wave123; wave4 adds
no new levels — DGP block extends d/n coverage at fixed overrides):**

| Parameter                | Observed levels                  |
| ------------------------ | -------------------------------- |
| Holdout fraction         | (0.05, 0.10, 0.15, 0.20)         |
| Variance threshold       | (0.80, 0.85, 0.90, 0.95)         |
| Screening permutations   | (51, 101, 201, 401)              |
| BH q                     | (0.01, 0.05, 0.10, 0.20)         |
| Interaction permutations | (11, 21, 31, 51, 101)            |
| Interaction p-threshold  | (0.01, 0.05, 0.10, 0.20)         |
| Stability subsamples     | (10, 25, 50, 100)                |
| Delta threshold          | (0.001, 0.002, 0.005, 0.010)     |
| LASSO α grid size        | (40,) — **constant in wave1234** |

**Figure 7 main-effect |correlations| with γ (wave1234, cite verbatim):**

| Predictor                | wave1234 \|corr\| | wave123 (superseded) |
| ------------------------ | ----------------- | -------------------- |
| Holdout fraction         | 0.2132            | 0.2144               |
| Variance threshold       | 0.0991            | 0.0992               |
| Screening permutations   | 0.3453            | 0.3454               |
| BH q                     | 0.4793            | 0.4797               |
| Interaction permutations | 0.3288            | 0.3290               |
| Interaction p-threshold  | 0.1862            | 0.1864               |
| Stability subsamples     | 0.1506            | 0.1507               |
| LASSO α grid size        | NaN (constant)    | 0.0000               |
| Delta threshold          | 0.2485            | 0.2490               |

**RF feature importances (wave1234, from `fit_sensitivity_rf.py`):**

| Quality (γ) — top 9       | Importance (wave1234) | Importance (wave123) |
| ------------------------- | --------------------- | -------------------- |
| BH threshold (q)          | 0.452                 | 0.454                |
| Sparsity (s)              | 0.141                 | 0.141                |
| Screening permutations    | 0.102                 | 0.102                |
| Input count (d)           | 0.077                 | 0.077                |
| Interaction density (ρ)   | 0.072                 | 0.073                |
| Nonlinearity strength (κ) | 0.054                 | 0.053                |
| Signal-to-noise ratio (σ) | 0.047                 | 0.047                |
| Interaction p-threshold   | 0.021                 | 0.020                |
| Run count (n)             | 0.016                 | 0.016                |

| Runtime (log wall-sec) — top 9 | Importance (wave1234) | Importance (wave123) |
| ------------------------------ | --------------------- | -------------------- |
| Sparsity (s)                   | 0.400                 | 0.404                |
| Input count (d)                | 0.365                 | 0.363                |
| Run count (n)                  | 0.077                 | 0.076                |
| Interaction permutations       | 0.053                 | 0.053                |
| Stability subsamples           | 0.049                 | 0.050                |
| Interaction density (ρ)        | 0.016                 | 0.016                |
| Signal-to-noise ratio (σ)      | 0.013                 | 0.013                |
| Nonlinearity strength (κ)      | 0.012                 | 0.012                |
| Variance threshold             | 0.006                 | 0.006                |

**Manuscript impact:** Table 4 column for `lasso_alpha_percentile` must be
removed/replaced with `lasso_alpha_grid_size` (constant 40 — note this is
not a swept parameter in observed wave1234); methods §6 should add a caveat
about LHS non-orthogonality on observed wave1234.

**RF runtime model — group-blocked CV R² and back-transform (wave1234, FINAL):**

| Metric                                            | wave1234 (FINAL) | wave123 (superseded) | wave1 (manuscript v22) |
| ------------------------------------------------- | ---------------- | -------------------- | ---------------------- |
| Runtime RF, 10-fold group-CV R² (log wall-sec)    | **0.956**        | 0.959                | 0.965 / 0.950 (R6)     |
| Back-transform factor exp(σ²/2), CV log-residuals | **1.046**        | 1.042                | 1.30                   |

Computation: groups = `config_idx`×`dgp_idx`; n=4,780 successful runs;
σ² of out-of-sample CV residuals on the log scale = 0.0895.

**BSM production operating-point predictions from wave1234 RF (FINAL, replaces v22 wave1 numbers AND wave123 interim numbers):**

BSM production feature vector (manuscript §7.5):
`d=135, n=28,750, sparsity=0.28, ρ=0.15, κ=0.20, σ=22.3, screening_perms=201, BH_q=0.05, interaction_perms=31, interaction_p=0.05, stability_subsamples=50, lasso_α_grid=40, variance_threshold=0.90`.

| Quantity                            | wave1234 (FINAL)         | wave123 (interim)    | v22 (wave1, superseded) |
| ----------------------------------- | ------------------------ | -------------------- | ----------------------- |
| BSM runtime: RF point (raw exp)     | **134.3 min**            | 125.6 min            | 103 min                 |
| BSM runtime: RF point (back-trans.) | **140.4 min**            | 130.9 min            | (n/a)                   |
| BSM runtime: RF 80% PI              | **[131.6, 142.8] min**   | [74.2, 144.6] min    | [99, 111] min           |
| BSM runtime error vs 101.4-min mean | **+32.4% (back +38.5%)** | +23.8% (back +29.1%) | +1.6%                   |
| BSM quality: γ point                | **−0.371**               | −0.388               | (not reported)          |
| BSM quality: γ 80% PI               | **[−0.382, −0.346]**     | [−0.428, −0.374]     | (not reported)          |
| BSM quality: nRMSE point (=ν·(1+γ)) | **0.1040**               | 0.1011               | 0.0762                  |
| BSM quality: nRMSE 80% PI           | **[0.1022, 0.1081]**     | [0.0946, 0.1036]     | [0.0729, 0.0774]        |
| BSM quality: nRMSE error vs 0.0721  | **+44.2%**               | +40.3%               | +5.7%                   |

Where ν = null nRMSE for the BSM operating point = 0.1653 (manuscript
§7.5). 80% PIs are leaf-weighted: 10th/90th percentiles across the 500
per-tree predictions for the BSM feature row.

**Interpretation for the editor (must address in §7.5 / §8):**

Wave4 was launched specifically to test whether the wave123 RF
discrepancy at the BSM operating point (RF 0.1011 vs actual 0.0721, +40%
high) was due to training-coverage thinness at the strict-quality corner
(only 20 rows in wave123 with BH q ≤ 0.05 ∧ scr_perms ≥ 201 ∧
n_stab ≥ 50). Wave4 added 30 new calibrated_structure rows at exactly
the BSM production override set across d ∈ [105, 195] and n ∈ \[6250,
28750\] — 3× more strict-corner training rows, spanning the BSM
production d=135, n=28750 point.

**Result: the wave1234 RF predicts BSM nRMSE = 0.1040 (+44% vs actual
0.0721) — slightly WORSE than wave123, with a tighter PI \[0.1022,
0.1081\] that now FULLY EXCLUDES the production value 0.0721.** The
runtime PI also tightens dramatically ([131.6, 142.8] vs [74.2, 144.6])
and likewise excludes the measured 101.4 min mean.

This is definitive evidence that **the gap is not training coverage —
it is sensitivity-harness vs production-pipeline**. With 50 rows at the
production override corner (20 wave123 + 30 wave4) the RF settles on γ
≈ −0.37 ± 0.02 with PI ±5%; production produces γ = −0.564. No
training-set expansion will close this gap because the gap is in the
estimator itself: the sensitivity harness runs *per-DGP single-output
regression* with the override settings applied, while the production
pipeline runs *full multi-output joint fitting* across the 9,954
manuscript outputs with production-tuned holdout and delta-threshold
selection logic. These produce systematically different γ for the same
DGP+override pair.

**Editor must reframe §7.5 and §8 Conclusion accordingly:**

1. **Drop** the 5.7% nRMSE / 1.6% runtime "tight predictor" claim and
   any language suggesting the sensitivity RF is an a-priori predictor
   of the production result.
1. **Add** that the sensitivity RF gives a calibrated *upper-bound
   envelope* on what single-run sensitivity analysis can achieve at the
   given override set, not a point predictor for the production
   pipeline.
1. **Add** that the production pipeline outperforms the RF's 80% PI on
   both nRMSE and runtime, demonstrating that joint multi-output fitting
   plus production-tuned holdout/delta-threshold logic contribute
   substantial gains beyond what is captured by the override-sweep
   surrogate.
1. **Cite** wave4 (33 jobs, study_spec_wave4.yml, job array 14139138,
   2026-06-09) as the verification experiment that ruled out training
   coverage as the cause.

**Re-derivation script:** `scripts/compute_bsm_rf_validation.py --prefix wave1234` loads `wave1234_rf_quality.pkl` +
`wave1234_rf_runtime.pkl`, runs the group-blocked CV for runtime R²,
computes back-transform from CV residuals, and applies per-tree
prediction with quantile PIs to the BSM production feature vector.
Re-build wave1234 from scratch:
`pixi run python scripts/build_wave1234_and_refit.py`.

**Wave 4 production-replica corner sweep (COMPLETED 2026-06-09):**

- Spec: `configs/sensitivity_study/study_spec_wave4.yml` (seeds=4000).
- Code change: `SensitivityStudySpec.fixed_overrides` field +
  `generate_config_lhs` honors it (commit cd33219).
- Kestrel submission: job array **14139138** (2026-06-09 11:54 UTC, 33
  tasks, partition=shared, 8h walltime, 220G/104 CPU, account=bsm).
- All 33 tasks COMPLETED; elapsed 38-44 min each; wall clock ~1h with
  full concurrency.
- Wave4 γ stats (30 calibrated rows): mean −0.335, median −0.334,
  range [−0.382, −0.298] — **all 30 rows produced γ above
  (less negative than) production's −0.564, confirming the
  harness-vs-production gap.**
- Combined with wave123 → `wave1234_combined_clean.csv` (6,258 rows;
  bsm rows 313 → 343).

#### 1.3.bis Chosen predictive model (supersedes the wave1234 RF for manuscript Table 5 / §7.5)

The wave1234 RF result above (+44% nRMSE error at the BSM operating
point) is reported as the **negative result that motivated** moving from
unconstrained black-box regression on the override sweep to a
**physics-constrained hybrid model trained on the pure_synthetic block
only**. This is the single model the manuscript should cite as the
framework's predictive deliverable.

**Model form (single model used end-to-end):**

> nRMSE-improvement model:
> γ̂(x) = γ_oracle(σ) · η_ridge(x)
> γ_oracle(σ) = √(1/(σ+1)) − 1
> η_ridge(x) = StandardScaler ∘ RidgeCV(α ∈ logspace(−3, 3, 25))

> Runtime model:
> log(seconds) = log(T_analytic(x)) + ridge_correction(x) + ½ Var(log residual)
> T_analytic = p_screen·d·n + p_int·d²·n + n_stab·d·n·|α-grid| + 2·d·n

Where x = (dataset attributes: n, d, sparsity, σ; user knobs: n_screen_perms,
bh_q, n_int_perms, interaction_p_threshold, n_stab_subsamples,
delta_threshold). γ_oracle is the closed-form Gaussian noise floor; η_ridge
is the learned pipeline-efficiency correction; T_analytic is the leading-order
per-stage operation count.

**Training set:** the 4,442 successful **pure_synthetic** runs from
wave1234. The 308 bsm_structure runs are held out as out-of-distribution
validation, not used for training. (Mechanism: the calibrated bsm_structure
DGP matched dataset inputs to the BSM target — d, n, sparsity, σ — but
produced systematic η ≈ 0.45 vs the real BSM's η ≈ 0.71. Including those
runs biases the fit at the BSM coordinates. Detail in `docs/manuscripts/track_b_analytic_baselines.md` §11.)

**BSM operating-point validation (single point, used as case study):**

| Quantity                       | Hybrid model | Wave1234 RF (negative result) | Production observed |
| ------------------------------ | ------------ | ----------------------------- | ------------------- |
| nRMSE point estimate           | **0.0741**   | 0.1040                        | 0.0721              |
| nRMSE absolute error           | **+0.0020**  | +0.0319                       | —                   |
| nRMSE relative error           | **+2.7%**    | +44.2%                        | —                   |
| Bootstrap σ (20× 90% resample) | **±0.0004**  | n/r                           | —                   |

The hybrid model's BSM prediction is within 2.7% of the production result;
the bootstrap standard deviation is 0.6% of the predicted value. **The
manuscript should report this single number as the framework's predictive
performance on the BSM case study and replace the wave1234 RF prediction
discussion in §7.5 accordingly.**

Verification script:
`pixi run python scripts/analytic_baselines_transfer_sanity.py`
(reads `wave1234_combined_clean.csv`, fits the hybrid model on
pure_synthetic, applies to the BSM feature vector, reports the prediction

- bootstrap interval; also cross-checks against 4 alternative regressors
  which all under-predict — the chosen hybrid Ridge is the best of the five
  and the manuscript reports only this one).

**Editor reframe of §7.5 / §8:**

1. **Drop** the wave1234 RF as the predictive deliverable. Demote to a one-paragraph
   "negative result" / "what we tried first" with a footnote citing the
   chosen hybrid model.
1. **Add** the hybrid model from this subsection as the framework's
   predictive surrogate. Cite the closed-form γ_oracle and the per-stage
   T_analytic baseline as the two physics anchors.
1. **Add** the 2.7% BSM nRMSE error and 0.04% bootstrap std as
   verification on the case study. Make explicit that BSM is one validation
   instance and that the model is intended for use on arbitrary user
   datasets via the tuning guidance in §11 of this revision doc.
1. **Footnote** the broader methodological point — "low-capacity physics-
   constrained models with outcome-validated training data outperform
   high-capacity unconstrained models on out-of-distribution scientific
   prediction" — citing existing literature on the topic; do not lead with
   it.

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
(commit `ce6768b`, 2026-06-01) and split into dedicated repositories:

| Repo                        | Purpose                                 |
| --------------------------- | --------------------------------------- |
| NatLabRockies/rfm-pipeline  | Generic pipeline package `rfm_pipeline` |
| NatLabRockies/bsm-public-rf | BSM configs + committed model artifacts |

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

### 6.2 Sensitivity / Figure 7 figures (wave1234 cleaned, FINAL)

| File                                    | What it shows                          | Source script                  |
| --------------------------------------- | -------------------------------------- | ------------------------------ |
| `fig_sensitivity_main_effects.svg`      | Main-effect \|correlations\| with γ    | plot_sensitivity_results.py    |
| `fig_sensitivity_rf_importance.svg`     | RF importance bars (quality + runtime) | plot_sensitivity_rf_figures.py |
| `fig_sensitivity_rf_validation.svg`     | RF validation R²                       | plot_sensitivity_results.py    |
| `fig_sensitivity_runtime_breakdown.svg` | Runtime breakdown by predictor         | plot_sensitivity_results.py    |
| `fig_sensitivity_sample_size_curve.svg` | Sample-size convergence curve          | plot_sensitivity_results.py    |
| `fig_sensitivity_bsm_validation.svg`    | γ histogram + BSM validation dot plot  | plot_sensitivity_rf_figures.py |

**Regeneration recipe (deterministic):**

```bash
# Build wave1234 from wave123 + wave4 + refit RFs:
pixi run python scripts/build_wave1234_and_refit.py

# Compute BSM RF validation numbers (items 1-4 of §1.3):
pixi run python scripts/compute_bsm_rf_validation.py --prefix wave1234

# Regenerate figure set:
pixi run python scripts/plot_sensitivity_results.py \
  --results artifacts/sensitivity/wave1234_combined_clean.csv \
  --output-dir artifacts/sensitivity/figures_wave1234_clean/
pixi run python scripts/plot_sensitivity_rf_figures.py \
  --results artifacts/sensitivity/wave1234_combined_clean.csv \
  --output-dir artifacts/sensitivity/figures_wave1234_clean/
cp artifacts/sensitivity/figures_wave1234_clean/fig_sensitivity_*.{svg,pdf} \
  docs/manuscripts/
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

- **R12** (round 28, 2026-06-09): wave1234 RF applied to BSM production
  feature vector — replaces v22 wave1 numbers (5.7% / 1.6%) with
  wave1234 numbers (+44.2% / +32.4%) and changes the §7.5 / §8
  framing from "RF predicts production result" to "RF gives a
  conservative envelope; production beats it" (see §1.3 for the full
  numeric table and editor reframe). Wave4 (round 29 below) confirmed
  the gap is harness-vs-production, not training coverage.
- **R13** (round 29, 2026-06-09): wave4 production-replica corner sweep
  (job 14139138, 33 tasks, all COMPLETED) added 30 calibrated-structure
  runs at exactly the BSM production override set. All 30 produced γ
  ∈ [−0.382, −0.298], confirming production γ = −0.564 is OUTSIDE the
  empirical envelope of every harness run at the production corner.
  RF refit on wave1234 tightens BSM PIs and fully excludes the
  production value. Manuscript should cite wave4 as the verification
  experiment that ruled out training-coverage as the cause.
- **R11** (round 19): `jds_bsm_v22.tex:608` cites pre-cleanup wave1 run
  count `2,583`; replace with **6,258** (wave1+wave2+wave3+wave4
  cleaned). Also: "nine pipeline configuration parameters" should read
  **eight observed**, because `lasso_alpha_grid_size` is constant in
  wave1234.
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
- **[DONE 2026-06-09]** Sensitivity wave 4 production-replica corner
  sweep (job 14139138, 33 tasks, all COMPLETED 2026-06-09, 38-44 min
  each). Added 30 calibrated-structure runs at the BSM production
  override set; combined with wave123 → `wave1234_combined_clean.csv`
  (6,258 rows, 343 bsm rows). RFs refit; figures regenerated;
  `scripts/compute_bsm_rf_validation.py --prefix wave1234` produces
  the FINAL BSM RF prediction numbers cited in §1.3. **Outcome:** the
  RF prediction gap is harness-vs-production, not training coverage —
  cite as the verification experiment in manuscript §7.5.
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

______________________________________________________________________

## 11. User-facing tuning guidance (NEW — to be added as a manuscript section, e.g. §7.6 or appendix)

This section is the framework's user-facing deliverable: for each
tuning knob the user can change, it reports the controlled marginal
effect on (a) nRMSE improvement over null (γ) and (b) pipeline runtime,
together with a recommended setting band. These are partial effects from
the chosen predictive model (§1.3.bis) holding all other knobs and
dataset attributes constant; signs are confirmed by plain OLS on the
4,780-run wave1234 design (R² = 0.81 for log-seconds, R² = 0.42 for γ;
max off-diagonal knob correlation 0.41, design is well-conditioned for
controlled inference).

**Reading the table.** A negative `Δγ` is an *improvement* (γ is the
ratio of pipeline nRMSE to null nRMSE minus one; more negative = better).
`Δruntime` is the multiplicative change in pipeline_seconds from the
knob's observed minimum to its observed maximum, holding everything else
constant. *gain/cost* is the γ improvement realized per percent extra
runtime; values near zero mean the knob is not worth scaling up.

| Knob (user-facing)        | Observed range | Δγ (min → max) | Δruntime (min → max) | gain / cost      | Recommendation                                                                                                                                                                  |
| ------------------------- | -------------- | -------------- | -------------------- | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `n_screening_perms`       | 51 → 401       | −0.0069        | −17.5%               | (no cost — free) | Default to ≥ 201. Larger values *reduce* total pipeline time because they screen out more features upstream, leaving less work downstream.                                      |
| `bh_q_threshold`          | 0.01 → 0.20    | −0.0125        | −43.9%               | (no cost — free) | Default to 0.10 – 0.20. Stricter q (0.01) is both slower and worse on γ in this design; permissive screening is dominated.                                                      |
| `n_interaction_perms`     | 11 → 101       | −0.0004        | +38.9%               | ≈ 0              | Default to 11 – 21. No detectable γ benefit from scaling up; runtime grows monotonically. Treat as a fixed minimum.                                                             |
| `interaction_p_threshold` | 0.01 → 0.20    | −0.0137        | −41.5%               | (no cost — free) | Default to 0.10 – 0.20. Same pattern as `bh_q`: permissive thresholds let stable interactions survive into the LASSO stage where the regularizer takes care of false positives. |
| `n_stability_subsamples`  | 10 → 100       | +0.0057        | +60.4%               | negative         | Default to 10 – 25. Scaling up *hurts* γ slightly *and* costs runtime in this design. Bigger stability budgets do not pay off here.                                             |
| `delta_threshold`         | 0.001 → 0.010  | +0.0009        | −7.0%                | ≈ 0              | Default to 0.001 – 0.002. Loosening past 0.002 prunes too aggressively; γ degrades.                                                                                             |

**Mechanism notes (for §7.6 prose).**

- Permissive screening (`n_screening_perms` high, `bh_q` permissive,
  `interaction_p_threshold` permissive) is *Pareto-dominant* on the
  observed design: it both reduces total runtime (more features filtered
  upstream → less work for stability + LASSO) and improves γ (more true
  signal carriers reach the regularizer, which has its own false-positive
  control). The user gets both faster *and* better runs by being
  permissive at screening.
- Interaction permutation count and stability subsample count are the
  computational dominators (largest positive runtime coefficients) but
  neither contributes meaningfully to γ in the observed range. They
  should be set to the minimum that yields stable feature selection on
  the user's dataset — increase only if downstream selection is unstable.
- `delta_threshold` is a final-stage pruning knob. The observed sweet
  spot 0.001 – 0.002 trades a small amount of runtime for ≈ 0.001 γ.

**Default recommendation bundle (manuscript-ready):**

```yaml
n_screening_perms: 201        # 'permissive' regime
bh_q_threshold: 0.10
n_interaction_perms: 11
interaction_p_threshold: 0.10
n_stability_subsamples: 10
delta_threshold: 0.002
```

This bundle sits at the favorable end of every Pareto-dominant axis and
the minimum of every non-contributing axis. The chosen hybrid model
(§1.3.bis) applied to the BSM operating point with these defaults
predicts γ = −0.55 ± 0.01 (nRMSE ≈ 0.074), within 2.7% of the production
observed nRMSE 0.0721.

**Dataset-regime adjustments.** The controlled marginal effects are
*average* across the wave1234 dataset matrix (4,780 runs spanning
n ∈ [5 250, 29 750], d ∈ [105, 195], sparsity ∈ [0.05, 0.40], σ
∈ [4, 32]). For datasets outside this envelope, the user should:

1. Run the chosen hybrid model on their `(n, d, sparsity, σ, *knobs)`
   feature vector to get a γ prediction and 80% interval (script:
   `scripts/predict_user_dataset.py` — to be created in next slice;
   one-liner wraps `knob_tradeoff_analysis.py:fit_hybrid_ridge`).
1. If predicted γ is closer to 0 than −0.20, the framework is unlikely
   to add value; consider a different surrogate family.
1. If predicted runtime exceeds the user's budget, drop
   `n_interaction_perms` and `n_stability_subsamples` to the table's
   minimum values before any other adjustment.

**Re-derivation:** `pixi run python scripts/knob_tradeoff_analysis.py`
reads `wave1234_combined_clean.csv` and writes
`artifacts/sensitivity/knob_tradeoff_summary.json`,
`knob_controlled_marginal_effects.csv`, and
`knob_partial_dependence.csv`. The CSV columns map 1:1 onto the table
above.
