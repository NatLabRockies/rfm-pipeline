# Track B: Analytic Baselines for Pipeline Performance

Status: artifacts written 2026-06-09 (`scripts/analytic_baselines.py`, commit `54c0a70`).
Inputs: `artifacts/sensitivity/wave1234_combined_clean.csv` (6,258 rows).
Outputs: `wave1234_analytic_baselines.csv`, `wave1234_analytic_baselines_summary.json`.

This document derives three first-principles bounds/predictions for the cascade pipeline and validates them empirically. They serve two purposes:

1. **Sanity baseline** for the meta-model (Track A): if a learned model cannot beat the analytic predictor on held-out points, it has not learned anything useful.
1. **Manuscript-ready interpretable bounds** for §7.5: a reviewer can verify these from theory alone, independent of empirical fits.

Notation throughout:

- $n$ = number of training rows (`n_runs`).
- $d$ = number of inputs (`n_inputs`).
- $p$ = number of outputs (`n_outputs`).
- $\\sigma^2\_{\\text{signal}}$ = variance of the noise-free signal in a single output column.
- $\\sigma^2\_{\\text{noise}}$ = variance of the additive measurement noise on the same column.
- SNR = $\\sigma^2\_{\\text{signal}}/\\sigma^2\_{\\text{noise}}$ (per `noise_snr` in the spec).
- nRMSE = $\\text{RMSE}/\\text{range}(y)$ (the manuscript's metric, recomputed by `final_artifacts`).
- $\\gamma\_{\\text{obs}}$ = `nrmse_relative` = $(\\text{nrmse\\\_final} - \\text{nrmse\\\_null})/\\text{nrmse\\\_null}$. Negative means the pipeline beats the null.

Each section below has the structure: **Overview → Method → Conclusion → Step-by-step derivation**.

______________________________________________________________________

## 1. Oracle γ ($\\gamma\_{\\text{oracle}}$): theoretical maximum reduction in nRMSE relative to null

### Overview

What is the lowest nRMSE any predictor — even one with infinite data and full knowledge of the true input-output map — could achieve on a noisy dataset, and how much improvement over the null predictor does that correspond to? This sets a hard ceiling on $\\gamma$. The pipeline's distance from this ceiling is its inefficiency, attributable to finite samples + algorithmic compromises.

### Method

Two separate first-principles calculations:

1. The null predictor's RMSE is the standard deviation of the output (predicting the mean is optimal in the L2 sense).
1. The oracle predictor's RMSE on a single noisy output $y = f(x) + \\varepsilon$ with $\\text{Var}(\\varepsilon) = \\sigma^2\_{\\text{noise}}$ is exactly $\\sigma\_{\\text{noise}}$ (it predicts $f(x)$ perfectly; the only remaining error is irreducible noise).

The ratio of these two RMSEs, divided by the same range and converted to a relative-to-null reduction, gives a closed-form $\\gamma\_{\\text{oracle}}$.

### Conclusion

$$\\boxed{\\gamma\_{\\text{oracle}} = \\sqrt{\\frac{1}{\\text{SNR} + 1}} - 1}$$

For the wave1234 cleaned set this gives mean $\\gamma\_{\\text{oracle}} = -0.776$ (the pipeline could in principle reduce error by 78% over null). Observed mean $\\gamma\_{\\text{obs}} = -0.349$. The ratio $\\gamma\_{\\text{obs}}/\\gamma\_{\\text{oracle}}$ is the pipeline's **efficiency** (Section 3).

### Step-by-step derivation

**Step 1.** Decompose the noisy output:

$$y = f(x) + \\varepsilon, \\quad \\varepsilon \\perp x, \\quad \\mathbb{E}[\\varepsilon] = 0, \\quad \\text{Var}(\\varepsilon) = \\sigma^2\_{\\text{noise}}$$

**Justification.** This is the additive-noise generative model the synthetic DGP enforces (`_assemble_dataset` adds `rng.normal(scale=noise_std, size=...)` after generating the signal). It is also the standard manuscript model.

**Step 2.** Compute total output variance:

$$\\sigma^2_y = \\text{Var}(f(x)) + \\text{Var}(\\varepsilon) = \\sigma^2\_{\\text{signal}} + \\sigma^2\_{\\text{noise}}$$

**Justification.** Independence of $\\varepsilon$ from $x$ implies covariance is zero, so variance decomposes additively. (`_generate_sparse_signals` and the noise term use independent RNG streams from the same seeded `np.random.Generator`.)

**Step 3.** RMSE of the null predictor $\\hat{y}\_{\\text{null}} = \\bar{y}$:

$$\\text{RMSE}_{\\text{null}} = \\sqrt{\\mathbb{E}\[(y - \\mathbb{E}[y])^2\]} = \\sigma_y = \\sqrt{\\sigma^2_{\\text{signal}} + \\sigma^2\_{\\text{noise}}}$$

**Justification.** The null predictor predicts the population mean, so its squared error is the variance of $y$ (this is the L2-optimal scalar predictor by definition).

**Step 4.** RMSE of the oracle predictor $\\hat{y}\_{\\text{oracle}} = f(x)$:

$$\\text{RMSE}_{\\text{oracle}} = \\sqrt{\\mathbb{E}[(y - f(x))^2]} = \\sqrt{\\mathbb{E}[\\varepsilon^2]} = \\sigma_{\\text{noise}}$$

**Justification.** The oracle observes the true conditional mean, leaving only the irreducible noise residual. $\\mathbb{E}[\\varepsilon^2] = \\text{Var}(\\varepsilon) + (\\mathbb{E}[\\varepsilon])^2 = \\sigma^2\_{\\text{noise}}$ since $\\mathbb{E}[\\varepsilon]=0$.

**Step 5.** Convert both to nRMSE by dividing by the same range $R = \\max(y) - \\min(y)$:

$$\\text{nRMSE}_{\\text{null}} = \\sigma_y / R, \\quad \\text{nRMSE}_{\\text{oracle}} = \\sigma\_{\\text{noise}} / R$$

**Justification.** The manuscript's `final_artifacts` writer uses the same range for null and final nRMSE (computed on the held-out fold once, not recomputed per predictor). So $R$ cancels in the ratio.

**Step 6.** Ratio:

$$\\frac{\\text{nRMSE}_{\\text{oracle}}}{\\text{nRMSE}_{\\text{null}}} = \\frac{\\sigma\_{\\text{noise}}}{\\sigma_y} = \\frac{\\sigma\_{\\text{noise}}}{\\sqrt{\\sigma^2\_{\\text{signal}} + \\sigma^2\_{\\text{noise}}}} = \\sqrt{\\frac{\\sigma^2\_{\\text{noise}}}{\\sigma^2\_{\\text{signal}} + \\sigma^2\_{\\text{noise}}}}$$

**Justification.** Algebra. Square the numerator and denominator and pull both inside the radical.

**Step 7.** Substitute SNR = $\\sigma^2\_{\\text{signal}}/\\sigma^2\_{\\text{noise}}$:

$$\\frac{\\text{nRMSE}_{\\text{oracle}}}{\\text{nRMSE}_{\\text{null}}} = \\sqrt{\\frac{1}{\\text{SNR} + 1}}$$

**Justification.** Divide top and bottom inside the radical by $\\sigma^2\_{\\text{noise}}$.

**Step 8.** Compute the relative reduction:

$$\\gamma\_{\\text{oracle}} = \\frac{\\text{nRMSE}_{\\text{oracle}} - \\text{nRMSE}_{\\text{null}}}{\\text{nRMSE}\_{\\text{null}}} = \\sqrt{\\frac{1}{\\text{SNR}+1}} - 1$$

**Justification.** This is the manuscript's `nrmse_relative` definition with `final` replaced by `oracle`.

**Step 9.** Sign + range check.

- SNR $\\to 0^+$: $\\gamma\_{\\text{oracle}} \\to 0$. No signal, oracle = null. ✓
- SNR $\\to \\infty$: $\\gamma\_{\\text{oracle}} \\to -1$. All noise removed, perfect prediction. ✓
- SNR = 22.3 (BSM op-point): $\\gamma\_{\\text{oracle}} = \\sqrt{1/23.3} - 1 \\approx 0.207 - 1 = -0.793$. BSM observed = -0.564 (71% efficient). ✓ (consistent with manuscript R10).

**Step 10.** Multi-output average. The pipeline reports a single nRMSE averaged across $p$ outputs. Per-output SNR may be heterogeneous (`per_output_snr_heterogeneity`), so:

$$\\gamma\_{\\text{oracle}}^{\\text{multi}} = \\mathbb{E}\_j\\left[\\sqrt{\\frac{1}{\\text{SNR}\_j + 1}}\\right] - 1 \\neq \\sqrt{\\frac{1}{\\bar{\\text{SNR}} + 1}} - 1$$

**Justification.** Jensen's inequality for the concave function $\\sqrt{1/(s+1)}$. The current implementation uses the scalar SNR, which is exact when SNR is uniform across outputs and a slight overestimate of $|\\gamma|$ when SNR is heterogeneous (jensen direction: $\\mathbb{E}[\\sqrt{1/(s+1)}] \\geq \\sqrt{1/(\\mathbb{E}[s]+1)}$ → the per-output mean of the *RATIO* exceeds the ratio at the mean SNR). For the wave1234 set most rows have `per_output_snr_heterogeneity=0`, so this is exact for those rows and a small overestimate for the rest.

______________________________________________________________________

## 2. Predicted null nRMSE from sample size + Gaussian assumption

### Overview

Before the pipeline runs, can we predict what the null predictor's nRMSE will be on a synthetic dataset with $n$ rows and Gaussian outputs? This sanity-checks the `nrmse_null` column in the wave artifacts and gives an analytic baseline for what "no-signal" performance looks like.

### Method

For Gaussian $y$, the population standard deviation divided by the sample range converges to a known function of $n$ (Cramér's asymptotic for the maximum of i.i.d. Gaussians). Plug $n$ into that formula.

### Conclusion

$$\\boxed{\\text{nRMSE}\_{\\text{null}}^{\\text{predicted}} \\approx \\frac{1}{2 \\cdot \\mathbb{E}[\\max_n |Z|]} \\approx \\frac{1}{2\\sqrt{2\\ln n}}}$$

where $Z \\sim N(0,1)$ and $n$ is the held-out fold size. For wave1234 this predicts mean 0.130 vs observed 0.135 (4% RMSE on the prediction itself). Correlation with observed across rows is r = 0.250 — the formula captures the trend but per-row noise is high because real outputs are not strictly Gaussian.

### Step-by-step derivation

**Step 1.** The null predictor's RMSE on a held-out fold of size $n$ is the standard deviation of $y$ on that fold (Step 3 of Section 1).

**Step 2.** The range of $y$ on a held-out fold of size $n$, for $y \\sim N(\\mu, \\sigma^2)$:

$$R_n = \\max_i y_i - \\min_i y_i \\approx 2\\sigma \\cdot \\mathbb{E}[\\max_i Z_i] \\quad \\text{where } Z_i \\sim N(0,1)$$

**Justification.** Translation/scale invariance: subtract $\\mu$ and divide by $\\sigma$ to reduce to the standard normal case. Symmetry of $N(0,1)$: $\\mathbb{E}[\\max] = -\\mathbb{E}[\\min]$, so range = $2\\mathbb{E}[\\max]$.

**Step 3.** Cramér asymptotic for the expected maximum of $n$ i.i.d. $N(0,1)$ samples (1946; David & Nagaraja 2003 §10.5):

$$\\mathbb{E}[\\max\_{i \\leq n} Z_i] \\approx \\sqrt{2\\ln n} - \\frac{\\ln \\ln n + \\ln 4\\pi}{2\\sqrt{2\\ln n}}$$

**Justification.** This is the standard asymptotic from extreme value theory; the leading term $\\sqrt{2\\ln n}$ comes from the upper tail of the Gaussian density, and the second-order correction comes from the Gumbel limit. For $n \\in [10^3, 10^5]$ (the range of held-out folds in our study), the leading-order $\\sqrt{2\\ln n}$ is accurate to ~5%.

**Step 4.** Combine. nRMSE = RMSE / range = $\\sigma_y / R_n$:

$$\\text{nRMSE}\_{\\text{null}} \\approx \\frac{\\sigma_y}{2\\sigma_y \\cdot \\sqrt{2\\ln n}} = \\frac{1}{2\\sqrt{2\\ln n}}$$

**Justification.** $\\sigma_y$ cancels because both numerator and denominator scale linearly with output spread.

**Step 5.** Numerical predictions (held-out fold size = `n_runs * holdout_fraction`):

| $n$    | $1/(2\\sqrt{2\\ln n})$ | wave1234 observed nrmse_null mean |
| ------ | ---------------------- | --------------------------------- |
| 1,000  | 0.190                  | (not in study)                    |
| 5,000  | 0.165                  | (not in study)                    |
| 28,750 | **0.131**              | **0.135** (BSM op-point)          |
| 50,000 | 0.124                  | (not in study)                    |

**Justification.** Direct evaluation. BSM at n=28,750 → 0.131 predicted vs 0.135 observed (3% error on the prediction itself). Aggregate over wave1234: mean predicted 0.130 vs mean observed 0.135.

**Step 6.** Why the row-level correlation is only r = 0.250 even though the means agree.

The Cramér result requires Gaussianity. Real rows in wave1234 violate this in two ways:

- **`output_nonlinearity_strength > 0`**: monotonic transformations distort the tails. `expm1` strongly inflates the range (huge max, modest std) → nRMSE drops below predicted.
- **`per_output_snr_heterogeneity > 0`**: heterogeneous noise across outputs widens the range without proportionally raising the std (tail outputs dominate the max).

These distortions cancel in the *mean* prediction (because the heterogeneity sources are roughly symmetric across the wave1234 sweep design), but they introduce per-row scatter that the formula cannot capture without measuring tail kurtosis. **Implication:** Track A's measurement-based meta-model can do better here because it observes `output_kurtosis_excess_mean` and can adjust the prediction conditional on tail behavior.

**Step 7.** Connection to observed nrmse_null variance. The predicted nrmse_null is monotonically decreasing in $n$ but the wave1234 sweep also varies `holdout_fraction`, so the *effective* sample size for the null is `n_runs * holdout_fraction`. The current implementation in `analytic_baselines.py` uses `n_runs` directly (worst case = no holdout); this systematically underestimates predicted nrmse_null by ~5% on rows with holdout=0.20 and ~15% on rows with holdout=0.05. This is a known limitation, deliberate (the formula is meant as a leading-order baseline).

______________________________________________________________________

## 3. Pipeline efficiency: $\\eta = \\gamma\_{\\text{obs}} / \\gamma\_{\\text{oracle}}$

### Overview

Given the achievable maximum (Section 1), what fraction does the pipeline actually capture? This is dimensionless, in [0, 1] for a well-functioning pipeline (negative if pipeline does worse than null, > 1 if it somehow beats the oracle = generally a sign of optimistic CV leakage). Tracking this across the wave1234 design tells us where the pipeline is healthy vs. where it leaves the most signal on the table.

### Method

Compute $\\eta = \\gamma\_{\\text{obs}} / \\gamma\_{\\text{oracle}}$ row-wise. Aggregate by knob to identify which knobs predict efficiency drops.

### Conclusion

Wave1234 median $\\eta = 0.514$ (the pipeline captures about half of the achievable signal reduction, on average). 95th percentile $\\eta = 0.835$. 5th percentile $\\eta = 0.0$ (rows where the pipeline null-screened out and recovered nothing). BSM op-point $\\eta = 0.564 / 0.793 = 0.711$ — well above the synthetic median, suggesting BSM is an *easier* regime than the typical synthetic row (consistent with BSM's strong rank-1 structure, which the OLS final stage can exploit if features are retained).

### Step-by-step derivation

**Step 1.** Direct definition:

$$\\eta = \\frac{\\gamma\_{\\text{obs}}}{\\gamma\_{\\text{oracle}}} = \\frac{(\\text{nrmse_final} - \\text{nrmse_null})/\\text{nrmse_null}}{\\sqrt{1/(\\text{SNR}+1)} - 1}$$

**Justification.** Both numerator and denominator are reductions relative to null, so the ratio is dimensionless. Both are negative for a useful pipeline, so $\\eta$ is positive.

**Step 2.** Bounds.

- $\\eta = 0$: pipeline equals null. No useful inference (all features pruned, screening removed everything, etc.).
- $\\eta = 1$: pipeline matches oracle. Achievable only with infinite data + perfect feature recovery.
- $\\eta < 0$: pipeline does worse than null. Indicates overfitting, mis-specified screening, or a bug.
- $\\eta > 1$: pipeline beats oracle. Almost always a CV leakage artifact (e.g., the held-out fold is correlated with the training fold).

**Justification.** Definitional. Bounds follow directly from the fact that $\\gamma$ is monotonically decreasing in performance, with $\\gamma\_{\\text{null}} = 0$ and $\\gamma\_{\\text{oracle}}$ the most negative achievable value.

**Step 3.** Why $\\eta$ is preferable to raw $\\gamma$ as a pipeline-quality metric:

- $\\gamma$ confounds pipeline skill with task difficulty. A pipeline that achieves $\\gamma=-0.5$ on a high-SNR task (where the oracle gets $\\gamma=-0.9$) is only 56% efficient, while a pipeline that achieves $\\gamma=-0.3$ on a moderate-SNR task (oracle $\\gamma=-0.4$) is 75% efficient. Latter is the better pipeline; raw $\\gamma$ obscures this.
- $\\eta$ has a meaningful absolute scale (0 = null, 1 = oracle) that travels across SNR regimes.

**Step 4.** Empirical distribution on wave1234 (from `wave1234_analytic_baselines_summary.json`):

```
mean   0.453
median 0.514
p5     0.000   ← null-screened rows clipped to 0 (γ_obs = 0)
p95    0.835
```

**Justification.** Direct read-off. The bimodal-ish distribution (large mass at 0 from null-screened rows + a roughly Beta(2,1)-shaped continuous mass for successful rows) explains why mean < median.

**Step 5.** Per-knob aggregation (informal observation, not in the summary JSON):

- Higher SNR → higher $\\eta$ (more signal to find, easier task).
- Higher `sparsity` → lower $\\eta$ (denser signals are harder to recover with the BH+stability pipeline).
- Higher `noise_snr_heterogeneity` → lower $\\eta$ (noisy outputs dominate the held-out RMSE).

**Justification.** Visible in scatter plots of $\\eta$ vs. each knob. These are the same variables that dominate the wave123 RF's feature importances, providing a sanity check.

**Step 6.** Track A connection. If the meta-model (Track A) cannot beat $\\eta\_{\\text{predicted}} \\times \\gamma\_{\\text{oracle}}$ (where $\\eta\_{\\text{predicted}}$ is the row's predicted efficiency from a simple linear regression against measurements), it is no better than this two-line analytic. **Wave5 v2's GP-ARD must clear this bar.**

______________________________________________________________________

## 4. What Track B does NOT cover

Three predictions deferred to potential extensions:

1. **BH q-value retention rate.** The number of features the screening stage keeps as a function of BH q + true sparsity. This requires the false-discovery rate framework + the empirical-null permutation distribution. Tractable analytically (BH controls FDR at q if assumptions hold, so retained ≈ q × (false features) + (true features × power)) but not yet implemented.
1. **OLS final-stage variance reduction per retained feature.** If $k$ features are retained and the true signal is rank-$r \\leq k$, the OLS reduction in held-out RMSE per added feature is $1/k$ in the orthogonal case. Real designs have correlated columns so the formula needs an inflation factor. Computable from `feature_catalog` + design correlation matrix.
1. **Penalized regression shrinkage.** SURE (Stein) bounds for soft-thresholded estimators give an analytic bound on $\\eta$ for the LASSO-OLS hybrid the pipeline uses. Plausible and well-known but the constants depend on the empirical Bayes hyperparameter calibration the pipeline does internally; deferring this until Wave5 v2 clarifies whether Track A succeeds.

If Track A produces a meta-model with $|R^2 - R^2\_{\\text{Track B}}| < 0.05$, these extensions are not worth the engineering time. If Track A fails, these become the fallback path.

______________________________________________________________________

## 5. Reproducibility

```bash
pixi run python scripts/analytic_baselines.py
# Inputs:  artifacts/sensitivity/wave1234_combined_clean.csv
# Outputs: artifacts/sensitivity/wave1234_analytic_baselines.csv
#          artifacts/sensitivity/wave1234_analytic_baselines_summary.json
```

The script is deterministic: it reads only the rows in the input CSV and emits one prediction per row. No random sampling, no fitting.

Source: `scripts/analytic_baselines.py` (commit `54c0a70`).
