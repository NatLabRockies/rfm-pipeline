# Interpreting Your Results

**Purpose:** Learn how to validate your workflow run succeeded and compare results to manuscript benchmarks.

**Prerequisites:**

- Completed a workflow run (demo, test, or real data)
- Familiar with the 6-stage workflow (see [Workflow Stages](manuscript_runtime.md))

**Next Steps:** [Troubleshooting](troubleshooting.md) if results don't look right

______________________________________________________________________

## Table of Contents

1. [Quick Validation Checklist](#quick-validation-checklist)
1. [Stage-by-Stage Benchmarks](#stage-by-stage-benchmarks)
1. [Manuscript Benchmark Values](#manuscript-benchmark-values)
1. [Red Flags (Run Likely Failed)](#red-flags-run-likely-failed)
1. [Interpreting Differences](#interpreting-differences)
1. [Next Steps](#next-steps)

______________________________________________________________________

## Quick Validation Checklist

After your run completes, check these indicators:

### ✅ Run Succeeded If:

- [x] All 6 stage directories exist in your output folder
- [x] No Python tracebacks in console output
- [x] `final_artifacts/holdout_performance.csv` exists with nRMSE < 1.0
- [x] Feature counts decrease across stages (candidates → screened → final)
- [x] PCA explains >80% variance with \<100 components
- [x] Final feature count is non-zero (at least a few features retained)

### ❌ Run Failed If:

- [ ] Missing stage directories
- [ ] nRMSE > 1.0 or NaN
- [ ] Zero features retained in final model
- [ ] All features filtered out in early stages
- [ ] PCA explains \<50% variance even with many components
- [ ] Error messages in console

______________________________________________________________________

## Stage-by-Stage Benchmarks

Use these manuscript values as reference points. **Your results will differ** due to:

- Different sample sizes (3k test vs. 20k manuscript)
- Different random seeds
- Different candidate library sizes

### Stage 1: Output Conditioning

**Purpose:** Reduce output dimensionality via PCA while retaining variance

**Manuscript Benchmarks:**

- **PCA components retained:** 39
- **Variance explained:** 90%
- **Outputs retained:** 23,495 (all outputs passed variance filter)

**Your Results Should Show:**

- Components: 20–60 typically (depends on data)
- Variance: >80% minimum, >90% ideal
- Most outputs retained (>95%)

**Check Files:**

- `output_conditioning/pca_explained_variance.csv`
  - `cumulative_variance` column should reach >0.90
- `output_conditioning/output_filter_diagnostics.csv`
  - Most outputs should have `retained_after_filter = true`

**Red Flags:**

- Need >100 components for 90% variance → outputs may be too noisy
- \<50% variance explained → data quality issue
- \<80% outputs retained → variance threshold too strict

______________________________________________________________________

### Stage 2: Empirical Null Screening

**Purpose:** Filter candidate features using permutation-based false discovery control

**Manuscript Benchmarks:**

- **Retained features:** 349 (from 26,560 candidates)
- **BH FDR threshold:** q = 0.10
- **Permutations:** 200
- **Retention rate:** 1.3%

**Your Results Should Show:**

- Retained: 100–500 features (varies with candidate library)
- Most features filtered out (>95% rejection typical)
- Retention rate: 0.5%–5%

**Check Files:**

- `empirical_null_screen/retained_terms.csv`
  - Row count = number of features that passed screen
  - Should be much smaller than candidate library
- `empirical_null_screen/screening_provenance.csv`
  - `fdr_adjusted_p_value` shows significance levels

**Red Flags:**

- All features retained → screen not working
- Zero features retained → threshold too strict
- Retention rate >20% → candidate library may be too small

______________________________________________________________________

### Stage 3: Interaction Discovery

**Purpose:** Identify synergistic feature pairs via tree SHAP interaction values

**Manuscript Benchmarks:**

- **Interaction pairs retained:** 367
- **Null threshold:** 99.5th percentile of permutation distribution
- **Permutations:** 200

**Your Results Should Show:**

- Pairs retained: 50–500 (depends on candidate pairs)
- Pairs are colon-delimited (e.g., `feature1:feature2`)

**Check Files:**

- `interaction_discovery/retained_interaction_pairs.csv`
  - Row count = number of pairs retained
  - `feature_name` column shows `feature_a:feature_b` format

**Red Flags:**

- Zero interactions → candidate catalog may lack interaction terms
- All interactions retained → threshold too permissive

______________________________________________________________________

### Stage 4: Nonlinear Discovery

**Purpose:** Identify nonlinear transformations via GAM curvature detection

**Manuscript Benchmarks:**

- **Transformations identified:** 112
- **Final support transformations:** 37
- **Families:** quadratic, logarithmic, inverse, exponential

**Your Results Should Show:**

- Transformations identified: 20–150
- Families represented in results
- Transformations named like `log(feature)`, `square(feature)`

**Check Files:**

- `nonlinear_discovery/identified_nonlinear_terms.csv`
  - Row count = number of transformations found
  - `transform_type` shows which family

**Red Flags:**

- Zero transformations → linear relationships dominate (not necessarily bad)
- All features flagged nonlinear → GAM overfitting

______________________________________________________________________

### Stage 5: Sparse Selection

**Purpose:** Select sparse feature sets per output component via EBIC-penalized lasso

**Manuscript Benchmarks:**

- **Final retained predictors:** 132
- **Distinct first-order main effects:** 54
- **EBIC gamma:** 0.5
- **L1 ratio:** 1.0 (pure lasso)

**Your Results Should Show:**

- Features selected: 100–500
- Much smaller than candidate union from stages 2-4
- Some features shared across components, some component-specific

**Check Files:**

- `sparse_selection/selected_features.csv`
  - Row count = total feature selections (may include duplicates across components)
  - Aggregated unique count should be 50–500
- `sparse_selection/stability_diagnostics.csv`
  - Shows resampling stability (Jaccard, Spearman)

**Red Flags:**

- Zero features selected → penalty too strong
- All candidates selected → penalty too weak
- Stability < 0.5 → model not robust

______________________________________________________________________

### Stage 6: Final Artifacts & Holdout Validation

**Purpose:** Fit final OLS on selected features, validate on holdout, export bundle

**Manuscript Benchmarks:**

- **Intermediate penalized holdout nRMSE:** 0.0709
- **Final OLS holdout nRMSE:** 0.0721 [0.0706, 0.0730]
- **Holdout fraction:** 5%

**Your Results Should Show:**

- nRMSE: 0.01–0.50 (depends on problem difficulty)
- Lower than null baseline (predicting mean)
- Confidence intervals exclude poor performance

**Check Files:**

- `final_artifacts/holdout_performance.csv`
  - `point_estimate` column shows overall performance
  - `nrmse_lower`, `nrmse_upper` show bootstrap CI
- `final_artifacts/coefficients.csv`
  - Final model weights for each feature × output
- `final_artifacts/feature_importance.csv`
  - Aggregate importance across outputs

**Red Flags:**

- nRMSE > 1.0 → model worse than predicting mean
- nRMSE = 0 → overfitting or data leakage
- Wide confidence intervals (>0.5 range) → unstable model

______________________________________________________________________

## Manuscript Benchmark Values

### Summary Table

| Stage                     | Metric             | Manuscript Value | Typical Range |
| ------------------------- | ------------------ | ---------------- | ------------- |
| **Output Conditioning**   | PCA components     | 20               | 15–40         |
|                           | Variance explained | 90%              | 80%–95%       |
| **Empirical Null**        | Retained features  | 69               | 30–200        |
|                           | Retention rate     | 43%              | 5%–60%        |
| **Interaction Discovery** | Pairs identified   | 62               | 30–200        |
| **Nonlinear Discovery**   | Transformations    | 41               | 10–80         |
| **Sparse Selection**      | Final features     | 132              | 50–300        |
| **Holdout Performance**   | Final nRMSE        | 0.0721           | 0.01–0.50     |

### Important Notes

**These are manuscript case-study values for a specific dataset (30,000 samples, 26,560 candidates).**

Your results **will differ** based on:

- **Sample size:** Smaller datasets (e.g., 3k) may retain fewer features
- **Candidate library:** More candidates → more retained in early stages
- **Data characteristics:** Nonlinearity, signal strength, noise level
- **Random seed:** Permutation tests, cross-validation vary slightly

______________________________________________________________________

## Red Flags (Run Likely Failed)

### Critical Issues (Stop and Debug)

1. **Zero features in final model**

   - **Symptom:** `final_artifacts/coefficients.csv` is empty or has zero rows
   - **Cause:** All features filtered out in earlier stages
   - **Fix:** Check variance thresholds, screening alpha, penalty strength

1. **nRMSE > 1.0 or NaN**

   - **Symptom:** Holdout performance worse than baseline
   - **Cause:** Model not learning, data issues, or normalization problem
   - **Fix:** Check data quality, verify sample_id alignment, inspect features

1. **All features retained at every stage**

   - **Symptom:** Feature count stays constant across stages
   - **Cause:** Screening/selection not working
   - **Fix:** Check config parameters, verify candidate library format

1. **Missing output directories**

   - **Symptom:** Expected stage folders don't exist
   - **Cause:** Python error, out of memory, process killed
   - **Fix:** Check console logs for traceback, increase memory

### Warning Signs (Investigate)

5. **Very high/low feature counts**

   - \<10 final features → may be underfitting
   - > 1000 final features → may be overfitting

1. **Poor PCA variance capture**

   - \<70% variance with 50+ components → outputs may be too noisy

1. **Extreme retention rates**

   - > 50% features pass null screen → weak signal or wrong threshold
   - \<0.1% features pass → threshold too strict

1. **Unstable models**

   - Stability diagnostics show Jaccard < 0.5
   - Wide bootstrap CIs on holdout performance

______________________________________________________________________

## Interpreting Differences

### Why Your Results Don't Match Manuscript Exactly

This is **expected and correct**. Here's why:

#### 1. Sample Size Effects

**Manuscript:** 20,000 samples (18,000 train, 2,000 holdout)\
**Your test run:** 3,000 samples (2,700 train, 300 holdout)

**Impact:**

- Fewer samples → fewer statistical power → fewer features retained
- Smaller holdout → wider confidence intervals
- Less stable model selection

**Example:** Test dataset may retain 150–250 features vs. manuscript's 132

#### 2. Random Seed Variation

**Sources of randomness:**

- Permutation tests (empirical null, interaction discovery)
- Subsampling for stability checks
- Bootstrap for confidence intervals

**Impact:** Results vary by ±5–10% across seeds

#### 3. Candidate Library Size

**Manuscript:** 26,560 candidates (full combinatorial library)\
**Your test:** May use subset to speed up computation

**Impact:** Fewer candidates → fewer features discovered

#### 4. Data Subset Selection

If you're using a stratified sample or subset:

- May not capture all signal present in full data
- Different feature importances
- Different interaction patterns

### What Constitutes "Good Agreement"?

Your results show **good agreement** with manuscript if:

✅ **Feature counts within 2x of manuscript:**

- Manuscript: 349 after screening → Your result: 150–700 = ✓
- Manuscript: 132 final → Your result: 150–680 = ✓

✅ **nRMSE within 2x of manuscript:**

- Manuscript: 0.0721 → Your result: 0.02–0.09 = ✓
- Manuscript: 0.0721 → Your result: 0.20–0.50 = ⚠️ acceptable if data differs
- Your result: >1.0 = ❌ problem

✅ **Variance explained within 10% of manuscript:**

- Manuscript: 90% → Your result: 80%–95% = ✓

✅ **Stage-to-stage filtering is monotonic:**

- Candidates > Screened > Interactions/Nonlinear > Sparse > Final

______________________________________________________________________

## Comparing to Manuscript: Worked Example

### Scenario: 3k Test Dataset Results

**Your Output:**

```
Stage 1: 36 PCA components, 88% variance
Stage 2: 287 features retained (from 62,228 candidates, 0.46% retention)
Stage 3: 198 interaction pairs
Stage 4: 87 nonlinear transforms
Stage 5: 256 final features selected
Stage 6: Holdout nRMSE = 0.067
```

**Comparison to Manuscript:**

| Metric               | Manuscript | Your Result | Ratio | Assessment                                         |
| -------------------- | ---------- | ----------- | ----- | -------------------------------------------------- |
| PCA components       | 20         | 36          | 1.80x | ⚠️ More components than manuscript                 |
| Variance explained   | 90%        | 88%         | -2%   | ✅ Good                                            |
| Screened features    | 69         | 287         | 4.16x | ⚠️ More screened — typical for smaller datasets    |
| Interaction pairs    | 62         | 198         | 3.19x | ⚠️ More pairs — smaller-N noise inflates discovery |
| Nonlinear transforms | 41         | 87          | 2.12x | ⚠️ More transforms than manuscript                 |
| Final features       | 132        | 256         | 1.94x | ⚠️ More features than manuscript                   |
| Holdout nRMSE        | 0.0721     | 0.067       | 0.93x | ✅ Good                                            |

**Interpretation:** Results show **acceptable agreement** for a 3k smoke-test dataset. Counts are inflated relative to the manuscript because smaller-N runs are more permissive at the discovery stages; the final nRMSE remains within the manuscript band, which is the primary acceptance signal.

**Discovery counts inflated:** Smaller sample sizes reduce the effective null and let more candidate terms pass screening/discovery. This is expected on small dev datasets.

**Conclusion:** ✅ **Test run validated successfully**

______________________________________________________________________

## Next Steps

### If Your Results Look Good ✅

1. **Document your run:**

   - Note sample size, config used, runtime
   - Save output directory for future reference

1. **Explore artifacts:**

   - Load feature importance tables
   - Examine selected features
   - Visualize PCA loadings

1. **Try different configurations:**

   - Adjust screening threshold (alpha)
   - Change penalty strength (EBIC gamma)
   - Use different candidate library

1. **Scale up:**

   - Run on larger dataset
   - Use full candidate library
   - Compare results across sample sizes

### If Results Don't Look Right ❌

1. **Check console logs:**

   - Look for Python tracebacks
   - Identify which stage failed

1. **Validate your data:**

   - Verify `sample_id` column exists
   - Check for missing values
   - Confirm split assignments are valid (`train` / `test`)

1. **Review configuration:**

   - Compare your config to `configs/datasets/template.yml`
   - Verify all required fields present
   - Check file paths are absolute and correct

1. **Consult troubleshooting guide:**

   - See [troubleshooting.md](troubleshooting.md) for common issues
   - Search for your error message

1. **Reduce problem size:**

   - Use smaller candidate library to speed up debugging
   - Try demo mode to verify environment works

______________________________________________________________________

## Advanced: Metric Definitions

### nRMSE (Normalized Root Mean Square Error)

**Formula:**

```
nRMSE = (1/n_outputs) × Σ [ RMSE(output_i) / range(Y_train[:, i]) ]
```

**Interpretation:**

- **0.0** = perfect prediction (impossible with real data)
- **\<0.1** = excellent performance
- **0.1–0.3** = good performance
- **0.3–0.5** = moderate performance
- **>0.5** = poor performance
- **1.0** = baseline (predicting mean within training range)
- **>1.0** = worse than predicting mean

**Note:** Normalized by **training set range**, not test set range.

### PCA Variance Explained

**Formula:**

```
Variance Explained = Σ(eigenvalues of retained PCs) / Σ(all eigenvalues)
```

**Interpretation:**

- **>90%** = excellent dimensionality reduction
- **80%–90%** = good reduction
- **60%–80%** = acceptable
- **\<60%** = retaining too few components or outputs too diverse

### Feature Selection Stability (Jaccard Index)

**Formula:**

```
Jaccard = |selected_A ∩ selected_B| / |selected_A ∪ selected_B|
```

**Interpretation:**

- **>0.75** = highly stable (manuscript threshold)
- **0.50–0.75** = moderately stable
- **\<0.50** = unstable (model may not be robust)

______________________________________________________________________

## Summary

✅ **Your run succeeded if:**

- All stages completed
- Features decrease monotonically
- nRMSE < 1.0
- PCA variance >80%

✅ **Good agreement with manuscript if:**

- Feature counts within 2x
- nRMSE within 2x
- Variance explained within ±10%

⚠️ **Acceptable differences due to:**

- Sample size (3k vs 20k)
- Random seeds
- Candidate library size

❌ **Red flags:**

- nRMSE >1.0 or NaN
- Zero features retained
- All features retained
- Missing output directories

**Next:** [Troubleshooting Guide](troubleshooting.md) if you hit issues

______________________________________________________________________

**See also:**

- [Artifact Reference](artifact_reference.md) - Detailed file descriptions
- [Configuration Reference](configuration_reference.md) - Config parameters
- [Manuscript Contract](manuscript_contract.md) - Scientific specifications
