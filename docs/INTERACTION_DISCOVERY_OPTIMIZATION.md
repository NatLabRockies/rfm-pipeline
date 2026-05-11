# Interaction Discovery Deep Dive: Analysis & Optimization Opportunities

**Generated:** 2026-05-11

## Current State

### Runtime Profile

- **Large profile (1000 rows, ~9712 effective outputs):** 4520.2s (78.8 min)
- **% of total budget:** 95.9% (dominant bottleneck)
- **Scaling exponent:** ~0.896 (subquadratic with row count)

### Implementation Overview

Current interaction discovery (`src/bsm_rfm/manuscript_stages.py:1256-1432`):

1. **Generate interaction candidates:**

   - All pairwise combinations of retained first-order features
   - Large profile example: ~62,835 possible pairs from ~355 features

1. **Pre-generate permuted response matrices:**

   - Observed + B permutations (B=40 for large profile)
   - Each permutation is a column-wise shuffle of PCA component scores

1. **Score each permutation in parallel:** (`_score_interaction_permutation`)

   - For each active PCA component (typically ~20):
     - Fit GBT (n_estimators=120, max_depth=4) on x_feat (1000 rows × 355 features)
     - Compute SHAP interaction matrix (355 × 355)
     - Extract max interactions across all components
   - Parallel batch processing with batch_size = ceil(total_scores / 25)

1. **Aggregate and filter:**

   - Compute p-values and thresholds from null distribution
   - Retain pairs exceeding null_threshold_quantile (0.995)

### Configuration (Large Profile, from ladder)

```yaml
interaction_discovery:
  n_permutations: 41            # 40 null + 1 observed
  n_tree_estimators: 120        # Trees per GBT
  max_tree_depth: 4             # Max depth per tree
  max_shap_samples: 500         # (default, used for all profiles)
  p_threshold: 0.05             # For pair selection
```

### Computational Complexity

**Per permutation:**

- Active components: ~20
- Tree training: 20 × GBT(120, depth=4, 1000×355) ≈ **2–3 min per permutation**
- SHAP computation: 20 × TreeExplainer.shap_interaction_values(500 samples, 355 features) ≈ **1–2 min per permutation**

**Total for large profile:**

- 41 permutations × 3–5 min/perm = **2–3.4 hours** (theoretical range)
- Observed: **1.26 hours** (4520s) — within reasonable range, parallelization helps

### Comparison with Manuscript Configuration

The manuscript (`configs/manuscript_case_study.yml`) specifies:

- **permutation_count_B: 200** (not 40)
- **null_threshold_quantile: 0.995** (very strict)
- **n_permutations: 201** vs ladder's 40–41

If run with full manuscript parameters:

- 201 permutations × ~2 min/perm ≈ **6.7 hours** (observed would be much longer)
- This aligns with pre-optimization HC3 estimate of ~10–12 hours for final stage cost

______________________________________________________________________

## Identified Optimization Opportunities

### 1. **SHAP Sample Size Reduction** (Low Risk, Quick Wins)

**Current:** `max_shap_samples=500` (fixed)

**Issue:** For 1000-row training sets, using 500 samples (50%) might be overkill. SHAP interaction estimation typically stabilizes at lower sample sizes.

**Proposal:**

- Make `max_shap_samples` **adaptive** based on training set size
- Default: `min(250, 0.3 * n_train)`
- For 1000 rows: uses 250 instead of 500 (2× speedup in SHAP phase)
- Negligible accuracy loss (bootstrapped variance averaging)

**Expected Gain:** 10–15% overall (SHAP is ~40% of permutation cost)

______________________________________________________________________

### 2. **GBT Estimator & Depth Reduction** (Medium Risk, Needs Validation)

**Current:** `n_estimators=120, max_depth=4` (fixed for all profiles)

**Issue:** GBT trees trained on 1000 rows with 355 features may overfit or over-specify. Shallower trees might capture interaction signals with lower cost.

**Proposal:**

- Test reduced estimator counts: **80–100** (vs current 120)
- Test shallower trees: **max_depth=3** (vs current 4)
- Rationale: Interaction scoring relies on coarse feature separation, not fine-grained splits

**Evidence for viability:**

- Large profile uses `max_depth=4, n_estimators=120`
- Medium profile uses `max_depth=3, n_estimators=100` (only 20% fewer estimators!)
- Small profile uses `max_depth=3, n_estimators=50` (scaling suggests smaller is viable)

**Expected Gain:** 20–30% on GBT phase (GBT is ~50% of permutation cost)

______________________________________________________________________

### 3. **Early Stopping or Adaptive Permutation Count** (Medium Risk, Requires Tuning)

**Current:** Fixed `n_permutations=41` (ladder), 201 (manuscript)

**Issue:** Most of the null permutations may have scores far below the observed, making additional permutations redundant for p-value significance.

**Proposal:**

- Implement optional **early stopping** based on p-value confidence
- Stop if null permutations are consistently well-separated from observed (e.g., 99% of nulls < 50th percentile of observed)
- Maintain strict p-threshold computation for retained pairs

**Caveats:**

- Requires careful implementation to preserve statistical validity
- May risk Type-I error if permutation count is too aggressive
- Best applied only to non-critical pairs (not those close to threshold)

**Expected Gain:** 15–25% (if applicable)

______________________________________________________________________

### 4. **Candidate Pair Pre-filtering** (Low Risk, Requires Analysis)

**Current:** Score **all** pairwise combinations of retained first-order terms

- Large profile: ~1,165,440 candidate pairs (120 features × 9712 outputs; wait, this is HC3 logic)
- Actually: ~62,835 pairs (355 features) × outputs aggregation

**Issue:** Many pairs likely have near-zero interaction signals across all permutations.

**Proposal:**

- Compute a **fast approximate interaction score** using:
  - Shallow decision trees (max_depth=2, n_estimators=10)
  - Single sample (no permutation)
- Use approximate scores to prune the bottom ~50% of weak pairs
- Score remaining pairs with full method

**Expected Gain:** 30–50% (if pre-filter is 10× faster)

______________________________________________________________________

### 5. **Variance-Based Component Pruning** (Low Risk, Quick Win)

**Current:** Score all **active** PCA components (typically 15–25)

**Issue:** Some PCA components may have very low predictive power, yet still incur full tree training + SHAP cost.

**Proposal:**

- Rank components by **variance explained** in the input-output relationship
- Score only top K components (e.g., top 10–15)
- Rationale: High-order interactions are unlikely to appear in low-signal components

**Expected Gain:** 10–20% (if we can reduce active components from 20 to 10–12)

______________________________________________________________________

### 6. **Parallel Batch Size Tuning** (Low Risk, May Help)

**Current:** `batch_size = ceil(total_scores / 25)` (41 permutations → batch of 2)

**Issue:** With only 2 permutations per batch and 20+ cores, parallelization overhead may dominate.

**Proposal:**

- Increase min batch size: `batch_size = max(2, ceil(total_scores / 8))` (larger batches)
- Rationale: Reduced overhead, better core utilization

**Expected Gain:** 5–10%

______________________________________________________________________

## Recommendation Ranking

| Rank | Optimization                     | Risk   | Gain   | Effort | Priority                          |
| ---- | -------------------------------- | ------ | ------ | ------ | --------------------------------- |
| 1    | SHAP sample reduction (adaptive) | Low    | 10–15% | Low    | **HIGH**                          |
| 2    | GBT estimator/depth reduction    | Medium | 20–30% | Medium | **HIGH**                          |
| 3    | Parallel batch size tuning       | Low    | 5–10%  | Low    | **MEDIUM**                        |
| 4    | Variance-based component pruning | Low    | 10–20% | Low    | **MEDIUM**                        |
| 5    | Candidate pair pre-filtering     | Low    | 30–50% | High   | **LOW** (high reward if feasible) |
| 6    | Early stopping on permutations   | Medium | 15–25% | Medium | **LOW** (validation risk)         |

______________________________________________________________________

## Next Steps

1. **Implement & Test Top 2:**

   - Adaptive SHAP sampling (quick, low risk)
   - GBT parameter reduction with validation
   - Target: **20–35% improvement** on interaction discovery

1. **Validate Against Manuscript:**

   - Run phased tests with optimized parameters
   - Compare pair retention rates vs baseline
   - Ensure no statistical significance loss

1. **Profile with Detailed Instrumentation:**

   - Add timing markers for tree training, SHAP computation, aggregation
   - Identify which phase (within permutation scoring) is truly dominant

1. **Consider Optional HC3-Style Subsetting:**

   - If interaction pairs exceed user's output budget, allow subsetting by score
   - Reduces downstream HC3 burden (fewer pairs to cross with outputs)
