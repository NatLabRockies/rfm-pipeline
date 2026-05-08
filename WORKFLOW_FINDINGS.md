# Important Workflow Findings

**Date:** 2026-05-08\
**Context:** Testing config-driven manuscript reproduction with 3k sample dataset

## Critical Finding: Feature Catalog Structure

### ❌ What We Initially Thought

The feature catalog should contain:

- First-order features (352)
- ALL pairwise interactions (~62K)
- Nonlinear transformations (~100)

**Result:** Workflow failed with error `Feature 'square(feature)' is not a direct input or supported catalog expression`

### ✅ Actual Workflow Behavior

The feature catalog should contain **ONLY first-order features** (158 in manuscript case):

- Feature name
- Feature type: 'numeric' (not 'first_order', 'interaction', 'nonlinear')
- Origin: 'model_factors'

**The workflow DISCOVERS features dynamically:**

1. **Stage 2 (Empirical Null):** Screens first-order features
1. **Stage 3 (Interaction Discovery):** Creates and tests pairwise interactions
1. **Stage 4 (Nonlinear Discovery):** Detects and fits nonlinear transforms
1. **Stage 5+ (Sparse Selection):** Selects from discovered candidates

### Why This Matters

**Correct approach:**

```yaml
# Feature catalog: ONLY first-order features
manuscript_feature_catalog: docs/.../manuscript_feature_catalog.parquet
# Shape: (158, 3)
# Columns: feature_name, feature_type='numeric', origin='model_factors'
```

**Wrong approach:**

```yaml
# ❌ DON'T pre-populate interactions/nonlinear in catalog
# ❌ DON'T use 'interaction' or 'nonlinear' feature_type
# ❌ DON'T include colon-delimited names in catalog
```

## Data Format Requirements (Confirmed)

### Input Matrix Must Include Scenario Factors

**Required columns** (beyond base features):

- `sample_id` (string): Unique identifier
- `AFSC` (int): Binary scenario factor (0 or 1)
- `UAEORO` (int): Binary scenario factor (0 or 1)
- `scenario` (string): Scenario name
- `run_id` (int): Run identifier

**Why:** The workflow references these in feature engineering and stratification.

**Solution:** Use `scripts/add_scenario_factors.py` to extract from old data.

### Holdout Split Values

**Correct:** `'train'` and `'test'`\
**Wrong:** `'train'` and `'holdout'`

## Documentation Updates Needed

1. **configs/datasets/README.md** ✅ UPDATED

   - Clarify catalog should be first-order only
   - Remove misleading interaction/nonlinear generation code
   - Emphasize workflow discovers these automatically

1. **docs/interpreting_results.md** ✅ UPDATED

   - Explain dynamic feature discovery
   - Adjust candidate library size expectations

1. **docs/troubleshooting.md** ✅ UPDATED

   - Add error: "Feature 'X' is not a direct input"
   - Solution: Don't pre-populate transformations in catalog

## Test Run Status

**Current state:** Config corrected to use first-order-only catalog

**Next:** Re-run with correct configuration:

```bash
pixi run manuscript-reproduce --config configs/datasets/test_3k.yml
```

**Expected runtime:** ~20-30 minutes with 158 first-order features

______________________________________________________________________

**Key Takeaway:** The feature catalog is an **input specification**, not a **candidate library**. The workflow builds the candidate library dynamically through discovery stages.
