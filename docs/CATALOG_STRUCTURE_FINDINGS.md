# Feature Catalog Structure - Critical Findings

## Date: 2026-05-08

## Summary

The feature catalog structure was unclear from existing documentation. Through iterative testing with the 3k dataset, we discovered the correct format.

## Wrong Approaches Tried

### Attempt 1: manuscript_feature_catalog.parquet (158 features)

- **File**: `docs/final_scripts_from_hpc/model_artifacts/final_ols_model/postfit_diagnostics/manuscript_feature_catalog.parquet`
- **Problem**: This contains SELECTED features after the workflow completed, not INPUT specifications
- **Result**: Failed at stage 2 - features in catalog not in input matrix (97 missing)
- **Error**: `Feature 'WW.Alt Price Sensi multiplier[ManureToHTL]' is not a direct input`

### Attempt 2: First-order features only (352 features)

- **File**: `artifacts/actual_input_feature_catalog.parquet`
- **Structure**: 352 rows, all `feature_type='numeric'`
- **Problem**: Interaction discovery stage expects catalog with interaction candidates pre-specified
- **Result**: Failed at stage 3 - "feature_catalog does not contain any two-factor interaction candidates"

## Correct Approach

### Full catalog with interactions (62,128 features)

- **File**: `artifacts/full_feature_catalog_with_interactions.parquet`
- **Structure**:
  - 352 first-order features with `feature_type='numeric'`
  - 61,776 pairwise interactions with `feature_type='interaction'`
  - Interaction names formatted as `"feature1:feature2"` (colon-delimited)
- **Generation**: All combinations of 352 input features = C(352,2) = 61,776 pairs
- **Result**: Successfully started all 6 workflow stages

## Catalog Requirements

For a custom dataset with N input features:

1. **First-order features** (N rows):

   ```python
   {'feature_name': 'feature_A', 'feature_type': 'numeric', 'origin': 'model_factors'}
   ```

1. **Pairwise interactions** (N\*(N-1)/2 rows):

   ```python
   {'feature_name': 'feature_A:feature_B', 'feature_type': 'interaction', 'origin': 'model_factors'}
   ```

1. **Total catalog size**: N + N\*(N-1)/2 features

## Workflow Behavior

The workflow does NOT generate interaction pairs dynamically. Instead:

1. **Stage 2 (Screening)**: Evaluates all first-order features from catalog
1. **Stage 3 (Interaction Discovery)**: Evaluates interaction pairs listed in catalog
1. **Stage 4 (Nonlinear Discovery)**: Tests nonlinear transforms of screened features
1. **Stage 5-6**: Final selection and OLS fitting

The catalog is a **candidate specification**, not a dynamic generation rule.

## Performance Note

With 61,776 interaction candidates on 3000 samples:

- Stage 1 (PCA): ~3-5 minutes
- Stage 2 (Screening): ~8-12 minutes
- Stage 3 (Interactions): ~5-8 minutes
- Total expected: 20-30 minutes

The workflow generates ~100K PerformanceWarning messages about DataFrame fragmentation. These are cosmetic and do not affect results.

## Code Reference

Interaction candidate extraction (src/rfm_pipeline/manuscript_stages.py:4169):

```python
def _interaction_candidate_pairs(feature_catalog: pd.DataFrame):
    """Return two-factor interaction candidates from a feature catalog."""
    if "feature_type" in feature_catalog.columns:
        candidate_rows = feature_catalog.loc[
            feature_catalog["feature_type"].astype(str).str.lower() == "interaction"
        ]
    # Parse colon-delimited names like "feature1:feature2"
    for feature_name in candidate_rows["feature_name"]:
        factors = feature_name.split(":")
        if len(factors) != 2:
            continue
        candidates.append((feature_name, factors[0], factors[1]))
```

## Documentation Updates Needed

1. **configs/datasets/README.md** - Add explicit catalog structure requirements
1. **docs/troubleshooting.md** - Add "feature_catalog does not contain any two-factor interaction candidates" error
1. **WORKFLOW_FINDINGS.md** - Update with correct catalog format
1. **Example script** - Add catalog generation utility

## Validation Status

- ✅ Catalog structure identified
- ✅ Test config updated
- ⏳ 3k test run in progress (Stage 1 complete, Stages 2-6 pending)
- ⏳ Results validation against manuscripts (pending run completion)
