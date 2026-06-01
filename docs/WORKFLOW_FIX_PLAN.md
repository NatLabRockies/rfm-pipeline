# CRITICAL: Workflow Implementation Gap - Fix Plan

**Date**: 2026-05-08\
**Priority**: P0 - BLOCKING\
**Status**: Planning

## Problem Statement

The current workflow implementation does NOT match the manuscript methodology. This is a critical gap that must be fixed before any validation or documentation work can be considered complete.

### Manuscript Workflow (SOURCE OF TRUTH)

From `docs/manuscripts/jds_bsm_v5_editor_revised.tex` lines 397, 437:

1. **Stage 2 - Empirical Null Screening**:

   - Input: 160 first-order features
   - Screen with permutation null on PCA components
   - Output: 63 retained first-order features

1. **Stage 3 - Interaction Discovery**:

   - Generate C(63, 2) ≈ 1,953 pairwise interaction candidates
   - Use TreeSHAP interaction values to score pairs
   - Use permutation null to select significant pairs
   - Output: 248 retained interaction pairs

1. **Stage 4 - Nonlinear Discovery**:

   - Generate nonlinear transforms (inverse, log, quadratic, sqrt) for retained features
   - Use GAM curvature detection to identify significant transforms
   - Output: 41 retained nonlinear transforms

1. **Final Feature Matrix**: 352 features (63 + 248 + 41)

### Current Implementation (INCORRECT)

**File**: `src/rfm_pipeline/manuscript_stages.py`

**Stage 2** (line 914: `screen_manuscript_empirical_null_terms`):

- Materializes ALL features from catalog (first-order + interactions + nonlinear)
- Screens everything together
- This is why 62K features were screened instead of 160

**Stage 3** (line 1140: `discover_manuscript_interactions`):

- Line 1187: `candidates = _interaction_candidate_pairs(feature_catalog)`
- Expects interactions PRE-SPECIFIED in catalog with `feature_type='interaction'`
- Does NOT generate pairs from retained first-order features

**Stage 4** (line 1379: `discover_manuscript_nonlinear_transformations`):

- Similar issue - expects nonlinear candidates pre-specified
- Does NOT generate transforms from retained features

## Root Cause

The workflow was implemented to operate on a **static catalog** rather than a **dynamic feature library** that expands based on screening results.

This fundamentally breaks the manuscript methodology where:

- Stage 2 reduces dimensionality (160 → 63)
- Stage 3 expands with interactions from reduced set
- Stage 4 expands with nonlinear from reduced set

## Impact

**Cannot validate workflow accuracy** because:

- Feature counts wrong (62K vs 160 at screening)
- Retention rates wrong (can't compare 61K retained vs 63 expected)
- Interaction discovery meaningless (not generated from screened features)
- Nonlinear discovery not implemented correctly

**Cannot replicate manuscript results** because:

- Different features being evaluated
- Different statistical power
- Different null distributions

## Fix Requirements

### 1. Refactor Stage 2: Screen First-Order Only

**Current**:

```python
def screen_manuscript_empirical_null_terms(
    input_matrix, feature_catalog, holdout_assignments, pca_scores, spec
):
    design = build_manuscript_feature_design(input_matrix, feature_catalog)  # Materializes ALL
    # ... screen everything
```

**Required**:

```python
def screen_manuscript_empirical_null_terms(
    input_matrix, feature_catalog, holdout_assignments, pca_scores, spec
):
    # Filter catalog to ONLY first-order features
    first_order_catalog = feature_catalog[feature_catalog["feature_type"] == "numeric"]
    design = build_manuscript_feature_design(input_matrix, first_order_catalog)
    # ... screen only first-order features
    # Return retained first-order feature names
```

### 2. Refactor Stage 3: Generate Interactions Dynamically

**Current**:

```python
def discover_manuscript_interactions(
    input_matrix, feature_catalog, holdout_assignments, pca_scores, retained_terms, spec
):
    candidates = _interaction_candidate_pairs(feature_catalog)  # Reads from catalog
    # ... score with SHAP
```

**Required**:

```python
def discover_manuscript_interactions(
    input_matrix, feature_catalog, holdout_assignments, pca_scores, retained_terms, spec
):
    # Generate ALL pairwise interactions from retained_terms
    retained_features = retained_terms["feature_name"].tolist()
    candidates = _generate_pairwise_interactions(retained_features)  # NEW FUNCTION
    # ... score with SHAP
    # Return retained interaction pairs
```

### 3. Refactor Stage 4: Generate Nonlinear Dynamically

**Current**:

```python
def discover_manuscript_nonlinear_transformations(
    input_matrix, feature_catalog, holdout_assignments, pca_scores, retained_terms, spec
):
    # Expects nonlinear candidates in catalog
```

**Required**:

```python
def discover_manuscript_nonlinear_transformations(
    input_matrix, feature_catalog, holdout_assignments, pca_scores, retained_terms, spec
):
    # Generate nonlinear transforms for retained first-order features
    retained_features = retained_terms[retained_terms["feature_type"] == "numeric"]["feature_name"]
    candidates = _generate_nonlinear_transforms(input_matrix, retained_features, spec)  # NEW
    # ... score with GAM curvature
    # Return retained transforms
```

### 4. Update Downstream Stages

**Stage 5 (LASSO) and Stage 6 (Final OLS)** need to work with:

- Retained first-order features (from Stage 2)
- Retained interactions (from Stage 3)
- Retained nonlinear (from Stage 4)

Not from a static catalog.

## Implementation Plan

### Phase 1: Analysis and Design (2-3 hours)

1. **Read manuscript sections** on each stage methodology
1. **Read archived HPC scripts** to understand original implementation
1. **Document exact algorithms** for each stage
1. **Design new function signatures** and data flow
1. **Identify breaking changes** for downstream code

### Phase 2: Core Refactoring (6-8 hours)

1. **Create new helper functions**:

   - `_generate_pairwise_interactions(features: list[str]) -> list[tuple]`
   - `_generate_nonlinear_transforms(input_matrix, features, transforms=['inverse', 'log', 'quadratic', 'sqrt']) -> DataFrame`
   - `_filter_first_order_catalog(catalog) -> DataFrame`

1. **Refactor Stage 2**:

   - Filter catalog to first-order only
   - Update return type to include retained feature metadata
   - Ensure downstream compatibility

1. **Refactor Stage 3**:

   - Generate interaction candidates from `retained_terms`
   - Keep SHAP scoring logic unchanged
   - Update return type to include interaction feature definitions

1. **Refactor Stage 4**:

   - Generate nonlinear candidates from retained first-order features
   - Implement GAM curvature detection (if not already present)
   - Update return type to include transform definitions

1. **Update Stage 5 and 6**:

   - Build design matrix from:
     - Retained first-order features
     - Retained interactions (materialized)
     - Retained nonlinear (materialized)
   - Not from static catalog

### Phase 3: Integration and Testing (4-6 hours)

1. **Update workflow orchestration** (`run_manuscript_reproduction_stage_chain`)

1. **Test on 300-sample dataset**:

   - Verify Stage 2 screens only first-order
   - Verify Stage 3 generates correct number of pairs
   - Verify Stage 4 generates transforms
   - Verify final feature count matches expected

1. **Test on full 20K dataset** (if time permits)

1. **Compare to manuscript benchmarks**

### Phase 4: Documentation and Validation (2-3 hours)

1. Update workflow documentation
1. Update catalog generation utility
1. Document changes in `WORKFLOW_FINDINGS.md`
1. Update validation reports

## Timeline

**Total Estimated Effort**: 14-20 hours

**Recommended Approach**:

- Focus on correctness over speed
- Test each stage individually before integration
- Use 300-sample dataset for fast iteration
- Validate against manuscript benchmarks continuously

## Success Criteria

✅ Stage 2 screens ONLY first-order features (should screen ~352, retain ~63 equivalent)\
✅ Stage 3 generates C(n_retained, 2) interaction pairs dynamically\
✅ Stage 3 retains ~248/1953 pairs (12.7% retention rate)\
✅ Stage 4 generates nonlinear transforms dynamically\
✅ Stage 4 retains ~41 transforms\
✅ Final feature matrix: ~352 features (63 + 248 + 41)\
✅ Holdout nRMSE comparable to manuscript (considering sample size)

## Risk Assessment

**High Risk**:

- Breaking changes to core workflow functions
- Downstream notebooks may depend on current behavior
- Test suite may assume current catalog structure

**Mitigation**:

- Create feature branch for refactoring
- Keep old functions as `_legacy_*` during transition
- Update tests incrementally
- Maintain backward compatibility where possible

## Next Steps

1. **STOP all other work** - this is P0 blocking
1. **Create detailed technical spec** for each function change
1. **Set up 300-sample test dataset** for fast iteration
1. **Begin Phase 1** - analysis and design
1. **Update manifest** to reflect this priority

## References

- Manuscript: `docs/manuscripts/jds_bsm_v5_editor_revised.tex` lines 69, 397, 437, 560-562
- Current code: `src/rfm_pipeline/manuscript_stages.py`
- Old scripts: `docs/final_scripts_from_hpc/` (need to review)
- Workflow findings: `docs/3K_TEST_VALIDATION_REPORT.md` (documents the misunderstanding)
