# Config-Driven Manuscript Reproduction Workflow - Summary

## What Was Created

### 1. Main Reproduction Script

**Location**: `scripts/run_manuscript_reproduction.py`

A clean, standalone Python script that runs the full manuscript reproduction workflow using config files.

**Usage**:

```bash
pixi run manuscript-reproduce
pixi run manuscript-reproduce --output-dir artifacts/my-run
```

### 2. Data Preprocessing Script

**Location**: `scripts/preprocess_multiindex_data.py`

Converts MultiIndex data (scenario, run_id) to the required format with `sample_id` column.

**Usage**:

```bash
pixi run manuscript-preprocess \
  --input-x /path/to/raw_X.parquet \
  --input-y /path/to/raw_Y.parquet \
  --output-x artifacts/preprocessed/X.parquet \
  --output-y artifacts/preprocessed/Y.parquet
```

### 3. Pixi Tasks

**Location**: `pixi.toml`

Added two convenient Pixi tasks:

- `manuscript-reproduce`: Run with optimal thread settings
- `manuscript-preprocess`: Preprocess data with MultiIndex

### 4. Configuration Infrastructure

**Location**: `configs/`

- `configs/local/manuscript_paths.local.yml`: User-specific data paths (already existed, now documented)
- `configs/datasets/README.md`: Documentation for dataset configurations
- Existing configs automatically used by the runtime system

### 5. Documentation

**Created Files**:

- `docs/RUNNING_MANUSCRIPT_REPRODUCTION.md`: Complete user guide
- `MANUSCRIPT_QUICK_START.md`: Quick reference at repo root
- `configs/datasets/README.md`: Dataset configuration guide

### 6. Preprocessed Real Data

**Location**: `artifacts/preprocessed_real_data/`

- `sample_7500.X.parquet`: Preprocessed input matrix with `sample_id`
- `sample_7500.Y.parquet`: Preprocessed output matrix with `sample_id`

## How to Use

### Simple One-Liner (Your Request)

```bash
pixi run manuscript-reproduce --output-dir artifacts/run-$(date +%Y%m%d-%H%M)
```

### With Different Datasets

#### Real Data (Current Setup)

```bash
# Already configured in configs/local/manuscript_paths.local.yml
pixi run manuscript-reproduce
```

#### Your Own Data

1. Preprocess if needed:

   ```bash
   pixi run manuscript-preprocess \
     --input-x /path/to/your_X.parquet \
     --input-y /path/to/your_Y.parquet \
     --output-x artifacts/my_data/X.parquet \
     --output-y artifacts/my_data/Y.parquet
   ```

1. Update `configs/local/manuscript_paths.local.yml` to point to your files

1. Run:

   ```bash
   pixi run manuscript-reproduce --output-dir artifacts/my-analysis
   ```

## Configuration System

The workflow uses a layered config system:

1. **Base configs** (tracked in git):

   - `configs/manuscript_runtime.yml`: Runtime settings
   - `configs/manuscript_case_study.yml`: Case study parameters
   - `configs/manuscript_data_contract.yml`: Required artifacts
   - `configs/manuscript_paths.template.yml`: Template paths

1. **Local override** (user-specific, not in git):

   - `configs/local/manuscript_paths.local.yml`: Your actual data paths

The `build_manuscript_notebook_context()` function automatically loads all these configs in the correct order, with local overrides taking precedence.

## Required Data Format

All data matrices must have `sample_id` as the first column:

```
   sample_id         feature1  feature2  ...
0  scenario_A_001   0.123     4.567     ...
1  scenario_A_002   0.234     5.678     ...
```

Required metadata artifacts:

- `input_metadata.parquet`: Describes input features
- `output_metadata.parquet`: Describes output responses
- `manuscript_feature_catalog.parquet`: Feature engineering specs (must include interactions)
- `fixed_holdout_assignments.parquet`: Train/holdout split

## Workflow Stages

The script runs all 6 manuscript stages:

1. **Output Conditioning**: Variance/SNR filtering, PCA (39 components, 90% variance)
1. **Empirical Null Screening**: 200 permutations, BH q=0.10, ~349 retained terms
1. **Interaction Discovery**: Tree SHAP-based, residualized products, ~367 pairs
1. **Nonlinear Discovery**: GAM curvature detection, parametric replacement, ~112 transformations
1. **Sparse Selection**: EBIC L1-penalized per-component, stability filtering, ~340 features
1. **Final Artifacts**: OLS inference, HC3 Wald intervals, tables, figures, audit

## Output Structure

```
artifacts/your-run/
├── output_conditioning/
│   ├── pca_scores.csv (30k samples × 39 PCs)
│   ├── pca_loadings.csv
│   ├── pca_explained_variance.csv
│   └── output_conditioning_summary.csv
├── empirical_null_screen/
│   ├── retained_terms.csv (~349 features)
│   ├── feature_screening_statistics.csv
│   ├── permutation_null_summary.csv
│   └── empirical_null_provenance.csv
├── interaction_discovery/
│   ├── discovered_interactions.csv (~367 pairs)
│   └── interaction_statistics.csv
├── nonlinear_discovery/
│   ├── nonlinear_transformations.csv (~112)
│   └── curvature_diagnostics.csv
├── sparse_selection/
│   ├── selected_features.csv (~340)
│   ├── stability_metrics.csv
│   └── ebic_selection_summary.csv
├── final_manuscript_artifacts/
│   ├── final_coef_matrix_standardized.csv
│   ├── final_coef_matrix_raw_scale.csv
│   ├── holdout_nrmse_summary.csv
│   ├── manuscript_tables/
│   └── manuscript_figures/
└── audit/
    ├── artifact_inventory.csv
    ├── audit_summary.csv (QA status: pass/fail)
    ├── metric_checks.csv
    └── audit_provenance.csv
```

## Next Steps

### To Run Full Real Data Manuscript

```bash
# Data is already preprocessed and configured
pixi run manuscript-reproduce --output-dir artifacts/full-real-run-$(date +%Y%m%d)
```

This will take 10-30 minutes depending on your system (permutations, bootstrap, GAM fitting).

### To Add a New Dataset

1. Prepare X, Y matrices with `sample_id` column
1. Create metadata files (see `docs/final_scripts_from_hpc/.../postfit_diagnostics/` for examples)
1. Update `configs/local/manuscript_paths.local.yml`
1. Run: `pixi run manuscript-reproduce --output-dir artifacts/new-dataset`

### To Run the Toy Example

The original `examples/end_to_end_reproducibility.py` still works for toy data:

```bash
PYTHONPATH=src pixi run python examples/end_to_end_reproducibility.py \
  --output-dir artifacts/toy-example \
  --run-manuscript-chain \
  --manuscript-output-dir artifacts/toy-example/manuscript
```

(Note: Toy data doesn't have full metadata, so manuscript chain will fail at interaction discovery)

## Documentation Index

- `MANUSCRIPT_QUICK_START.md` - Quick reference (repo root)
- `docs/RUNNING_MANUSCRIPT_REPRODUCTION.md` - Complete guide
- `configs/datasets/README.md` - Dataset configuration details
- `scripts/run_manuscript_reproduction.py` - Script with inline docs
- `scripts/preprocess_multiindex_data.py` - Preprocessing script docs

## Troubleshooting

See `docs/RUNNING_MANUSCRIPT_REPRODUCTION.md` section "Troubleshooting" for:

- Missing PyArrow
- Missing sample_id column
- Feature catalog issues
- Metadata mismatches
- Performance tuning
