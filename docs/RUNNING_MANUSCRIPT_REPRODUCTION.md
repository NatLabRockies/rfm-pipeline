# Running Manuscript Reproduction

This guide explains how to run the manuscript reproduction workflow against different datasets.

## Quick Start

### Run with Local Config

The simplest way to run manuscript reproduction is using your local configuration:

```bash
pixi run manuscript-reproduce
```

This uses `configs/local/manuscript_paths.local.yml` to locate your data files.

### Run with Custom Output Directory

```bash
pixi run manuscript-reproduce --output-dir artifacts/my-run-2024
```

### Run with Explicit PYTHONPATH

If you need explicit control:

```bash
PYTHONPATH=src pixi run manuscript-reproduce --output-dir artifacts/test-run
```

## Configuration

### Required Artifacts

The manuscript reproduction workflow requires these data artifacts:

1. **Input Matrix** (`case_study_input_matrix`): Features/predictors with `sample_id` column
1. **Output Matrix** (`case_study_output_matrix`): Responses with `sample_id` column
1. **Input Metadata** (`input_metadata`): Describes input features
1. **Output Metadata** (`output_metadata`): Describes output responses
1. **Feature Catalog** (`manuscript_feature_catalog`): Feature engineering specifications
1. **Holdout Assignments** (`fixed_holdout_assignments`): Train/test split

### Setting Up Local Configuration

Create or edit `configs/local/manuscript_paths.local.yml`:

```yaml
# Point to your data files
case_study_input_matrix: /path/to/your/X.parquet
case_study_output_matrix: /path/to/your/Y.parquet
input_metadata: /path/to/your/input_metadata.parquet
output_metadata: /path/to/your/output_metadata.parquet
manuscript_feature_catalog: /path/to/your/feature_catalog.parquet
fixed_holdout_assignments: /path/to/your/holdout_assignments.parquet
output_root: /path/to/output/directory
```

## Data Preprocessing

### If Your Data Has MultiIndex

If your raw data uses a MultiIndex (e.g., `scenario` and `run_id`), preprocess it first:

```bash
pixi run manuscript-preprocess \
  --input-x /path/to/raw_X.parquet \
  --input-y /path/to/raw_Y.parquet \
  --output-x artifacts/preprocessed/X.parquet \
  --output-y artifacts/preprocessed/Y.parquet
```

Then update `configs/local/manuscript_paths.local.yml` to point to the preprocessed files.

### Required Data Format

Both X and Y matrices must have a `sample_id` column as the first column:

```
   sample_id         feature1  feature2  ...
0  scenario_A_001   0.123     4.567     ...
1  scenario_A_002   0.234     5.678     ...
...
```

## Workflow Stages

The manuscript reproduction runs these stages:

1. **Output Conditioning**: Variance/SNR filtering, PCA dimensionality reduction
1. **Empirical Null Screening**: Permutation-based feature screening
1. **Interaction Discovery**: Residualized product interaction terms
1. **Nonlinear Discovery**: GAM-based curvature detection and parametric replacement
1. **Sparse Selection**: L1-penalized per-component models with stability filtering
1. **Final Artifacts**: OLS inference, tables, figures, and audit reports

## Output Structure

After running, the output directory contains:

```
artifacts/my-run/
├── output_conditioning/
│   ├── pca_scores.csv
│   ├── pca_loadings.csv
│   └── ...
├── empirical_null_screen/
│   ├── retained_terms.csv
│   ├── feature_screening_statistics.csv
│   └── ...
├── interaction_discovery/
├── nonlinear_discovery/
├── sparse_selection/
├── final_manuscript_artifacts/
│   ├── final_coef_matrix.csv
│   ├── manuscript_tables/
│   └── manuscript_figures/
└── audit/
    ├── artifact_inventory.csv
    ├── audit_summary.csv
    └── metric_checks.csv
```

## Performance Tuning

### Memory Efficiency

For large datasets, limit BLAS/OpenMP threads to reduce memory usage:

```bash
pixi run manuscript-reproduce  # Already sets threads=1 by default
```

### Parallel Processing

If your system has sufficient memory, you can try more threads:

```bash
pixi run python scripts/run_manuscript_reproduction.py --threads 4
```

## Troubleshooting

### Missing PyArrow

```
ImportError: Unable to find a usable engine; tried using: 'pyarrow', 'fastparquet'
```

**Solution**: Reinstall the Pixi environment:

```bash
pixi install --locked
```

### Missing sample_id Column

```
ValueError: output_matrix must include a sample_id column.
```

**Solution**: Preprocess your data using `pixi run manuscript-preprocess` (see above).

### Feature Catalog Missing Interactions

```
ValueError: feature_catalog does not contain any two-factor interaction candidates.
```

**Solution**: Ensure your feature catalog includes interaction terms. See
`docs/final_scripts_from_hpc/model_artifacts/final_ols_model/postfit_diagnostics/manuscript_feature_catalog.parquet`
for a working example.

### Metadata Mismatch

If column names in your X/Y matrices don't match the metadata catalogs, the workflow will fail.
Ensure your metadata files describe exactly the columns present in your data matrices (excluding `sample_id`).

## Example: Full Real Data Workflow

```bash
# 1. Preprocess raw data (if needed)
pixi run manuscript-preprocess \
  --input-x /Box\ Sync/.../sample_7500.X.parquet \
  --input-y /Box\ Sync/.../sample_7500.Y.parquet \
  --output-x artifacts/preprocessed/X.parquet \
  --output-y artifacts/preprocessed/Y.parquet

# 2. Update configs/local/manuscript_paths.local.yml to point to:
#    - artifacts/preprocessed/X.parquet
#    - artifacts/preprocessed/Y.parquet
#    - docs/final_scripts_from_hpc/.../all_input_metadata.parquet
#    - docs/final_scripts_from_hpc/.../output_metadata.parquet
#    - docs/final_scripts_from_hpc/.../manuscript_feature_catalog.parquet
#    - docs/final_scripts_from_hpc/.../fixed_holdout_assignments.parquet

# 3. Run manuscript reproduction
pixi run manuscript-reproduce --output-dir artifacts/real-data-run-$(date +%Y%m%d)
```

## See Also

- `configs/datasets/README.md` - Dataset configuration details
- `examples/end_to_end_reproducibility.py` - Toy data example
- `tools/check_manuscript_reproduction.py` - Smoke test for manuscript reproduction
- `docs/manuscript_runtime.md` - Technical details on the runtime system
