# Manuscript Reproduction Quick Reference

## One-Line Commands

### Run with Config File

```bash
pixi run manuscript-reproduce --config configs/datasets/real_data.yml
```

### Run with Custom Output Directory

```bash
pixi run manuscript-reproduce --config configs/datasets/real_data.yml --output-dir artifacts/my-run
```

### Run with Help

```bash
pixi run python scripts/run_manuscript_reproduction.py --help
```

## Config File Format

Single YAML file with all paths:

```yaml
# configs/datasets/your_dataset.yml
case_study_input_matrix: /path/to/X.parquet
case_study_output_matrix: /path/to/Y.parquet
input_metadata: /path/to/input_metadata.parquet
output_metadata: /path/to/output_metadata.parquet
manuscript_feature_catalog: /path/to/feature_catalog.parquet
fixed_holdout_assignments: /path/to/holdout_assignments.parquet
output_root: /path/to/output  # optional
```

See `configs/datasets/template.yml` for a template.

## Preprocessing Raw Data

If your data has MultiIndex format:

```bash
pixi run manuscript-preprocess \
  --input-x /path/to/raw_X.parquet \
  --input-y /path/to/raw_Y.parquet \
  --output-x artifacts/preprocessed/X.parquet \
  --output-y artifacts/preprocessed/Y.parquet
```

## Quick Workflow

1. **Create config**: Copy `configs/datasets/template.yml` to `configs/datasets/my_data.yml`
1. **Edit paths**: Point to your data files
1. **Run**: `pixi run manuscript-reproduce --config configs/datasets/my_data.yml`

## Available Configs

- `configs/datasets/real_data.yml` - Full BSM real data
- `configs/datasets/template.yml` - Template for your own data

## Workflow Stages

1. Output Conditioning (PCA, filtering)
1. Empirical Null Screening (permutation tests)
1. Interaction Discovery (residualized products)
1. Nonlinear Discovery (GAM-based)
1. Sparse Selection (L1-penalized + stability)
1. Final Artifacts (OLS, tables, figures, audit)

## Documentation

- Full guide: `docs/RUNNING_MANUSCRIPT_REPRODUCTION.md`
- Dataset configs: `configs/datasets/README.md`
- Example workflow: `examples/end_to_end_reproducibility.py`

## Troubleshooting

### Missing PyArrow

```bash
pixi install --locked
```

### Missing sample_id

Use `pixi run manuscript-preprocess` to add it.

### Feature Catalog Issues

Ensure your feature catalog includes interaction terms (`feature_type == 'interaction'`).
