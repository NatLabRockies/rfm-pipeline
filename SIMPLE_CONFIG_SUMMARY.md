# Simple Config-Based Workflow - Final Summary

## What Changed

Simplified the config system to use **one config file per dataset** instead of searching through multiple config files.

## New Usage

### One Command

```bash
pixi run manuscript-reproduce --config configs/datasets/real_data.yml
```

Or with custom output:

```bash
pixi run manuscript-reproduce --config configs/datasets/real_data.yml --output-dir artifacts/my-run
```

### Config File Format

Simple YAML with all paths in one place:

```yaml
# configs/datasets/your_dataset.yml
case_study_input_matrix: /path/to/X.parquet
case_study_output_matrix: /path/to/Y.parquet
input_metadata: /path/to/input_metadata.parquet
output_metadata: /path/to/output_metadata.parquet
manuscript_feature_catalog: /path/to/feature_catalog.parquet
fixed_holdout_assignments: /path/to/holdout_assignments.parquet
output_root: /path/to/default/output  # optional
```

## Quick Workflow

1. Copy template: `cp configs/datasets/template.yml configs/datasets/my_data.yml`
1. Edit paths in `my_data.yml`
1. Run: `pixi run manuscript-reproduce --config configs/datasets/my_data.yml`

## Available Configs

- `configs/datasets/real_data.yml` - BSM real data (preprocessed and ready)
- `configs/datasets/template.yml` - Template for your own data

## Examples

### Run Real Data

```bash
pixi run manuscript-reproduce --config configs/datasets/real_data.yml --output-dir artifacts/$(date +%Y%m%d)
```

### Create Your Own Dataset Config

```bash
# 1. Preprocess if needed
pixi run manuscript-preprocess \
  --input-x /path/to/raw_X.parquet \
  --input-y /path/to/raw_Y.parquet \
  --output-x artifacts/my_data/X.parquet \
  --output-y artifacts/my_data/Y.parquet

# 2. Create config
cp configs/datasets/template.yml configs/datasets/my_data.yml
# Edit my_data.yml to point to your files

# 3. Run
pixi run manuscript-reproduce --config configs/datasets/my_data.yml
```

## Benefits

- **Simpler**: One file per dataset, no searching through multiple configs
- **Clearer**: All paths in one place
- **Flexible**: Easy to switch between datasets
- **Traceable**: Config file shows exactly what data was used

## Files Created

- `configs/datasets/real_data.yml` - Real data config
- `configs/datasets/template.yml` - Template
- Updated `scripts/run_manuscript_reproduction.py` - Now requires --config
- Updated `MANUSCRIPT_QUICK_START.md` - New instructions

## Documentation

- `MANUSCRIPT_QUICK_START.md` - Quick reference
- `docs/RUNNING_MANUSCRIPT_REPRODUCTION.md` - Full guide
- `configs/datasets/README.md` - Dataset config guide
