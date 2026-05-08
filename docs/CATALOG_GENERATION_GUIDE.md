# Feature Catalog Generation Utility

## Overview

The `scripts/generate_feature_catalog.py` utility automates creation of feature catalogs for the manuscript reproduction workflow. It handles first-order features, interaction pairs, and nonlinear transforms with configurable strategies.

## Quick Start

```bash
# Recommended: SHAP-based interaction selection with safe nonlinear transforms
pixi run python scripts/generate_feature_catalog.py \
  --input-matrix artifacts/my_data/X.parquet \
  --output-matrix artifacts/my_data/Y.parquet \
  --output-catalog artifacts/my_data/feature_catalog.parquet \
  --interaction-strategy top-shap \
  --max-interactions 500 \
  --nonlinear-strategy safe
```

## Usage Patterns

### 1. Small Dataset (\<100 features)

For small feature sets, you can use all pairwise interactions:

```bash
pixi run python scripts/generate_feature_catalog.py \
  --input-matrix artifacts/small_data/X.parquet \
  --output-catalog artifacts/small_data/feature_catalog.parquet \
  --interaction-strategy all \
  --nonlinear-strategy safe
```

**Warning**: With 100 features, this creates C(100,2) = 4,950 interaction pairs. Runtime ~5-10 minutes.

### 2. Medium Dataset (100-300 features)

Use SHAP to select top interaction candidates:

```bash
pixi run python scripts/generate_feature_catalog.py \
  --input-matrix artifacts/medium_data/X.parquet \
  --output-matrix artifacts/medium_data/Y.parquet \
  --output-catalog artifacts/medium_data/feature_catalog.parquet \
  --interaction-strategy top-shap \
  --max-interactions 500 \
  --nonlinear-strategy safe
```

**Note**: Requires lightgbm and shap packages: `pixi add lightgbm shap`

### 3. Large Dataset (>300 features)

Use domain knowledge to specify interaction pairs:

```bash
# Create interaction_pairs.csv with columns: feature_1, feature_2
# Example:
#   feature_1,feature_2
#   Temperature,Pressure
#   FlowRate,Concentration

pixi run python scripts/generate_feature_catalog.py \
  --input-matrix artifacts/large_data/X.parquet \
  --output-catalog artifacts/large_data/feature_catalog.parquet \
  --interaction-strategy from-file \
  --interaction-file artifacts/large_data/interaction_pairs.csv \
  --nonlinear-strategy safe
```

### 4. First-Order Features Only

Skip interactions and nonlinear transforms entirely:

```bash
pixi run python scripts/generate_feature_catalog.py \
  --input-matrix artifacts/data/X.parquet \
  --output-catalog artifacts/data/feature_catalog.parquet \
  --interaction-strategy none \
  --nonlinear-strategy none
```

**Note**: Workflow will have limited discovery capability without these candidates.

## Arguments Reference

### Required

- `--input-matrix PATH` - Input matrix (X.parquet) with sample_id column
- `--output-catalog PATH` - Where to write catalog (.parquet)

### Interaction Selection

- `--interaction-strategy {all,top-shap,from-file,none}`

  - `all`: All pairwise combinations (WARNING: large for >100 features)
  - `top-shap`: Use SHAP scores to select top N pairs (recommended, requires --output-matrix)
  - `from-file`: Load pairs from CSV (requires --interaction-file)
  - `none`: No interaction candidates

- `--max-interactions N` - Max pairs for top-shap strategy (default: 500)

- `--interaction-file PATH` - CSV with columns: feature_1, feature_2

- `--output-matrix PATH` - Output matrix (Y.parquet) for SHAP analysis

### Nonlinear Transforms

- `--nonlinear-strategy {all,safe,none}`
  - `all`: Add inverse, log, quadratic, sqrt for every feature
  - `safe`: Add only transforms with valid domains (recommended)
  - `none`: No nonlinear transforms

**Safe filters:**

- **inverse** (1/x): feature has no zeros
- **log** (log(x)): feature is strictly positive
- **quadratic** (x²): always safe
- **sqrt** (√x): feature is non-negative

### Options

- `--random-seed N` - Random seed for SHAP analysis (default: 42)

## Output

Creates a feature catalog with three feature types:

1. **numeric** - First-order base features
1. **interaction** - Interaction pairs formatted as `"feature1:feature2"`
1. **nonlinear** - Transforms formatted as `"feature_name_transform"`

Example catalog structure:

| feature_name          | feature_type | origin        |
| --------------------- | ------------ | ------------- |
| Temperature           | numeric      | model_factors |
| Pressure              | numeric      | model_factors |
| Temperature:Pressure  | interaction  | model_factors |
| Temperature_quadratic | nonlinear    | model_factors |
| Pressure_log          | nonlinear    | model_factors |

## Intelligent Filtering

The utility automatically handles:

### 1. Special Columns

Excludes from catalog:

- `sample_id`
- `scenario`
- `run_id`
- `AFSC`
- `UAEORO`

### 2. Existing Transforms

If input data already contains features with transform suffixes (`_inverse`, `_log`, `_quadratic`, `_sqrt`), they are:

- Excluded from first-order catalog
- Not re-transformed during nonlinear generation

Example: If X has `Temperature_quadratic`, the script:

- Treats it as a derived feature, not base input
- Won't create `Temperature_quadratic_quadratic`

### 3. Validation

Checks for:

- Required columns (feature_name, feature_type, origin)
- Valid feature_type values
- Duplicate feature names
- Proper interaction format (`feature1:feature2`)

## Performance Guidelines

| Features | Interaction Strategy | Catalog Size | Runtime | Workflow Runtime |
| -------- | -------------------- | ------------ | ------- | ---------------- |
| 50       | all                  | ~1,275       | \<1 min | ~5 min           |
| 100      | all                  | ~5,050       | \<1 min | ~10 min          |
| 200      | top-shap (500)       | ~1,250       | ~2 min  | ~10 min          |
| 350      | top-shap (500)       | ~1,850       | ~5 min  | ~15 min          |
| 350      | all                  | ~62,000      | \<1 min | ~45+ min ⚠️      |

**Recommendation**: For >200 features, use `top-shap` with max-interactions=500-1000 or curate pairs manually.

## Integration with Workflow

After generating catalog, update your dataset config:

```yaml
# configs/datasets/my_dataset.yml
case_study_input_matrix: /path/to/X.parquet
case_study_output_matrix: /path/to/Y.parquet
input_metadata: /path/to/input_metadata.parquet
output_metadata: /path/to/output_metadata.parquet
manuscript_feature_catalog: /path/to/feature_catalog.parquet  # ← Generated catalog
fixed_holdout_assignments: /path/to/holdout_assignments.parquet
output_root: /path/to/output
```

Then run:

```bash
pixi run manuscript-reproduce --config configs/datasets/my_dataset.yml
```

## Troubleshooting

### "No module named 'lightgbm'"

SHAP strategy requires additional packages:

```bash
pixi add lightgbm shap
```

Or use a different strategy:

- `--interaction-strategy all` (small datasets)
- `--interaction-strategy from-file` (curated pairs)
- `--interaction-strategy none` (skip interactions)

### "Catalog contains duplicate feature names"

This shouldn't happen after filtering, but if it does:

1. Check for duplicate columns in input matrix
1. Check interaction file for duplicate pairs
1. Report issue with example data

### "Feature X is not a direct input"

During workflow execution, this means:

1. Catalog references a feature not in input matrix
1. Feature name mismatch (check spelling, case)
1. Feature was filtered as special column

Verify catalog features match input matrix:

```python
import pandas as pd
X = pd.read_parquet("X.parquet")
catalog = pd.read_parquet("feature_catalog.parquet")

special = {"sample_id", "scenario", "run_id", "AFSC", "UAEORO"}
available = set(X.columns) - special
required = set(catalog[catalog["feature_type"] == "numeric"]["feature_name"])
missing = required - available

if missing:
    print(f"Catalog requires {len(missing)} features not in X:")
    for f in list(missing)[:10]:
        print(f"  - {f}")
```

### Large Catalog Warning

If catalog has >10,000 interaction pairs, workflow runtime increases significantly (20-60+ minutes).

Solutions:

1. Reduce `--max-interactions` (try 500-1000)
1. Use curated interaction file
1. Filter features before catalog generation

## Examples

### Example 1: Reproduce 3K Test Dataset

```bash
pixi run python scripts/generate_feature_catalog.py \
  --input-matrix artifacts/test_dataset_3k/X.parquet \
  --output-catalog artifacts/test_dataset_3k/proper_feature_catalog.parquet \
  --interaction-strategy none \
  --nonlinear-strategy safe
```

Output:

- 309 first-order features (41 derived features auto-filtered)
- 0 interactions
- 1,156 nonlinear transforms
- **Total: 1,465 features**

### Example 2: Full Manuscript Reproduction

```bash
# For 352 base features with curated interactions
pixi run python scripts/generate_feature_catalog.py \
  --input-matrix artifacts/preprocessed_real_data/sample_7500.X.parquet \
  --output-matrix artifacts/preprocessed_real_data/sample_7500.Y.parquet \
  --output-catalog artifacts/preprocessed_real_data/feature_catalog.parquet \
  --interaction-strategy top-shap \
  --max-interactions 500 \
  --nonlinear-strategy safe
```

Output:

- 352 first-order features
- 500 SHAP-ranked interactions
- ~1,320 nonlinear transforms
- **Total: ~2,172 features**

## See Also

- `configs/datasets/README.md` - Dataset configuration guide
- [docs/3K_TEST_VALIDATION_REPORT.md](3K_TEST_VALIDATION_REPORT.md) - Validation results
- [docs/CATALOG_STRUCTURE_FINDINGS.md](CATALOG_STRUCTURE_FINDINGS.md) - Catalog requirements discovery
- [docs/troubleshooting.md](troubleshooting.md) - Common issues and solutions
