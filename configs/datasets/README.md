# Dataset Configurations

**Purpose:** Configuration files for running manuscript reproduction workflows against different datasets.

**Quick Start:** See [test_3k.yml](test_3k.yml) for a working example with 3,000 samples.

______________________________________________________________________

## Table of Contents

1. [Quick Start](#quick-start)
1. [Configuration File Format](#configuration-file-format)
1. [Data Requirements](#data-requirements)
1. [Example Configurations](#example-configurations)
1. [Preprocessing Your Data](#preprocessing-your-data)
1. [Common Pitfalls](#common-pitfalls)
1. [Troubleshooting](#troubleshooting)

______________________________________________________________________

## Quick Start

### Run with a Pre-Built Configuration

```bash
# Test dataset (3k samples, fast ~20 min)
pixi run manuscript-reproduce --config configs/datasets/test_3k.yml

# Real data (20k samples, slow ~2 hrs)
pixi run manuscript-reproduce --config configs/datasets/real_data.yml

# For memory efficiency, set thread limits:
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    pixi run manuscript-reproduce --config configs/datasets/test_3k.yml
```

### Create Your Own Configuration

1. Copy the template: `cp configs/datasets/template.yml configs/datasets/my_data.yml`
1. Edit paths to point to your data files
1. Run: `pixi run manuscript-reproduce --config configs/datasets/my_data.yml`

______________________________________________________________________

## Configuration File Format

A dataset configuration is a YAML file with 6 required fields plus 1 optional:

```yaml
# Required: Input feature matrix (N × M with sample_id column)
case_study_input_matrix: /absolute/path/to/X.parquet

# Required: Output response matrix (N × K with sample_id column)
case_study_output_matrix: /absolute/path/to/Y.parquet

# Required: Input feature metadata
input_metadata: /absolute/path/to/input_metadata.parquet

# Required: Output response metadata
output_metadata: /absolute/path/to/output_metadata.parquet

# Required: Feature catalog with interactions and nonlinear transforms
manuscript_feature_catalog: /absolute/path/to/feature_catalog.parquet

# Required: Train/test split assignments
fixed_holdout_assignments: /absolute/path/to/holdout_assignments.parquet

# Optional: Default output directory (default: artifacts/manuscript-output)
output_root: /absolute/path/to/output
```

**Important:**

- All paths must be **absolute** (not relative)
- All files must be in **Parquet format**
- All matrices must include a `sample_id` column (see [Data Requirements](#data-requirements))

______________________________________________________________________

## Configuration Files

### Local Override (Primary Method)

The primary method for configuring data paths is through the local override file:

- **File**: `configs/local/manuscript_paths.local.yml`
- **Status**: User-specific, not tracked in git
- **Purpose**: Points to real data files on your local filesystem

This file should contain paths to all required artifacts:

- `case_study_input_matrix`: Input features matrix (X)
- `case_study_output_matrix`: Output responses matrix (Y)
- `input_metadata`: Input feature metadata
- `output_metadata`: Output response metadata
- `manuscript_feature_catalog`: Feature engineering catalog
- `fixed_holdout_assignments`: Train/holdout split assignments
- `output_root`: Default output directory

## Data Requirements

All data files must meet these requirements:

### 1. Input Matrix (`X.parquet`)

**Shape:** (N_samples, 1 + N_features)\
**Required columns:**

- `sample_id` (string): Unique identifier for each sample
- Feature columns: Numeric values for each input feature

**Format requirements:**

- First column MUST be `sample_id`
- All feature values must be numeric (float64 or int64)
- No missing values (NaN) allowed
- Must include scenario factors (e.g., AFSC, UAEORO) if used in manuscript

**Example:**

```
sample_id,AFSC,UAEORO,scenario,run_id,feature_1,feature_2,...
AFSCoff_UAEOROoff_0,0,1,AFSCoff_UAEOROoff,0,1.23,4.56,...
AFSCoff_UAEOROoff_1,0,1,AFSCoff_UAEOROoff,1,2.34,5.67,...
```

### 2. Output Matrix (`Y.parquet`)

**Shape:** (N_samples, 1 + N_outputs)\
**Required columns:**

- `sample_id` (string): Must match X matrix identically
- Output columns: Numeric values for each output response

**Format requirements:**

- First column MUST be `sample_id`
- Sample IDs must align with X matrix
- All output values must be numeric
- NaN values allowed (filtered by variance threshold)

**Example:**

```
sample_id,output_1,output_2,output_3,...
AFSCoff_UAEOROoff_0,123.4,567.8,901.2,...
AFSCoff_UAEOROoff_1,234.5,678.9,012.3,...
```

### 3. Input Metadata (`input_metadata.parquet`)

**Shape:** (N_features, 2+)\
**Required columns:**

- `input_name` (string): Must match X matrix column names exactly
- Additional metadata columns as needed

**Example:**

```
input_name,description,units,category
AFSC,Aviation fuel supply constraint,binary,scenario
feature_1,Description of feature 1,kg,physical
feature_2,Description of feature 2,USD,economic
```

### 4. Output Metadata (`output_metadata.parquet`)

**Shape:** (N_outputs, 2+)\
**Required columns:**

- `output_name` (string): Must match Y matrix column names exactly
- Additional metadata columns as needed

### 5. Feature Catalog (`feature_catalog.parquet`)

**Shape:** (N_candidates, 3)\
**Required columns:**

- `feature_name` (string): Candidate feature name
- `feature_type` (string): One of {'first_order', 'interaction', 'nonlinear'}
- `origin` (string): Source features and transformation

**Critical requirements:**

- **First-order features:** Must match input matrix columns
- **Interactions:** Must use colon-delimited format (`feature1:feature2`)
- **Nonlinear:** Must use function notation (`log(feature)`, `square(feature)`)

**Example:**

```
feature_name,feature_type,origin
AFSC,first_order,model_factors
feature_1,first_order,model_factors
feature_1:feature_2,interaction,"feature_1, feature_2"
log(feature_1),nonlinear,log of feature_1
square(feature_2),nonlinear,square of feature_2
```

**Generating interactions:**
For a full pairwise interaction library:

```python
import pandas as pd

# Load base features
base_features = input_metadata['input_name'].tolist()

# Create catalog
catalog = []

# First-order
for name in base_features:
    catalog.append({
        'feature_name': name,
        'feature_type': 'first_order',
        'origin': name
    })

# All pairwise interactions (colon-delimited)
for i, name1 in enumerate(base_features):
    for name2 in base_features[i+1:]:
        catalog.append({
            'feature_name': f"{name1}:{name2}",
            'feature_type': 'interaction',
            'origin': f"{name1}, {name2}"
        })

# Nonlinear transformations (example: first 50 features)
for name in base_features[:50]:
    for transform in ['square', 'log']:
        catalog.append({
            'feature_name': f"{transform}({name})",
            'feature_type': 'nonlinear',
            'origin': f"{transform} of {name}"
        })

pd.DataFrame(catalog).to_parquet('feature_catalog.parquet')
```

### 6. Holdout Assignments (`holdout_assignments.parquet`)

**Shape:** (N_samples, 2)\
**Required columns:**

- `sample_id` (string): Must match X and Y matrices exactly
- `split` (string): One of {'train', 'test'}

**Format requirements:**

- Every sample in X/Y must have an assignment
- Split values are `'train'` and `'test'` (NOT `'holdout'`)
- Typical split: 90% train, 10% test

**Example:**

```
sample_id,split
AFSCoff_UAEOROoff_0,train
AFSCoff_UAEOROoff_1,train
AFSCoff_UAEOROoff_2,test
```

**Generating holdout assignments:**

```python
import pandas as pd
import numpy as np

# Load X to get sample IDs
X = pd.read_parquet('X.parquet')

# Create stratified split (90/10)
holdout = pd.DataFrame({'sample_id': X['sample_id'].values})
np.random.seed(42)
holdout['split'] = np.random.choice(['train', 'test'],
                                    size=len(holdout),
                                    p=[0.9, 0.1])

holdout.to_parquet('holdout_assignments.parquet')
```

______________________________________________________________________

## Example Configurations

### Example 1: Test Dataset (3k samples)

**File:** `configs/datasets/test_3k.yml`

```yaml
case_study_input_matrix: /Users/you/rfm-pipeline/artifacts/test_dataset_3k/X.parquet
case_study_output_matrix: /Users/you/rfm-pipeline/artifacts/test_dataset_3k/Y.parquet
input_metadata: /Users/you/rfm-pipeline/docs/final_scripts_from_hpc/model_artifacts/final_ols_model/postfit_diagnostics/all_input_metadata.parquet
output_metadata: /Users/you/rfm-pipeline/docs/final_scripts_from_hpc/model_artifacts/final_ols_model/postfit_diagnostics/output_metadata.parquet
manuscript_feature_catalog: /Users/you/rfm-pipeline/artifacts/test_dataset_3k/full_feature_catalog.parquet
fixed_holdout_assignments: /Users/you/rfm-pipeline/artifacts/test_dataset_3k/holdout_assignments.parquet
output_root: /Users/you/rfm-pipeline/artifacts/test_3k_output
```

**Use case:** Quick validation (~20 minutes)

### Example 2: Real Data (20k samples)

**File:** `configs/datasets/real_data.yml`

```yaml
case_study_input_matrix: /Users/you/rfm-pipeline/artifacts/preprocessed_real_data/sample_7500.X.parquet
case_study_output_matrix: /Users/you/rfm-pipeline/artifacts/preprocessed_real_data/sample_7500.Y.parquet
input_metadata: /Users/you/rfm-pipeline/docs/final_scripts_from_hpc/model_artifacts/final_ols_model/postfit_diagnostics/all_input_metadata.parquet
output_metadata: /Users/you/rfm-pipeline/docs/final_scripts_from_hpc/model_artifacts/final_ols_model/postfit_diagnostics/output_metadata.parquet
manuscript_feature_catalog: /Users/you/rfm-pipeline/artifacts/preprocessed_real_data/full_feature_catalog.parquet
fixed_holdout_assignments: /Users/you/rfm-pipeline/docs/final_scripts_from_hpc/model_artifacts/final_ols_model/postfit_diagnostics/fixed_holdout_assignments.parquet
output_root: /Users/you/rfm-pipeline/artifacts/real_data_output
```

**Use case:** Full manuscript reproduction (~2 hours)

### Example 3: Custom User Data

**File:** `configs/datasets/my_custom_data.yml`

```yaml
case_study_input_matrix: /data/projects/my_study/X_preprocessed.parquet
case_study_output_matrix: /data/projects/my_study/Y_preprocessed.parquet
input_metadata: /data/projects/my_study/input_metadata.parquet
output_metadata: /data/projects/my_study/output_metadata.parquet
manuscript_feature_catalog: /data/projects/my_study/feature_catalog.parquet
fixed_holdout_assignments: /data/projects/my_study/holdout_split.parquet
output_root: /data/projects/my_study/workflow_output
```

**Use case:** Applying workflow to your own data

______________________________________________________________________

## Preprocessing Your Data

If your data doesn't match the required format, use these preprocessing steps:

### Converting MultiIndex Data

If your data has a MultiIndex (e.g., `scenario` + `run_id`), convert to `sample_id`:

**Using the preprocessing script:**

```bash
pixi run python scripts/add_scenario_factors.py
```

**Or manually:**

```python
import pandas as pd

# Load data with MultiIndex
X_raw = pd.read_parquet('raw_X.parquet')  # Index: (scenario, run_id)
Y_raw = pd.read_parquet('raw_Y.parquet')

# Create sample_id from index
X_raw['sample_id'] = (X_raw.index.get_level_values('scenario').astype(str) + '_' +
                      X_raw.index.get_level_values('run_id').astype(str))
Y_raw['sample_id'] = (Y_raw.index.get_level_values('scenario').astype(str) + '_' +
                      Y_raw.index.get_level_values('run_id').astype(str))

# Reset index to make scenario/run_id regular columns
X_reset = X_raw.reset_index()
Y_reset = Y_raw.reset_index()

# Reorder columns (sample_id first)
X_cols = ['sample_id'] + [c for c in X_reset.columns if c != 'sample_id']
Y_cols = ['sample_id'] + [c for c in Y_reset.columns if c != 'sample_id']

X_final = X_reset[X_cols]
Y_final = Y_reset[Y_cols]

# Save
X_final.to_parquet('X_preprocessed.parquet', index=False)
Y_final.to_parquet('Y_preprocessed.parquet', index=False)
```

### Extracting Scenario Factors

If your data has scenario factors (like AFSC, UAEORO) encoded in strings:

```python
# Example: scenario = "AFSCon_UAEOROoff"
X['AFSC'] = X['scenario'].str.contains('AFSCon').astype(int)
X['UAEORO'] = X['scenario'].str.contains('UAEOROon').astype(int)
```

### Creating Feature Catalog

See [Data Requirements](#data-requirements) section above for full script to generate pairwise interactions.

______________________________________________________________________

## Common Pitfalls

### ❌ Pitfall 1: Relative Paths

**Wrong:**

```yaml
case_study_input_matrix: artifacts/test_dataset_3k/X.parquet  # ❌ Relative
```

**Correct:**

```yaml
case_study_input_matrix: /Users/you/rfm-pipeline/artifacts/test_dataset_3k/X.parquet  # ✅ Absolute
```

**Fix:** Use `$(pwd)/artifacts/...` in your YAML or expand to absolute path.

### ❌ Pitfall 2: Missing sample_id

**Error:**

```
ValueError: case_study_input_matrix must include a sample_id column.
```

**Cause:** Your X or Y matrix doesn't have `sample_id` as first column.

**Fix:** Preprocess your data (see [Preprocessing](#preprocessing-your-data))

### ❌ Pitfall 3: Wrong Split Values

**Wrong:**

```
sample_id,split
sample_001,train
sample_002,holdout  # ❌ Should be 'test'
```

**Correct:**

```
sample_id,split
sample_001,train
sample_002,test  # ✅ Correct
```

### ❌ Pitfall 4: Missing Scenario Factors

**Error:**

```
ValueError: Feature 'AFSC' is not a direct input or supported catalog expression.
```

**Cause:** Your X matrix doesn't include AFSC, UAEORO columns that the feature catalog references.

**Fix:** Ensure your input matrix includes all scenario factors referenced in the catalog.

### ❌ Pitfall 5: Wrong Interaction Format

**Wrong:**

```
feature_name,feature_type,origin
feature1*feature2,interaction,...  # ❌ Wrong delimiter
feature1_x_feature2,interaction,...  # ❌ Wrong delimiter
```

**Correct:**

```
feature_name,feature_type,origin
feature1:feature2,interaction,...  # ✅ Colon delimiter
```

### ❌ Pitfall 6: Sample ID Mismatch

**Error:**

```
ValueError: sample_id values in holdout_assignments do not match input_matrix
```

**Cause:** Sample IDs differ between X, Y, and holdout assignments.

**Fix:** Verify sample IDs match exactly:

```python
X_ids = set(pd.read_parquet('X.parquet')['sample_id'])
Y_ids = set(pd.read_parquet('Y.parquet')['sample_id'])
H_ids = set(pd.read_parquet('holdout.parquet')['sample_id'])

assert X_ids == Y_ids == H_ids, "Sample IDs don't match!"
```

______________________________________________________________________

## Data Requirements

All data matrices must include a `sample_id` column that uniquely identifies each sample.
If your raw data uses a MultiIndex (e.g., `scenario` and `run_id`), convert it:

```python
import pandas as pd

# Load raw data with MultiIndex
x = pd.read_parquet('raw_X.parquet')
y = pd.read_parquet('raw_Y.parquet')

# Reset index and create sample_id
x_reset = x.reset_index()
y_reset = y.reset_index()

x_reset.insert(0, 'sample_id', x_reset['scenario'] + '_' + x_reset['run_id'].astype(str))
y_reset.insert(0, 'sample_id', y_reset['scenario'] + '_' + y_reset['run_id'].astype(str))

# Drop original index columns
x_final = x_reset.drop(columns=['scenario', 'run_id'])
y_final = y_reset.drop(columns=['scenario', 'run_id'])

# Save preprocessed data
x_final.to_parquet('preprocessed_X.parquet')
y_final.to_parquet('preprocessed_Y.parquet')
```

### Metadata Files

The manuscript reproduction workflow requires properly aligned metadata files that describe:

1. **Input Metadata** (`input_metadata.parquet`):

   - `input_name`: Feature column names
   - Additional metadata about each input feature

1. **Output Metadata** (`output_metadata.parquet`):

   - `output_name`: Response column names
   - Additional metadata about each output

1. **Feature Catalog** (`manuscript_feature_catalog.parquet`):

   - `feature_name`: Engineered feature names
   - `feature_type`: Type (first_order, interaction, nonlinear)
   - `origin`: Source feature(s) and transformation
   - Must include interaction candidates for the workflow to complete

1. **Holdout Assignments** (`fixed_holdout_assignments.parquet`):

   - `sample_id`: Sample identifiers (must match X/Y matrices)
   - `split`: 'train' or 'holdout'

See `docs/final_scripts_from_hpc/model_artifacts/final_ols_model/postfit_diagnostics/`
for examples of properly formatted metadata files.

## Troubleshooting

### Missing sample_id Column

**Error:**

```
ValueError: output_matrix must include a sample_id column.
```

**Solution:** Preprocess your data to include `sample_id` as first column (see [Preprocessing](#preprocessing-your-data)).

### Missing PyArrow

**Error:**

```
ImportError: Unable to find a usable engine; tried using: 'pyarrow', 'fastparquet'
```

**Solution:** Reinstall the Pixi environment:

```bash
pixi install --locked
```

### Feature Catalog Missing Interactions

**Error:**

```
ValueError: feature_catalog does not contain any two-factor interaction candidates.
```

**Solution:** Ensure your feature catalog includes rows where `feature_type == 'interaction'` with colon-delimited names.

### Missing Scenario Factors (AFSC, UAEORO)

**Error:**

```
ValueError: Feature 'AFSC' is not a direct input or supported catalog expression.
```

**Solution:** Your input matrix must include scenario factors (AFSC, UAEORO, scenario, run_id). Use `scripts/add_scenario_factors.py` or see [Preprocessing](#preprocessing-your-data).

### Configuration Validation Failed

**Error:**

```
KeyError: 'case_study_input_matrix'
```

**Solution:** Verify your YAML file has all 6 required fields (see [Configuration File Format](#configuration-file-format)).

### File Not Found

**Error:**

```
FileNotFoundError: [Errno 2] No such file or directory: '/path/to/X.parquet'
```

**Solution:**

1. Verify paths in your config are **absolute** (not relative)
1. Check files actually exist at those paths
1. Fix typos in file names

### Out of Memory

**Error:**

```
MemoryError
```

**Solution:**

1. Use thread limits (see [Quick Start](#quick-start))
1. Reduce candidate library size
1. Use smaller test dataset first
1. Close other applications
1. Use machine with more RAM (16GB+ recommended)

______________________________________________________________________

## Validation Checklist

Before running your workflow, verify:

- [ ] All file paths in config are **absolute** paths
- [ ] All files exist at specified paths
- [ ] X matrix has `sample_id` as first column
- [ ] Y matrix has `sample_id` as first column
- [ ] Sample IDs match across X, Y, and holdout files
- [ ] Feature catalog includes first-order, interaction, and nonlinear terms
- [ ] Interactions use colon-delimited format (`feature1:feature2`)
- [ ] Holdout split values are `'train'` and `'test'` (not `'holdout'`)
- [ ] Input metadata has `input_name` column matching X columns
- [ ] Output metadata has `output_name` column matching Y columns
- [ ] All data files are in Parquet format (not CSV)

______________________________________________________________________

## Next Steps

1. **Validate your configuration:**

   ```bash
   pixi run python - <<'PY'
   import yaml
   with open('configs/datasets/my_data.yml') as f:
       config = yaml.safe_load(f)
   required = ['case_study_input_matrix', 'case_study_output_matrix',
               'input_metadata', 'output_metadata',
               'manuscript_feature_catalog', 'fixed_holdout_assignments']
   missing = [k for k in required if k not in config]
   if missing:
       print(f"❌ Missing required fields: {missing}")
   else:
       print("✅ Configuration valid")
   PY
   ```

1. **Run the workflow:**

   ```bash
   pixi run manuscript-reproduce --config configs/datasets/my_data.yml
   ```

1. **Interpret results:**

   - See [Interpreting Your Results](../../docs/interpreting_results.md)
   - Compare to manuscript benchmarks
   - Validate feature counts and nRMSE

______________________________________________________________________

**See also:**

- [Interpreting Results](../../docs/interpreting_results.md) - Understanding output
- [Artifact Reference](../../docs/artifact_reference.md) - What each file contains
- [Troubleshooting Guide](../../docs/troubleshooting.md) - Common issues and fixes
- [Full Reproduction Guide](../../docs/RUNNING_MANUSCRIPT_REPRODUCTION.md) - Complete workflow documentation
