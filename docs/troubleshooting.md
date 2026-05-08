# Troubleshooting Guide

**Purpose:** Solutions to common problems when running the BSM reduced-form modeling workflow.

**Prerequisites:** You've attempted to run the workflow and encountered an error.

**Next Steps:** If your issue isn't listed here, open a GitHub issue with your error message and configuration.

______________________________________________________________________

## Table of Contents

1. [Installation & Environment Issues](#installation--environment-issues)
1. [Data Format Issues](#data-format-issues)
1. [Runtime Errors](#runtime-errors)
1. [Performance & Memory Issues](#performance--memory-issues)
1. [Result Quality Issues](#result-quality-issues)
1. [Frequently Asked Questions](#frequently-asked-questions)

______________________________________________________________________

## Installation & Environment Issues

### Pixi command not found

**Error:**

```
bash: pixi: command not found
```

**Cause:** Pixi not installed or not in PATH.

**Solution:**

```bash
# Install Pixi
curl -fsSL https://pixi.sh/install.sh | bash

# Add to PATH (for current session)
export PATH="$HOME/.pixi/bin:$PATH"

# Add to PATH permanently (add to ~/.bashrc or ~/.zshrc)
echo 'export PATH="$HOME/.pixi/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### Pixi lock file conflicts

**Error:**

```
Error: lock file is out of sync with pixi.toml
```

**Solution:**

```bash
# Reinstall with locked dependencies
pixi install --locked

# If that fails, remove and reinstall
rm -rf .pixi
pixi install --locked
```

### Missing PyArrow dependency

**Error:**

```
ImportError: Unable to find a usable engine; tried using: 'pyarrow', 'fastparquet'
```

**Cause:** PyArrow not installed in Pixi environment.

**Solution:**

```bash
pixi install --locked
```

### Python module not found

**Error:**

```
ModuleNotFoundError: No module named 'bsm_rfm'
```

**Cause:** Running Python without `PYTHONPATH=src`.

**Solution:**

```bash
# Always use PYTHONPATH=src when running scripts
PYTHONPATH=src pixi run python scripts/run_manuscript_reproduction.py --config ...

# Or use the pixi task
pixi run manuscript-reproduce --config ...
```

______________________________________________________________________

## Data Format Issues

### Missing sample_id column

**Error:**

```
ValueError: case_study_input_matrix must include a sample_id column.
```

**Cause:** Your X or Y matrix doesn't have `sample_id` as the first column.

**Solution:**

**Option 1:** Use the preprocessing script (if your data has scenario/run_id):

```bash
PYTHONPATH=src pixi run python scripts/add_scenario_factors.py
```

**Option 2:** Manually add sample_id:

```python
import pandas as pd

X = pd.read_parquet('X.parquet')
Y = pd.read_parquet('Y.parquet')

# Create unique sample_id
X.insert(0, 'sample_id', [f"sample_{i}" for i in range(len(X))])
Y.insert(0, 'sample_id', [f"sample_{i}" for i in range(len(Y))])

X.to_parquet('X_with_sample_id.parquet', index=False)
Y.to_parquet('Y_with_sample_id.parquet', index=False)
```

### Sample ID mismatch

**Error:**

```
ValueError: sample_id values in holdout_assignments do not match input_matrix
```

**Cause:** Sample IDs differ between X, Y, and holdout assignment files.

**Solution:** Verify and fix alignment:

```python
import pandas as pd

X = pd.read_parquet('X.parquet')
Y = pd.read_parquet('Y.parquet')
holdout = pd.read_parquet('holdout.parquet')

X_ids = set(X['sample_id'])
Y_ids = set(Y['sample_id'])
H_ids = set(holdout['sample_id'])

print(f"X samples: {len(X_ids)}")
print(f"Y samples: {len(Y_ids)}")
print(f"Holdout samples: {len(H_ids)}")

# Find mismatches
print(f"In X but not Y: {X_ids - Y_ids}")
print(f"In Y but not X: {Y_ids - X_ids}")
print(f"In holdout but not X: {H_ids - X_ids}")

# Fix: keep only common samples
common_ids = X_ids & Y_ids & H_ids
X_fixed = X[X['sample_id'].isin(common_ids)]
Y_fixed = Y[Y['sample_id'].isin(common_ids)]
holdout_fixed = holdout[holdout['sample_id'].isin(common_ids)]

# Save fixed versions
X_fixed.to_parquet('X_fixed.parquet', index=False)
Y_fixed.to_parquet('Y_fixed.parquet', index=False)
holdout_fixed.to_parquet('holdout_fixed.parquet', index=False)
```

### Missing scenario factors (AFSC, UAEORO)

**Error:**

```
ValueError: Feature 'AFSC' is not a direct input or supported catalog expression.
```

**Cause:** Your input matrix is missing scenario factor columns that the feature catalog references.

**Solution:**

**Option 1:** Use the scenario factor extraction script:

```bash
PYTHONPATH=src pixi run python scripts/add_scenario_factors.py
```

**Option 2:** Extract from scenario string manually:

```python
import pandas as pd

X = pd.read_parquet('X.parquet')

# Extract binary factors from scenario column
# Example: "AFSCon_UAEOROoff" -> AFSC=1, UAEORO=0
X['AFSC'] = X['scenario'].str.contains('AFSCon').astype(int)
X['UAEORO'] = X['scenario'].str.contains('UAEOROon').astype(int)

X.to_parquet('X_with_factors.parquet', index=False)
```

### Wrong split values in holdout assignments

**Error:**

```
ValueError: split column must contain only 'train' and 'test' values
```

**Cause:** Your holdout file uses `'holdout'` instead of `'test'`.

**Solution:**

```python
import pandas as pd

holdout = pd.read_parquet('holdout.parquet')

# Replace 'holdout' with 'test'
holdout['split'] = holdout['split'].replace('holdout', 'test')

# Verify only train/test remain
assert set(holdout['split'].unique()) <= {'train', 'test'}

holdout.to_parquet('holdout_fixed.parquet', index=False)
```

### Feature catalog missing interactions

**Error:**

```
ValueError: feature_catalog does not contain any two-factor interaction candidates.
```

**Cause:** Your feature catalog only has first-order features, no interaction terms.

**Solution:** Generate interaction candidates (see `configs/datasets/README.md` for full script):

```python
import pandas as pd

input_meta = pd.read_parquet('input_metadata.parquet')
base_features = input_meta['input_name'].tolist()

catalog = []

# First-order
for name in base_features:
    catalog.append({
        'feature_name': name,
        'feature_type': 'first_order',
        'origin': name
    })

# Pairwise interactions (colon-delimited)
for i, name1 in enumerate(base_features):
    for name2 in base_features[i+1:]:
        catalog.append({
            'feature_name': f"{name1}:{name2}",
            'feature_type': 'interaction',
            'origin': f"{name1}, {name2}"
        })

pd.DataFrame(catalog).to_parquet('feature_catalog_with_interactions.parquet')
```

______________________________________________________________________

## Runtime Errors

### Configuration file not found

**Error:**

```
FileNotFoundError: [Errno 2] No such file or directory: 'configs/datasets/my_data.yml'
```

**Cause:** Path to config file is wrong or file doesn't exist.

**Solution:**

```bash
# Check file exists
ls -l configs/datasets/my_data.yml

# Use absolute path if relative path fails
pixi run manuscript-reproduce --config $(pwd)/configs/datasets/my_data.yml
```

### Data file not found

**Error:**

```
FileNotFoundError: [Errno 2] No such file or directory: '/path/to/X.parquet'
```

**Cause:** Path in config file is wrong or file doesn't exist.

**Solution:**

1. Verify file exists: `ls -l /path/to/X.parquet`
1. Check for typos in config YAML
1. Ensure paths are **absolute** not relative
1. Expand `~` to full home directory path

### Process killed / Segmentation fault

**Error:**

```
Killed
```

or

```
Segmentation fault (core dumped)
```

**Cause:** Out of memory or system limits.

**Solution:**

1. Close other applications
1. Use thread limits (see [Performance Issues](#performance--memory-issues))
1. Use smaller test dataset first
1. Increase system swap space
1. Run on machine with more RAM (16GB+ recommended)

### Workflow hangs / no progress

**Symptom:** Process runs for hours without output or progress.

**Cause:** Large candidate library (62K+ features) requires significant computation time.

**Solution:**

1. **Be patient:** Empirical null screening with 62K candidates can take 20-30 minutes
1. Check CPU usage: `top` or Activity Monitor
1. If CPU usage is high (>90%), it's working — wait longer
1. If CPU usage is low, process may be stuck — restart with smaller library
1. Use smaller test dataset first to validate workflow

______________________________________________________________________

## Performance & Memory Issues

### Out of memory

**Error:**

```
MemoryError
```

or system becomes unresponsive

**Cause:** Workflow uses too much RAM for available system memory.

**Solution:**

**Option 1:** Use thread limits (reduces memory usage):

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    pixi run manuscript-reproduce --config configs/datasets/test_3k.yml
```

**Option 2:** Reduce candidate library size:

```python
# Use fewer interaction pairs
import pandas as pd

catalog = pd.read_parquet('feature_catalog.parquet')

# Keep all first-order
first_order = catalog[catalog['feature_type'] == 'first_order']

# Keep only top 5000 interaction pairs (example)
interactions = catalog[catalog['feature_type'] == 'interaction'].head(5000)

# Keep all nonlinear
nonlinear = catalog[catalog['feature_type'] == 'nonlinear']

# Combine
reduced_catalog = pd.concat([first_order, interactions, nonlinear])
reduced_catalog.to_parquet('feature_catalog_reduced.parquet')
```

**Option 3:** Use smaller test dataset:

- Start with 1k samples instead of 3k or 20k
- Validate workflow succeeds
- Scale up gradually

**Option 4:** Upgrade hardware:

- Minimum: 8GB RAM
- Recommended: 16GB RAM
- Ideal: 32GB+ RAM

### Workflow too slow

**Symptom:** Workflow takes hours to complete.

**Cause:** Large dataset, large candidate library, or inefficient threading.

**Solution:**

**Quick wins:**

1. Use thread limits (can actually speed up by reducing contention):

   ```bash
   env OMP_NUM_THREADS=1 ... pixi run manuscript-reproduce --config ...
   ```

1. Use smaller test dataset for development/debugging

**Expected runtimes:**

- Demo mode (synthetic data): ~2 minutes
- 3k samples, 62K candidates: ~20-30 minutes
- 20k samples, 26K candidates: ~2 hours
- 20k samples, 62K candidates: ~4 hours

**Which stage is slowest:**

- Empirical null screening (stage 2): 50-70% of total time
- Interaction discovery (stage 3): 20-30% of total time
- Other stages: 10-20% combined

______________________________________________________________________

## Result Quality Issues

### nRMSE > 1.0 or NaN

**Symptom:** Final holdout nRMSE is greater than 1.0 or NaN.

**Cause:** Model not learning, data issues, or normalization problem.

**Diagnosis:**

```python
import pandas as pd

# Load holdout performance
perf = pd.read_parquet('artifacts/output/final_artifacts/holdout_performance.csv')
print(perf[['output_name', 'nrmse_point']])

# Check for NaN
print(f"NaN nRMSE count: {perf['nrmse_point'].isna().sum()}")

# Check for > 1.0
print(f"nRMSE > 1.0 count: {(perf['nrmse_point'] > 1.0).sum()}")
```

**Solutions:**

1. **Check data quality:** Look for NaN, inf, or extreme outliers in X and Y
1. **Verify sample_id alignment:** Ensure X, Y, holdout all match
1. **Check feature variance:** Features with zero variance cause problems
1. **Inspect selected features:** Are any features selected? Check `final_support_features.csv`
1. **Compare to baseline:** Is model better than predicting mean?

### Zero features in final model

**Symptom:** `final_support_features.csv` is empty or has zero rows.

**Cause:** All features filtered out in earlier stages.

**Diagnosis:**

```bash
# Check feature counts per stage
wc -l artifacts/output/empirical_null_screen/retained_terms.csv
wc -l artifacts/output/sparse_selection/final_stable_support.csv
wc -l artifacts/output/final_artifacts/final_support_features.csv
```

**Solutions:**

1. **Relax screening threshold:** Increase BH q from 0.10 to 0.20
1. **Reduce penalty:** Decrease EBIC gamma from 0.5 to 0.3
1. **Check candidate library:** Is it too small? Add more features
1. **Check data quality:** Low signal-to-noise ratio?

### All features retained at every stage

**Symptom:** Feature count stays the same across stages (no filtering).

**Cause:** Screening/selection not working properly.

**Solutions:**

1. **Check config parameters:** Verify thresholds are set correctly
1. **Verify candidate library format:** Check interaction names use colons
1. **Inspect screening statistics:** Look at p-values in provenance files

### Poor stability (Jaccard < 0.5)

**Symptom:** Stability diagnostics show low Jaccard index.

**Cause:** Model selection is unstable across resamples.

**Solutions:**

1. **Increase regularization:** Higher EBIC gamma or penalty strength
1. **More resamples:** Increase from 100 to 200 subsamples
1. **Larger subsample fraction:** Use 90% instead of 80%
1. **Reduce candidate library:** Fewer features = more stable selection
1. **More data:** Small sample size reduces stability

______________________________________________________________________

## Frequently Asked Questions

### How long should it take?

**Demo mode:** ~2 minutes\
**3k samples:** 20-30 minutes\
**20k samples:** 2-4 hours

Depends on:

- Sample size
- Candidate library size (26K vs 62K)
- Machine specs (CPU, RAM)
- Thread settings

### How much memory do I need?

**Minimum:** 8GB RAM\
**Recommended:** 16GB RAM\
**Ideal:** 32GB+ RAM

Memory usage scales with:

- Sample size × Number of outputs
- Candidate library size
- Number of threads

### Can I resume a failed run?

**No, the workflow doesn't support checkpointing.** You must restart from the beginning.

To avoid losing progress:

1. Test with small dataset first
1. Use thread limits to reduce crashes
1. Monitor memory usage
1. Save intermediate outputs if needed

### Can I run stages individually?

**Not directly with the config-driven workflow.** The workflow runs all 6 stages sequentially.

For individual stages, use the manuscript notebooks:

- `notebooks/manuscript/02_output_conditioning.ipynb`
- `notebooks/manuscript/03_empirical_null_screen.ipynb`
- etc.

### Why don't my results match the manuscript?

**This is expected!** Results will differ due to:

- Different sample sizes (3k vs 20k)
- Random seed variations
- Different candidate libraries

See [Interpreting Results](interpreting_results.md) for guidance on what constitutes "good agreement."

**Good agreement means:**

- Feature counts within 2x of manuscript
- nRMSE within 2x of manuscript
- PCA variance within ±10%

### What if I want to use my own data?

1. Format your data to match requirements (see `configs/datasets/README.md`)
1. Create metadata files
1. Generate feature catalog
1. Create config YAML
1. Run workflow

**Key requirements:**

- X and Y must have `sample_id` column
- Interactions must use colon-delimited format
- Holdout assignments use `'train'` and `'test'` values

### Can I modify the workflow parameters?

**Some parameters** are configurable in `configs/manuscript_case_study.yml`:

- PCA variance threshold
- Screening alpha (BH FDR)
- EBIC gamma (penalty strength)
- Stability thresholds

**Other parameters** are frozen in the manuscript contract and should not be changed without good reason.

### Where can I get help?

1. **Check this guide** for common issues
1. **Read the docs:**
   - [Interpreting Results](interpreting_results.md)
   - [Artifact Reference](artifact_reference.md)
   - `configs/datasets/README.md`
1. **Search GitHub issues:** Your question may already be answered
1. **Open a new issue:** Include error message, config file, and system specs

______________________________________________________________________

## Error Message Index

Quick lookup for specific error messages:

| Error Message                       | Solution Link                                                     |
| ----------------------------------- | ----------------------------------------------------------------- |
| `pixi: command not found`           | [Pixi not installed](#pixi-command-not-found)                     |
| `sample_id column` missing          | [Missing sample_id](#missing-sample_id-column)                    |
| `sample_id values...do not match`   | [Sample ID mismatch](#sample-id-mismatch)                         |
| `Feature 'AFSC' is not`             | [Missing scenario factors](#missing-scenario-factors-afsc-uaeoro) |
| `split column must contain`         | [Wrong split values](#wrong-split-values-in-holdout-assignments)  |
| `two-factor interaction candidates` | [Missing interactions](#feature-catalog-missing-interactions)     |
| `FileNotFoundError`                 | [File not found](#data-file-not-found)                            |
| `MemoryError`                       | [Out of memory](#out-of-memory)                                   |
| `Killed` or `Segmentation fault`    | [Process killed](#process-killed--segmentation-fault)             |
| `ModuleNotFoundError: bsm_rfm`      | [Python module not found](#python-module-not-found)               |
| `lock file is out of sync`          | [Lock file conflicts](#pixi-lock-file-conflicts)                  |

______________________________________________________________________

**See also:**

- [Interpreting Results](interpreting_results.md) - Understanding your output
- `configs/datasets/README.md` - Preparing your data
- [Artifact Reference](artifact_reference.md) - What each file contains
