# Configuration Reference

This document describes every configuration file in the repository and how to customize them for your workflow.

## Configuration File Structure

The repository uses YAML configuration files in two locations:

| Location         | Purpose                        | Mutability                        |
| ---------------- | ------------------------------ | --------------------------------- |
| `configs/`       | Tracked defaults and templates | Read-only (checked into git)      |
| `configs/local/` | Your local overrides           | Not tracked (add to `.gitignore`) |

The runtime loads configuration in this priority order:

1. **Defaults** from `configs/` (lowest priority)
1. **Local overrides** from `configs/local/` (highest priority, if present)

This allows you to customize parameters without modifying tracked files.

## Core Configuration Files

### `configs/manuscript_case_study.yml`

**Purpose:** Frozen workflow contract for the manuscript case study.

**When to edit:** Only when changing the scientific workflow (rare). Usually read-only.

**Key fields:**

```yaml
output_conditioning:
  variance_filter_threshold: 0.01    # Min train variance to retain output
  pca_variance_retained: 0.99        # PCA dimensionality reduction target

empirical_null_screen:
  method: "permutation"              # Null screening method
  alpha: 0.10                        # BH FDR threshold
  n_permutations: 1000               # Permutation samples for null dist.

sparse_selection:
  ebic_gamma: 0.5                    # EBIC penalty
  l1_ratio: 1.0                      # Lasso (1.0) vs. Ridge (0.0)

final_ols:
  hc3_alpha: 0.05                    # HC3 Wald inferential filter (95%)
```

**Example change:** To use a stricter p-value threshold:

```yaml
empirical_null_screen:
  alpha: 0.05  # Changed from 0.10
```

### `configs/manuscript_runtime.yml`

**Purpose:** Notebook execution order and runtime paths.

**When to edit:** Only to change notebook execution sequence or runtime paths (rare).

**Key fields:**

```yaml
notebook_execution_order:
  - notebook: "00_case_study_data_intake.ipynb"
    output_key: "artifact_root"
  - notebook: "01_candidate_library_audit.ipynb"
  - notebook: "02_output_conditioning.ipynb"
  # ... stages 3-8 follow

runtime_resolution_mode: "auto"  # or "demo" or "real"
```

### `configs/manuscript_data_contract.yml`

**Purpose:** Data schema and validation constraints.

**When to edit:** Only when input data schema changes (rare).

**Key fields:**

```yaml
x_train:
  shape: [18000, 352]               # Expected feature matrix dimensions
  dtype: "float64"
  required_columns: null             # or list of required column names

y_train:
  shape: [18000, 23]                # Expected output matrix dimensions
  dtype: "float64"
```

______________________________________________________________________

## Local Configuration (Your Customizations)

### `configs/local/manuscript_paths.local.yml`

**Purpose:** Point to your local data files and output directory.

**When to create:** When running with real data (optional for demo).

**How to create:**

```bash
cp configs/manuscript_paths.template.yml configs/local/manuscript_paths.local.yml
```

**Edit the file:**

```yaml
# Example configuration
paths:
  # ABSOLUTE paths to your data files
  x_train: "/Users/you/data/X_train.parquet"
  y_train: "/Users/you/data/Y_train.parquet"
  x_holdout: "/Users/you/data/X_holdout.parquet"
  y_holdout: "/Users/you/data/Y_holdout.parquet"

  # Where to write outputs
  artifact_root: "/Users/you/bsm-outputs"
```

**Important notes:**

- Use **absolute paths** (start with `/` on macOS/Linux, `C:\` on Windows)
- Use **double quotes** around paths with spaces
- Use `~` (home directory) for paths like `~/data/X_train.parquet`
- Files must exist and be readable
- Artifact root will be created if missing

### `configs/local/custom_case_study.yml` (Optional)

**Purpose:** Override workflow parameters for your custom data.

**When to create:** When running with custom data and different parameters.

**Example:**

```yaml
output_conditioning:
  variance_filter_threshold: 0.05   # Stricter filter for your data
  pca_variance_retained: 0.95       # Less aggressive PCA

empirical_null_screen:
  alpha: 0.05                       # Stricter BH threshold
  n_permutations: 500               # Fewer permutations (faster)
```

______________________________________________________________________

## Runtime Mode Detection

The runtime automatically detects which mode to run based on file presence:

```
Does configs/local/manuscript_paths.local.yml exist?
  ├─ YES: Does every referenced file exist and is readable?
  │        ├─ YES → Mode: "real" (uses your data)
  │        └─ NO  → Mode: "demo" (falls back to synthetic)
  └─ NO: Mode: "demo" (uses built-in synthetic data)
```

### Check Your Runtime Mode

```bash
pixi run python -c "
from bsm_rfm import resolve_manuscript_runtime
from pathlib import Path

rt = resolve_manuscript_runtime(Path.cwd())
print(f'Mode: {rt.mode}')
print(f'X train shape: {rt.x_train.shape}')
print(f'Artifact root: {rt.artifact_root}')
"
```

### Force Demo Mode (for testing)

```bash
# Temporarily rename the local config
mv configs/local/manuscript_paths.local.yml configs/local/manuscript_paths.local.yml.bak

# Run in demo mode
pixi run jupyter notebook notebooks/manuscript/

# Restore the config
mv configs/local/manuscript_paths.local.yml.bak configs/local/manuscript_paths.local.yml
```

______________________________________________________________________

## Example Configurations

### Configuration 1: Demo Mode (Default)

**Files:** None needed (uses built-ins)

**Runtime:** ~2 min for full 9-stage chain

**Use case:** First run, validation, CI testing

```bash
# Just run it—no config needed
PYTHONPATH=src python examples/end_to_end_reproducibility.py \
  --output-dir artifacts/demo-run \
  --run-manuscript-chain \
  --manuscript-output-dir artifacts/demo-run/manuscript
```

### Configuration 2: Real Data from Shared Server

**Files:** Create `configs/local/manuscript_paths.local.yml`:

```yaml
paths:
  x_train: "/Volumes/SharedDrive/BSM_Data/X_train.parquet"
  y_train: "/Volumes/SharedDrive/BSM_Data/Y_train.parquet"
  x_holdout: "/Volumes/SharedDrive/BSM_Data/X_holdout.parquet"
  y_holdout: "/Volumes/SharedDrive/BSM_Data/Y_holdout.parquet"
  artifact_root: "/Users/you/bsm-outputs"
```

**Verify:**

```bash
pixi run python -c "
from pathlib import Path
import yaml
with open('configs/local/manuscript_paths.local.yml') as f:
    cfg = yaml.safe_load(f)
for key, path in cfg['paths'].items():
    p = Path(path).expanduser()
    print(f'{key}: exists={p.exists()}')"
```

**Run:**

```bash
pixi run jupyter notebook notebooks/manuscript/
# Notebooks auto-detect and use real data
```

### Configuration 3: Custom Data with Modified Parameters

**Files:**

1. Create `configs/local/manuscript_paths.local.yml` pointing to your data
1. Create `configs/local/custom_case_study.yml` with modified parameters:

```yaml
output_conditioning:
  variance_filter_threshold: 0.02   # Stricter for your data
  pca_variance_retained: 0.95       # Adjust PCA

empirical_null_screen:
  alpha: 0.05                       # Different BH threshold
  n_permutations: 2000              # More permutations

sparse_selection:
  ebic_gamma: 0.25                  # Different EBIC tuning
```

**Run:**

```bash
pixi run python -c "
from bsm_rfm import run_manuscript_reproduction_audit_stage, resolve_manuscript_runtime
from pathlib import Path

rt = resolve_manuscript_runtime(Path.cwd())
result = run_manuscript_reproduction_audit_stage(rt)
print(f'All stages completed. Audit status: {result.audit_result.qa_status}')
"
```

______________________________________________________________________

## Configuration Validation

### Check Runtime Paths

```bash
pixi run python << 'EOF'
from bsm_rfm import resolve_manuscript_runtime
from pathlib import Path

rt = resolve_manuscript_runtime(Path.cwd())
print(f"Mode: {rt.mode}")
print(f"X train: {rt.x_train.shape} {rt.x_train_path if rt.mode == 'real' else '(demo)'}")
print(f"Y train: {rt.y_train.shape} {rt.y_train_path if rt.mode == 'real' else '(demo)'}")
print(f"Holdout X: {rt.x_holdout.shape}")
print(f"Holdout Y: {rt.y_holdout.shape}")
print(f"Artifact root: {rt.artifact_root}")
EOF
```

### Validate Configuration Parameters

```bash
pixi run python << 'EOF'
from bsm_rfm import load_manuscript_case_study_config
from pathlib import Path

config = load_manuscript_case_study_config(Path.cwd())
print("Output Conditioning:")
for key, val in config.output_conditioning.items():
    print(f"  {key}: {val}")
print("Empirical Null Screen:")
for key, val in config.empirical_null_screen.items():
    print(f"  {key}: {val}")
EOF
```

______________________________________________________________________

## Troubleshooting Configuration Issues

### Issue: "File not found" when using real data

**Diagnosis:**

```bash
python -c "
from pathlib import Path
path = Path('/Users/you/data/X_train.parquet').expanduser()
print(f'Path: {path}')
print(f'Exists: {path.exists()}')
print(f'Is file: {path.is_file()}')
print(f'Readable: {path.stat().st_mode & 0o400}')
"
```

**Fixes:**

- Check path is absolute (not relative)
- Verify file exists: `ls /path/to/file`
- Check permissions: `chmod 644 /path/to/file`
- Use `~` for home: `~/data/X_train.parquet` expands correctly

### Issue: Still using demo data after setting local config

**Solution:**

```bash
# Force reload
rm -rf ~/.cache/jupyter  # Clear notebook cache
pkill jupyter            # Restart kernel

# Verify config is found
python -c "
from pathlib import Path
cfg_path = Path('configs/local/manuscript_paths.local.yml')
print(f'Config exists: {cfg_path.exists()}')
"

# Then restart notebook
pixi run jupyter notebook notebooks/manuscript/
```

### Issue: Parameters not taking effect

**Solution:**

1. Verify override file is in `configs/local/`, not `configs/`
1. Use correct YAML indentation (2 spaces, not tabs)
1. Restart Python kernel
1. Test configuration loads:

```bash
pixi run python -c "
from bsm_rfm import load_manuscript_case_study_config
from pathlib import Path
cfg = load_manuscript_case_study_config(Path.cwd())
# Print to verify it's your custom config
print(cfg.empirical_null_screen)
"
```

______________________________________________________________________

## Common Configuration Patterns

### Pattern 1: Reproduce Public Demo Exactly

No configuration needed. Just run:

```bash
PYTHONPATH=src python examples/end_to_end_reproducibility.py \
  --output-dir artifacts/exact-demo --run-manuscript-chain \
  --manuscript-output-dir artifacts/exact-demo/manuscript
```

Output: Deterministic, reproducible on any machine.

### Pattern 2: Run with Your Real Data

```bash
# 1. Set up local config
cp configs/manuscript_paths.template.yml configs/local/manuscript_paths.local.yml
# Edit configs/local/manuscript_paths.local.yml with your paths

# 2. Verify
pixi run python -c "from bsm_rfm import resolve_manuscript_runtime; from pathlib import Path; rt = resolve_manuscript_runtime(Path.cwd()); print(f'Mode: {rt.mode}; X shape: {rt.x_train.shape}')"

# 3. Run
PYTHONPATH=src python examples/end_to_end_reproducibility.py \
  --output-dir artifacts/my-data --run-manuscript-chain \
  --manuscript-output-dir artifacts/my-data/manuscript
```

### Pattern 3: Custom Data, Different Parameters

```bash
# 1. Set up data paths
cp configs/manuscript_paths.template.yml configs/local/manuscript_paths.local.yml
# Edit with your paths

# 2. Set up custom parameters
cat > configs/local/custom_case_study.yml << 'EOF'
output_conditioning:
  pca_variance_retained: 0.95
empirical_null_screen:
  alpha: 0.05
EOF

# 3. Run
PYTHONPATH=src python << 'EOF'
from bsm_rfm import run_manuscript_reproduction_audit_stage, resolve_manuscript_runtime
from pathlib import Path

rt = resolve_manuscript_runtime(Path.cwd())
result = run_manuscript_reproduction_audit_stage(rt)
print(f"Status: {result.audit_result.qa_status}")
EOF
```

______________________________________________________________________

## Reference: All Configuration Fields

| Config     | Section               | Field                     | Type  | Default       | Purpose                                 |
| ---------- | --------------------- | ------------------------- | ----- | ------------- | --------------------------------------- |
| case_study | output_conditioning   | variance_filter_threshold | float | 0.01          | Minimum train variance to retain output |
| case_study | output_conditioning   | pca_variance_retained     | float | 0.99          | PCA cumulative variance target          |
| case_study | empirical_null_screen | method                    | str   | "permutation" | Null screening method                   |
| case_study | empirical_null_screen | alpha                     | float | 0.10          | Benjamini-Hochberg FDR threshold        |
| case_study | empirical_null_screen | n_permutations            | int   | 1000          | Permutations for null distribution      |
| case_study | sparse_selection      | ebic_gamma                | float | 0.5           | EBIC penalty weight                     |
| case_study | sparse_selection      | l1_ratio                  | float | 1.0           | Elastic-net L1 ratio (1=Lasso)          |
| case_study | final_ols             | hc3_alpha                 | float | 0.05          | HC3 Wald filter (1 - confidence level)  |
| paths      | -                     | x_train                   | path  | demo          | Training feature matrix                 |
| paths      | -                     | y_train                   | path  | demo          | Training output matrix                  |
| paths      | -                     | x_holdout                 | path  | demo          | Holdout feature matrix                  |
| paths      | -                     | y_holdout                 | path  | demo          | Holdout output matrix                   |
| paths      | -                     | artifact_root             | path  | `temp`        | Output directory for results            |
