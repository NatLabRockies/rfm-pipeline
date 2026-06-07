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

**Key fields** (sample — see the file for the full frozen contract):

```yaml
case_study:
  output_conditioning:
    variance_threshold_status: frozen_repo_decision_matching_live_code
    pca_variance_target: 0.90              # PCA dimensionality reduction target

  empirical_null_screen:
    statistic: coefficient_row_l2_norm
    bh_q_screen: 0.05                      # BH FDR threshold (manuscript baseline)
    permutation_count_B: 200               # null permutations; observed adds +1

  sparse_selection:
    ebic_gamma: 0.5                        # EBIC penalty
    public_implementation_method: ebic_l1_component_union_with_subsample_stability

  final_inferential_filter:
    interval_method: hc3_wald_95_percent_drop_if_zero_compatible_for_all_outputs
```

**Example change:** To loosen the BH FDR threshold:

```yaml
case_study:
  empirical_null_screen:
    bh_q_screen: 0.10                      # Loosened from manuscript baseline 0.05
```

> **Note.** The keys above are the contract-document field names consumed by
> `case_study_config_from_workflow_config` and the per-stage `spec_from_case_study_config`
> helpers in `src/rfm_pipeline/manuscript_stages.py`. They differ from the
> `WorkflowConfig` dataclass field names (e.g. `bh_q_threshold`) used by the
> generic non-case-study YAMLs under `configs/validation_*.yml`.

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
  artifact_root: "/Users/you/rfm-outputs"
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
from rfm_pipeline import resolve_manuscript_runtime
from pathlib import Path

rt = resolve_manuscript_runtime(Path.cwd())
print(f'Mode: {rt.mode}')
print(f'Output root: {rt.output_root}')
print(f'Artifacts available: {sorted(rt.artifact_paths)}')
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
  x_train: "/Volumes/SharedDrive/MyData/X_train.parquet"
  y_train: "/Volumes/SharedDrive/MyData/Y_train.parquet"
  x_holdout: "/Volumes/SharedDrive/MyData/X_holdout.parquet"
  y_holdout: "/Volumes/SharedDrive/MyData/Y_holdout.parquet"
  artifact_root: "/Users/you/rfm-outputs"
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
from rfm_pipeline import run_manuscript_reproduction_audit_stage, resolve_manuscript_runtime
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
from rfm_pipeline import resolve_manuscript_runtime
from pathlib import Path

rt = resolve_manuscript_runtime(Path.cwd())
print(f"Mode: {rt.mode}")
print(f"Output root: {rt.output_root}")
print(f"Repo root: {rt.repo_root}")
print(f"Local override used: {rt.local_override_used}")
print(f"Unresolved placeholders: {rt.unresolved_placeholders}")
print("Artifact paths:")
for key in sorted(rt.artifact_paths):
    print(f"  {key}: {rt.artifact_paths[key]}")
EOF
```

### Validate Configuration Parameters

```bash
pixi run python << 'EOF'
from rfm_pipeline import load_manuscript_case_study_config
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
from rfm_pipeline import load_manuscript_case_study_config
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
pixi run python -c "from rfm_pipeline import resolve_manuscript_runtime; from pathlib import Path; rt = resolve_manuscript_runtime(Path.cwd()); print(f'Mode: {rt.mode}; output_root: {rt.output_root}')"

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
from rfm_pipeline import run_manuscript_reproduction_audit_stage, resolve_manuscript_runtime
from pathlib import Path

rt = resolve_manuscript_runtime(Path.cwd())
result = run_manuscript_reproduction_audit_stage(rt)
print(f"Status: {result.audit_result.qa_status}")
EOF
```

______________________________________________________________________

## Reference: All Configuration Fields

| Config     | Section                  | Field                                   | Type  | Value (manuscript)                                          | Purpose                             |
| ---------- | ------------------------ | --------------------------------------- | ----- | ----------------------------------------------------------- | ----------------------------------- |
| case_study | output_conditioning      | variance_filter.epsilon_var             | float | 1e-12                                                       | Min train variance to retain output |
| case_study | output_conditioning      | snr_filter.epsilon_snr                  | float | 0.01                                                        | Minimum SNR for retention           |
| case_study | output_conditioning      | temporary_reduction.retained_components | int   | 20                                                          | PCA component count                 |
| case_study | empirical_null_screen    | bh_q_screen                             | float | 0.05                                                        | BH FDR threshold                    |
| case_study | empirical_null_screen    | permutation_count_B                     | int   | 200                                                         | Null permutations (observed + B)    |
| case_study | empirical_null_screen    | statistic                               | str   | coefficient_row_l2_norm                                     | Screening statistic                 |
| case_study | sparse_selection         | ebic_gamma                              | float | 0.5                                                         | EBIC penalty weight                 |
| case_study | stability                | jaccard_threshold                       | float | 0.75                                                        | Stability Jaccard cutoff            |
| case_study | stability                | spearman_threshold                      | float | 0.9                                                         | Stability Spearman cutoff           |
| case_study | final_inferential_filter | interval_method                         | str   | hc3_wald_95_percent_drop_if_zero_compatible_for_all_outputs | HC3 Wald rule                       |
| case_study | interaction_discovery    | null_threshold_quantile                 | float | 0.995                                                       | SHAP-interaction null quantile      |
| case_study | nonlinear_discovery      | curvature_rule                          | str   | edf_gt_1_and_smooth_pvalue_lt_0p01                          | GAM curvature acceptance            |

> **Note.** The keys above are from the **case-study contract YAML**
> (`configs/manuscript_case_study.yml`); they are frozen-from-manuscript
> values consumed by `manuscript_runtime.load_manuscript_case_study_config`.
> The `WorkflowConfig` dataclass YAMLs (`configs/validation_*.yml`, HPC
> YAMLs) use different field names — e.g. `bh_q_threshold` (not
> `bh_q_screen`), `n_permutations` (not `permutation_count_B`), `lasso_alpha_grid_size`,
> `n_stability_subsamples`, `subsample_fraction` — and live under top-level
> `stages:`. The earlier section in this document describes both forms; see
> `src/rfm_pipeline/config.py` for the authoritative dataclass schema.

| paths | - | x_train | path | demo | Training feature matrix |
| paths | - | y_train | path | demo | Training output matrix |
| paths | - | x_holdout | path | demo | Holdout feature matrix |
| paths | - | y_holdout | path | demo | Holdout output matrix |
| paths | - | artifact_root | path | `temp` | Output directory for results |
