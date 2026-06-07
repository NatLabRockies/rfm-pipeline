# Setup and First Run Guide

This guide walks you through setting up your environment and running the complete reduced-form modeling workflow for the first time, from start to finish.

## Table of Contents

1. Prerequisites
1. Clone and Environment Setup
1. Understanding the Three Paths
1. Path 1: Demo Workflow (Recommended First Run)
1. Path 2: Adding Real Data
1. Path 3: Custom Data Workflow
1. Troubleshooting
1. What to Expect

## Prerequisites

You will need:

- **macOS, Linux, or Windows WSL2** environment
- **Git** (check with `git --version`)
- **Internet connection** to download dependencies

You do NOT need to pre-install Python, conda, or any data science tools—Pixi handles all of that.

## Clone and Environment Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/NatLabRockies/rfm-pipeline.git
cd rfm-pipeline
```

### Step 2: Install Pixi

Pixi is the environment and task manager for this repository. It automatically handles all Python dependencies.

```bash
# On macOS
curl -fsSL https://pixi.sh/install.sh | bash

# On Linux
curl -fsSL https://pixi.sh/install.sh | bash

# On Windows (WSL2)
curl -fsSL https://pixi.sh/install.sh | bash
```

After installation, restart your terminal or run:

```bash
source ~/.bashrc  # or ~/.zshrc if using zsh
```

Verify installation:

```bash
pixi --version
```

### Step 3: Initialize the Repository Environment

From inside the cloned repository directory:

```bash
pixi install --locked
```

This creates a fully isolated environment with all Python packages, Jupyter, and tools pinned to exact versions. **First-time setup takes 3–5 minutes.**

### Step 4: Verify Installation

```bash
# Test the main gate (should pass)
./test_repo.sh --check
```

If this passes, you're ready to run the workflow.

## Understanding the Three Paths

The package supports three execution paths:

| Path                  | Data                                 | Use Case                          | Time    |
| --------------------- | ------------------------------------ | --------------------------------- | ------- |
| **Demo**              | Built-in toy data                    | First run, validation, CI testing | ~2 min  |
| **Real Data (Local)** | Your local files via config override | Full case study on your machine   | ~10 min |
| **Custom Data**       | Your own dataset                     | Apply workflow to new data        | Varies  |

You start with **Demo** (requires no data files), then optionally add **Real Data**, then **Custom Data**.

## Path 1: Demo Workflow (Recommended First Run)

The demo workflow runs end-to-end on built-in synthetic data. No configuration needed.

### Step 1: Run the Canonical Workflow Example

This is the simplest possible entry point—runs the core workflow and writes results:

```bash
cd <repo-root>
pixi run python examples/end_to_end_reproducibility.py \
  --output-dir artifacts/my-first-run
```

**Expected output:**

- Creates `artifacts/my-first-run/` directory
- Writes screening results, final OLS model, and bundle
- Prints nRMSE metrics to console
- Completes in ~30 seconds

### Step 2: Run the Complete Manuscript-Reproduction Chain

This runs all 9 notebook stages and produces final tables/figures:

```bash
pixi run python examples/end_to_end_reproducibility.py \
  --output-dir artifacts/my-first-run \
  --run-manuscript-chain \
  --manuscript-output-dir artifacts/my-first-run/manuscript
```

**Expected output:**

- All 9 stages execute in dependency order
- Creates `artifacts/my-first-run/manuscript/` with:
  - `output_conditioning/` — filtered outputs and PCA representation
  - `empirical_null_screen/` — screened features
  - `interaction_discovery/` — interaction pairs
  - `nonlinear_discovery/` — nonlinear terms
  - `sparse_selection_stability/` — final stable support
  - `final_manuscript_artifacts/` — final model, tables, figures
  - `reproduction_audit/` — artifact manifest and QA checks
- Completes in ~2 minutes

### Step 3: Inspect the Results

```bash
# List all generated artifacts
find artifacts/my-first-run -name "*.csv" -o -name "*.svg" | head -20

# Check the audit summary (QA pass/fail)
cat artifacts/my-first-run/manuscript/reproduction_audit/audit_summary.csv

# Inspect final coefficients
head -20 artifacts/my-first-run/manuscript/final_manuscript_artifacts/final_model/coefficient_matrix_standardized.csv

# View a figure (on macOS)
open artifacts/my-first-run/manuscript/final_manuscript_artifacts/figures/figure_model_performance.svg
```

### Step 4: Run the Notebooks Interactively (Optional)

To walk through the workflow step-by-step in Jupyter:

```bash
pixi run jupyter notebook notebooks/manuscript/
```

Then open and run:

- `00_case_study_data_intake.ipynb` — loads or generates demo data
- `01_candidate_library_audit.ipynb` — inspects feature catalog
- `02_output_conditioning.ipynb` — applies PCA to outputs
- `03_empirical_null_screen.ipynb` — screens features
- ... (stages 4–8 follow)

The notebooks auto-detect demo mode and use synthetic data. No additional setup required.

**Estimated time: 10 minutes to run all notebooks interactively.**

## Path 2: Adding Real Data

Once you've verified the demo workflow works, you can add your own case-study data.

### Step 1: Locate Your Data Files

You will need four CSV or Parquet files:

| File                | Description                    | Rows    | Cols |
| ------------------- | ------------------------------ | ------- | ---- |
| `X_train.parquet`   | Training features (predictors) | ~18,000 | ~350 |
| `Y_train.parquet`   | Training outputs (responses)   | ~18,000 | ~23  |
| `X_holdout.parquet` | Holdout/test features          | ~2,000  | ~350 |
| `Y_holdout.parquet` | Holdout/test outputs           | ~2,000  | ~23  |

### Step 2: Create a Local Configuration Override

The repository ships with `configs/manuscript_paths.template.yml`. Create a local override to point to your files:

```bash
# Copy the template
cp configs/manuscript_paths.template.yml configs/local/manuscript_paths.local.yml
```

### Step 3: Edit the Local Configuration

Open `configs/local/manuscript_paths.local.yml` and update file paths:

```yaml
# Example (adjust paths for your machine):
paths:
  x_train: "/path/to/my-data/X_train.parquet"
  y_train: "/path/to/my-data/Y_train.parquet"
  x_holdout: "/path/to/my-data/X_holdout.parquet"
  y_holdout: "/path/to/my-data/Y_holdout.parquet"
  artifact_root: "<repo-root>/artifacts/real-data-output"
```

### Step 4: Validate the Configuration

```bash
pixi run python -c "
from rfm_pipeline import resolve_manuscript_runtime
from pathlib import Path

rt = resolve_manuscript_runtime(Path.cwd())
print(f'Runtime mode: {rt.mode}')
print(f'Output root: {rt.output_root}')
print(f'Local override used: {rt.local_override_used}')
print(f'Artifact paths: {sorted(rt.artifact_paths)}')
"
```

**Expected output** (if real data found):

```
Runtime mode: real
Output root: /path/to/artifacts/manuscript-case-study
Local override used: True
Artifact paths: ['case_study_input_matrix', 'case_study_output_matrix', ...]
```

If you still see demo data, check:

- File paths are correct and absolute
- Files exist and are readable
- No trailing spaces in YAML

### Step 5: Run with Real Data

```bash
# Run the canonical workflow
pixi run python examples/end_to_end_reproducibility.py \
  --output-dir artifacts/my-real-data-run \
  --run-manuscript-chain \
  --manuscript-output-dir artifacts/my-real-data-run/manuscript
```

Or run interactively:

```bash
pixi run jupyter notebook notebooks/manuscript/
```

The notebooks automatically detect real data and use it.

## Path 3: Custom Data Workflow

To apply the workflow to a completely different dataset:

### Step 1: Prepare Your Data

You need four aligned DataFrames:

```python
import pandas as pd

# Your data must have:
X_train = pd.read_csv("your_training_features.csv", index_col=0)  # shape: (n_train, p)
Y_train = pd.read_csv("your_training_outputs.csv", index_col=0)   # shape: (n_train, m)
X_holdout = pd.read_csv("your_test_features.csv", index_col=0)    # shape: (n_holdout, p)
Y_holdout = pd.read_csv("your_test_outputs.csv", index_col=0)     # shape: (n_holdout, m)

# Verify alignment:
assert X_train.shape[1] == X_holdout.shape[1], "Feature counts must match"
assert Y_train.shape[1] == Y_holdout.shape[1], "Output counts must match"
assert X_train.index.equals(Y_train.index), "Train indices must align"
```

### Step 2: Use the Minimal Workflow API

```python
from pathlib import Path
import pandas as pd
from rfm_pipeline import run_canonical_workflow, write_postfit_bundle

# Load your data
X_train = pd.read_csv("X_train.csv", index_col=0)
Y_train = pd.read_csv("Y_train.csv", index_col=0)
X_holdout = pd.read_csv("X_holdout.csv", index_col=0)
Y_holdout = pd.read_csv("Y_holdout.csv", index_col=0)

# Run the workflow
result = run_canonical_workflow(
    X_train=X_train,
    Y_train=Y_train,
    X_holdout=X_holdout,
    Y_holdout=Y_holdout,
    dataset_tag="my-custom-data",
)

# Write results
written = write_postfit_bundle(result.artifacts, Path("artifacts/my-custom-data"))

# Print metrics
print(f"Holdout nRMSE: {result.holdout_summary['point_estimate']:.4f}")
print(f"Final support size: {len(result.final_ols_result.retained_features)}")
```

See `docs/quickstart.md` for full API documentation.

## Troubleshooting

### Issue: "pixi: command not found"

**Solution:** Pixi was not added to your PATH. Restart your terminal or run:

```bash
source ~/.bashrc  # or ~/.zshrc for zsh
pixi --version
```

### Issue: "Could not locate the rfm-pipeline repository root"

**Solution:** Make sure you're running commands from inside the cloned repository:

```bash
cd /path/to/rfm-pipeline
```

### Issue: "ModuleNotFoundError: No module named 'rfm_pipeline'"

**Solution:** Ensure `PYTHONPATH` is set and Pixi environment is active:

```bash
cd /path/to/rfm-pipeline
pixi run python examples/end_to_end_reproducibility.py ...
```

Or use Pixi's persistent shell (note: repo policy disallows `pixi shell` for project tooling — prefer `pixi run` for reproducibility):

```bash
# Discouraged — kept for reference only:
# pixi shell && python examples/end_to_end_reproducibility.py ...
pixi run python examples/end_to_end_reproducibility.py ...
```

### Issue: "FileNotFoundError" when running with real data

**Solution:** Check your local config file:

```bash
# Verify the file exists
cat configs/local/manuscript_paths.local.yml

# Test each path
pixi run python -c "
from pathlib import Path
import yaml
with open('configs/local/manuscript_paths.local.yml') as f:
    cfg = yaml.safe_load(f)
for key, path in cfg['paths'].items():
    p = Path(path).expanduser()
    print(f'{key}: {p.exists()}')"
```

### Issue: Notebooks still use demo data after setting local config

**Solution:** Force a fresh Python kernel:

```bash
pixi run jupyter notebook --NotebookApp.kernel_spec_manager_class='IPython.kernel.kernelspec.KernelSpecManager'
```

Then restart the kernel: `Kernel → Restart & Clear Output`

### Issue: Tests fail with "File not found" errors

**Solution:** This is expected in CI where only demo data is available. Verify locally:

```bash
./test_repo.sh --check
```

Should pass with demo data.

## What to Expect

### Demo Workflow Timeline

| Step                  | Duration    | Output                           |
| --------------------- | ----------- | -------------------------------- |
| Clone + Pixi install  | 5–10 min    | Isolated environment ready       |
| Canonical workflow    | ~30 sec     | `artifacts/my-first-run/`        |
| Full manuscript chain | ~2 min      | All 9 stages with tables/figures |
| Notebook walkthrough  | ~10 min     | Interactive exploration          |
| **Total first run**   | **~15 min** | Complete end-to-end results      |

### Expected Artifacts

After the full manuscript chain, inspect:

```bash
artifacts/my-first-run/manuscript/
├── output_conditioning/
│   ├── pca_scores.csv (response PCA representation)
│   └── pca_loadings.csv
├── empirical_null_screen/
│   ├── retained_terms.csv (screened feature list)
│   └── feature_screening_statistics.csv
├── interaction_discovery/
│   ├── retained_interaction_pairs.csv
│   └── interaction_discovery_provenance.csv (note: public surrogate)
├── nonlinear_discovery/
│   ├── retained_transformations.csv
│   └── nonlinear_discovery_provenance.csv (note: public surrogate)
├── sparse_selection_stability/
│   ├── final_stable_support.csv (features to retain)
│   └── stability_feature_summary.csv
├── final_manuscript_artifacts/
│   ├── final_model/
│   │   ├── final_support_features.csv
│   │   ├── coefficient_matrix_standardized.csv
│   │   └── hc3_wald_intervals.csv (95% confidence intervals)
│   ├── tables/
│   │   ├── workflow_stage_summary.csv (Table 1 data)
│   │   └── model_performance.csv (Table 2 data)
│   └── figures/
│       ├── figure_model_performance.svg
│       └── figure_support_composition.svg
└── reproduction_audit/
    ├── artifact_manifest.csv (all output files and hashes)
    ├── metric_checks.csv (QA pass/fail status)
    └── audit_summary.csv (overall status)
```

### Key Files to Inspect

- **`reproduction_audit/audit_summary.csv`** — Overall workflow status (should be `qa_status = 'passed'`)
- **`final_manuscript_artifacts/final_model/coefficient_matrix_standardized.csv`** — Your fitted model
- **`final_manuscript_artifacts/tables/model_performance.csv`** — Holdout nRMSE and bootstrap CIs
- **`final_manuscript_artifacts/figures/figure_model_performance.svg`** — Final model performance plot

## Next Steps

- Review `docs/manuscript_alignment_audit.md` to understand where the workflow is manuscript-exact vs. approximate
- See `docs/configuration_reference.md` for all tunable parameters
- Check `docs/artifact_reference.md` for detailed artifact schema documentation
- Run `examples/end_to_end_reproducibility.py --help` for advanced options

## Getting Help

- **Setup issues?** Check the Troubleshooting section above
- **Data format questions?** See `docs/artifact_reference.md`
- **API questions?** See `docs/quickstart.md` and `docs/module_plan.md`
- **Scientific questions?** See `docs/manuscript_alignment_audit.md` and `docs/debiased_lasso_contract.md`
