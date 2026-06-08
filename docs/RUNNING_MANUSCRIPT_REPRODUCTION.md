# Running Manuscript Reproduction

This guide explains how to run the manuscript reproduction workflow against different datasets.

## Quick Start

### Run with Config-Driven Unified Entry Point

The standard entrypoint is config-driven:

```bash
pixi run python tools/run_manuscript_pipeline.py configs/validation_300_sample_no_caps.yml
```

For long runs with timestamped tracking and durable logs:

```bash
pixi run workflow-run -- --config configs/validation_300_sample_no_caps.yml
```

### Runtime Investigation Ladder (single command)

Use this to estimate full-run runtime from small/medium/large profiling runs on your own dataset:

```bash
pixi run runtime-investigation \
  --base-config configs/validation_300_sample_no_caps.yml \
  --dataset-path /absolute/path/to/your_dataset_root \
  --output-root artifacts/runtime_investigation \
  --label my-dataset
```

For full details, see `docs/RUNTIME_INVESTIGATION_WORKFLOW.md`.

### Unified HPC Workflow Driver (single local command)

For HPC submissions and artifact collection without manual SSH/scp choreography:

```bash
pixi run hpc-workflow -- \
  --config configs/hpc/kestrel_workflow_orchestration.yml \
  --action submit
```

Then use:

```bash
pixi run hpc-workflow -- --config configs/hpc/kestrel_workflow_orchestration.yml --action status
pixi run hpc-workflow -- --config configs/hpc/kestrel_workflow_orchestration.yml --action collect
```

Set `pullback.mode` in the orchestration config to `manifest_only`, `reporting_bundle`, or `full` depending on local storage constraints.

Quick smoke path (2-node distributed test):

```bash
pixi run hpc-workflow -- \
  --config configs/hpc/kestrel_cpu_scale_2_smoke.yml \
  --action submit --generate-only --dry-run
```

This uses the lightweight `kestrel_cpu_scale_2_smoke.yml` orchestration
config (30-minute walltime target) to validate command plumbing without
submitting real jobs. For `interaction_discovery`, the small workflow
also pre-materializes prerequisite `output_conditioning` and
`empirical_null_screen` artifacts before SLURM submission when
`prepare_interaction_inputs: true` is set in the config.

### Run with Custom Output Directory

```bash
pixi run python tools/run_manuscript_pipeline.py \
  configs/validation_300_sample_no_caps.yml \
  --output-dir artifacts/my-run-2024
```

### Run with Config + Tracking + Label

```bash
pixi run workflow-run -- \
  --config configs/validation_300_sample_no_caps.yml \
  --run-label validation-no-caps
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

Thread pool size is controlled via standard environment variables before
launching the reproduction script:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  pixi run python scripts/run_manuscript_reproduction.py --config your-config.yml
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

## Parallel Execution with Thread Limiting

When running with `n_jobs=-1` (all CPUs), you must limit threads for each worker to prevent memory exhaustion and worker crashes. Use environment variables:

```bash
# Required for parallel execution (n_jobs=-1)
env \
  OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 \
  JOBLIB_TEMP_FOLDER=/tmp/joblib \
  pixi run python tools/run_manuscript_pipeline.py configs/validation_300_sample_no_caps.yml
```

**Why**: Each joblib worker spawns multiple threads for numpy/scipy operations. Without limiting these, thread count = (n_workers × threads_per_worker), which can exceed physical cores and cause memory pressure.

**Error if missing**: `joblib.externals.loky.process_executor.TerminatedWorkerError: A worker process managed by the executor was unexpectedly terminated`

**Alternative**: Use `n_jobs=2` or `n_jobs=4` in config instead of `-1` for fewer, more stable workers.

## Tracked Runs for Long-Term Debugging

`pixi run workflow-run` writes durable records:

- `artifacts/workflow_runs/<run-id>/pipeline.log`
- `artifacts/workflow_runs/<run-id>/run_metadata.json`
- `artifacts/workflow_runs/<run-id>/run_summary.json`
- `artifacts/workflow_runs/run_history.jsonl` (append-only run ledger)

The pipeline artifact directory also records marker files:

- `run_started.json`
- `run_complete.json`
- `run_failed.json`
- `run_interrupted.json`
