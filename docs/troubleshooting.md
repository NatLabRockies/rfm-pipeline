# Troubleshooting

## Installation and imports

### `pixi: command not found`

Install [Pixi](https://pixi.sh), restart the terminal, and verify:

```bash
pixi --version
pixi install --locked
```

### `ModuleNotFoundError: rfm_pipeline`

From a source checkout, run project commands through Pixi:

```bash
pixi run python examples/basic_workflow.py
```

For an external environment, install the package first.

## Input errors

### Train and holdout shapes do not align

Check column identity and order before fitting:

```python
assert X_train.columns.equals(X_holdout.columns)
assert Y_train.columns.equals(Y_holdout.columns)
assert len(X_train) == len(Y_train)
assert len(X_holdout) == len(Y_holdout)
```

Also check the row identifiers you use outside the numeric matrices. The
canonical API assumes the caller has already aligned rows.

### Missing `sample_id` or split errors

The staged runner requires `sample_id` in `X.parquet`, `Y.parquet`, and
`holdout_assignments.parquet`. The assignments file also needs `split`.
Accepted holdout labels include `holdout`, `test`, `val`, and
`validation`; they are normalized internally.

### Missing feature catalog

Place `actual_input_feature_catalog.parquet` in the configured dataset
directory. Its feature names must match columns or supported expressions used
by the staged workflow. It must include the columns documented in the
[Configuration reference](configuration_reference.md).

## Fit and result errors

### Screening retained no features

Confirm that:

- features and outputs are numeric and finite;
- train rows are correctly aligned;
- candidate features contain variation;
- the training set contains signal; and
- screening settings are appropriate for the sample size.

Do not bypass the error by inventing a feature. Diagnose the data or
prespecified screening configuration.

### nRMSE is non-finite

Check the holdout for missing values and confirm that at least one output has a
training-reference range above the minimum. Use the same output eligibility
policy when comparing runs.

## Long or interrupted runs

### Memory pressure

Set a conservative worker count and enable out-of-core loading:

```yaml
runtime:
  n_jobs: 1
  max_loaded_table_mb: 8000
  out_of_core:
    enabled: true
    chunk_size_mb: 512
    max_memory_budget_mb: 8000
```

Use an explicit `temp_dir` on storage with enough capacity when spill-to-disk
is enabled.

### Resume from completed stages

The staged runner supports stage windows:

```bash
pixi run python tools/run_manuscript_pipeline.py configs/my-workflow.yml \
  --start-stage sparse_selection
```

Resume only when the prior stage artifacts belong to the same data and
configuration. Inspect the run markers and summaries first.

### A stage appears stuck

Check `runtime_diagnostics/stage_progress.json`, the tracked workflow log, CPU
use, available memory, and free temporary-storage space. Runtime depends
strongly on rows, outputs, candidate pairs, permutations, resamples, and
hardware; historical BSM timings are not general estimates.

## Get help

Open a [GitHub issue](https://github.com/NatLabRockies/rfm-pipeline/issues)
with:

- the shortest reproducible command;
- package version or commit;
- operating system and Python version;
- relevant config with private paths removed;
- the full error and terminal marker; and
- input shapes and column names, without restricted data.
