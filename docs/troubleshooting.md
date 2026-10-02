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
assert X_train.index.equals(Y_train.index)
assert X_holdout.index.equals(Y_holdout.index)
```

The canonical API rejects duplicate labels and mismatched index or column
order instead of silently realigning tables.

## Fit and result errors

### Screening retained no features

Confirm that:

- features and outputs are numeric and finite;
- train rows are correctly aligned;
- candidate features contain variation;
- the training set contains signal; and
- screening settings are appropriate for the sample size.

Do not bypass the error by inventing a feature. Diagnose the data and screening
settings.

### nRMSE is non-finite

Check the holdout for missing values and confirm that at least one output has a
training-reference range above the minimum. Use the same output eligibility
policy when comparing runs.

### Fitting is slow or uses too much memory

Reduce the number of candidate features, outputs, cross-validation folds, or
bootstrap replicates. `fit_final_ols(...)` also accepts `output_batch_size` to
fit output columns in smaller batches. Benchmark on a representative subset
before scaling to the full dataset.

## Get help

Open a [GitHub issue](https://github.com/NatLabRockies/rfm-pipeline/issues)
with:

- the shortest reproducible command;
- package version or commit;
- operating system and Python version;
- relevant fitting options with private paths removed;
- the full error; and
- input shapes and column names, without restricted data.
