# Quickstart

This package exposes a tested canonical reduced-form workflow that runs screening,
final OLS fitting, holdout evaluation, post-fit artifact assembly, and bundle writing.

## Minimal workflow

```python
from pathlib import Path

import pandas as pd

from bsm_rfm import load_postfit_bundle, run_canonical_workflow, write_postfit_bundle

X_train = pd.read_parquet("X_train.parquet")
Y_train = pd.read_parquet("Y_train.parquet")
X_holdout = pd.read_parquet("X_holdout.parquet")
Y_holdout = pd.read_parquet("Y_holdout.parquet")

run = run_canonical_workflow(
    X_train,
    Y_train,
    X_holdout,
    Y_holdout,
    dataset_tag="bsm-demo",
)
written = write_postfit_bundle(run.artifacts, Path("artifacts/bsm-demo"))
reloaded = load_postfit_bundle(Path("artifacts/bsm-demo"))
```

## What `run_canonical_workflow(...)` does

The current canonical workflow performs these implemented stages:

1. multitask elastic-net screening on the training data
1. final output-wise OLS fitting on the retained features
1. holdout macro nRMSE evaluation with bootstrap confidence intervals
1. canonical post-fit artifact assembly with a manifest payload

## Inputs and outputs

`run_canonical_workflow(...)` expects four aligned raw-scale pandas DataFrames:

- `X_train`
- `Y_train`
- `X_holdout`
- `Y_holdout`

It returns a `CanonicalWorkflowRun` containing:

- `screening_result`
- `final_ols_result`
- `holdout_summary`
- `artifacts`
- `artifact_format`

## Bundle writing and reload

`write_postfit_bundle(...)` writes the manifest and canonical tabular artifacts to disk.
When Parquet support is unavailable, tabular artifacts automatically fall back to CSV while
preserving the manifest file map.

`load_postfit_bundle(...)` is the visualization-side read helper for the canonical exported
post-fit tables.

## Current provenance boundary

The package implements the screening/final-fit/evaluation/export path directly. The upstream
Delta permutation-null screen is still represented through the recovered source adapter in
`bsm_rfm.null_screening`, while the recovered notebook-specific feature-expansion specification
is still only partially promoted into a source-driven canonical default.

For a machine-readable summary of those current limits, call
`bsm_rfm.workflow_scope_boundary_table()`.

## Reproducibility example

For a deterministic end-to-end example that also writes and reloads the bundle, run
`examples/end_to_end_reproducibility.py` from the repo source tree:

```bash
PYTHONPATH=src python examples/end_to_end_reproducibility.py --output-dir artifacts/toy-reproducibility-example
```
