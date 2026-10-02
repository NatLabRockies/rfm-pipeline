# rfm-pipeline

`rfm-pipeline` fits interpretable, multi-output reduced-form models from
prepared training and holdout tables. It screens features with a multitask
elastic net, fits output-wise ordinary least squares models, evaluates holdout
error, and exports portable coefficient and metadata tables.

## Quick start

```bash
git clone https://github.com/NatLabRockies/rfm-pipeline.git
cd rfm-pipeline
pixi install --locked
pixi run python examples/basic_workflow.py \
  --output-dir artifacts/basic-workflow
```

The example generates deterministic data, fits a model, prints predictions and
holdout metrics, and writes a bundle that can predict without refitting.

## Use your data

```python
from pathlib import Path

import pandas as pd

from rfm_pipeline import predict_from_postfit_bundle, run_canonical_workflow, write_postfit_bundle

X_train = pd.read_parquet("X_train.parquet")
Y_train = pd.read_parquet("Y_train.parquet")
X_holdout = pd.read_parquet("X_holdout.parquet")
Y_holdout = pd.read_parquet("Y_holdout.parquet")

run = run_canonical_workflow(
    X_train,
    Y_train,
    X_holdout,
    Y_holdout,
    dataset_tag="my-model",
)

bundle = Path("artifacts/my-model")
write_postfit_bundle(run.artifacts, bundle)
predictions = predict_from_postfit_bundle(bundle, X_holdout)
print(run.holdout_summary)
```

Rows must be aligned; train and holdout columns must match; column names must
be non-empty strings; and modeled values must be numeric and finite. The
workflow expects prepared feature columns; use the feature-expansion utilities
when you want to add named transformations or interactions before fitting.

## Documentation

| Need                            | Guide                                                |
| ------------------------------- | ---------------------------------------------------- |
| Install and run the example     | [Setup and first run](docs/setup_and_first_run.md)   |
| Fit and predict with DataFrames | [Python API quickstart](docs/quickstart.md)          |
| See a released application      | [BSM application](docs/bsm_application.md)           |
| Understand generated files      | [Artifact reference](docs/artifact_reference.md)     |
| Interpret evaluation results    | [Interpreting results](docs/interpreting_results.md) |
| Diagnose common failures        | [Troubleshooting](docs/troubleshooting.md)           |
| Browse functions and classes    | [API reference](docs/api.rst)                        |

## Validate and contribute

```bash
pixi run gate-fast  # routine style and test checks
pixi run gate       # release checks, including docs and package builds
```

- Cite the software using [`CITATION.cff`](CITATION.cff).
- Read [`CONTRIBUTING.md`](CONTRIBUTING.md) before proposing changes.
- See [`CHANGELOG.md`](CHANGELOG.md) for release history.

Licensed under the [MIT License](LICENSE).
