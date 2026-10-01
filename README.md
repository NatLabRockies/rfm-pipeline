# rfm-pipeline

`rfm-pipeline` fits interpretable, multi-output reduced-form models from
simulation data. It provides feature screening, output-wise linear fitting,
holdout evaluation, portable coefficient tables, and an optional staged
workflow for interaction and nonlinear-feature discovery.

This repository creates models; it does not ship a fitted model. For the
ready-to-use Biomass Scenario Model reduced-form model, use
[`bsm-public-rf`](https://github.com/NatLabRockies/bsm-public-rf).

## Start here

| Goal                                      | Start with                                                         |
| ----------------------------------------- | ------------------------------------------------------------------ |
| Run a complete synthetic example          | [`examples/basic_workflow.py`](examples/basic_workflow.py)         |
| Fit prepared train/holdout tables         | [Python API quickstart](docs/quickstart.md)                        |
| Discover interactions and nonlinear terms | [Staged workflow configuration](docs/configuration_reference.md)   |
| Locate or interpret generated files       | [Artifact reference](docs/artifact_reference.md)                   |
| Scale a validated run                     | [HPC and distributed execution](docs/HPC_DISTRIBUTED_EXECUTION.md) |

## Install

Python 3.10–3.12 is supported. Pixi provides the locked environment used by
the examples and validation commands.

```bash
git clone https://github.com/NatLabRockies/rfm-pipeline.git
cd rfm-pipeline
pixi install --locked
```

To install only the package into an existing environment:

```bash
python -m pip install 'git+https://github.com/NatLabRockies/rfm-pipeline.git'
```

## Run the example

```bash
pixi run python examples/basic_workflow.py \
  --output-dir artifacts/basic-workflow
```

The example creates deterministic train and holdout data, fits the canonical
workflow, predicts the holdout rows, writes a manifest-aware coefficient
bundle, reloads it, and prints the holdout summary.

## Fit your data

Use the canonical API when your feature columns are already prepared:

```python
from pathlib import Path

import pandas as pd

from rfm_pipeline import predict_final_ols, run_canonical_workflow, write_postfit_bundle

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

predictions = predict_final_ols(run.final_ols_result, X_holdout)
write_postfit_bundle(run.artifacts, Path("artifacts/my-model"))
print(run.holdout_summary)
```

The canonical workflow applies multitask elastic-net screening, fits OLS for
each output, and computes holdout macro nRMSE with a bootstrap interval. Input
and output rows must already be aligned; train and holdout columns must match;
and modeled values must be numeric and finite.

Use the [staged workflow](docs/configuration_reference.md) only when you also
need output conditioning, empirical-null screening, interaction discovery,
nonlinear discovery, stability selection, or resumable stage artifacts.

## Documentation

| Need                          | Guide                                                      |
| ----------------------------- | ---------------------------------------------------------- |
| Installation and first run    | [Setup and first run](docs/setup_and_first_run.md)         |
| Canonical API options         | [Python API quickstart](docs/quickstart.md)                |
| Staged-run configuration      | [Configuration reference](docs/configuration_reference.md) |
| Generated files               | [Artifact reference](docs/artifact_reference.md)           |
| Result interpretation         | [Interpreting results](docs/interpreting_results.md)       |
| Failures and recovery         | [Troubleshooting](docs/troubleshooting.md)                 |
| Current implementation limits | [Workflow scope boundary](docs/scope_boundary.md)          |

The documentation site starts at [`docs/index.md`](docs/index.md).

## Validate, cite, and contribute

```bash
pixi run gate-fast  # routine development checks
pixi run gate       # complete repository validation
```

The complete gate validates the package, examples, notebooks, documentation,
and built wheel. It does not launch external HPC work.

- Cite the software using [`CITATION.cff`](CITATION.cff).
- See [`CONTRIBUTING.md`](CONTRIBUTING.md) before proposing changes.
- See [`CHANGELOG.md`](CHANGELOG.md) for release history.
- Report defects through [GitHub Issues](https://github.com/NatLabRockies/rfm-pipeline/issues).

Licensed under the [MIT License](LICENSE).
