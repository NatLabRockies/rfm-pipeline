# rfm-pipeline

`rfm-pipeline` fits interpretable reduced-form models for datasets with many
inputs and outputs. It provides screening, linear fitting, holdout evaluation,
artifact export, and optional staged discovery of interactions and nonlinear
terms.

## What the package provides

Two execution surfaces serve different needs:

- **Canonical Python API:** multitask elastic-net screening, output-wise OLS,
  holdout macro nRMSE with bootstrap intervals, and a portable model bundle.
  This is the best starting point for a new dataset.
- **Staged research workflow:** output conditioning, empirical-null screening,
  interaction and nonlinear discovery, stability selection, final OLS/HC3
  filtering, resumable execution, and distributed building blocks. Use this
  when you need the full research pipeline.

## Install

Python 3.10–3.12 is supported. A locked source environment is the recommended
way to run the repository examples:

```bash
git clone https://github.com/NatLabRockies/rfm-pipeline.git
cd rfm-pipeline
pixi install --locked
```

To install the package from GitHub into an existing environment:

```bash
python -m pip install 'git+https://github.com/NatLabRockies/rfm-pipeline.git'
```

## Run the two-minute example

```bash
pixi run python examples/basic_workflow.py \
  --output-dir artifacts/basic-workflow
```

The example creates a tiny train/holdout dataset, fits the canonical workflow,
writes a manifest-aware bundle, reloads it, and prints the holdout summary.

## Use your own data

```python
from pathlib import Path

import pandas as pd

from rfm_pipeline import run_canonical_workflow, write_postfit_bundle

X_train = pd.read_parquet("X_train.parquet")
Y_train = pd.read_parquet("Y_train.parquet")
X_holdout = pd.read_parquet("X_holdout.parquet")
Y_holdout = pd.read_parquet("Y_holdout.parquet")

run = run_canonical_workflow(
    X_train,
    Y_train,
    X_holdout,
    Y_holdout,
    dataset_tag="my-study",
)
write_postfit_bundle(run.artifacts, Path("artifacts/my-study"))
print(run.holdout_summary)
```

Rows and columns must already be aligned: train inputs with train outputs,
holdout inputs with holdout outputs, and identical feature/output columns
across splits.

## Find the right guide

| Need                             | Guide                                                              |
| -------------------------------- | ------------------------------------------------------------------ |
| Install and verify the package   | [Setup and first run](docs/setup_and_first_run.md)                 |
| Fit a model with the Python API  | [Quickstart](docs/quickstart.md)                                   |
| Run the staged workflow          | [Configuration reference](docs/configuration_reference.md)         |
| Find a generated file            | [Artifact reference](docs/artifact_reference.md)                   |
| Interpret a run                  | [Interpreting results](docs/interpreting_results.md)               |
| Scale across shards or SLURM     | [HPC and distributed execution](docs/HPC_DISTRIBUTED_EXECUTION.md) |
| Understand implementation limits | [Workflow scope boundary](docs/scope_boundary.md)                  |
| Resolve a failure                | [Troubleshooting](docs/troubleshooting.md)                         |

The full documentation site starts at [`docs/index.md`](docs/index.md).

## Validate a checkout

```bash
pixi run gate
```

The full gate is intentionally comprehensive and can take a long time. It
checks formatting, tests, the BSM case study, notebooks, documentation, and the
built wheel; it does not run the full BSM production analysis.

## Citation and license

Cite the software using [`CITATION.cff`](CITATION.cff). See
[`CHANGELOG.md`](CHANGELOG.md) for release history. Licensed under the
[MIT License](LICENSE).
