# rfm-pipeline

A Python package for running reduced-form modeling (RFM) pipelines:
empirical-null screening, interaction discovery, nonlinear discovery,
sparse selection with stability filtering, and final table/figure generation.

Designed for large-scale simulation or observational datasets where the
number of candidate inputs and outputs is large and the structure of
input–output relationships is unknown.

## Installation

> **Note:** PyPI publication is pending. Until then, install from source.

From source using [Pixi](https://pixi.sh) (recommended):

```bash
git clone https://github.com/NatLabRockies/rfm-pipeline.git
cd rfm-pipeline
pixi install --locked
```

Or via pip from GitHub (requires git):

```bash
pip install git+https://github.com/NatLabRockies/rfm-pipeline.git
```

Once published to PyPI:

```bash
pip install rfm-pipeline  # coming soon
```

## Quick start

Use `run_canonical_workflow(...)` for the core API, or the script entry point for the
full config-driven manuscript workflow.

```python
from pathlib import Path

import pandas as pd

from rfm_pipeline import run_canonical_workflow

X_train = pd.read_parquet("X_train.parquet")
Y_train = pd.read_parquet("Y_train.parquet")
X_holdout = pd.read_parquet("X_holdout.parquet")
Y_holdout = pd.read_parquet("Y_holdout.parquet")

run = run_canonical_workflow(
    X_train,
    Y_train,
    X_holdout,
    Y_holdout,
    dataset_tag="my-run",
)
print(f"Holdout nRMSE: {run.holdout_summary['point_estimate'].iloc[0]:.4f}")
```

For the full config-driven manuscript reproduction, use the script entry point:

```bash
pixi run python scripts/run_manuscript_reproduction.py \
  --config configs/datasets/my_dataset.yml

# equivalent pixi task
pixi run manuscript-reproduce --config configs/datasets/my_dataset.yml
```

## Dataset config format

```yaml
case_study_input_matrix: /path/to/X.parquet
case_study_output_matrix: /path/to/Y.parquet
input_metadata: /path/to/input_metadata.parquet
output_metadata: /path/to/output_metadata.parquet
manuscript_feature_catalog: /path/to/feature_catalog.parquet
fixed_holdout_assignments: /path/to/holdout_assignments.parquet
output_root: /path/to/output  # optional
```

See `configs/datasets/template.yml` for a fully-annotated template.

## Pipeline stages

| Stage                        | Description                                         |
| ---------------------------- | --------------------------------------------------- |
| `output_conditioning`        | Normalize and validate inputs/outputs               |
| `empirical_null_screening`   | Permutation-based null screening with BH correction |
| `interaction_discovery`      | Tree-SHAP interaction scoring                       |
| `nonlinear_discovery`        | GAM-based nonlinear term detection                  |
| `sparse_selection`           | EBIC-selected L1 models with subsample stability    |
| `final_manuscript_artifacts` | HC3 Wald filter, OLS refit, holdout nRMSE, figures  |

## HPC / distributed execution

For large datasets, individual stages can be distributed across SLURM array
jobs. See `configs/sensitivity_study/study_spec.yml` for an example sensitivity
study configuration and `scripts/submit_sensitivity_study.sh` for submission.

## Sensitivity study

The package includes a built-in sensitivity study framework for evaluating
how pipeline hyperparameters affect NRMSE across synthetic DGPs:

```bash
pixi run python scripts/generate_sensitivity_study.py \
  --spec configs/sensitivity_study/study_spec.yml \
  --study-dir /path/to/sensitivity_output

SENSITIVITY_SPEC=configs/sensitivity_study/study_spec.yml \
  bash scripts/submit_sensitivity_study.sh --submit-all
```

## Repository gate

```bash
./test_repo.sh           # format + validate
./test_repo.sh --check   # validate only (no mutations)
./test_repo.sh --ci      # CI entrypoint
```

## Documentation

- `docs/setup_and_first_run.md` — Step-by-step setup guide
- `docs/configuration_reference.md` — Config field reference
- `docs/manuscript_alignment_audit.md` — Scientific alignment status
- `docs/ENGINEERING_MANIFEST.md` — Development roadmap

## Citation

If you use this software, please cite it using the metadata in `CITATION.cff`.

See `CHANGELOG.md` for release history.

## Scientific alignment status

The pipeline stages are tested and produce deterministic outputs. The current
status, summarized from `docs/manuscript_alignment_audit.md`:

- **Manuscript aligned:** tree-SHAP interaction discovery via gradient-boosted
  trees, GAM-based nonlinear discovery via cubic smoothing splines, OLS-based
  final fit and HC3 inferential filter, ablation comparisons, and bootstrap
  confidence intervals.
- **Partially aligned (provenance reconciled; private-script equivalence not
  yet externally validated):** empirical-null screening (matches manuscript
  BH q = 0.05 and 201 permutations) and PCA-based sparse stability selection.
- **Public surrogate (not a validated implementation of the private notebook
  reference):** the de-biased LASSO inference path remains a documented
  surrogate; users requiring exact manuscript-method equivalence should
  treat it as a generic L1/EBIC stability selector until externalization.

See `docs/manuscript_alignment_audit.md` for stage-by-stage detail.

## License

MIT. See `LICENSE`.
