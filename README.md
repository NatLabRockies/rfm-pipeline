# rfm-pipeline

A Python package for running reduced-form modeling (RFM) pipelines:
empirical-null screening, interaction discovery, nonlinear discovery,
sparse selection with stability filtering, and final table/figure generation.

Designed for large-scale simulation or observational datasets where the
number of candidate inputs and outputs is large and the structure of
input–output relationships is unknown.

## Installation

```bash
pip install rfm-pipeline
```

Or from source using [Pixi](https://pixi.sh):

```bash
git clone https://github.com/NatLabRockies/rfm-pipeline.git
cd rfm-pipeline
pixi install --locked
```

## Quick start

```python
from rfm_pipeline import ManuscriptPipelineConfig, run_manuscript_pipeline

cfg = ManuscriptPipelineConfig.from_yaml("configs/datasets/my_dataset.yml")
run_manuscript_pipeline(cfg, output_dir="artifacts/my-run")
```

Or via the command line:

```bash
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

| Stage                                 | Description                                         |
| ------------------------------------- | --------------------------------------------------- |
| `output_conditioning`                 | Normalize and validate inputs/outputs               |
| `empirical_null_screening`            | Permutation-based null screening with BH correction |
| `interaction_discovery`               | Tree-SHAP interaction scoring                       |
| `nonlinear_discovery`                 | GAM-based nonlinear term detection                  |
| `sparse_selection_and_stability`      | EBIC-selected L1 models with subsample stability    |
| `final_manuscript_tables_and_figures` | HC3 Wald filter, OLS refit, holdout nRMSE, figures  |

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
  --output-dir /path/to/sensitivity_output

pixi run python scripts/submit_sensitivity_study.sh \
  --study-dir /path/to/sensitivity_output
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

The pipeline stages are tested and produce deterministic outputs, but several
stages are public approximations of the manuscript-exact methods. See
`docs/manuscript_alignment_audit.md` for the current status ledger.

The package should not yet be described as a full exact implementation of every
manuscript method. Known scientific exactness gaps include the tree-SHAP
interaction workflow, GAM EDF/p-value nonlinear discovery, and the
de-biased-LASSO stability selection. Higher-level reconciliation remains before
the package claims complete manuscript reproduction.

## License

MIT. See `LICENSE`.
