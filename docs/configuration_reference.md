# Staged workflow configuration

The canonical Python API does not require YAML. This page covers the advanced
staged runner, which adds output conditioning, empirical-null screening,
interaction and nonlinear discovery, stability selection, and final
artifacts.

Some command and artifact names retain `manuscript` for compatibility. The
typed configuration itself is reusable.

## Start from the template

```bash
cp configs/workflow.template.yml configs/my-workflow.yml
```

Set `dataset.path` to a directory containing:

| File                                   | Required contents                                                                           |
| -------------------------------------- | ------------------------------------------------------------------------------------------- |
| `X.parquet`                            | `sample_id` plus numeric input columns                                                      |
| `Y.parquet`                            | matching `sample_id` plus numeric output columns                                            |
| `holdout_assignments.parquet`          | `sample_id` and `split`; `train`, `holdout`, `test`, `val`, and `validation` are normalized |
| `actual_input_feature_catalog.parquet` | feature catalog consumed by the discovery stages                                            |

Then run:

```bash
pixi run python tools/run_manuscript_pipeline.py configs/my-workflow.yml
```

Use a distinct `output.artifact_dir` for each run.

## Minimal configuration

```yaml
dataset:
  type: custom
  path: /absolute/path/to/dataset

runtime:
  n_jobs: 1

output:
  artifact_dir: ./artifacts/my-workflow-run/
  seed: 123
```

Unspecified values use the typed defaults in
`src/rfm_pipeline/config.py`.

## Configuration sections

| Section                           | Controls                                                                              |
| --------------------------------- | ------------------------------------------------------------------------------------- |
| `dataset`                         | Dataset label and root directory                                                      |
| `algorithm`                       | PCA variance target or fixed component count                                          |
| `runtime`                         | Worker count, memory limit, output batching, out-of-core I/O, and distributed backend |
| `stages.empirical_null_screening` | Permutation count, BH threshold, and optional retained-term cap                       |
| `stages.interaction_discovery`    | Error control, permutations, tree size/depth, and active-component limits             |
| `stages.nonlinear_discovery`      | EDF threshold and transform library                                                   |
| `stages.sparse_selection`         | Stability resamples, subsample fraction, thresholds, and candidate cap                |
| `stages.final_artifacts`          | Bootstrap settings, HC3 output subset, and pruning controls                           |
| `validation`                      | Optional fast-mode overrides                                                          |
| `output`                          | Artifact directory, random seed, and verbosity                                        |
| `categorical_inputs`              | Named categorical predictors and optional levels                                      |

The template shows the common fields. The dataclasses in
`rfm_pipeline.config` are the authoritative complete schema.

## Resume or stop at a stage

```bash
# Run through nonlinear discovery.
pixi run python tools/run_manuscript_pipeline.py configs/my-workflow.yml \
  --stop-stage nonlinear_discovery

# Continue from artifacts already written by the earlier stages.
pixi run python tools/run_manuscript_pipeline.py configs/my-workflow.yml \
  --start-stage sparse_selection
```

Valid stage names are:

1. `output_conditioning`
1. `empirical_null_screen`
1. `interaction_discovery`
1. `nonlinear_discovery`
1. `sparse_selection`
1. `final_manuscript_artifacts`

The final name is historical; it contains the final model and reporting
artifacts for any staged run.

## Track a long local run

```bash
pixi run workflow-run \
  --config configs/my-workflow.yml \
  --run-label my-study
```

The tracker records the command, log, stage observations, terminal status, and
elapsed time under `artifacts/workflow_runs/`.

## Two config formats that should not be mixed

- `configs/workflow.template.yml` and `configs/validation_*.yml` use the
  typed `WorkflowConfig` schema described above.
- `configs/datasets/template.yml` is a legacy BSM-compatible adapter for
  `scripts/run_manuscript_reproduction.py`. It uses six direct artifact
  paths and pins the BSM case-study contract.

For new non-BSM work, use the Python API or `workflow.template.yml`. See the
[`configs/datasets` guide](https://github.com/NatLabRockies/rfm-pipeline/tree/main/configs/datasets)
only when you need the case-study adapter.
