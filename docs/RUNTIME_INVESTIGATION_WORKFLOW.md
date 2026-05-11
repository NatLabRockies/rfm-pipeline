# Runtime Investigation Workflow

Use this workflow to estimate full-run runtime from a controlled small → medium → large ladder on your own dataset.

## One-command entrypoint

```bash
pixi run runtime-investigation \
  --base-config configs/validation_300_sample_no_caps.yml \
  --dataset-path /absolute/path/to/your_dataset_root \
  --output-root artifacts/runtime_investigation \
  --label my-dataset
```

For realistic estimates, use a base config with `validation.fast_mode: false`.
`fast_mode: true` is useful for smoke checks of the ladder plumbing only.

`dataset-path` must point to a folder containing:

- `X.parquet`
- `Y.parquet`
- `feature_catalog.parquet`
- `holdout_assignments.parquet`

## What the command does

1. Generates ladder configs under `.../configs/`:
   - `small.yml`
   - `medium.yml`
   - `large.yml`
1. Runs each profile sequentially through `tools/run_manuscript_pipeline.py`.
1. Collects runtime metrics from run markers and stage diagnostics.
1. Produces a projected base-config runtime from observed sparse/final stage rates.
1. Writes reports and monitor command artifacts.

## Output layout

```text
artifacts/runtime_investigation/<timestamp>-<label>/
  configs/
  runs/
    small/
    medium/
    large/
  report/
    runtime_investigation_summary.csv
    runtime_investigation_summary.md
    runtime_projection.json
    monitor_command.txt
```

## Monitoring during execution

Use the generated monitor command:

```bash
cat artifacts/runtime_investigation/<timestamp>-<label>/report/monitor_command.txt
```

Or run directly (10-minute refresh):

```bash
scripts/watch_final_cost_ladder.sh \
  artifacts/runtime_investigation/<timestamp>-<label>/runs \
  600
```

## Useful flags

```bash
# Generate configs only (no runs)
pixi run runtime-investigation --base-config ... --dataset-path ... --generate-only

# Re-run profiles even if run_complete.json exists
pixi run runtime-investigation --base-config ... --dataset-path ... --force-rerun

# Stage-window runtime probe
pixi run runtime-investigation --base-config ... --dataset-path ... \
  --start-stage sparse_selection --stop-stage final_manuscript_artifacts
```

## Runtime projection model

Projection is computed from completed ladder profiles only:

- sparse rate = median(`sparse_seconds / sparse_units`)
- final rate = median(`final_seconds / final_units`)
- projected sparse seconds = sparse rate × target sparse units
- projected final seconds = final rate × target final units

Target units come from the base config (`n_stability_subsamples`, `max_candidate_terms`, `bootstrap_count`).
