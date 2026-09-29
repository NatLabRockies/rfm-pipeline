# Case-study dataset adapter

This directory supports the historical
`scripts/run_manuscript_reproduction.py` entry point. It is useful for the
BSM-compatible stage chain, but it is not the recommended starting point for a
new generic workflow.

For new work, use either:

- the four-DataFrame Python API in
  [`docs/quickstart.md`](../../docs/quickstart.md); or
- `configs/workflow.template.yml` for the typed staged runner.

## Adapter config

Copy `template.yml` and provide six Parquet paths:

```yaml
case_study_input_matrix: /absolute/path/to/X.parquet
case_study_output_matrix: /absolute/path/to/Y.parquet
input_metadata: /absolute/path/to/input_metadata.parquet
output_metadata: /absolute/path/to/output_metadata.parquet
manuscript_feature_catalog: /absolute/path/to/feature_catalog.parquet
fixed_holdout_assignments: /absolute/path/to/holdout_assignments.parquet
output_root: /absolute/path/to/output  # optional
```

Run it with:

```bash
pixi run manuscript-reproduce --config configs/datasets/my-data.yml
```

This adapter loads the repository's frozen BSM case-study settings from
`configs/manuscript_case_study.yml`. Changing only the six paths does not
turn it into a neutral workflow configuration.

## Required table relationships

- Input and output matrices contain matching `sample_id` values.
- Input metadata names match input columns.
- Output metadata names match output columns.
- Feature-catalog terms resolve against the input matrix.
- Holdout assignments cover every sample and use train/test-compatible labels.

See [Staged workflow configuration](../../docs/configuration_reference.md) for
the generic dataset-directory layout.
