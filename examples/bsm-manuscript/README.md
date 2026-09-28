# BSM manuscript workflow example

This directory is the canonical, reproducible Biomass Scenario Model (BSM)
case study for `rfm-pipeline`. It contains the study-specific configuration,
execution wrappers, validation tests, publication artifacts, and rendered
figures that previously lived in `bsm-public-rf`.

Repository responsibilities are now intentionally separate:

- [`rfm-pipeline`](https://github.com/NatLabRockies/rfm-pipeline) owns the
  generic workflow and this complete publication example.
- [`bsm-public-rf`](https://github.com/NatLabRockies/bsm-public-rf) distributes
  the consumable BSM reduced-form model and the files required for inference.
- [`bsm-public-rf-manuscript`](https://github.com/NatLabRockies/bsm-public-rf-manuscript)
  owns the article and submission documents.

## Validate the committed example

Run these commands from the `rfm-pipeline` repository root:

```bash
pixi run bsm-manuscript-example-tests
pixi run bsm-manuscript-figure-check
```

The test command validates the committed artifact bundle and study
configuration. The figure check verifies the five manuscript-facing PDF
outputs without requiring the private simulator inputs or an HPC allocation.

To regenerate the figures from the committed CSV artifacts:

```bash
pixi run bsm-manuscript-figures
```

SVG output is always produced. PDF output is added when a supported Chromium
or Chrome executable is available.

## Full workflow

The complete workflow requires the BSM simulator data and, for the production
configuration, a SLURM environment. Start with
[`configs/manuscript_paths_template.yml`](configs/manuscript_paths_template.yml)
and the operational guide in
[`docs/G11_FINAL_EXECUTION.md`](docs/G11_FINAL_EXECUTION.md).

The example is fail-closed: local validation does not submit scheduler jobs,
and the production controls remain subject to the scientific gate records
carried with the example.

## Layout

- `configs/`: BSM study, campaign, and optional Kestrel configurations.
- `scripts/`: BSM adapters, reproduction entry points, and HPC wrappers.
- `tests/`: study-specific contract and artifact validation.
- `artifacts/`: committed model outputs, tables, and figure-source data.
- `figures/`: rendered publication figures.
- `docs/`: execution controls, provenance, and historical review records.

See [`MIGRATION.md`](MIGRATION.md) for the source commit, ownership mapping,
and branch-integration audit.
