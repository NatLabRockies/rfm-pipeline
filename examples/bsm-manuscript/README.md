# BSM workflow case study

This directory is the complete Biomass Scenario Model application of
`rfm-pipeline`. It preserves the study configuration, execution wrappers,
validation tests, model outputs, and rendered publication figures as one
reproducible example.

You do not need this directory to fit a generic reduced-form model. If you only
need BSM predictions, use
[`bsm-public-rf`](https://github.com/NatLabRockies/bsm-public-rf).

## What you can do locally

From the `rfm-pipeline` repository root:

```bash
# Validate the committed configs and artifacts.
pixi run bsm-manuscript-example-tests

# Validate the five committed figure outputs.
pixi run bsm-manuscript-figure-check

# Regenerate figures from committed CSV inputs.
pixi run bsm-manuscript-figures
```

The Pixi task names retain `manuscript` for compatibility. SVG output is
always produced; PDF output is added when a supported Chromium or Chrome
executable is available.

## Full study execution

The production workflow requires controlled BSM simulator data and a SLURM
environment. Start with
[`configs/manuscript_paths_template.yml`](configs/manuscript_paths_template.yml)
and [`docs/G11_FINAL_EXECUTION.md`](docs/G11_FINAL_EXECUTION.md).

The execution path is fail-closed. Local validation does not submit scheduler
jobs or authorize a production campaign, and the scientific gate records in
this example remain controlling.

## Layout

| Directory | Contents |
| --- | --- |
| `configs/` | Study, campaign, and optional Kestrel settings |
| `scripts/` | BSM adapters, reproduction commands, and HPC wrappers |
| `tests/` | Study-specific contract and artifact checks |
| `artifacts/` | Committed model outputs and figure-source tables |
| `figures/` | Rendered publication figures |
| `docs/` | Execution controls, provenance, and review records |

[`MIGRATION.md`](MIGRATION.md) records the source commit and ownership audit;
it is maintainer provenance rather than an onboarding guide.
