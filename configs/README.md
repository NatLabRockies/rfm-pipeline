# Configuration Files

Configuration YAML files for the unified manuscript pipeline runner.

## Quick Start

Run the pipeline with a config:

```bash
pixi run python tools/run_manuscript_pipeline.py configs/validation_300_sample_no_caps.yml
```

## Provided Configs

### `validation_300_sample_fast.yml`

Fast validation for CI and smoke testing:

- 5 null permutations (vs 1000)
- 8 stability resamples (vs 100)
- 20 bootstrap iterations (vs 200)
- Serial execution (1 worker)
- Output: `./artifacts/validation_300_sample_fast/`

**Use**: Quick testing, continuous integration

### `validation_300_sample_no_caps.yml`

Full validation with parallel execution:

- 1000 null permutations
- 100 stability resamples
- 200 bootstrap iterations
- All CPUs (`n_jobs: -1`)
- Output: `./artifacts/validation_300_sample_no_caps/`

**Use**: Production validation, reproducibility baseline

### `validation_300_sample_serial.yml`

Full validation with serial execution:

- Same parameters as `no_caps`
- Serial execution (1 worker)
- Output: `./artifacts/validation_300_sample_serial/`

**Use**: Memory-constrained environments, baseline comparisons

## Creating Custom Configs

1. Copy an existing config
1. Modify parameters (see `docs/CONFIGURATION_REFERENCE.md` for all options)
1. Update `output.artifact_dir` to avoid conflicts
1. Run: `pixi run python tools/run_manuscript_pipeline.py your_config.yml`

## Config Structure

All configs follow this structure:

```yaml
dataset:           # Dataset type and path
algorithm:         # Algorithm hyperparameters (variance threshold, components)
runtime:           # Parallelization (n_jobs, batch size)
stages:            # Per-stage parameters (permutations, bootstrap count, etc.)
validation:        # Fast-mode overrides for testing
output:            # Output directory, seed, verbosity
```

See `docs/CONFIGURATION_REFERENCE.md` for complete parameter documentation.

## CLI Overrides

Override config values from command line:

```bash
# Override output directory
pixi run python tools/run_manuscript_pipeline.py configs/validation_300_sample_no_caps.yml \
  --output-dir /tmp/custom

# Override random seed
pixi run python tools/run_manuscript_pipeline.py configs/validation_300_sample_no_caps.yml \
  --seed 999

# Enable fast-mode overrides
pixi run python tools/run_manuscript_pipeline.py configs/validation_300_sample_no_caps.yml \
  --fast
```

## Reproducibility

To reproduce an exact run:

1. Use the same config file
1. Use the same `--seed` value
1. Save the exact command

Example:

```bash
pixi run python tools/run_manuscript_pipeline.py configs/validation_300_sample_no_caps.yml \
  --seed 123 \
  --output-dir ./artifacts/exact_run_2026/
```
