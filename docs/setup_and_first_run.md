# Setup and first run

## Requirements

- macOS, Linux, or Windows
- Git
- [Pixi](https://pixi.sh) for the locked repository environment

Pixi installs the supported Python version and all dependencies.

## Install the repository

```bash
git clone https://github.com/NatLabRockies/rfm-pipeline.git
cd rfm-pipeline
pixi install --locked
```

Verify the package import:

```bash
pixi run import-smoke
```

## Run the smallest complete example

```bash
pixi run python examples/basic_workflow.py \
  --output-dir artifacts/basic-workflow
```

A successful run prints holdout predictions and an evaluation summary, then
writes:

```text
artifacts/basic-workflow/
├── manifest.json
└── postfit_diagnostics/
    ├── coef_matrix_raw_scale.csv
    ├── coef_matrix_standardized.csv
    ├── nrmse_summary.csv
    └── ...
```

This example is synthetic and deterministic. Continue with the
[Python API quickstart](quickstart.md) to use your own DataFrames.

## Install only the package

If you do not need repository tasks or examples:

```bash
pip install 'git+https://github.com/NatLabRockies/rfm-pipeline.git'
```

## Development validation

The full repository gate is:

```bash
pixi run gate
```

It runs style checks, unit tests, examples, documentation, packaging, and a
wheel-import check. It does not run long model-fitting jobs.

If setup fails, use the short [Troubleshooting guide](troubleshooting.md).
