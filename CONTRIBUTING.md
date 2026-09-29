# Contributing to rfm-pipeline

Thank you for your interest in contributing!

## Getting started

1. Fork the repository and create a feature branch from `main`.
1. Install the development environment:
   ```bash
   git clone https://github.com/NatLabRockies/rfm-pipeline.git
   cd rfm-pipeline
   pixi install --locked
   ```
1. Run the development gate to verify your setup:
   ```bash
   pixi run gate-fast
   ```

> **Note on optional tests.** Tests that exercise SLURM/HPC orchestration or
> GPU code paths are skipped automatically when the relevant environment
> variables or hardware are unavailable. Local `pytest` runs therefore
> report fewer collected tests than the full HPC test matrix. To opt into
> them, run the suite on a SLURM-enabled login node (sets `SLURM_*` env
> vars) and/or on a host with a CUDA-capable GPU.

## Making changes

- Write tests before implementing (TDD).
- Run targeted tests during development: `pixi run python -m pytest tests/<your_test>.py`
- Run `pixi run gate-fast` before pushing. CI runs the complete validation suite.
- Follow existing code style (ruff enforces formatting and linting automatically).

## Pull requests

- Open a PR against `main` with a clear description of what changed and why.
- Link any related issues.
- All CI checks must pass before merging.
- Keep changes focused — one concern per PR.

## Reporting bugs

Open a GitHub issue with:

- A minimal reproducible example
- Expected vs. actual behavior
- Python version, OS, and package version (`rfm_pipeline.__version__`)

## Feature requests

Open a GitHub issue describing:

- The use case
- Why the current API is insufficient
- A proposed interface (optional)
