______________________________________________________________________

## applyTo: "**/{pixi.toml,pyproject.toml,.github/workflows/*.yml,.github/workflows/*.yaml,scripts/**/*.sh,scripts/\*\*/*.py}"

# Pixi and validation instructions

If `pixi.toml` exists, use repo-local Pixi.

Allowed project tooling pattern:

- `pixi install --locked`
- `pixi run ...`
- `pixi run gate`

Do not use bare `python`, `python3`, `pytest`, `ruff`, `mypy`, `pre-commit`, `pip`, `uv`, `poetry`, `conda`, `pixi global`, `pixi shell`, or `pixi exec` for project work unless explicitly approved.

CI should align with `pixi run gate` whenever possible.
