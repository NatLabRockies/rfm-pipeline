______________________________________________________________________

## applyTo: "**/{pixi.toml,pyproject.toml,test_repo.sh,.github/workflows/*.yml,.github/workflows/*.yaml,scripts/**/*.sh,scripts/\*\*/*.py}"

# Pixi and validation instructions

If `pixi.toml` exists, use repo-local Pixi.

Allowed project tooling pattern:

- `pixi install --locked`
- `pixi run ...`
- `./test_repo.sh`

Do not use bare `python`, `python3`, `pytest`, `ruff`, `mypy`, `pre-commit`, `pip`, `uv`, `poetry`, `conda`, `pixi global`, `pixi shell`, or `pixi exec` for project work unless explicitly approved.

CI should align with `test_repo.sh` whenever possible.
