# BSM reduced-form model refactor scaffold

This repository scaffold establishes the engineering contract for the BSM reduced-form modeling refactor.

## Environment management

The repository is designed around Pixi. The canonical local and CI entrypoint is:

```bash
./test_repo.sh
```

To repair formatting and notebook hygiene before committing:

```bash
./test_repo.sh --fix
```

## What the repo gate enforces

- Ruff linting and formatting for Python and notebook code
- mdformat formatting checks for Markdown
- stripped notebook outputs and execution counts
- repository hygiene checks for generated Python artifacts and text whitespace issues
- unit tests
- workflow smoke tests
- Sphinx documentation builds

## Documentation

The package documentation is built with Sphinx and MyST. Public Python APIs are expected to use NumPy-style docstrings.

## Current scope

This scaffold covers the foundational utilities that the later canonical workflow refactor will build on:

- data alignment and split utilities
- train-only standardization
- selected-feature structure parsing
- nRMSE helpers and bootstrap summaries
- artifact manifest helpers
- visualization-side artifact loading

The scientific workflow itself still needs to be ported from the recovered scripts and notebooks into canonical modules.

## Provenance notes

- The canonical 20k modeling subset is a balanced stratified sample: 5,000 rows drawn within each AFSC/UAEORO boolean combination.
- The canonical upstream null-screening stage comes from the recovered `null_distribution.py` source script and is wrapped through `bsm_rfm.null_screening` rather than reimplemented ad hoc in notebooks.

## Repository gate

Use `./test_repo.sh` as the canonical local gate.

- `./test_repo.sh` runs the default prepare-and-validate path. It formats Python and Markdown,
  strips notebook outputs, then runs the full validation chain.
- `./test_repo.sh --fix` is an explicit alias for the default prepare-and-validate path.
- `./test_repo.sh --check` runs the same validation chain without mutating the working tree.
- `./test_repo.sh --ci` is the non-mutating GitHub Actions entrypoint.
- `./test_repo.sh --clean` rebuilds the Pixi environment, then runs the default path.

The validation chain includes repository hygiene checks, Ruff lint/format validation,
Markdown formatting validation, notebook hygiene validation, Python compilation, unit tests,
workflow smoke tests, notebook execution tests, Sphinx documentation builds, package builds,
and `git diff --check`.
