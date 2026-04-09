# Overview

This scaffold establishes the local and CI engineering contract for the BSM reduced-form workflow refactor.

## Canonical local gate

Run the full repository gate:

```bash
./test_repo.sh
```

Run the repair path before committing:

```bash
./test_repo.sh --fix
```

## Expected standards

- Pixi-managed environment and task execution
- NumPy-style docstrings on public Python APIs
- Ruff for Python and notebook linting and formatting
- mdformat for Markdown formatting
- stripped notebook outputs before commit
- Sphinx documentation build in the local and CI gates

## Current scientific boundary

The repo now exposes the recovered workflow-stage ordering and case-study counts through `bsm_rfm.workflow`. It also exposes an explicit notebook-derived feature-expansion contract through `bsm_rfm.feature_expansion`. This makes the audited scientific boundary importable and testable even though the later regularized-screening and final-OLS stages have not yet been ported into canonical package code.
