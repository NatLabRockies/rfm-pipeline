______________________________________________________________________

## applyTo: "notebooks/\*\*/\*.ipynb"

# Notebooks

Notebooks are production artifacts when included in the workflow.

Rules:

- use Pixi for execution
- no absolute local paths
- no hidden state
- no broad unrelated rewrites
- clear outputs unless repo convention tracks outputs
- notebooks should call reusable logic from `src/` where practical

If a notebook is part of validation, the repository gate or CI should execute it or smoke-test it.
