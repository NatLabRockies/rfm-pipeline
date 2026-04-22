# End-to-end reproducibility example

The repository ships a deterministic toy example that exercises the canonical public
workflow end to end. It is intended for CI and demonstration; the manuscript reproduction layer
adds separate real-data notebooks for the full BSM case study.

1. build aligned train and holdout DataFrames
1. run `run_canonical_workflow(...)`
1. write the canonical post-fit bundle with `write_postfit_bundle(...)`
1. reload the written tables with `load_postfit_bundle(...)`

## Run from the repo source tree

```bash
PYTHONPATH=src python examples/end_to_end_reproducibility.py \
  --output-dir artifacts/toy-reproducibility-example
```

The command writes a canonical bundle containing:

- `manifest.json`
- `postfit_diagnostics/all_input_metadata.*`
- `postfit_diagnostics/selected_input_metadata.*`
- `postfit_diagnostics/output_metadata.*`
- `postfit_diagnostics/coef_matrix_standardized.*`
- `postfit_diagnostics/coef_matrix_raw_scale.*`
- `postfit_diagnostics/x_standardization.*`
- `postfit_diagnostics/y_standardization.*`
- `postfit_diagnostics/nrmse_summary.*`

## Python entrypoint

The example script exposes `run_reproducibility_example(...)` for tests and notebook reuse.

```python
from pathlib import Path

from examples.end_to_end_reproducibility import run_reproducibility_example

result = run_reproducibility_example(Path("artifacts/toy-reproducibility-example"))
print(result["manifest"]["dataset_tag"])
print(result["loaded"]["nrmse_summary"])
```

The output bundle is deterministic because the example uses fixed synthetic data and fixed
screening/bootstrap random seeds.
