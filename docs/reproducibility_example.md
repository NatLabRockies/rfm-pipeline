# End-to-end reproducibility example

The repository ships a deterministic toy example that exercises the canonical public
workflow end to end. It is intended for CI and demonstration. The toy holdout responses include a fixed deterministic residual so the reported holdout nRMSE is nonzero and exercises the evaluation path rather than a perfect interpolation case. The same script can also run
the complete demo manuscript-reproduction stage chain, which mirrors the source-backed
Phase 3 notebook stages and writes every manuscript artifact family.

1. build aligned train and holdout DataFrames
1. run `run_canonical_workflow(...)`
1. write the canonical post-fit bundle with `write_postfit_bundle(...)`
1. reload the written tables with `load_postfit_bundle(...)`

## Run from the repo source tree

```bash
PYTHONPATH=src python examples/end_to_end_reproducibility.py \
  --output-dir artifacts/toy-reproducibility-example
```

To run the canonical example and the complete demo manuscript-reproduction chain together:

```bash
PYTHONPATH=src python examples/end_to_end_reproducibility.py \
  --output-dir artifacts/toy-reproducibility-example \
  --run-manuscript-chain \
  --manuscript-output-dir artifacts/toy-manuscript-reproduction-example
```

The manuscript-chain path also writes `reproduction_audit/`, including an artifact manifest,
metric checks, and a one-row audit summary. The audit explicitly checks that every artifact
exists, every artifact is nonempty, the demo final-OLS holdout nRMSE is finite and positive,
and the bootstrap interval is ordered around the point estimate.

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

The example script exposes `run_reproducibility_example(...)` and `run_manuscript_reproduction_example(...)` for tests and notebook reuse.

```python
from pathlib import Path

from examples.end_to_end_reproducibility import run_reproducibility_example

result = run_reproducibility_example(Path("artifacts/toy-reproducibility-example"))
print(result["manifest"]["dataset_tag"])
print(result["loaded"]["nrmse_summary"])
```

The output bundle is deterministic because the example uses fixed synthetic data and fixed
screening/bootstrap random seeds.

## Manuscript-reproduction chain output

When `--run-manuscript-chain` is supplied, the script writes the deterministic demo outputs for:

- `output_conditioning/`
- `empirical_null_screen/`
- `interaction_discovery/`
- `nonlinear_discovery/`
- `sparse_selection/`
- `final_manuscript_artifacts/`
- `reproduction_audit/`

This path is the public smoke-test companion to the real-data manuscript notebooks.
