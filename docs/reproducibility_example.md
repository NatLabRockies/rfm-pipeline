# Reproducibility examples

## Neutral first example

`examples/basic_workflow.py` is the shortest complete example. It creates a
small synthetic dataset, fits the canonical API, writes the bundle, reloads
it, and prints the holdout summary.

```bash
pixi run python examples/basic_workflow.py \
  --output-dir artifacts/basic-workflow
```

The reusable entry point is:

```python
from pathlib import Path

from examples.basic_workflow import run_example

run, tables = run_example(Path("artifacts/basic-workflow"))
print(run.holdout_summary)
print(tables["coef_matrix_raw_scale"])
```

Fixed data and random seeds make the output deterministic. The example is for
orientation and contract testing, not scientific benchmarking.

## Extended deterministic example

`examples/end_to_end_reproducibility.py` exercises the same public bundle
contract with a second toy dataset:

```bash
pixi run python examples/end_to_end_reproducibility.py \
  --output-dir artifacts/reproducibility-example
```

It exposes `run_reproducibility_example(...)` for test and notebook reuse.
