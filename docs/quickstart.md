# Python API quickstart

Use the canonical API when your features are already prepared and split into
training and holdout sets.

## Input contract

`run_canonical_workflow(...)` accepts four pandas DataFrames:

- `X_train`: training features
- `Y_train`: training outputs
- `X_holdout`: holdout features
- `Y_holdout`: holdout outputs

Before calling it, ensure:

- `X_train` and `Y_train` have the same row order;
- `X_holdout` and `Y_holdout` have the same row order;
- train and holdout feature columns match exactly;
- train and holdout output columns match exactly; and
- all modeled values are numeric and finite.

The API uses DataFrame column names as the feature and output identifiers.

## Fit and export

```python
from pathlib import Path

import pandas as pd

from rfm_pipeline import run_canonical_workflow, write_postfit_bundle

X_train = pd.read_parquet("X_train.parquet")
Y_train = pd.read_parquet("Y_train.parquet")
X_holdout = pd.read_parquet("X_holdout.parquet")
Y_holdout = pd.read_parquet("Y_holdout.parquet")

run = run_canonical_workflow(
    X_train,
    Y_train,
    X_holdout,
    Y_holdout,
    dataset_tag="my-study",
    screening_random_state=123,
    bootstrap_random_state=123,
)

write_postfit_bundle(run.artifacts, Path("artifacts/my-study"))
print(run.screening_result.selected_features)
print(run.holdout_summary)
```

The result contains the screening fit, final OLS fit, holdout summary, and
in-memory export tables. The writer records the actual file paths in
`manifest.json`.

## Reload a bundle

```python
from pathlib import Path

from rfm_pipeline import load_postfit_bundle

tables = load_postfit_bundle(Path("artifacts/my-study"))
coefficients = tables["coef_matrix_raw_scale"]
performance = tables["nrmse_summary"]
```

Parquet is used when requested and available; CSV is the fallback. The loader
handles either format.

## Common options

| Option                | Default      | Purpose                              |
| --------------------- | ------------ | ------------------------------------ |
| `screening_cv`        | `5`          | Cross-validation folds for screening |
| `screening_l1_ratio`  | `(0.9, 1.0)` | Elastic-net mixing values            |
| `screening_alphas`    | `100`        | Alpha count or explicit alpha grid   |
| `n_boot`              | `1000`       | Holdout bootstrap replicates         |
| `alpha`               | `0.05`       | Bootstrap interval error level       |
| `artifact_format`     | `"auto"`     | `"parquet"`, `"csv"`, or automatic   |
| `upstream_provenance` | `None`       | Metadata to retain in the manifest   |

See the `run_canonical_workflow` API reference for the complete signature.

## Need feature discovery too?

The canonical API screens the feature columns it receives; it does not
automatically generate interactions or nonlinear transformations. For the
larger staged workflow, continue with the
[Configuration reference](configuration_reference.md). For a ready-to-run
neutral example, see [Reproducibility examples](reproducibility_example.md).
