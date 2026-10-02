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

- `X_train` and `Y_train` have identical row indexes in the same order;
- `X_holdout` and `Y_holdout` have identical row indexes in the same order;
- train and holdout feature columns match exactly;
- train and holdout output columns match exactly;
- feature and output column names are non-empty strings; and
- all modeled values are numeric and finite.

The API uses DataFrame column names as the feature and output identifiers.

## Fit and export

```python
from pathlib import Path

import pandas as pd

from rfm_pipeline import predict_from_postfit_bundle, run_canonical_workflow, write_postfit_bundle

X_train = pd.read_parquet("X_train.parquet")
Y_train = pd.read_parquet("Y_train.parquet")
X_holdout = pd.read_parquet("X_holdout.parquet")
Y_holdout = pd.read_parquet("Y_holdout.parquet")

run = run_canonical_workflow(
    X_train,
    Y_train,
    X_holdout,
    Y_holdout,
    dataset_tag="my-model",
    screening_random_state=123,
    bootstrap_random_state=123,
)

bundle = Path("artifacts/my-model")
write_postfit_bundle(run.artifacts, bundle)
predictions = predict_from_postfit_bundle(bundle, X_holdout)
print(run.screening_result.selected_features)
print(predictions)
print(run.holdout_summary)
```

The result contains the screening fit, fitted OLS model, holdout summary, and
in-memory export tables. The writer records feature/output order and actual
file paths in `manifest.json`; the bundle predictor uses that contract.

## Reload a bundle

```python
from pathlib import Path

import pandas as pd

from rfm_pipeline import load_postfit_bundle, predict_from_postfit_bundle

bundle = Path("artifacts/my-model")
tables = load_postfit_bundle(bundle)
coefficients = tables["coef_matrix_raw_scale"]
performance = tables["nrmse_summary"]
new_inputs = pd.read_parquet("X_new.parquet")
predictions = predict_from_postfit_bundle(bundle, new_inputs)
```

Parquet is used when requested and available; CSV is the fallback. The loader
handles either format. `predict_from_postfit_bundle(...)` validates the
manifest, aligns retained features, and returns outputs in manifest order.

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

## Add transformations or interactions

The workflow screens the feature columns it receives; it does not automatically
generate interactions or nonlinear transformations. Use
`FeatureExpansionSpec` and `apply_feature_expansion(...)` to define those
columns explicitly:

```python
from rfm_pipeline import QUADRATIC, apply_feature_expansion, default_feature_expansion_spec

spec = default_feature_expansion_spec(
    base_features=list(X_train.columns),
    add_transforms={"temperature": [QUADRATIC]},
    interaction_pairs=(("temperature", "pressure"),),
)
X_train = apply_feature_expansion(X_train, spec).expanded_frame
X_holdout = apply_feature_expansion(X_holdout, spec).expanded_frame
```

Then pass the expanded tables to the workflow. See the
[reproducibility example](reproducibility_example.md) for a complete fit.
