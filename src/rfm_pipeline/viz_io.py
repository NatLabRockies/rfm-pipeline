"""Load and evaluate canonical post-fit artifact bundles."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def canonical_bundle_loader_keys() -> list[str]:
    """Return the canonical logical table names exposed by the post-fit bundle loader."""
    return [
        "all_input_metadata",
        "selected_input_metadata",
        "output_metadata",
        "coef_matrix_standardized",
        "coef_matrix_raw_scale",
        "x_standardization",
        "y_standardization",
        "nrmse_summary",
    ]


def _load_manifest(root: Path) -> dict[str, object]:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Expected manifest.json under {root}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json must contain a JSON object")
    if not isinstance(manifest.get("files"), dict):
        raise ValueError("manifest.json must contain a 'files' object")
    return manifest


def _artifact_path(root: Path, value: object, *, key: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"manifest file path for '{key}' must be a non-empty string")
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError(f"manifest file path for '{key}' must be relative")
    bundle_root = root.resolve()
    resolved = (bundle_root / relative).resolve()
    if resolved != bundle_root and bundle_root not in resolved.parents:
        raise ValueError(f"manifest file path for '{key}' escapes the bundle directory")
    if resolved.suffix not in {".csv", ".parquet"}:
        raise ValueError(f"manifest file path for '{key}' must end in .csv or .parquet")
    if not resolved.is_file():
        raise FileNotFoundError(f"Bundle artifact for '{key}' does not exist: {resolved}")
    return resolved


def _read_artifact(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _load_artifacts(root: Path, keys: list[str]) -> dict[str, pd.DataFrame]:
    manifest = _load_manifest(root)
    files = manifest["files"]
    assert isinstance(files, dict)
    missing = [key for key in keys if key not in files]
    if missing:
        raise ValueError(f"manifest.json is missing canonical file entries: {missing}")
    return {key: _read_artifact(_artifact_path(root, files[key], key=key)) for key in keys}


def load_postfit_bundle(root: Path) -> dict[str, pd.DataFrame]:
    """Load canonical artifact tables using the bundle manifest.

    Parameters
    ----------
    root
        Export bundle directory containing ``manifest.json``.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Mapping from logical artifact name to loaded table.
    """
    bundle_root = Path(root)
    return _load_artifacts(bundle_root, canonical_bundle_loader_keys())


def predict_from_postfit_bundle(root: Path, X: pd.DataFrame) -> pd.DataFrame:
    """Predict from raw retained features using a written post-fit bundle.

    Parameters
    ----------
    root
        Export bundle directory containing ``manifest.json``.
    X
        Raw-scale feature matrix containing the retained feature columns.

    Returns
    -------
    pandas.DataFrame
        Predicted outputs preserving the input row index.

    Raises
    ------
    ValueError
        Raised when the bundle contract or input values are invalid.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")
    if X.columns.has_duplicates:
        raise ValueError("X must not contain duplicate column names")

    bundle_root = Path(root)
    manifest = _load_manifest(bundle_root)
    retained = manifest.get("retained_features")
    outputs = manifest.get("output_names")
    if not isinstance(retained, list) or not all(isinstance(name, str) for name in retained):
        raise ValueError("manifest.json must contain a string list named 'retained_features'")
    if not isinstance(outputs, list) or not all(isinstance(name, str) for name in outputs):
        raise ValueError("manifest.json must contain a string list named 'output_names'")
    if len(retained) != len(set(retained)) or len(outputs) != len(set(outputs)):
        raise ValueError("manifest feature and output names must be unique")

    missing = [name for name in retained if name not in X.columns]
    if missing:
        raise ValueError(f"X is missing retained feature columns: {missing}")
    X_aligned = X.loc[:, retained].apply(pd.to_numeric, errors="raise")
    X_values = X_aligned.to_numpy(dtype=float)
    if not np.isfinite(X_values).all():
        raise ValueError("X must contain only finite numeric values")

    tables = _load_artifacts(
        bundle_root,
        ["coef_matrix_raw_scale", "x_standardization", "y_standardization"],
    )
    coefficients = tables["coef_matrix_raw_scale"]
    x_scaling = tables["x_standardization"]
    y_scaling = tables["y_standardization"]
    expected_coefficient_columns = ["output_name", *retained]
    if coefficients.columns.tolist() != expected_coefficient_columns:
        raise ValueError("raw coefficient columns do not match the manifest feature order")
    if coefficients["output_name"].astype(str).tolist() != outputs:
        raise ValueError("raw coefficient rows do not match the manifest output order")
    if x_scaling.get("feature_name", pd.Series(dtype=str)).astype(str).tolist() != retained:
        raise ValueError("x_standardization rows do not match the manifest feature order")
    if y_scaling.get("output_name", pd.Series(dtype=str)).astype(str).tolist() != outputs:
        raise ValueError("y_standardization rows do not match the manifest output order")
    if "mean" not in x_scaling or "mean" not in y_scaling:
        raise ValueError("standardization tables must contain a 'mean' column")

    coefficient_values = coefficients.loc[:, retained].to_numpy(dtype=float)
    x_means = x_scaling["mean"].to_numpy(dtype=float)
    y_means = y_scaling["mean"].to_numpy(dtype=float)
    if not all(np.isfinite(values).all() for values in (coefficient_values, x_means, y_means)):
        raise ValueError("bundle prediction values must be finite")

    predictions = (X_values - x_means) @ coefficient_values.T + y_means
    return pd.DataFrame(predictions, index=X.index, columns=outputs)
