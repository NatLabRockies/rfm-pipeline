"""Final OLS foundations and post-fit artifact helpers.

This module keeps the recovered notebook-derived post-fit contract explicit while also
providing a small canonical implementation of the final OLS fit, prediction, holdout
metric summarization, and artifact-bundle assembly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from .artifacts import PipelineManifest, make_metadata_frame
from .data import align_xy
from .metrics import bootstrap_macro_nrmse_ci


@dataclass(frozen=True)
class PostfitArtifactSpec:
    """One expected artifact in the notebook-derived post-fit export bundle.

    Parameters
    ----------
    artifact_name
        Canonical artifact stem under ``postfit_diagnostics``.
    category
        Broad artifact category used for documentation and validation.
    description
        Human-readable description of the artifact's role.
    """

    artifact_name: str
    category: str
    description: str


@dataclass(frozen=True)
class FinalOLSContract:
    """Recovered contract for the notebook-derived final OLS stage.

    Parameters
    ----------
    provenance
        Provenance label for the recovered workflow stage.
    estimator
        Summary of the final fitted model family.
    selected_feature_handoff
        Description of how the selected feature set reaches the final OLS stage.
    holdout_fraction
        External holdout fraction used by the notebook workflow.
    selected_feature_count
        Recovered feature count handed to final OLS.
    coefficient_scales
        Coefficient representations expected in the exported artifact bundle.
    artifact_specs
        Canonical post-fit artifacts expected by downstream visualization helpers.
    diagnostics
        Additional post-fit diagnostics and metadata exported by the notebook.
    """

    provenance: str
    estimator: str
    selected_feature_handoff: str
    holdout_fraction: float
    selected_feature_count: int
    coefficient_scales: tuple[str, ...]
    artifact_specs: tuple[PostfitArtifactSpec, ...]
    diagnostics: tuple[str, ...]


@dataclass(frozen=True)
class FinalOLSFitResult:
    """Fitted final OLS model with raw-scale and standardized representations.

    Parameters
    ----------
    feature_names
        Original-order retained feature names used in the final fit.
    output_names
        Original-order modeled output names.
    coef_raw_scale
        Raw-scale coefficient matrix with one row per output.
    intercept_raw_scale
        Raw-scale intercept vector with one value per output.
    coef_standardized
        Coefficient matrix after transforming both inputs and outputs to z-scores.
    intercept_standardized
        Intercepts after transforming both inputs and outputs to z-scores.
    x_means
        Training-set input means.
    x_scales
        Training-set input scaling factors.
    y_means
        Training-set output means.
    y_scales
        Training-set output scaling factors.
    n_training_rows
        Number of aligned training rows used in the fit.
    """

    feature_names: tuple[str, ...]
    output_names: tuple[str, ...]
    coef_raw_scale: np.ndarray
    intercept_raw_scale: np.ndarray
    coef_standardized: np.ndarray
    intercept_standardized: np.ndarray
    x_means: np.ndarray
    x_scales: np.ndarray
    y_means: np.ndarray
    y_scales: np.ndarray
    n_training_rows: int


def _coerce_numeric_frame(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    """Return a finite numeric DataFrame.

    Parameters
    ----------
    frame
        Candidate feature or response frame.
    name
        Human-readable name used in validation errors.

    Returns
    -------
    pandas.DataFrame
        Numeric copy of the supplied frame.

    Raises
    ------
    ValueError
        Raised when the input is empty or non-finite.
    """
    if frame.shape[0] == 0 or frame.shape[1] == 0:
        raise ValueError(f"{name} must be non-empty.")
    numeric = frame.apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(f"{name} must contain only finite numeric values.")
    return numeric


def notebook_final_ols_contract() -> FinalOLSContract:
    """Return the recovered notebook-derived final OLS contract.

    Returns
    -------
    FinalOLSContract
        Explicit contract for the final OLS handoff and post-fit export boundary.
    """
    return FinalOLSContract(
        provenance="notebook_derived",
        estimator="per-output ordinary least squares",
        selected_feature_handoff=(
            "selected inputs from the notebook sparse-screening stage are handed to a final OLS fit"
        ),
        holdout_fraction=0.10,
        selected_feature_count=346,
        coefficient_scales=("standardized", "raw_scale"),
        artifact_specs=(
            PostfitArtifactSpec(
                artifact_name="all_input_metadata",
                category="metadata",
                description="Original-order metadata for all candidate inputs.",
            ),
            PostfitArtifactSpec(
                artifact_name="selected_input_metadata",
                category="metadata",
                description="Original-order metadata for selected screening inputs.",
            ),
            PostfitArtifactSpec(
                artifact_name="output_metadata",
                category="metadata",
                description="Original-order metadata for modeled outputs.",
            ),
            PostfitArtifactSpec(
                artifact_name="coef_matrix_standardized",
                category="coefficients",
                description=(
                    "Coefficient matrix on the standardized scale for downstream "
                    "inspection and visualization."
                ),
            ),
            PostfitArtifactSpec(
                artifact_name="coef_matrix_raw_scale",
                category="coefficients",
                description=(
                    "Coefficient matrix transformed back to the raw scale for interpretation."
                ),
            ),
            PostfitArtifactSpec(
                artifact_name="x_standardization",
                category="standardization",
                description="Means and scales used to standardize final-model inputs.",
            ),
            PostfitArtifactSpec(
                artifact_name="y_standardization",
                category="standardization",
                description="Means and scales used to standardize modeled outputs.",
            ),
            PostfitArtifactSpec(
                artifact_name="nrmse_summary",
                category="evaluation",
                description="Holdout nRMSE summaries and associated evaluation metadata.",
            ),
        ),
        diagnostics=(
            "all_input_order",
            "selected_input_order",
            "retained_input_order",
            "output_order",
            "transform_vectors",
            "postfit_diagnostics",
        ),
    )


def canonical_postfit_artifact_names() -> list[str]:
    """Return canonical notebook-derived post-fit artifact names.

    Returns
    -------
    list[str]
        Ordered artifact stems under ``postfit_diagnostics``.
    """
    return [spec.artifact_name for spec in notebook_final_ols_contract().artifact_specs]


def postfit_artifact_table() -> pd.DataFrame:
    """Tabulate the recovered notebook-derived post-fit artifact contract.

    Returns
    -------
    pandas.DataFrame
        One row per canonical artifact with ordering preserved.
    """
    contract = notebook_final_ols_contract()
    rows = []
    for position, spec in enumerate(contract.artifact_specs):
        rows.append(
            {
                "artifact_name": spec.artifact_name,
                "artifact_position": position,
                "category": spec.category,
                "description": spec.description,
            }
        )
    return pd.DataFrame(rows)


def fit_final_ols(X: pd.DataFrame, Y: pd.DataFrame) -> FinalOLSFitResult:
    """Fit the canonical final OLS model on aligned raw-scale inputs and outputs.

    Parameters
    ----------
    X
        Final retained feature matrix in raw units.
    Y
        Modeled output matrix in raw units.

    Returns
    -------
    FinalOLSFitResult
        Fitted model plus raw-scale and standardized representations.
    """
    X_aligned, Y_aligned = align_xy(X, Y)
    X_numeric = _coerce_numeric_frame(X_aligned, name="X")
    Y_numeric = _coerce_numeric_frame(Y_aligned, name="Y")

    X_values = X_numeric.to_numpy(dtype=float)
    Y_values = Y_numeric.to_numpy(dtype=float)

    model = LinearRegression(fit_intercept=True)
    model.fit(X_values, Y_values)

    x_scaler = StandardScaler().fit(X_values)
    y_scaler = StandardScaler().fit(Y_values)

    coef_raw = np.asarray(model.coef_, dtype=float)
    if coef_raw.ndim == 1:
        coef_raw = coef_raw[np.newaxis, :]
    intercept_raw = np.asarray(model.intercept_, dtype=float)
    intercept_raw = np.atleast_1d(intercept_raw)

    x_means = np.asarray(x_scaler.mean_, dtype=float)
    x_scales = np.asarray(x_scaler.scale_, dtype=float)
    y_means = np.asarray(y_scaler.mean_, dtype=float)
    y_scales = np.asarray(y_scaler.scale_, dtype=float)

    coef_standardized = coef_raw * (x_scales[np.newaxis, :] / y_scales[:, np.newaxis])
    intercept_standardized = (intercept_raw + coef_raw @ x_means - y_means) / y_scales

    return FinalOLSFitResult(
        feature_names=tuple(str(column) for column in X_numeric.columns),
        output_names=tuple(str(column) for column in Y_numeric.columns),
        coef_raw_scale=coef_raw.copy(),
        intercept_raw_scale=intercept_raw.copy(),
        coef_standardized=coef_standardized.copy(),
        intercept_standardized=intercept_standardized.copy(),
        x_means=x_means.copy(),
        x_scales=x_scales.copy(),
        y_means=y_means.copy(),
        y_scales=y_scales.copy(),
        n_training_rows=len(X_numeric),
    )


def predict_final_ols(result: FinalOLSFitResult, X: pd.DataFrame) -> pd.DataFrame:
    """Predict outputs from a fitted final OLS model.

    Parameters
    ----------
    result
        Fitted final OLS result.
    X
        Raw-scale feature matrix containing the retained feature columns.

    Returns
    -------
    pandas.DataFrame
        Predicted outputs preserving the row index from ``X``.

    Raises
    ------
    ValueError
        Raised when the required retained feature columns are unavailable.
    """
    missing = [feature for feature in result.feature_names if feature not in X.columns]
    if missing:
        raise ValueError(f"Missing retained feature columns: {missing}")

    X_eval = X.loc[:, list(result.feature_names)].apply(pd.to_numeric, errors="raise")
    values = X_eval.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("X must contain only finite numeric values.")

    pred = values @ result.coef_raw_scale.T + result.intercept_raw_scale
    return pd.DataFrame(pred, index=X_eval.index, columns=list(result.output_names))


def make_holdout_nrmse_summary(
    result: FinalOLSFitResult,
    X_holdout: pd.DataFrame,
    Y_holdout: pd.DataFrame,
    Y_ref: pd.DataFrame,
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    random_state: int = 123,
    sample_size: int | None = None,
) -> pd.DataFrame:
    """Build a one-row holdout nRMSE summary for a fitted final OLS model.

    Parameters
    ----------
    result
        Fitted final OLS result.
    X_holdout
        Holdout feature matrix.
    Y_holdout
        Holdout response matrix.
    Y_ref
        Reference response matrix whose fixed columnwise ranges define the nRMSE
        normalization denominator.
    n_boot
        Number of bootstrap replicates.
    alpha
        Two-sided bootstrap error level.
    random_state
        Seed for bootstrap resampling.
    sample_size
        Optional bootstrap draw size.

    Returns
    -------
    pandas.DataFrame
        Single-row evaluation summary preserving key holdout metadata.
    """
    predictions = predict_final_ols(result, X_holdout)
    Y_eval = Y_holdout.loc[predictions.index, list(result.output_names)]
    Y_ref_eval = Y_ref.loc[:, list(result.output_names)]

    metric_summary = bootstrap_macro_nrmse_ci(
        Y_eval.to_numpy(dtype=float),
        predictions.to_numpy(dtype=float),
        Y_ref_eval.to_numpy(dtype=float),
        n_boot=n_boot,
        alpha=alpha,
        random_state=random_state,
        sample_size=sample_size,
    )
    return pd.DataFrame(
        [
            {
                "holdout_rows": int(len(Y_eval)),
                "n_features": int(len(result.feature_names)),
                "n_outputs": int(len(result.output_names)),
                **metric_summary,
            }
        ]
    )


def build_postfit_artifacts(
    result: FinalOLSFitResult,
    *,
    dataset_tag: str,
    all_input_features: list[str],
    selected_features: list[str] | None = None,
    evaluation_summary: dict[str, Any] | pd.DataFrame | None = None,
    upstream_provenance: dict[str, Any] | None = None,
) -> dict[str, pd.DataFrame | dict[str, Any]]:
    """Assemble the canonical post-fit artifact bundle from a fitted model.

    Parameters
    ----------
    result
        Fitted final OLS result.
    dataset_tag
        Human-readable label for the current dataset or run family.
    all_input_features
        Full original-order candidate input list before screening.
    selected_features
        Screening-stage selected features in original order. Defaults to the retained
        final OLS features.
    evaluation_summary
        Optional one-row summary payload for the ``nrmse_summary`` artifact.
    upstream_provenance
        Optional provenance metadata stored in the manifest.

    Returns
    -------
    dict[str, pandas.DataFrame | dict[str, Any]]
        Artifact bundle keyed by the canonical notebook-derived artifact names plus a
        ``manifest`` entry.
    """
    selected = list(selected_features or result.feature_names)
    retained = list(result.feature_names)

    if isinstance(evaluation_summary, pd.DataFrame):
        nrmse_summary = evaluation_summary.copy()
    elif evaluation_summary is None:
        nrmse_summary = pd.DataFrame()
    else:
        nrmse_summary = pd.DataFrame([evaluation_summary])

    files = {
        artifact_name: f"postfit_diagnostics/{artifact_name}.parquet"
        for artifact_name in canonical_postfit_artifact_names()
    }

    artifacts: dict[str, pd.DataFrame | dict[str, Any]] = {
        "all_input_metadata": make_metadata_frame(all_input_features, "input_name"),
        "selected_input_metadata": make_metadata_frame(selected, "input_name"),
        "output_metadata": make_metadata_frame(list(result.output_names), "output_name"),
        "coef_matrix_standardized": make_coefficient_matrix_frame(
            result.coef_standardized,
            output_names=list(result.output_names),
            feature_names=retained,
        ),
        "coef_matrix_raw_scale": make_coefficient_matrix_frame(
            result.coef_raw_scale,
            output_names=list(result.output_names),
            feature_names=retained,
        ),
        "x_standardization": make_standardization_frame(
            retained,
            result.x_means,
            result.x_scales,
            name_column="feature_name",
        ),
        "y_standardization": make_standardization_frame(
            list(result.output_names),
            result.y_means,
            result.y_scales,
            name_column="output_name",
        ),
        "nrmse_summary": nrmse_summary,
    }
    artifacts["manifest"] = PipelineManifest(
        dataset_tag=dataset_tag,
        n_all_input_features=len(all_input_features),
        n_selected_features=len(selected),
        n_retained_features=len(retained),
        n_outputs=len(result.output_names),
        all_input_features=list(all_input_features),
        selected_features=selected,
        retained_features=retained,
        output_names=list(result.output_names),
        files=files,
        metrics={},
        evaluation={},
        upstream_provenance=dict(upstream_provenance or {}),
    ).to_dict()
    return artifacts


def make_coefficient_matrix_frame(
    coef_matrix: np.ndarray,
    output_names: list[str],
    feature_names: list[str],
) -> pd.DataFrame:
    """Convert a coefficient matrix into a labeled artifact table.

    Parameters
    ----------
    coef_matrix
        Two-dimensional array with one row per output and one column per feature.
    output_names
        Ordered output names matching the rows of ``coef_matrix``.
    feature_names
        Ordered feature names matching the columns of ``coef_matrix``.

    Returns
    -------
    pandas.DataFrame
        DataFrame with an ``output_name`` column and one coefficient column per
        feature.

    Raises
    ------
    ValueError
        Raised when the matrix shape does not match the supplied output or feature
        labels.
    """
    matrix = np.asarray(coef_matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("coef_matrix must be two-dimensional")
    expected_shape = (len(output_names), len(feature_names))
    if matrix.shape != expected_shape:
        raise ValueError(f"coef_matrix shape {matrix.shape} does not match labels {expected_shape}")

    return pd.DataFrame(matrix, columns=list(feature_names)).assign(output_name=output_names)[
        ["output_name", *feature_names]
    ]


def make_standardization_frame(
    names: list[str],
    means: np.ndarray,
    scales: np.ndarray,
    *,
    name_column: str,
) -> pd.DataFrame:
    """Build a canonical standardization artifact table.

    Parameters
    ----------
    names
        Ordered feature or output names.
    means
        Means aligned to ``names``.
    scales
        Standard deviations or scaling factors aligned to ``names``.
    name_column
        Column name used for the ordered identifiers.

    Returns
    -------
    pandas.DataFrame
        Standardization table with original ordering preserved.

    Raises
    ------
    ValueError
        Raised when the lengths of names, means, and scales do not match.
    """
    means_arr = np.asarray(means, dtype=float)
    scales_arr = np.asarray(scales, dtype=float)
    n_names = len(names)
    if means_arr.shape != (n_names,) or scales_arr.shape != (n_names,):
        raise ValueError("names, means, and scales must have the same one-dimensional length")

    return pd.DataFrame(
        {
            name_column: list(names),
            "original_position": list(range(n_names)),
            "mean": means_arr,
            "scale": scales_arr,
        }
    )
