"""Regularized feature screening for multi-output reduced-form models."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, MultiTaskElasticNetCV
from sklearn.preprocessing import StandardScaler

from .data import align_xy


@dataclass(frozen=True)
class ScreeningSelectionResult:
    """Result from multi-output regularized feature screening.

    Parameters
    ----------
    feature_names
        Original-order candidate feature names entering the screening fit.
    output_names
        Original-order output names entering the screening fit.
    selected_mask
        Boolean feature mask indicating whether each candidate was retained.
    selected_features
        Retained feature names in original order.
    coefficient_matrix
        Multi-output coefficient matrix on the fitted standardized scale.
    intercepts
        Per-output intercepts from the fitted estimator.
    alpha_
        Selected global penalty strength from ``MultiTaskElasticNetCV``.
    l1_ratio_
        Selected elastic-net mixing parameter.
    cv_folds
        Number of cross-validation folds used during the screening fit.
    """

    feature_names: tuple[str, ...]
    output_names: tuple[str, ...]
    selected_mask: np.ndarray
    selected_features: tuple[str, ...]
    coefficient_matrix: np.ndarray
    intercepts: np.ndarray
    alpha_: float
    l1_ratio_: float
    cv_folds: int


def _coerce_numeric_frame(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    """Return a numeric DataFrame with stable column ordering.

    Parameters
    ----------
    frame
        Candidate feature or response frame.
    name
        Human-readable name used in error messages.

    Returns
    -------
    pandas.DataFrame
        Numeric copy of the supplied frame.

    Raises
    ------
    ValueError
        Raised when the input is empty or contains non-numeric data.
    """
    if frame.shape[0] == 0 or frame.shape[1] == 0:
        raise ValueError(f"{name} must be non-empty.")
    numeric = frame.apply(pd.to_numeric, errors="raise")
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite numeric values.")
    return numeric


def fit_multitask_elastic_net_screen(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    *,
    cv: int = 5,
    l1_ratio: Sequence[float] | float = (0.2, 0.5, 0.8, 1.0),
    alphas: Sequence[float] | int = 100,
    max_iter: int = 5000,
    tol: float = 1e-4,
    random_state: int = 123,
    selection_tol: float = 1e-8,
) -> ScreeningSelectionResult:
    """Fit a multi-output elastic-net screening model.

    Parameters
    ----------
    X
        Candidate input matrix with one column per feature.
    Y
        Multi-output response matrix aligned to ``X``.
    cv
        Requested number of cross-validation folds. The effective number of folds is
        capped at the number of aligned training rows.
    l1_ratio
        Elastic-net mixing values searched by ``MultiTaskElasticNetCV``.
    alphas
        Penalty grid specification passed through to ``MultiTaskElasticNetCV``.
    max_iter
        Maximum coordinate-descent iterations.
    tol
        Optimization tolerance passed to the estimator.
    random_state
        Random seed used by the estimator when stochastic behavior is enabled.
    selection_tol
        Absolute coefficient threshold used to convert the fitted coefficient matrix
        into a stable selected-feature mask.

    Returns
    -------
    ScreeningSelectionResult
        Fitted screening result with selected features preserved in original order.

    Raises
    ------
    ValueError
        Raised when the inputs cannot support a valid multi-output screening fit.
    """
    X_aligned, Y_aligned = align_xy(X, Y)
    X_numeric = _coerce_numeric_frame(X_aligned, name="X")
    Y_numeric = _coerce_numeric_frame(Y_aligned, name="Y")

    n_rows = len(X_numeric)
    cv_folds = min(int(cv), n_rows)
    if cv_folds < 2:
        raise ValueError("At least two aligned rows are required for screening cross-validation.")

    x_values = X_numeric.to_numpy(dtype=float)
    y_values = Y_numeric.to_numpy(dtype=float)
    x_scaled = StandardScaler().fit_transform(x_values)
    y_scaled = StandardScaler().fit_transform(y_values)

    estimator = MultiTaskElasticNetCV(
        l1_ratio=l1_ratio,
        alphas=alphas,
        cv=cv_folds,
        fit_intercept=True,
        max_iter=int(max_iter),
        tol=float(tol),
        random_state=int(random_state),
    )
    estimator.fit(x_scaled, y_scaled)

    coef_matrix = np.asarray(estimator.coef_, dtype=float)
    if coef_matrix.ndim != 2:
        raise ValueError("Expected a two-dimensional coefficient matrix from screening fit.")

    selected_mask = np.any(np.abs(coef_matrix) > float(selection_tol), axis=0)
    feature_names = tuple(str(column) for column in X_numeric.columns)
    output_names = tuple(str(column) for column in Y_numeric.columns)
    selected_features = tuple(
        feature_name
        for feature_name, keep in zip(feature_names, selected_mask, strict=True)
        if keep
    )

    return ScreeningSelectionResult(
        feature_names=feature_names,
        output_names=output_names,
        selected_mask=selected_mask.astype(bool, copy=True),
        selected_features=selected_features,
        coefficient_matrix=coef_matrix.copy(),
        intercepts=np.asarray(estimator.intercept_, dtype=float).copy(),
        alpha_=float(estimator.alpha_),
        l1_ratio_=float(estimator.l1_ratio_),
        cv_folds=cv_folds,
    )


def screening_selection_table(result: ScreeningSelectionResult) -> pd.DataFrame:
    """Summarize the executable screening-selection result as a table.

    Parameters
    ----------
    result
        Fitted screening result from :func:`fit_multitask_elastic_net_screen`.

    Returns
    -------
    pandas.DataFrame
        Feature-level summary preserving original feature order.
    """
    nonzero_output_count = np.sum(np.abs(result.coefficient_matrix) > 0.0, axis=0)
    max_abs_coef = np.max(np.abs(result.coefficient_matrix), axis=0)
    return pd.DataFrame(
        {
            "feature_name": list(result.feature_names),
            "original_position": list(range(len(result.feature_names))),
            "selected": result.selected_mask.astype(bool),
            "nonzero_output_count": nonzero_output_count.astype(int),
            "max_abs_standardized_coef": max_abs_coef.astype(float),
        }
    )


@dataclass(frozen=True)
class ScreeningImportanceResult:
    """Feature importance scores from an OLS refit on the enriched candidate set.

    Provides a simple, interpretable ranking of enriched candidates by their
    mean absolute standardized OLS coefficient magnitude across responses.
    Suitable as initial importance scores for the support-selection rule in
    :func:`~rfm_pipeline.final_ols.select_support_via_refit`.

    Parameters
    ----------
    feature_names
        Original-order enriched candidate feature names.
    coef_magnitudes
        Mean absolute standardized OLS coefficient across responses, one value
        per feature. Larger values indicate stronger linear association with the
        response set under standardization.
    """

    feature_names: tuple[str, ...]
    coef_magnitudes: np.ndarray


def compute_enriched_coef_magnitudes(
    X: pd.DataFrame,
    Y: pd.DataFrame,
) -> ScreeningImportanceResult:
    """Compute standardized OLS coefficient magnitudes for enriched candidates.

    Fits OLS on the provided (X, Y) training data and returns the mean absolute
    standardized coefficient across responses as a feature-importance ranking.
    This ranking is suitable as the initial importance signal for the refit-based
    support-selection rule in :func:`~rfm_pipeline.final_ols.select_support_via_refit`.

    Parameters
    ----------
    X
        Enriched candidate feature matrix (training data).
    Y
        Response matrix (training data) aligned to ``X``.

    Returns
    -------
    ScreeningImportanceResult
        Feature names with corresponding mean absolute standardized OLS
        coefficient magnitudes.

    Raises
    ------
    ValueError
        Raised when the inputs are empty or contain non-finite numeric values.
    """
    X_numeric = _coerce_numeric_frame(X, name="X")
    Y_numeric = _coerce_numeric_frame(Y, name="Y")

    X_values = X_numeric.to_numpy(dtype=float)
    Y_values = Y_numeric.to_numpy(dtype=float)

    x_scaler = StandardScaler().fit(X_values)
    y_scaler = StandardScaler().fit(Y_values)
    X_scaled = x_scaler.transform(X_values)
    Y_scaled = y_scaler.transform(Y_values)

    model = LinearRegression(fit_intercept=True).fit(X_scaled, Y_scaled)
    coef = np.asarray(model.coef_, dtype=float)
    if coef.ndim == 1:
        coef = coef[np.newaxis, :]  # → (1, n_features) for single-output case
    coef_magnitudes = np.mean(np.abs(coef), axis=0)

    return ScreeningImportanceResult(
        feature_names=tuple(str(c) for c in X_numeric.columns),
        coef_magnitudes=coef_magnitudes.copy(),
    )
