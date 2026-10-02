"""Final OLS fitting, prediction, evaluation, and artifact helpers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from .artifacts import PipelineManifest, make_metadata_frame
from .data import SealedSplitResult, align_xy
from .features import DesignMatrixSpec, build_design_matrix, resolve_spec_levels
from .metrics import bootstrap_macro_nrmse_ci


@dataclass(frozen=True)
class PostfitArtifactSpec:
    """One table in the portable post-fit artifact bundle.

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
    # P0-S03: design spec for categorical-aware round-trip prediction
    design_spec: DesignMatrixSpec | None = None
    categorical_feature_columns: tuple[str, ...] = field(default_factory=tuple)  # type: ignore[assignment]


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


POSTFIT_ARTIFACT_SPECS = (
    PostfitArtifactSpec(
        artifact_name="all_input_metadata",
        category="metadata",
        description="Original-order metadata for all candidate inputs.",
    ),
    PostfitArtifactSpec(
        artifact_name="selected_input_metadata",
        category="metadata",
        description="Original-order metadata for selected inputs.",
    ),
    PostfitArtifactSpec(
        artifact_name="output_metadata",
        category="metadata",
        description="Original-order metadata for modeled outputs.",
    ),
    PostfitArtifactSpec(
        artifact_name="coef_matrix_standardized",
        category="coefficients",
        description="Coefficient matrix on the standardized scale.",
    ),
    PostfitArtifactSpec(
        artifact_name="coef_matrix_raw_scale",
        category="coefficients",
        description="Coefficient matrix transformed to the raw input and output scales.",
    ),
    PostfitArtifactSpec(
        artifact_name="x_standardization",
        category="standardization",
        description="Means and scales used to standardize model inputs.",
    ),
    PostfitArtifactSpec(
        artifact_name="y_standardization",
        category="standardization",
        description="Means and scales used to standardize modeled outputs.",
    ),
    PostfitArtifactSpec(
        artifact_name="nrmse_summary",
        category="evaluation",
        description="Holdout nRMSE summary and bootstrap interval.",
    ),
)


def canonical_postfit_artifact_names() -> list[str]:
    """Return the portable post-fit artifact names in write order.

    Returns
    -------
    list[str]
        Ordered artifact stems.
    """
    return [spec.artifact_name for spec in POSTFIT_ARTIFACT_SPECS]


def postfit_artifact_table() -> pd.DataFrame:
    """Describe the portable post-fit artifact bundle.

    Returns
    -------
    pandas.DataFrame
        One row per canonical artifact with ordering preserved.
    """
    rows = []
    for position, spec in enumerate(POSTFIT_ARTIFACT_SPECS):
        rows.append(
            {
                "artifact_name": spec.artifact_name,
                "artifact_position": position,
                "category": spec.category,
                "description": spec.description,
            }
        )
    return pd.DataFrame(rows)


def fit_final_ols(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    *,
    output_batch_size: int | None = None,
) -> FinalOLSFitResult:
    """Fit the canonical final OLS model on aligned raw-scale inputs and outputs.

    Parameters
    ----------
    X
        Final retained feature matrix in raw units.
    Y
        Modeled output matrix in raw units.
    output_batch_size
        When set, solve the OLS in batches of this many output columns and stack the
        results. Useful when Y has many columns that would not fit in a full coefficient
        matrix in memory. ``None`` (default) processes all columns in a single call.

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

    x_scaler = StandardScaler().fit(X_values)
    y_scaler = StandardScaler().fit(Y_values)

    x_means = np.asarray(x_scaler.mean_, dtype=float)
    x_scales = np.asarray(x_scaler.scale_, dtype=float)
    y_means = np.asarray(y_scaler.mean_, dtype=float)
    y_scales = np.asarray(y_scaler.scale_, dtype=float)

    n_outputs = Y_values.shape[1]
    if output_batch_size is None or output_batch_size >= n_outputs:
        model = LinearRegression(fit_intercept=True)
        model.fit(X_values, Y_values)
        coef_raw = np.asarray(model.coef_, dtype=float)
        if coef_raw.ndim == 1:
            coef_raw = coef_raw[np.newaxis, :]
        intercept_raw = np.atleast_1d(np.asarray(model.intercept_, dtype=float))
    else:
        # Solve output columns in batches to avoid materializing the full coef matrix.
        coef_batches = []
        intercept_batches = []
        batches = range(0, n_outputs, output_batch_size)
        for start in batches:
            end = min(start + output_batch_size, n_outputs)
            batch_model = LinearRegression(fit_intercept=True)
            batch_model.fit(X_values, Y_values[:, start:end])
            batch_coef = np.asarray(batch_model.coef_, dtype=float)
            if batch_coef.ndim == 1:
                batch_coef = batch_coef[np.newaxis, :]
            coef_batches.append(batch_coef)
            intercept_batches.append(np.atleast_1d(np.asarray(batch_model.intercept_, dtype=float)))
        coef_raw = np.vstack(coef_batches)
        intercept_raw = np.concatenate(intercept_batches)

    # Guard against zero y-scales (constant or near-zero-variance outputs).  For such
    # outputs the raw-scale coefficient vector is already near-zero, so setting the
    # denominator to 1 yields a meaningful (near-zero) standardised representation.
    y_scales_safe = np.where(y_scales > 0, y_scales, 1.0)
    coef_standardized = coef_raw * (x_scales[np.newaxis, :] / y_scales_safe[:, np.newaxis])
    intercept_standardized = (intercept_raw + coef_raw @ x_means - y_means) / y_scales_safe

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


def fit_final_ols_with_design(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    spec: DesignMatrixSpec,
    *,
    output_batch_size: int | None = None,
) -> FinalOLSFitResult:
    """Fit the final OLS model from raw inputs using a design-matrix spec.

    Applies *spec* (categorical encoding + interactions) to *X* first, then fits
    OLS on the resulting numeric design matrix.  The spec and derived categorical
    column names are stored in the returned result so that :func:`predict_from_raw_records`
    can round-trip through the same encoding at evaluation time.

    Parameters
    ----------
    X
        Raw input frame (may contain categorical/block columns declared in *spec*).
    Y
        Modeled output frame in raw units.
    spec
        Design-matrix specification describing categorical encodings and interactions.
    output_batch_size
        Forwarded to :func:`fit_final_ols`.

    Returns
    -------
    FinalOLSFitResult
        Fitted result with ``design_spec`` and ``categorical_feature_columns`` set.
    """
    resolved_spec = resolve_spec_levels(X, spec)
    dm = build_design_matrix(X, resolved_spec)
    base = fit_final_ols(dm.matrix, Y, output_batch_size=output_batch_size)
    return FinalOLSFitResult(
        feature_names=base.feature_names,
        output_names=base.output_names,
        coef_raw_scale=base.coef_raw_scale,
        intercept_raw_scale=base.intercept_raw_scale,
        coef_standardized=base.coef_standardized,
        intercept_standardized=base.intercept_standardized,
        x_means=base.x_means,
        x_scales=base.x_scales,
        y_means=base.y_means,
        y_scales=base.y_scales,
        n_training_rows=base.n_training_rows,
        design_spec=resolved_spec,
        categorical_feature_columns=dm.categorical_columns,
    )


def predict_from_raw_records(
    result: FinalOLSFitResult,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """Predict from raw records, applying the stored design spec if present.

    When *result* was produced by :func:`fit_final_ols_with_design`, the stored
    :attr:`~FinalOLSFitResult.design_spec` is used to encode categorical inputs
    before calling :func:`predict_final_ols`.  When *result* has no ``design_spec``
    (i.e. was produced by :func:`fit_final_ols` directly), the call falls back to
    :func:`predict_final_ols` unchanged.

    This is the primary entry point for scenario-contrast evaluation: pass two
    records that differ only in a categorical/block column and the predictions will
    reflect the fitted main effect for that column.

    Parameters
    ----------
    result
        Fitted OLS result, optionally carrying a design spec.
    X
        Raw input frame.  May contain categorical columns when *result* carries a
        design spec.

    Returns
    -------
    pandas.DataFrame
        Predicted outputs preserving the row index from *X*.
    """
    if result.design_spec is None:
        return predict_final_ols(result, X)
    dm = build_design_matrix(X, result.design_spec)
    return predict_final_ols(result, dm.matrix)


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
    metrics: dict[str, Any] | None = None,
    evaluation: dict[str, Any] | None = None,
    artifact_format: str = "parquet",
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
    metrics
        Optional metric payloads stored in the manifest.
    evaluation
        Optional evaluation metadata stored in the manifest.
    artifact_format
        Tabular artifact format recorded in the manifest file map. Must be either
        ``"parquet"`` or ``"csv"``.

    Returns
    -------
    dict[str, pandas.DataFrame | dict[str, Any]]
        Artifact bundle keyed by the canonical post-fit artifact names plus a
        ``manifest`` entry.
    """
    if artifact_format not in {"parquet", "csv"}:
        raise ValueError("artifact_format must be either 'parquet' or 'csv'.")

    selected = list(selected_features or result.feature_names)
    retained = list(result.feature_names)

    if isinstance(evaluation_summary, pd.DataFrame):
        nrmse_summary = evaluation_summary.copy()
    elif evaluation_summary is None:
        nrmse_summary = pd.DataFrame()
    else:
        nrmse_summary = pd.DataFrame([evaluation_summary])

    files = {
        artifact_name: f"postfit_diagnostics/{artifact_name}.{artifact_format}"
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
        metrics=dict(metrics or {}),
        evaluation=dict(evaluation or {}),
        upstream_provenance=dict(upstream_provenance or {}),
    ).to_dict()
    # P0-S03: record categorical encoding metadata when result carries a design spec.
    if result.design_spec is not None and result.design_spec.categorical_inputs:
        artifacts["manifest"]["categorical_inputs"] = [
            {"name": decl.name, "levels": decl.levels}
            for decl in result.design_spec.categorical_inputs
        ]
        artifacts["manifest"]["categorical_feature_columns"] = list(
            result.categorical_feature_columns
        )
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


@dataclass(frozen=True)
class SupportSelectionResult:
    """Result from the refit-based support-selection rule.

    Parameters
    ----------
    feature_names
        Original-order enriched candidate feature names entering selection.
    response_names
        Response column names used in the refit evaluation.
    selected_features
        Feature names retained after applying the best threshold.
    selected_mask
        Boolean mask over ``feature_names`` indicating retained features.
    best_threshold
        Coefficient-magnitude threshold chosen by internal validation.
    threshold_sensitivity
        DataFrame with columns ``threshold``, ``n_selected``, ``val_score``
        documenting performance across all candidate thresholds on the internal
        validation partition.
    initial_coef_magnitudes
        Mean absolute standardized OLS coefficient magnitudes across responses,
        one value per enriched candidate. Used as the feature-importance ranking
        that drives threshold-based selection.
    """

    feature_names: tuple[str, ...]
    response_names: tuple[str, ...]
    selected_features: tuple[str, ...]
    selected_mask: np.ndarray
    best_threshold: float
    threshold_sensitivity: pd.DataFrame
    initial_coef_magnitudes: np.ndarray


def select_threshold_on_internal_validation(
    split: SealedSplitResult,
    thresholds: Sequence[float],
    scorer: Callable[[float, pd.DataFrame, pd.DataFrame], float],
    *,
    seed: int | None = None,
) -> float:
    """Choose a threshold using only the internal-validation partition.

    Lower scorer values are preferred. Ties are resolved reproducibly with
    ``seed``. The function refuses to run after the sealed test partition has
    been exposed.
    """
    from .data import SealedTestAccessError

    if not split._sealed:
        raise SealedTestAccessError(
            "Threshold selection requires a sealed test partition; use only "
            "the training and internal-validation data during model selection."
        )
    if not thresholds:
        raise ValueError("thresholds must be a non-empty sequence.")

    candidates = [float(value) for value in thresholds]
    scores = [scorer(value, split.train, split.val) for value in candidates]
    best_score = min(scores)
    tied = [value for value, score in zip(candidates, scores, strict=True) if score == best_score]
    if len(tied) == 1:
        return tied[0]

    rng = np.random.default_rng(seed)
    return tied[int(rng.integers(len(tied)))]


def select_support_via_refit(
    feature_cols: Sequence[str],
    response_cols: Sequence[str],
    sealed_split: SealedSplitResult,
    thresholds: Sequence[float],
    *,
    seed: int | None = None,
) -> SupportSelectionResult:
    """Select sparse support from an enriched candidate set via refit-based criterion.

    Implements the validated support-selection rule (P0-S11 / F6): replaces the
    no-refit marginal-impact approximation with a justified OLS-refit criterion
    whose coefficient-magnitude threshold is selected by internal validation only,
    never the sealed test partition.

    Algorithm
    ---------
    1. Fit OLS on the enriched candidate feature set using the training partition.
    2. Compute mean absolute standardized OLS coefficient magnitudes as feature
       importance scores.
    3. For each candidate threshold: retain features whose magnitude exceeds the
       threshold, refit OLS on the training partition, evaluate RMSE on the
       internal validation partition.
    4. Select the best threshold using the P0-S05 internal-validation machinery;
       the sealed test partition is never accessed.
    5. Return the resulting sparse support and a threshold-sensitivity table.

    Parameters
    ----------
    feature_cols
        Enriched candidate feature column names. Must be present in the split data.
    response_cols
        Response column names. Must be present in the split data.
    sealed_split
        Three-way sealed split. Must still be sealed (test not yet exposed) when
        this function is called.
    thresholds
        Candidate coefficient-magnitude threshold values. Lower thresholds retain
        more features; higher thresholds enforce sparsity more aggressively.
    seed
        Seed for deterministic tie-breaking in threshold selection.

    Returns
    -------
    SupportSelectionResult
        Selected sparse support with best threshold and full sensitivity table.

    Raises
    ------
    SealedTestAccessError
        If ``sealed_split`` has already been unsealed.
    ValueError
        If ``thresholds`` is empty or required columns are missing from the split data.
    """
    from .data import SealedTestAccessError

    feature_cols = list(feature_cols)
    response_cols = list(response_cols)

    if not thresholds:
        raise ValueError("thresholds must be a non-empty sequence.")
    thresholds_list = [float(t) for t in thresholds]

    # Guard: raises SealedTestAccessError if already unsealed.
    if not sealed_split._sealed:
        raise SealedTestAccessError(
            "select_support_via_refit: the supplied split has already been unsealed. "
            "Threshold selection must use only train + internal-validation data; "
            "unsealing before selection risks test-data leakage."
        )

    train_df = sealed_split.train

    missing_features = [c for c in feature_cols if c not in train_df.columns]
    missing_responses = [c for c in response_cols if c not in train_df.columns]
    if missing_features:
        raise ValueError(f"Feature columns not found in split data: {missing_features}")
    if missing_responses:
        raise ValueError(f"Response columns not found in split data: {missing_responses}")

    # Compute initial OLS coefficient magnitudes on training data.
    X_train = train_df[feature_cols].to_numpy(dtype=float)
    Y_train = train_df[response_cols].to_numpy(dtype=float)

    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(Y_train)
    X_scaled = x_scaler.transform(X_train)
    Y_scaled = y_scaler.transform(Y_train)

    initial_model = LinearRegression(fit_intercept=True).fit(X_scaled, Y_scaled)
    coef = np.asarray(initial_model.coef_, dtype=float)
    if coef.ndim == 1:
        coef = coef[np.newaxis, :]  # → (1, n_features) for single-output case
    # Mean absolute standardized coefficient across responses.
    coef_magnitudes = np.mean(np.abs(coef), axis=0)  # shape (n_features,)

    _feature_cols = feature_cols
    _response_cols = response_cols
    _coef_magnitudes = coef_magnitudes

    def _refit_scorer(threshold: float, train_df_: pd.DataFrame, val_df_: pd.DataFrame) -> float:
        """Refit OLS on features above threshold; return validation RMSE."""
        selected = [
            f for f, mag in zip(_feature_cols, _coef_magnitudes, strict=True) if mag > threshold
        ]
        if not selected:
            return float("inf")

        Xtr = train_df_[selected].to_numpy(dtype=float)
        Ytr = train_df_[_response_cols].to_numpy(dtype=float)
        Xval = val_df_[selected].to_numpy(dtype=float)
        Yval = val_df_[_response_cols].to_numpy(dtype=float)

        model = LinearRegression(fit_intercept=True).fit(Xtr, Ytr)
        pred = model.predict(Xval)
        if pred.ndim == 1:
            pred = pred[:, np.newaxis]
        residuals = Yval - pred
        return float(np.sqrt(np.mean(residuals**2)))

    # Select best threshold via internal validation only (P0-S05 machinery).
    best_threshold = select_threshold_on_internal_validation(
        sealed_split, thresholds_list, _refit_scorer, seed=seed
    )

    # Build threshold-sensitivity table using train/val data.
    val_df = sealed_split.val
    sensitivity_rows = []
    for t in thresholds_list:
        score = _refit_scorer(t, train_df, val_df)
        n_sel = int(np.sum(coef_magnitudes > t))
        sensitivity_rows.append({"threshold": t, "n_selected": n_sel, "val_score": score})
    threshold_sensitivity = pd.DataFrame(sensitivity_rows)

    # Finalize sparse support.
    selected_mask = coef_magnitudes > best_threshold
    selected_features = tuple(
        f for f, keep in zip(feature_cols, selected_mask, strict=True) if keep
    )

    return SupportSelectionResult(
        feature_names=tuple(feature_cols),
        response_names=tuple(response_cols),
        selected_features=selected_features,
        selected_mask=selected_mask.copy(),
        best_threshold=best_threshold,
        threshold_sensitivity=threshold_sensitivity,
        initial_coef_magnitudes=coef_magnitudes.copy(),
    )


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
