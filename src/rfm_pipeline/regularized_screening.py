"""Recovered regularized-screening contracts and executable foundations.

This module keeps the audited workflow contracts explicit while also providing a small,
canonical executable implementation of the archived multi-output screening family. The
notebook-derived PCA/debiased-LASSO workflow remains notebook-derived until a verified
standalone implementation is ported.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import MultiTaskElasticNetCV
from sklearn.preprocessing import StandardScaler

from .data import align_xy


@dataclass(frozen=True)
class ScreeningWorkflowContract:
    """Structured description of one recovered regularized-screening workflow.

    Attributes
    ----------
    workflow_name
        Human-readable workflow label.
    provenance
        Source provenance classification for the workflow.
    source_artifact
        Archived script or notebook from which the contract was recovered.
    estimator_family
        High-level estimator family used for screening.
    response_representation
        Response-space representation used by the workflow before selection.
    holdout_fraction
        External holdout fraction used by the workflow.
    candidate_input_count
        Number of candidate input columns at screening start when recovered.
    output_count
        Number of outputs entering the workflow stage.
    output_count_after_culling
        Number of outputs retained after notebook-side culling, when applicable.
    selected_feature_count
        Number of retained features after the final sparse-screening stage,
        when recovered.
    tuning_subset_file
        Optional archived filename used for output-subset tuning.
    scoring_rule
        Primary scoring or model-selection rule used by the workflow.
    notes
        Additional concise scientific notes from the audit.
    """

    workflow_name: str
    provenance: str
    source_artifact: str
    estimator_family: str
    response_representation: str
    holdout_fraction: float
    candidate_input_count: int | None
    output_count: int | None
    output_count_after_culling: int | None
    selected_feature_count: int | None
    tuning_subset_file: str | None
    scoring_rule: str | None
    notes: tuple[str, ...]


@dataclass(frozen=True)
class ScreeningSelectionResult:
    """Result from the executable multi-output screening foundation.

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


def archived_multitask_enet_contract() -> ScreeningWorkflowContract:
    """Return the recovered archived-script screening contract.

    Returns
    -------
    ScreeningWorkflowContract
        Contract recovered from the archived
        ``multivariate_mmreg_pipeline.with_subset.py`` script.
    """
    return ScreeningWorkflowContract(
        workflow_name="archived_multitask_elastic_net",
        provenance="source-script",
        source_artifact="multivariate_mmreg_pipeline.with_subset.py",
        estimator_family="MultiTaskElasticNetCV",
        response_representation="direct standardized multi-output response",
        holdout_fraction=0.05,
        candidate_input_count=352,
        output_count=23495,
        output_count_after_culling=None,
        selected_feature_count=None,
        tuning_subset_file="tune_vars.csv",
        scoring_rule="cross-validation within MultiTaskElasticNetCV",
        notes=(
            "Tunes on a target subset before refitting on all outputs.",
            "Represents the canonical archived regularized screening script.",
        ),
    )


def notebook_sparse_screening_contract() -> ScreeningWorkflowContract:
    """Return the recovered notebook-derived sparse-screening contract.

    Returns
    -------
    ScreeningWorkflowContract
        Contract recovered from ``LASSO_to_OLS_v9.ipynb``.
    """
    return ScreeningWorkflowContract(
        workflow_name="notebook_pca_debiased_lasso",
        provenance="notebook-derived",
        source_artifact="LASSO_to_OLS_v9.ipynb",
        estimator_family="PCA plus de-biased LASSO with final OLS handoff",
        response_representation="PCA-compressed output representation",
        holdout_fraction=0.10,
        candidate_input_count=352,
        output_count=23495,
        output_count_after_culling=9782,
        selected_feature_count=346,
        tuning_subset_file=None,
        scoring_rule="EBIC-guided sparse path search with ALO/KKT diagnostics",
        notes=(
            "Uses a deterministic 18,000 / 2,000 train-holdout split.",
            "Uses notebook-side bootstrap uncertainty for macro nRMSE reporting.",
            "Produces the richer export schema already expected by downstream viz code.",
        ),
    )


def canonical_screening_contracts() -> tuple[ScreeningWorkflowContract, ...]:
    """Return the recovered screening contracts in audit order.

    Returns
    -------
    tuple[ScreeningWorkflowContract, ...]
        Archived-script contract followed by the notebook-derived contract.
    """
    return (
        archived_multitask_enet_contract(),
        notebook_sparse_screening_contract(),
    )


def screening_contract_table() -> pd.DataFrame:
    """Return the recovered screening contracts as a tabular summary.

    Returns
    -------
    pandas.DataFrame
        One row per recovered workflow contract.
    """
    return pd.DataFrame.from_records(
        [
            {
                "workflow_name": contract.workflow_name,
                "provenance": contract.provenance,
                "source_artifact": contract.source_artifact,
                "estimator_family": contract.estimator_family,
                "response_representation": contract.response_representation,
                "holdout_fraction": contract.holdout_fraction,
                "candidate_input_count": contract.candidate_input_count,
                "output_count": contract.output_count,
                "output_count_after_culling": contract.output_count_after_culling,
                "selected_feature_count": contract.selected_feature_count,
                "tuning_subset_file": contract.tuning_subset_file,
                "scoring_rule": contract.scoring_rule,
            }
            for contract in canonical_screening_contracts()
        ]
    )


def screening_divergence_table() -> pd.DataFrame:
    """Summarize material divergences between recovered screening paths.

    Returns
    -------
    pandas.DataFrame
        Row-wise divergence summary comparing the archived script with the
        notebook-derived workflow.
    """
    archived, notebook = canonical_screening_contracts()
    records = [
        {
            "dimension": "holdout_fraction",
            "archived_script": archived.holdout_fraction,
            "notebook_workflow": notebook.holdout_fraction,
            "scientific_implication": "Different external validation design.",
        },
        {
            "dimension": "estimator_family",
            "archived_script": archived.estimator_family,
            "notebook_workflow": notebook.estimator_family,
            "scientific_implication": (
                "Different sparse-screening estimator and optimization path."
            ),
        },
        {
            "dimension": "response_representation",
            "archived_script": archived.response_representation,
            "notebook_workflow": notebook.response_representation,
            "scientific_implication": "Notebook compresses outputs before screening.",
        },
        {
            "dimension": "output_count_after_culling",
            "archived_script": archived.output_count_after_culling,
            "notebook_workflow": notebook.output_count_after_culling,
            "scientific_implication": "Notebook culls outputs before PCA-LASSO search.",
        },
        {
            "dimension": "selected_feature_count",
            "archived_script": archived.selected_feature_count,
            "notebook_workflow": notebook.selected_feature_count,
            "scientific_implication": "Notebook records a recovered final feature count.",
        },
        {
            "dimension": "scoring_rule",
            "archived_script": archived.scoring_rule,
            "notebook_workflow": notebook.scoring_rule,
            "scientific_implication": "Model selection criteria differ materially.",
        },
        {
            "dimension": "tuning_subset_file",
            "archived_script": archived.tuning_subset_file,
            "notebook_workflow": notebook.tuning_subset_file,
            "scientific_implication": "Archived script tunes on an explicit output subset file.",
        },
    ]
    return pd.DataFrame.from_records(records)


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
    """Fit a canonical executable screening model based on the archived script family.

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
