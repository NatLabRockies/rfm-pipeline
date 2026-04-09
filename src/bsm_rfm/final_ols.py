"""Notebook-derived final OLS and post-fit export helpers.

This module does not yet reimplement the full notebook workflow. It captures the
recovered final-OLS handoff and post-fit artifact contract as explicit, tested code so
that later refactor steps can build on a stable boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


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
