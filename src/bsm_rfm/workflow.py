"""Canonical workflow provenance for the BSM reduced-form case study.

This module records the recovered stage ordering and case-study counts that are
already established by the repo audit. It does not claim that every stage has
already been ported into canonical package code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class WorkflowStage:
    """Provenance record for one workflow stage.

    Parameters
    ----------
    stage_key
        Stable machine-readable stage identifier.
    stage_label
        Human-readable stage name.
    source_status
        Provenance classification for the stage.
    primary_source
        Recovered script, notebook, or audit document anchoring this stage.
    implemented_in_package
        Whether this stage already exists as importable package code.
    notes
        Short explanatory note describing the current boundary.
    """

    stage_key: str
    stage_label: str
    source_status: str
    primary_source: str
    implemented_in_package: bool
    notes: str


_CANONICAL_STAGES: tuple[WorkflowStage, ...] = (
    WorkflowStage(
        stage_key="null_screening",
        stage_label="Upstream delta/null screening",
        source_status="source-derived",
        primary_source="null_distribution.py",
        implemented_in_package=True,
        notes=("Canonical upstream screening stage delegated through bsm_rfm.null_screening."),
    ),
    WorkflowStage(
        stage_key="feature_expansion",
        stage_label="Feature expansion",
        source_status="notebook-derived",
        primary_source="make_nonlinear_features.ipynb",
        implemented_in_package=False,
        notes=(
            "Adds nonlinear terms, interactions, and scenario flags after the "
            "recovered null-screening stage."
        ),
    ),
    WorkflowStage(
        stage_key="modeling_subset",
        stage_label="Balanced modeling subset creation",
        source_status="audit-resolved",
        primary_source="repo workflow audit",
        implemented_in_package=True,
        notes=(
            "Balanced 20k subset built by sampling 5,000 rows within each "
            "AFSC/UAEORO boolean combination."
        ),
    ),
    WorkflowStage(
        stage_key="regularized_screening",
        stage_label="Regularized screening",
        source_status="notebook-derived",
        primary_source="LASSO_to_OLS_v9.ipynb",
        implemented_in_package=False,
        notes=(
            "Notebook workflow replaces the archived multi-task elastic-net "
            "script with PCA plus sparse screening."
        ),
    ),
    WorkflowStage(
        stage_key="final_ols",
        stage_label="Final OLS handoff",
        source_status="notebook-derived",
        primary_source="LASSO_to_OLS_v9.ipynb",
        implemented_in_package=False,
        notes=(
            "Final selected features are handed to interpretable OLS fits with "
            "exported coefficients and diagnostics."
        ),
    ),
    WorkflowStage(
        stage_key="evaluation_export",
        stage_label="Evaluation and export",
        source_status="notebook-derived",
        primary_source="LASSO_to_OLS_v9.ipynb",
        implemented_in_package=False,
        notes=(
            "Macro nRMSE summaries, holdout diagnostics, and artifact schema "
            "live downstream of the final OLS stage."
        ),
    ),
    WorkflowStage(
        stage_key="downstream_visualization",
        stage_label="Downstream visualization",
        source_status="consumer-contract",
        primary_source="visualization notebook expectations",
        implemented_in_package=True,
        notes=(
            "Read-only artifact loading is stabilized in bsm_rfm.viz_io even "
            "though the richer export contract is still being ported."
        ),
    ),
)


_CANONICAL_CASE_STUDY_NUMBERS: tuple[tuple[str, int], ...] = (
    ("upstream_null_screening_rows", 300000),
    ("modeling_subset_rows", 20000),
    ("boolean_strata", 4),
    ("rows_per_boolean_stratum", 5000),
    ("train_rows_notebook_split", 18000),
    ("holdout_rows_notebook_split", 2000),
    ("candidate_input_count", 352),
    ("full_output_count", 23495),
    ("diagnostic_output_subset", 555),
    ("outputs_retained_before_pca_lasso", 9782),
    ("outputs_culled_before_pca_lasso", 13713),
    ("selected_feature_count", 346),
    ("selected_first_order_count", 62),
    ("selected_nonlinear_count", 40),
    ("selected_second_order_count", 244),
    ("null_permutation_count", 200),
    ("null_resample_size", 2000),
)


def canonical_workflow_stages() -> tuple[WorkflowStage, ...]:
    """Return the recovered workflow-stage sequence.

    Returns
    -------
    tuple[WorkflowStage, ...]
        Immutable ordered stage records for the current audited workflow.
    """
    return _CANONICAL_STAGES


def workflow_stage_table() -> pd.DataFrame:
    """Return the recovered workflow stages as a table.

    Returns
    -------
    pandas.DataFrame
        One row per canonical stage in execution order.
    """
    return pd.DataFrame(asdict(stage) for stage in _CANONICAL_STAGES)


def canonical_case_study_numbers() -> dict[str, int]:
    """Return the recovered case-study counts from the workflow audit.

    Returns
    -------
    dict[str, int]
        Mapping from stable metric names to audited integer values.
    """
    return {key: value for key, value in _CANONICAL_CASE_STUDY_NUMBERS}


def case_study_number_table() -> pd.DataFrame:
    """Return the recovered case-study counts as a table.

    Returns
    -------
    pandas.DataFrame
        Two-column table with metric names and integer values.
    """
    return pd.DataFrame(
        _CANONICAL_CASE_STUDY_NUMBERS,
        columns=["metric_name", "value"],
    )


__all__ = [
    "WorkflowStage",
    "canonical_workflow_stages",
    "workflow_stage_table",
    "canonical_case_study_numbers",
    "case_study_number_table",
]
