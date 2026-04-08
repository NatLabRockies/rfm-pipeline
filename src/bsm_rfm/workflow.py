"""Canonical workflow provenance for the BSM reduced-form refactor.

This module captures the currently verified stage ordering, provenance boundaries,
and recovered case-study numbers. It does not implement the full modeling
workflow. Instead, it provides a tested, importable source of truth for what is
already known from the recovered scripts, notebooks, and audit notes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class WorkflowStage:
    """Structured description of one stage in the reduced-form workflow.

    Parameters
    ----------
    stage_key
        Stable machine-readable stage identifier.
    stage_name
        Human-readable stage name.
    provenance_status
        Provenance classification such as ``source-derived`` or
        ``notebook-derived``.
    implementation_status
        Current refactor status for the stage in this public package.
    canonical_source
        Primary recovered script, notebook, or audit-resolved provenance note.
    summary
        Concise description of what the stage does.
    key_settings
        Verified stage settings and case-study numbers associated with the stage.
    outputs
        Key artifacts or outputs produced by the stage.
    """

    stage_key: str
    stage_name: str
    provenance_status: str
    implementation_status: str
    canonical_source: str
    summary: str
    key_settings: dict[str, Any] = field(default_factory=dict)
    outputs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return the workflow stage as a JSON-serializable dictionary."""
        return asdict(self)


def canonical_workflow_stages() -> list[WorkflowStage]:
    """Return the currently verified canonical workflow stage sequence.

    Returns
    -------
    list[WorkflowStage]
        Ordered stage definitions reflecting the recovered source scripts,
        notebook-derived stages, and audit-resolved provenance boundaries.
    """
    return [
        WorkflowStage(
            stage_key="upstream_null_screening",
            stage_name="Upstream null screening",
            provenance_status="source-derived",
            implementation_status="implemented_adapter",
            canonical_source="null_distribution.py",
            summary=(
                "SALib Delta sensitivity screening with permutation-null cutoffs "
                "used as the canonical upstream screening stage."
            ),
            key_settings={
                "dataset_size": 300000,
                "B": 200,
                "alpha": 0.05,
                "method": "fwer-max",
                "N_new": 2000,
            },
            outputs=[
                "influential-input reports",
                "permutation-null cutoff tables",
                "new LHS sample of size 2000",
            ],
        ),
        WorkflowStage(
            stage_key="feature_expansion",
            stage_name="Feature expansion",
            provenance_status="artifact-recovered",
            implementation_status="not_yet_ported",
            canonical_source="make_nonlinear_features.ipynb",
            summary=(
                "Builds the nonlinear and interaction feature library after the "
                "upstream influential-input stage and adds AFSC/UAEORO scenario flags."
            ),
            key_settings={
                "input_artifact": "influential_factors.300k.null_010.nl.csv",
                "output_artifact": "sa_068.null_010.X.nl.csv",
            },
            outputs=[
                "expanded candidate feature matrix",
                "scenario flag columns",
            ],
        ),
        WorkflowStage(
            stage_key="modeling_subset_creation",
            stage_name="Modeling subset creation",
            provenance_status="audit-resolved",
            implementation_status="implemented_subset_sampler",
            canonical_source="balanced AFSC/UAEORO stratified subset path",
            summary=(
                "Creates the 20,000-row modeling subset by drawing 5,000 rows within "
                "each AFSC/UAEORO boolean combination from the 300,000-run archive."
            ),
            key_settings={
                "input_dataset_size": 300000,
                "output_dataset_size": 20000,
                "n_strata": 4,
                "n_per_stratum": 5000,
            },
            outputs=[
                "balanced 20k modeling subset",
            ],
        ),
        WorkflowStage(
            stage_key="regularized_screening",
            stage_name="Regularized screening",
            provenance_status="notebook-derived",
            implementation_status="not_yet_ported",
            canonical_source="LASSO_to_OLS_v9.ipynb",
            summary=(
                "Performs notebook-derived PCA compression and de-biased LASSO style "
                "screening rather than the archived MultiTaskElasticNetCV script path."
            ),
            key_settings={
                "holdout_fraction": 0.10,
                "candidate_input_count": 352,
                "full_output_count": 23495,
                "kept_outputs_before_search": 9782,
                "culled_outputs_before_search": 13713,
                "selected_feature_count": 346,
            },
            outputs=[
                "selected screening features",
                "screening diagnostics",
            ],
        ),
        WorkflowStage(
            stage_key="final_ols",
            stage_name="Final OLS refit",
            provenance_status="notebook-derived",
            implementation_status="not_yet_ported",
            canonical_source="LASSO_to_OLS_v9.ipynb",
            summary=(
                "Refits the final interpretable OLS models after the notebook-derived "
                "screening stage."
            ),
            key_settings={
                "selected_feature_count": 346,
                "first_order_count": 62,
                "nonlinear_count": 40,
                "second_order_count": 244,
            },
            outputs=[
                "raw-scale coefficient matrix",
                "standardized coefficient matrix",
                "intercepts",
            ],
        ),
        WorkflowStage(
            stage_key="evaluation_export",
            stage_name="Evaluation and export",
            provenance_status="notebook-derived",
            implementation_status="partially_ported_helpers_only",
            canonical_source="LASSO_to_OLS_v9.ipynb",
            summary=(
                "Computes holdout evaluation summaries and exports the richer post-fit "
                "artifact bundle consumed downstream by visualization code."
            ),
            key_settings={
                "bootstrap_replicates": 1000,
                "bootstrap_ci_type": "percentile",
                "normalization_reference": "fixed Y_ref range",
            },
            outputs=[
                "nRMSE summaries",
                "postfit diagnostics bundle",
                "artifact metadata tables",
            ],
        ),
        WorkflowStage(
            stage_key="downstream_visualization",
            stage_name="Downstream visualization",
            provenance_status="consumer-contract",
            implementation_status="implemented_loader_only",
            canonical_source="postfit_diagnostics export contract",
            summary=(
                "Loads canonical exported artifact tables for downstream plots, tables, "
                "and manuscript assembly without re-running the modeling code."
            ),
            key_settings={},
            outputs=[
                "visualization-ready metadata tables",
                "visualization-ready coefficient tables",
            ],
        ),
    ]


def workflow_stage_table() -> pd.DataFrame:
    """Return the canonical workflow stage sequence as a tabular summary.

    Returns
    -------
    pandas.DataFrame
        One row per workflow stage, preserving the canonical stage order.
    """
    return pd.DataFrame.from_records(stage.to_dict() for stage in canonical_workflow_stages())


def canonical_case_study_numbers() -> dict[str, Any]:
    """Return the verified case-study numbers currently supported by the audit.

    Returns
    -------
    dict[str, Any]
        Stable mapping of recovered case-study numbers and settings that are
        currently treated as canonical or explicitly notebook-derived.
    """
    return {
        "upstream_screening_dataset_size": 300000,
        "modeling_subset_size": 20000,
        "n_boolean_strata": 4,
        "n_per_boolean_stratum": 5000,
        "notebook_train_size": 18000,
        "notebook_holdout_size": 2000,
        "candidate_input_count": 352,
        "full_output_count": 23495,
        "diagnostic_plot_output_count": 555,
        "kept_outputs_before_search": 9782,
        "culled_outputs_before_search": 13713,
        "selected_feature_count": 346,
        "selected_first_order_count": 62,
        "selected_nonlinear_count": 40,
        "selected_second_order_count": 244,
        "null_permutation_count": 200,
        "null_resample_size": 2000,
        "bootstrap_replicates": 1000,
    }


def case_study_number_table() -> pd.DataFrame:
    """Return the verified case-study numbers as a two-column table.

    Returns
    -------
    pandas.DataFrame
        Table with ``name`` and ``value`` columns for manuscript or audit use.
    """
    numbers = canonical_case_study_numbers()
    return pd.DataFrame(
        {
            "name": list(numbers.keys()),
            "value": list(numbers.values()),
        }
    )
