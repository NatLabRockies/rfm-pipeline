"""Canonical workflow provenance recovered from the audited case-study archive.

This module does not implement the full scientific workflow. Instead, it exposes the
recovered stage sequence and case-study numbers as importable, tested package data so
later scientific modules can depend on an explicit provenance contract rather than on
markdown notes alone.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class WorkflowStage:
    """Recovered stage-level provenance for the reduced-form workflow.

    Parameters
    ----------
    name
        Stable stage identifier.
    order
        Canonical order in the recovered end-to-end workflow.
    provenance
        Provenance label describing the strongest available source for the stage.
    source_artifact
        Audited script, notebook, or report artifact supporting the stage.
    status
        Current module-port status in this refactor scaffold.
    description
        Concise description of the stage responsibility.
    """

    name: str
    order: int
    provenance: str
    source_artifact: str
    status: str
    description: str


@dataclass(frozen=True)
class CaseStudyNumber:
    """Recovered quantitative fact from the audited workflow archive.

    Parameters
    ----------
    key
        Stable identifier for the recovered quantity.
    value
        Numeric value recovered from the audited source materials.
    unit
        Human-readable unit or count label.
    provenance
        Provenance label for the number.
    note
        Short explanation of how the quantity fits into the workflow.
    """

    key: str
    value: int
    unit: str
    provenance: str
    note: str


def canonical_workflow_stages() -> tuple[WorkflowStage, ...]:
    """Return the recovered canonical workflow-stage sequence.

    Returns
    -------
    tuple of WorkflowStage
        Ordered stage definitions recovered from the audited archive and design notes.
    """
    return (
        WorkflowStage(
            name="upstream_null_screening",
            order=1,
            provenance="source-derived",
            source_artifact="null_distribution.py",
            status="implemented_adapter",
            description="Permutation-null Delta screening over the upstream 300k sample.",
        ),
        WorkflowStage(
            name="feature_expansion",
            order=2,
            provenance="notebook-derived",
            source_artifact="make_nonlinear_features.ipynb",
            status="spec_recovered_not_fully_ported",
            description=("Create scenario flags, nonlinear terms, and second-order interactions."),
        ),
        WorkflowStage(
            name="modeling_subset_creation",
            order=3,
            provenance="audit-resolved",
            source_artifact="workflow_audit.md",
            status="implemented_foundation",
            description=(
                "Balanced 20k modeling subset built by sampling 5k rows in each boolean "
                "scenario stratum."
            ),
        ),
        WorkflowStage(
            name="regularized_screening",
            order=4,
            provenance="notebook-derived",
            source_artifact="LASSO_to_OLS_v9.ipynb",
            status="not_ported",
            description="Sparse screening and dimensionality reduction before final OLS.",
        ),
        WorkflowStage(
            name="final_ols",
            order=5,
            provenance="notebook-derived",
            source_artifact="LASSO_to_OLS_v9.ipynb",
            status="not_ported",
            description=("Fit interpretable output-wise OLS models on the retained feature set."),
        ),
        WorkflowStage(
            name="evaluation_export",
            order=6,
            provenance="notebook-derived",
            source_artifact="LASSO_to_OLS_v9.ipynb",
            status="not_ported",
            description=(
                "Compute point metrics and bootstrap summaries, then write export tables."
            ),
        ),
        WorkflowStage(
            name="downstream_visualization",
            order=7,
            provenance="consumer-contract",
            source_artifact="viz_io consumer expectations",
            status="implemented_foundation",
            description="Read canonical exported artifacts for postfit visualization.",
        ),
    )


def workflow_stage_table() -> pd.DataFrame:
    """Return the canonical workflow-stage sequence as a table.

    Returns
    -------
    pandas.DataFrame
        One row per recovered stage ordered by the canonical workflow sequence.
    """
    return pd.DataFrame(asdict(stage) for stage in canonical_workflow_stages()).sort_values(
        "order",
        ignore_index=True,
    )


def canonical_case_study_numbers() -> tuple[CaseStudyNumber, ...]:
    """Return recovered quantitative facts for the audited case study.

    Returns
    -------
    tuple of CaseStudyNumber
        Stable, importable case-study counts that are already documented in the audit.
    """
    return (
        CaseStudyNumber(
            key="upstream_sample_size",
            value=300_000,
            unit="rows",
            provenance="source-derived",
            note="Standardized upstream sample used for null screening.",
        ),
        CaseStudyNumber(
            key="modeling_subset_size",
            value=20_000,
            unit="rows",
            provenance="audit-resolved",
            note="Balanced modeling subset derived from the upstream sample.",
        ),
        CaseStudyNumber(
            key="boolean_strata",
            value=4,
            unit="strata",
            provenance="audit-resolved",
            note="AFSC/UAEORO boolean scenario combinations.",
        ),
        CaseStudyNumber(
            key="rows_per_boolean_stratum",
            value=5_000,
            unit="rows",
            provenance="audit-resolved",
            note="Rows drawn independently within each boolean scenario combination.",
        ),
        CaseStudyNumber(
            key="notebook_train_rows",
            value=18_000,
            unit="rows",
            provenance="notebook-derived",
            note="Notebook train split after the 10 percent external holdout.",
        ),
        CaseStudyNumber(
            key="notebook_holdout_rows",
            value=2_000,
            unit="rows",
            provenance="notebook-derived",
            note="Notebook external holdout split size.",
        ),
        CaseStudyNumber(
            key="candidate_input_count",
            value=352,
            unit="inputs",
            provenance="notebook-derived",
            note="Expanded feature count loaded by the notebook at model start.",
        ),
        CaseStudyNumber(
            key="full_output_count",
            value=23_495,
            unit="outputs",
            provenance="notebook-derived",
            note="Total outputs before notebook output culling.",
        ),
        CaseStudyNumber(
            key="culled_output_count",
            value=9_782,
            unit="outputs",
            provenance="notebook-derived",
            note="Outputs retained for the notebook PCA-LASSO search.",
        ),
        CaseStudyNumber(
            key="selected_feature_count",
            value=346,
            unit="features",
            provenance="notebook-derived",
            note="Retained feature count after the notebook screening stage.",
        ),
        CaseStudyNumber(
            key="first_order_selected_features",
            value=62,
            unit="features",
            provenance="notebook-derived",
            note="First-order terms in the recovered selected-feature summary.",
        ),
        CaseStudyNumber(
            key="nonlinear_selected_features",
            value=40,
            unit="features",
            provenance="notebook-derived",
            note="Nonlinear transformations in the recovered selected-feature summary.",
        ),
        CaseStudyNumber(
            key="second_order_selected_features",
            value=244,
            unit="features",
            provenance="notebook-derived",
            note="Second-order interaction terms in the recovered selected-feature summary.",
        ),
        CaseStudyNumber(
            key="null_permutation_count",
            value=200,
            unit="replicates",
            provenance="source-derived",
            note="Permutation-null count used in null_distribution.py.",
        ),
        CaseStudyNumber(
            key="null_resample_size",
            value=2_000,
            unit="rows",
            provenance="source-derived",
            note="Influential-resampling size used after upstream screening.",
        ),
    )


def case_study_number_table() -> pd.DataFrame:
    """Return recovered case-study numbers as a table.

    Returns
    -------
    pandas.DataFrame
        One row per recovered quantity ordered as documented by the workflow audit.
    """
    return pd.DataFrame(asdict(item) for item in canonical_case_study_numbers())
