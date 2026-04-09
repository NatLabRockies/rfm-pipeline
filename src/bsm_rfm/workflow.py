"""Canonical workflow provenance for the reduced-form modeling case study."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class WorkflowStage:
    """Describe one workflow stage and its current implementation status.

    Attributes
    ----------
    stage_key
        Stable programmatic stage identifier.
    stage_label
        Human-readable stage label.
    provenance
        Provenance classification for the current implementation boundary.
    implementation_status
        Whether the stage is implemented in canonical package code, wrapped through a
        recovered source script, or still only documented.
    source_artifact
        Recovered source artifact or notebook associated with the stage.
    notes
        Short audit note explaining the current boundary.
    """

    stage_key: str
    stage_label: str
    provenance: str
    implementation_status: str
    source_artifact: str
    notes: str


@dataclass(frozen=True)
class CaseStudyNumber:
    """Verified case-study number recovered during the repo audit."""

    key: str
    value: int
    units: str
    provenance: str
    notes: str


def canonical_workflow_stages() -> list[WorkflowStage]:
    """Return the audited canonical workflow stage sequence.

    Returns
    -------
    list[WorkflowStage]
        Workflow stages in their audited canonical order.
    """
    return [
        WorkflowStage(
            stage_key="upstream_null_screening",
            stage_label="Upstream null screening",
            provenance="source-derived",
            implementation_status="wrapped_source_script",
            source_artifact="null_distribution.py",
            notes=(
                "Canonical delta sensitivity with permutation null. Package code wraps "
                "the recovered source interface rather than reimplementing it."
            ),
        ),
        WorkflowStage(
            stage_key="feature_expansion",
            stage_label="Feature expansion",
            provenance="notebook-derived",
            implementation_status="implemented_config_boundary",
            source_artifact="make_nonlinear_features.ipynb",
            notes=(
                "Interaction and nonlinear feature generation is notebook-derived, but "
                "the package now provides an explicit configuration boundary so those "
                "choices are labeled instead of treated as implicit canonical truth."
            ),
        ),
        WorkflowStage(
            stage_key="modeling_subset_creation",
            stage_label="Modeling subset creation",
            provenance="audit-resolved",
            implementation_status="implemented",
            source_artifact="workflow audit + recovered subset rule",
            notes=(
                "Balanced 20k modeling subset formed by sampling 5,000 rows within each "
                "AFSC/UAEORO boolean combination."
            ),
        ),
        WorkflowStage(
            stage_key="regularized_screening",
            stage_label="Regularized screening",
            provenance="notebook-derived",
            implementation_status="documented_only",
            source_artifact="LASSO_to_OLS_v9.ipynb",
            notes=(
                "PCA plus de-biased LASSO search is scientifically important but not yet "
                "ported into canonical modules."
            ),
        ),
        WorkflowStage(
            stage_key="final_ols",
            stage_label="Final OLS",
            provenance="notebook-derived",
            implementation_status="documented_only",
            source_artifact="LASSO_to_OLS_v9.ipynb",
            notes=(
                "Final selected-feature OLS export remains notebook-derived until the "
                "fitting and export contract are ported into package code."
            ),
        ),
        WorkflowStage(
            stage_key="evaluation_export",
            stage_label="Evaluation and export",
            provenance="consumer-contract",
            implementation_status="partially_implemented",
            source_artifact="artifact schema + notebook exports",
            notes=(
                "Metrics, manifests, and viz-side loaders exist, but the full postfit "
                "evaluation/export workflow is not yet ported."
            ),
        ),
        WorkflowStage(
            stage_key="downstream_visualization",
            stage_label="Downstream visualization",
            provenance="consumer-contract",
            implementation_status="implemented_read_only_loader",
            source_artifact="canonical exported artifacts",
            notes=(
                "Visualization-side loading helpers are implemented for artifact "
                "consumption, not for fitting the upstream model."
            ),
        ),
    ]


def workflow_stage_table() -> pd.DataFrame:
    """Return workflow stages as a tabular audit artifact."""
    return pd.DataFrame(asdict(stage) for stage in canonical_workflow_stages())


def canonical_case_study_numbers() -> list[CaseStudyNumber]:
    """Return verified workflow numbers recovered from the archive audit."""
    return [
        CaseStudyNumber(
            key="upstream_rows",
            value=300000,
            units="rows",
            provenance="source-derived",
            notes="Standardized upstream screening matrix size in null_distribution.py.",
        ),
        CaseStudyNumber(
            key="modeling_subset_rows",
            value=20000,
            units="rows",
            provenance="audit-resolved",
            notes="Balanced modeling subset recovered from the audited workflow.",
        ),
        CaseStudyNumber(
            key="boolean_strata",
            value=4,
            units="strata",
            provenance="audit-resolved",
            notes="AFSC/UAEORO combinations used for balanced sampling.",
        ),
        CaseStudyNumber(
            key="rows_per_boolean_stratum",
            value=5000,
            units="rows",
            provenance="audit-resolved",
            notes="Per-stratum draw size used to form the 20k modeling subset.",
        ),
        CaseStudyNumber(
            key="notebook_train_rows",
            value=18000,
            units="rows",
            provenance="notebook-derived",
            notes="Notebook external training split size under the 10 percent holdout.",
        ),
        CaseStudyNumber(
            key="notebook_holdout_rows",
            value=2000,
            units="rows",
            provenance="notebook-derived",
            notes="Notebook external holdout size under the 10 percent split.",
        ),
        CaseStudyNumber(
            key="candidate_inputs",
            value=352,
            units="features",
            provenance="notebook-derived",
            notes="Candidate input count loaded at modeling start in the notebook.",
        ),
        CaseStudyNumber(
            key="full_outputs",
            value=23495,
            units="outputs",
            provenance="notebook-derived",
            notes="Output count loaded at modeling start in the notebook.",
        ),
        CaseStudyNumber(
            key="selected_features",
            value=346,
            units="features",
            provenance="notebook-derived",
            notes="Selected-feature count after the final notebook LASSO stage.",
        ),
        CaseStudyNumber(
            key="selected_first_order_features",
            value=62,
            units="features",
            provenance="notebook-derived",
            notes="Selected first-order feature count parsed from notebook artifacts.",
        ),
        CaseStudyNumber(
            key="selected_nonlinear_features",
            value=40,
            units="features",
            provenance="notebook-derived",
            notes=("Selected nonlinear transformation count parsed from notebook artifacts."),
        ),
        CaseStudyNumber(
            key="selected_second_order_features",
            value=244,
            units="features",
            provenance="notebook-derived",
            notes="Selected second-order interaction count parsed from notebook artifacts.",
        ),
        CaseStudyNumber(
            key="null_permutations",
            value=200,
            units="permutations",
            provenance="source-derived",
            notes="Permutation count B in the recovered null-screening script.",
        ),
        CaseStudyNumber(
            key="null_resample_size",
            value=2000,
            units="rows",
            provenance="source-derived",
            notes="New influential-input resampling size N_NEW in null_distribution.py.",
        ),
        CaseStudyNumber(
            key="bootstrap_replicates",
            value=1000,
            units="bootstrap_samples",
            provenance="notebook-derived",
            notes="Bootstrap replicate count used for notebook nRMSE intervals.",
        ),
    ]


def case_study_number_table() -> pd.DataFrame:
    """Return case-study numbers as a tabular audit artifact."""
    return pd.DataFrame(asdict(item) for item in canonical_case_study_numbers())
