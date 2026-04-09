"""Recovered regularized-screening workflow contracts.

This module does not implement the full downstream screening estimators yet. Instead,
it makes the recovered scientific contract explicit and testable by encoding the two
materially different screening paths found in the audited archive:

- the archived standalone ``multivariate_mmreg_pipeline.with_subset.py`` script
- the notebook-derived ``LASSO_to_OLS_v9.ipynb`` workflow

The goal is to keep these paths distinct until a canonical implementation is ported
from verified scientific source artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


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
