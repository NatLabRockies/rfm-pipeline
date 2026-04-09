"""Notebook-derived regularized-screening workflow contracts.

This module does not claim that the notebook workflow is the only canonical scientific
truth for the case study. Instead, it captures the recovered, auditable configuration
boundary for the downstream regularized-screening stage so later refactors can test
against a stable contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import pandas as pd

ScreeningProvenance = Literal["archived_script", "notebook_derived", "audit_resolved"]


@dataclass(frozen=True)
class RegularizedScreeningSpec:
    """Configuration contract for a recovered regularized-screening workflow.

    Attributes
    ----------
    workflow_name
        Human-readable name for the recovered workflow variant.
    provenance
        Provenance label describing where the workflow definition came from.
    model_family
        Main regression family used in the screening stage.
    response_representation
        Representation used for the multi-output response matrix before screening.
    holdout_fraction
        External holdout fraction used by the workflow.
    candidate_input_count
        Number of candidate inputs entering the screening stage.
    full_output_count
        Number of outputs available before any culling.
    output_count_after_culling
        Number of outputs retained after any pre-screen culling step.
    selected_feature_count
        Number of features retained by the regularized-screening workflow.
    tuning_subset_file
        Optional file used to define a tuning subset of outputs.
    l1_ratio_grid
        Candidate l1-ratio values explored during hyperparameter search.
    alpha_fraction_grid
        Candidate alpha-fraction values explored during hyperparameter search.
    selected_l1_ratio
        Selected l1-ratio recovered from the workflow, when known.
    selected_alpha_fraction
        Selected alpha-fraction recovered from the workflow, when known.
    selected_alpha_absolute
        Selected absolute alpha value recovered from the workflow, when known.
    model_selection_criterion
        Main selection criterion used to choose the final screening setting.
    notes
        Additional concise provenance notes needed to interpret the workflow.
    """

    workflow_name: str
    provenance: ScreeningProvenance
    model_family: str
    response_representation: str
    holdout_fraction: float
    candidate_input_count: int
    full_output_count: int
    output_count_after_culling: int | None
    selected_feature_count: int | None
    tuning_subset_file: str | None
    l1_ratio_grid: tuple[float, ...]
    alpha_fraction_grid: tuple[float, ...]
    selected_l1_ratio: float | None
    selected_alpha_fraction: float | None
    selected_alpha_absolute: float | None
    model_selection_criterion: str | None
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view of the screening specification."""
        payload = asdict(self)
        payload["l1_ratio_grid"] = list(self.l1_ratio_grid)
        payload["alpha_fraction_grid"] = list(self.alpha_fraction_grid)
        payload["notes"] = list(self.notes)
        return payload


@dataclass(frozen=True)
class ScreeningWorkflowComparison:
    """Summary of the material differences between recovered workflow variants."""

    archived_script: RegularizedScreeningSpec
    notebook_workflow: RegularizedScreeningSpec

    def divergence_summary(self) -> dict[str, tuple[Any, Any]]:
        """Return the key archived-script versus notebook divergences."""
        return {
            "model_family": (
                self.archived_script.model_family,
                self.notebook_workflow.model_family,
            ),
            "response_representation": (
                self.archived_script.response_representation,
                self.notebook_workflow.response_representation,
            ),
            "holdout_fraction": (
                self.archived_script.holdout_fraction,
                self.notebook_workflow.holdout_fraction,
            ),
            "tuning_subset_file": (
                self.archived_script.tuning_subset_file,
                self.notebook_workflow.tuning_subset_file,
            ),
            "model_selection_criterion": (
                self.archived_script.model_selection_criterion,
                self.notebook_workflow.model_selection_criterion,
            ),
        }


def archived_multitask_elastic_net_spec() -> RegularizedScreeningSpec:
    """Return the recovered archived-script screening specification."""
    return RegularizedScreeningSpec(
        workflow_name="archived multitask elastic-net script",
        provenance="archived_script",
        model_family="MultiTaskElasticNetCV",
        response_representation="direct standardized multi-output response",
        holdout_fraction=0.05,
        candidate_input_count=352,
        full_output_count=23495,
        output_count_after_culling=None,
        selected_feature_count=None,
        tuning_subset_file="tune_vars.csv",
        l1_ratio_grid=(),
        alpha_fraction_grid=(),
        selected_l1_ratio=None,
        selected_alpha_fraction=None,
        selected_alpha_absolute=None,
        model_selection_criterion=None,
        notes=(
            "Recovered source script uses a 5% holdout by default.",
            "Script tunes on a target subset, then refits on all outputs.",
        ),
    )


def notebook_regularized_screening_spec() -> RegularizedScreeningSpec:
    """Return the recovered notebook-derived screening specification."""
    return RegularizedScreeningSpec(
        workflow_name="notebook PCA plus debiased-lasso screening",
        provenance="notebook_derived",
        model_family="PCA plus de-biased LASSO with final OLS handoff",
        response_representation="PCA-compressed output representation",
        holdout_fraction=0.10,
        candidate_input_count=352,
        full_output_count=23495,
        output_count_after_culling=9782,
        selected_feature_count=346,
        tuning_subset_file=None,
        l1_ratio_grid=(1.0, 0.95, 0.9),
        alpha_fraction_grid=(0.75, 0.5, 0.25, 0.10, 0.05, 0.02, 0.01),
        selected_l1_ratio=1.0,
        selected_alpha_fraction=0.10,
        selected_alpha_absolute=2.8541,
        model_selection_criterion="EBIC",
        notes=(
            "Notebook uses a deterministic 10% external holdout.",
            "Notebook applies output culling before the PCA-LASSO search.",
            "Recovered notebook workflow includes KKT checks and BH-FDR rowwise selection.",
        ),
    )


def recovered_regularized_screening_comparison() -> ScreeningWorkflowComparison:
    """Return the archived-script versus notebook workflow comparison."""
    return ScreeningWorkflowComparison(
        archived_script=archived_multitask_elastic_net_spec(),
        notebook_workflow=notebook_regularized_screening_spec(),
    )


def screening_spec_table() -> pd.DataFrame:
    """Return a tidy table of the recovered screening workflow variants."""
    specs = [
        archived_multitask_elastic_net_spec(),
        notebook_regularized_screening_spec(),
    ]
    rows = [spec.to_dict() for spec in specs]
    return pd.DataFrame(rows)


def divergence_table() -> pd.DataFrame:
    """Return a tidy table of the key workflow divergences."""
    comparison = recovered_regularized_screening_comparison()
    rows = []
    for field_name, values in comparison.divergence_summary().items():
        rows.append(
            {
                "field": field_name,
                "archived_script": values[0],
                "notebook_workflow": values[1],
            }
        )
    return pd.DataFrame(rows)
