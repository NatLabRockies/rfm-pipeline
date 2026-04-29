"""Frozen contract for the recovered notebook de-biased-LASSO stage.

This module is intentionally contract-only. It records source-backed facts and
blockers before any deterministic de-biased-LASSO implementation replaces the
current public EBIC/L1 sparse-selection surrogate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class DebiasedLassoStageContract:
    """Recovered contract for the manuscript sparse-selection target stage.

    The contract separates known notebook facts from unresolved implementation
    details. In particular, it prevents the current public EBIC/L1 surrogate from
    being silently promoted to manuscript-exact de-biased LASSO before the source
    notebook cells are audited.
    """

    stage_name: str
    source_workflow_reference: str
    source_artifact: str
    public_stage_function: str
    current_public_method: str
    implementation_status: str
    source_workflow_equivalence_status: str
    source_artifact_available_in_public_repo: bool
    response_representation: str
    candidate_input_count: int
    output_count: int
    output_count_after_culling: int
    retained_component_count: int
    train_sample_count: int
    holdout_sample_count: int
    holdout_fraction: float
    l1_ratio: float
    ebic_gamma: float
    alpha_fraction_grid: tuple[float, ...]
    selected_alpha_fraction_reference: float
    selected_alpha_absolute_reference: float
    scoring_rule: str
    diagnostics: tuple[str, ...]
    debiasing_definition_status: str
    selected_feature_count_reference: int
    final_support_count_reference: int
    required_before_exact_claim: tuple[str, ...]

    def can_claim_exact_implementation(self) -> bool:
        """Return whether the public stage may claim exact notebook equivalence."""
        return (
            self.implementation_status == "implemented"
            and self.source_workflow_equivalence_status == "validated"
            and self.debiasing_definition_status == "frozen_from_source_notebook"
        )

    def as_record(self) -> dict[str, Any]:
        """Return a stable record suitable for provenance tables or tests."""
        record = asdict(self)
        record["alpha_fraction_grid"] = tuple(self.alpha_fraction_grid)
        record["diagnostics"] = tuple(self.diagnostics)
        record["required_before_exact_claim"] = tuple(self.required_before_exact_claim)
        record["can_claim_exact_implementation"] = self.can_claim_exact_implementation()
        return record


def notebook_debiased_lasso_stage_contract() -> DebiasedLassoStageContract:
    """Return the frozen recovered-notebook de-biased-LASSO contract."""
    return DebiasedLassoStageContract(
        stage_name="sparse_selection_and_stability",
        source_workflow_reference="notebook_pca_debiased_lasso",
        source_artifact="LASSO_to_OLS_v9.ipynb",
        public_stage_function="select_manuscript_sparse_support",
        current_public_method="ebic_l1_component_union_with_subsample_stability",
        implementation_status="contract_frozen_not_implemented",
        source_workflow_equivalence_status="not_yet_validated",
        source_artifact_available_in_public_repo=False,
        response_representation="39 PCA component scores after output culling",
        candidate_input_count=352,
        output_count=23495,
        output_count_after_culling=9782,
        retained_component_count=39,
        train_sample_count=18000,
        holdout_sample_count=2000,
        holdout_fraction=0.10,
        l1_ratio=1.00,
        ebic_gamma=0.5,
        alpha_fraction_grid=(0.75, 0.50, 0.25, 0.10, 0.05, 0.02, 0.01),
        selected_alpha_fraction_reference=0.10,
        selected_alpha_absolute_reference=2.8541,
        scoring_rule="EBIC-guided sparse path search with ALO/KKT diagnostics",
        diagnostics=("ALO", "KKT"),
        debiasing_definition_status="unresolved_requires_source_notebook_audit",
        selected_feature_count_reference=346,
        final_support_count_reference=340,
        required_before_exact_claim=(
            "recover and inspect LASSO_to_OLS_v9.ipynb source cells",
            "freeze the exact de-biasing estimator definition",
            "add deterministic tests for alpha-path, EBIC, diagnostics, and artifacts",
            "validate emitted support and diagnostics against the recovered notebook run",
            "preserve the stability and final-OLS handoff artifact schema",
        ),
    )
