"""Tests for the frozen de-biased-LASSO implementation contract."""

from __future__ import annotations

from pathlib import Path

from rfm_pipeline.debiased_lasso_contract import notebook_debiased_lasso_stage_contract


def test_debiased_lasso_contract_freezes_recovered_notebook_facts() -> None:
    contract = notebook_debiased_lasso_stage_contract()

    assert contract.stage_name == "sparse_selection_and_stability"
    assert contract.source_workflow_reference == "notebook_pca_debiased_lasso"
    assert contract.source_artifact == "LASSO_to_OLS_v9.ipynb"
    assert contract.public_stage_function == "select_manuscript_sparse_support"
    assert contract.current_public_method == "ebic_l1_component_union_with_subsample_stability"
    assert contract.implementation_status == "contract_frozen_not_implemented"
    assert contract.source_workflow_equivalence_status == "not_yet_validated"
    assert contract.source_artifact_available_in_public_repo is False

    assert contract.candidate_input_count == 352
    assert contract.output_count == 23495
    assert contract.output_count_after_culling == 9782
    assert contract.retained_component_count == 39
    assert contract.train_sample_count == 18000
    assert contract.holdout_sample_count == 2000
    assert contract.holdout_fraction == 0.10
    assert contract.l1_ratio == 1.00
    assert contract.ebic_gamma == 0.5
    assert contract.alpha_fraction_grid == (0.75, 0.50, 0.25, 0.10, 0.05, 0.02, 0.01)
    assert contract.selected_alpha_fraction_reference == 0.10
    assert contract.selected_alpha_absolute_reference == 2.8541
    assert contract.selected_feature_count_reference == 346
    assert contract.final_support_count_reference == 340


def test_debiased_lasso_contract_blocks_exactness_claims() -> None:
    contract = notebook_debiased_lasso_stage_contract()

    assert contract.debiasing_definition_status == "unresolved_requires_source_notebook_audit"
    assert contract.can_claim_exact_implementation() is False
    assert "recover and inspect LASSO_to_OLS_v9.ipynb source cells" in (
        contract.required_before_exact_claim
    )
    assert "freeze the exact de-biasing estimator definition" in (
        contract.required_before_exact_claim
    )


def test_debiased_lasso_contract_record_is_stable() -> None:
    row = notebook_debiased_lasso_stage_contract().as_record()

    assert row["alpha_fraction_grid"] == (0.75, 0.50, 0.25, 0.10, 0.05, 0.02, 0.01)
    assert row["diagnostics"] == ("ALO", "KKT")
    assert row["can_claim_exact_implementation"] is False
    assert row["debiasing_definition_status"] == "unresolved_requires_source_notebook_audit"


def test_debiased_lasso_contract_document_exists_and_names_blockers() -> None:
    text = Path("docs/debiased_lasso_contract.md").read_text(encoding="utf-8")

    assert "contract_frozen_not_implemented" in text
    assert "unresolved_requires_source_notebook_audit" in text
    assert "EBIC-guided sparse path search with ALO/KKT diagnostics" in text
    assert "0.75, 0.50, 0.25, 0.10, 0.05, 0.02, 0.01" in text
    assert "LASSO_to_OLS_v9.ipynb" in text
