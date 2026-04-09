"""Tests for recovered regularized-screening workflow contracts."""

from __future__ import annotations

from bsm_rfm.regularized_screening import (
    archived_multitask_enet_contract,
    canonical_screening_contracts,
    notebook_sparse_screening_contract,
    screening_contract_table,
    screening_divergence_table,
)


def test_archived_multitask_contract_matches_recovered_script_audit() -> None:
    contract = archived_multitask_enet_contract()
    assert contract.provenance == "source-script"
    assert contract.source_artifact == "multivariate_mmreg_pipeline.with_subset.py"
    assert contract.estimator_family == "MultiTaskElasticNetCV"
    assert contract.holdout_fraction == 0.05
    assert contract.candidate_input_count == 352
    assert contract.output_count == 23495
    assert contract.tuning_subset_file == "tune_vars.csv"


def test_notebook_sparse_contract_matches_recovered_notebook_audit() -> None:
    contract = notebook_sparse_screening_contract()
    assert contract.provenance == "notebook-derived"
    assert contract.source_artifact == "LASSO_to_OLS_v9.ipynb"
    assert contract.holdout_fraction == 0.10
    assert contract.output_count_after_culling == 9782
    assert contract.selected_feature_count == 346
    assert "EBIC" in (contract.scoring_rule or "")


def test_screening_contracts_preserve_audit_order() -> None:
    contracts = canonical_screening_contracts()
    assert [contract.workflow_name for contract in contracts] == [
        "archived_multitask_elastic_net",
        "notebook_pca_debiased_lasso",
    ]


def test_screening_contract_table_exposes_key_recovered_fields() -> None:
    table = screening_contract_table()
    assert table["workflow_name"].tolist() == [
        "archived_multitask_elastic_net",
        "notebook_pca_debiased_lasso",
    ]
    assert table["holdout_fraction"].tolist() == [0.05, 0.10]
    assert table["candidate_input_count"].tolist() == [352, 352]


def test_screening_divergence_table_surfaces_material_differences() -> None:
    table = screening_divergence_table()
    assert set(table["dimension"]) >= {
        "holdout_fraction",
        "estimator_family",
        "response_representation",
        "output_count_after_culling",
        "selected_feature_count",
        "scoring_rule",
        "tuning_subset_file",
    }
    holdout = table.loc[table["dimension"] == "holdout_fraction"].iloc[0]
    assert holdout["archived_script"] == 0.05
    assert holdout["notebook_workflow"] == 0.10
