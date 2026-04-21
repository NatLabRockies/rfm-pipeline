"""Tests for recovered and executable regularized-screening helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bsm_rfm.regularized_screening import (
    archived_multitask_enet_contract,
    canonical_screening_contracts,
    fit_multitask_elastic_net_screen,
    notebook_sparse_screening_contract,
    screening_contract_table,
    screening_divergence_table,
    screening_selection_table,
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


def test_fit_multitask_elastic_net_screen_recovers_signal_features() -> None:
    rng = np.random.RandomState(7)
    X = pd.DataFrame(
        rng.normal(size=(80, 4)),
        columns=["x1", "x2", "x3", "x4"],
    )
    Y = pd.DataFrame(
        {
            "y1": 3.0 * X["x1"] - 2.0 * X["x3"],
            "y2": -1.5 * X["x1"] + 1.0 * X["x3"],
        }
    )

    result = fit_multitask_elastic_net_screen(
        X,
        Y,
        cv=4,
        l1_ratio=(0.9, 1.0),
        alphas=50,
        max_iter=10000,
        selection_tol=1e-4,
        random_state=11,
    )

    assert set(result.selected_features) == {"x1", "x3"}
    assert result.feature_names == ("x1", "x2", "x3", "x4")
    assert result.output_names == ("y1", "y2")
    assert result.cv_folds == 4


def test_screening_selection_table_preserves_feature_order_and_selection_metadata() -> None:
    rng = np.random.RandomState(5)
    X = pd.DataFrame(rng.normal(size=(40, 3)), columns=["a", "b", "c"])
    Y = pd.DataFrame({"y": 4.0 * X["b"]})

    result = fit_multitask_elastic_net_screen(
        X,
        Y,
        cv=3,
        l1_ratio=1.0,
        alphas=25,
        max_iter=10000,
        selection_tol=1e-5,
        random_state=13,
    )
    table = screening_selection_table(result)

    assert table["feature_name"].tolist() == ["a", "b", "c"]
    assert table["original_position"].tolist() == [0, 1, 2]
    selected_row = table.loc[table["feature_name"] == "b"].iloc[0]
    assert bool(selected_row["selected"])
    assert int(selected_row["nonzero_output_count"]) == 1
