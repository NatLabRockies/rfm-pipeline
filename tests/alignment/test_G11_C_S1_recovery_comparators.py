"""G11-C-S1: frozen recovery comparators use training data only."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from rfm_pipeline.campaign_contract import G11_CONTRACT
from rfm_pipeline.recovery_study import (
    run_recovery_comparators,
    write_recovery_comparator_result,
)


def _fixture(seed: int = 31):
    rng = np.random.default_rng(seed)
    x_train = rng.normal(size=(60, 6))
    x_eval = rng.normal(size=(20, 6))
    oracle_train = np.column_stack([x_train[:, 0], x_train[:, 1], x_train[:, 0] * x_train[:, 1]])
    oracle_eval = np.column_stack([x_eval[:, 0], x_eval[:, 1], x_eval[:, 0] * x_eval[:, 1]])
    algebraic_train = np.column_stack([x_train, x_train[:, 0] * x_train[:, 1]])
    algebraic_eval = np.column_stack([x_eval, x_eval[:, 0] * x_eval[:, 1]])
    coefficient = rng.normal(size=(3, 3))
    y_train = oracle_train @ coefficient + rng.normal(scale=0.1, size=(60, 3))
    proposed = oracle_eval @ coefficient
    return (
        x_train,
        y_train,
        x_eval,
        oracle_train,
        oracle_eval,
        algebraic_train,
        algebraic_eval,
        proposed,
    )


def test_comparators_have_no_evaluation_truth_argument_and_are_deterministic(
    tmp_path: Path,
) -> None:
    assert "Y_eval" not in inspect.signature(run_recovery_comparators).parameters
    fixture = _fixture()

    first = run_recovery_comparators(
        X_train=fixture[0],
        Y_train=fixture[1],
        X_eval=fixture[2],
        oracle_train_design=fixture[3],
        oracle_eval_design=fixture[4],
        algebraic_train_design=fixture[5],
        algebraic_eval_design=fixture[6],
        proposed_predictions=fixture[7],
        execution_contract=G11_CONTRACT,
        seed=123,
    )
    second = run_recovery_comparators(
        X_train=fixture[0],
        Y_train=fixture[1],
        X_eval=fixture[2],
        oracle_train_design=fixture[3],
        oracle_eval_design=fixture[4],
        algebraic_train_design=fixture[5],
        algebraic_eval_design=fixture[6],
        proposed_predictions=fixture[7],
        execution_contract=G11_CONTRACT,
        seed=123,
    )

    assert tuple(first.predictions) == G11_CONTRACT.recovery_comparators
    for name in G11_CONTRACT.recovery_comparators:
        assert first.predictions[name].shape == (20, 3)
        np.testing.assert_allclose(first.predictions[name], second.predictions[name])
    assert first.selected_hyperparameters == second.selected_hyperparameters
    assert first.selected_hyperparameters["boosted_tree"]["response_components"] == min(
        G11_CONTRACT.boosted_tree_response_components,
        fixture[1].shape[1],
    )
    assert first.schedule_sha256 == second.schedule_sha256
    paths = write_recovery_comparator_result(first, tmp_path / "comparators")
    assert paths["predictions"].is_file()
    assert paths["metadata"].is_file()


def test_comparator_design_mismatch_fails_before_any_fit() -> None:
    fixture = _fixture()
    with pytest.raises(ValueError, match="oracle evaluation"):
        run_recovery_comparators(
            X_train=fixture[0],
            Y_train=fixture[1],
            X_eval=fixture[2],
            oracle_train_design=fixture[3],
            oracle_eval_design=fixture[4][:, :-1],
            algebraic_train_design=fixture[5],
            algebraic_eval_design=fixture[6],
            proposed_predictions=fixture[7],
            execution_contract=G11_CONTRACT,
            seed=123,
        )
