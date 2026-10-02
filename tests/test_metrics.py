"""Tests for test metrics."""

from __future__ import annotations

import numpy as np
import pytest

import rfm_pipeline.metrics as metrics_module
from rfm_pipeline.metrics import (
    bootstrap_macro_nrmse_ci,
    build_output_eligibility_ledger,
    macro_nrmse_from_ledger,
    macro_nrmse_with_ref,
    make_null_mean_prediction,
    per_output_nrmse_frame,
)


def test_macro_nrmse_with_ref_matches_manual_calculation():
    y_true = np.array([[0.0], [1.0], [2.0], [3.0]])
    y_pred = np.array([[0.0], [1.0], [2.0], [4.0]])
    y_ref = np.array([[0.0], [5.0]])
    macro, info, rmse = macro_nrmse_with_ref(y_true, y_pred, y_ref)
    assert pytest.approx(macro) == (0.5 / 5.0)
    assert info == {"k_used": 1, "k_total": 1}
    assert pytest.approx(rmse[0]) == 0.5


def test_macro_nrmse_rejects_shape_broadcasting() -> None:
    y_true = np.zeros((4, 1))
    y_pred = np.zeros(4)
    y_ref = np.zeros((3, 1))

    with pytest.raises(ValueError, match="two-dimensional.*matching shapes"):
        macro_nrmse_with_ref(y_true, y_pred, y_ref)


def test_bootstrap_macro_nrmse_ci_returns_expected_fields():
    y_true = np.array([[0.0], [1.0], [2.0], [3.0]])
    y_pred = np.array([[0.0], [1.2], [1.8], [3.2]])
    y_ref = np.array([[0.0], [3.0]])
    summary = bootstrap_macro_nrmse_ci(
        y_true,
        y_pred,
        y_ref,
        n_boot=40,
        sample_size=3,
        random_state=9,
    )
    assert summary["n_boot"] == 40
    assert summary["bootstrap_sample_size"] == 3
    assert summary["ci_type"] == "percentile"
    assert summary["ci_lower"] <= summary["point_estimate"] <= summary["ci_upper"]


def test_make_null_mean_prediction_repeats_training_mean():
    y_train = np.array([[1.0, 2.0], [3.0, 6.0]])
    pred = make_null_mean_prediction(y_train, n_rows=4)
    assert pred.shape == (4, 2)
    assert np.allclose(pred[0], [2.0, 4.0])
    assert np.allclose(pred[-1], [2.0, 4.0])


def test_bootstrap_macro_nrmse_ci_reuses_checkpointed_replicates(
    monkeypatch,
    tmp_path,
):
    y_true = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    y_pred = np.array([[0.1], [1.2], [1.9], [3.1], [4.2], [5.1]])
    y_ref = np.array([[0.0], [5.0]])

    original_metric = metrics_module.macro_nrmse_with_ref
    call_counter = {"count": 0}

    def _counting_metric(*args, **kwargs):  # noqa: ANN002, ANN003
        call_counter["count"] += 1
        return original_metric(*args, **kwargs)

    monkeypatch.setattr(metrics_module, "macro_nrmse_with_ref", _counting_metric)

    checkpoint_dir = tmp_path / "bootstrap_ckpt"
    first = bootstrap_macro_nrmse_ci(
        y_true,
        y_pred,
        y_ref,
        n_boot=20,
        sample_size=4,
        random_state=9,
        checkpoint_dir=checkpoint_dir,
    )
    assert call_counter["count"] > 1

    call_counter["count"] = 0
    second = bootstrap_macro_nrmse_ci(
        y_true,
        y_pred,
        y_ref,
        n_boot=20,
        sample_size=4,
        random_state=9,
        checkpoint_dir=checkpoint_dir,
    )
    assert call_counter["count"] == 1
    assert first == second


def test_bootstrap_macro_nrmse_ci_supports_active_replicate_subsets(
    monkeypatch,
    tmp_path,
):
    y_true = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    y_pred = np.array([[0.1], [1.2], [1.9], [3.1], [4.2], [5.1]])
    y_ref = np.array([[0.0], [5.0]])

    original_metric = metrics_module.macro_nrmse_with_ref
    call_counter = {"count": 0}

    def _counting_metric(*args, **kwargs):  # noqa: ANN002, ANN003
        call_counter["count"] += 1
        return original_metric(*args, **kwargs)

    monkeypatch.setattr(metrics_module, "macro_nrmse_with_ref", _counting_metric)
    checkpoint_dir = tmp_path / "bootstrap_subset_ckpt"
    active_subset = {1, 3, 5}

    bootstrap_macro_nrmse_ci(
        y_true,
        y_pred,
        y_ref,
        n_boot=8,
        sample_size=4,
        random_state=9,
        checkpoint_dir=checkpoint_dir,
        active_bootstrap_indices=active_subset,
    )
    assert call_counter["count"] == 1 + len(active_subset)

    call_counter["count"] = 0
    bootstrap_macro_nrmse_ci(
        y_true,
        y_pred,
        y_ref,
        n_boot=8,
        sample_size=4,
        random_state=9,
        checkpoint_dir=checkpoint_dir,
        active_bootstrap_indices=active_subset,
    )
    assert call_counter["count"] == 1


def test_bootstrap_checkpoints_are_keyed_by_array_contents(tmp_path) -> None:
    y_true = np.array([[0.0], [1.0], [2.0], [3.0]])
    y_pred_a = np.array([[0.0], [0.0], [3.0], [3.0]])
    y_pred_b = np.array([[1.0], [1.0], [2.0], [2.0]])
    y_ref = np.array([[0.0], [3.0]])
    assert y_pred_a.mean() == y_pred_b.mean()

    checkpoint_dir = tmp_path / "bootstrap_ckpt"
    bootstrap_macro_nrmse_ci(
        y_true,
        y_pred_a,
        y_ref,
        n_boot=20,
        random_state=9,
        checkpoint_dir=checkpoint_dir,
    )
    checkpointed = bootstrap_macro_nrmse_ci(
        y_true,
        y_pred_b,
        y_ref,
        n_boot=20,
        random_state=9,
        checkpoint_dir=checkpoint_dir,
    )
    fresh = bootstrap_macro_nrmse_ci(
        y_true,
        y_pred_b,
        y_ref,
        n_boot=20,
        random_state=9,
    )

    assert checkpointed == fresh
    assert len(list(checkpoint_dir.iterdir())) == 2


def test_per_output_metrics_reject_shape_broadcasting() -> None:
    with pytest.raises(ValueError, match="matching shapes"):
        per_output_nrmse_frame(
            np.zeros((4, 2)),
            np.zeros((4, 1)),
            np.zeros((3, 2)),
            ["a", "b"],
        )


def test_eligibility_ledger_requires_two_dimensional_reference() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        build_output_eligibility_ledger(np.zeros(4))


def test_ledger_metric_rejects_row_count_mismatch() -> None:
    Y = np.zeros((4, 2))
    ledger = build_output_eligibility_ledger(np.array([[0.0, 0.0], [1.0, 1.0]]))

    with pytest.raises(ValueError, match="matching shapes"):
        macro_nrmse_from_ledger(Y, np.zeros((3, 2)), ledger)
