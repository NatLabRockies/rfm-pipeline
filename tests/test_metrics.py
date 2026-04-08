"""Tests for test metrics."""

from __future__ import annotations

import numpy as np
import pytest

from bsm_rfm.metrics import (
    bootstrap_macro_nrmse_ci,
    macro_nrmse_with_ref,
    make_null_mean_prediction,
)


def test_macro_nrmse_with_ref_matches_manual_calculation():
    y_true = np.array([[0.0], [1.0], [2.0], [3.0]])
    y_pred = np.array([[0.0], [1.0], [2.0], [4.0]])
    y_ref = np.array([[0.0], [5.0]])
    macro, info, rmse = macro_nrmse_with_ref(y_true, y_pred, y_ref)
    assert pytest.approx(macro) == (0.5 / 5.0)
    assert info == {"k_used": 1, "k_total": 1}
    assert pytest.approx(rmse[0]) == 0.5


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
