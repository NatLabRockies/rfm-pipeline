"""Tests for baseline model interfaces and comparison helpers.

Tests:
- Each baseline (Ridge, PLS, ElasticNet, PerStratum) fits and predicts on synthetic data.
- predict() output shape matches Y_eval shape.
- model_size_bytes() returns a positive integer after fitting.
- compare_baselines() returns a DataFrame with the required schema columns.
- Harness records: name, rmse, r2, model_size_bytes, fit_time_s, eval_time_s, peak_memory_mb.
- All rmse values are finite and non-negative.
- All fit_time_s and eval_time_s are non-negative.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rfm_pipeline.baselines import (
    ElasticNetBaseline,
    PerStratumFirstOrderBaseline,
    PLSBaseline,
    RidgeBaseline,
    compare_baselines,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = {
    "name",
    "rmse",
    "r2",
    "model_size_bytes",
    "fit_time_s",
    "eval_time_s",
    "peak_memory_mb",
}


@pytest.fixture()
def synthetic_multioutput():
    """Small synthetic multi-output regression dataset.

    Returns (X_train, Y_train, X_eval, Y_eval) each as NumPy arrays.
    X: (n, 10), Y: (n, 4)
    """
    rng = np.random.default_rng(42)
    n_train, n_eval, n_features, n_outputs = 120, 40, 10, 4
    X_train = rng.standard_normal((n_train, n_features))
    W = rng.standard_normal((n_features, n_outputs))
    Y_train = X_train @ W + 0.1 * rng.standard_normal((n_train, n_outputs))
    X_eval = rng.standard_normal((n_eval, n_features))
    Y_eval = X_eval @ W + 0.1 * rng.standard_normal((n_eval, n_outputs))
    return X_train, Y_train, X_eval, Y_eval


@pytest.fixture()
def strata_labels():
    """Stratum labels for train (120 rows) and eval (40 rows) with 4 equal strata."""
    rng = np.random.default_rng(7)
    train_labels = np.repeat(["A", "B", "C", "D"], 30)
    rng.shuffle(train_labels)
    eval_labels = np.repeat(["A", "B", "C", "D"], 10)
    rng.shuffle(eval_labels)
    return train_labels, eval_labels


# ---------------------------------------------------------------------------
# Ridge baseline
# ---------------------------------------------------------------------------


def test_ridge_fits_predicts(synthetic_multioutput):
    X_train, Y_train, X_eval, Y_eval = synthetic_multioutput
    bl = RidgeBaseline(alpha=1.0)
    bl.fit(X_train, Y_train)
    Y_hat = bl.predict(X_eval)
    assert Y_hat.shape == Y_eval.shape, "Ridge predict shape mismatch"
    assert np.all(np.isfinite(Y_hat)), "Ridge predict produced non-finite values"


def test_ridge_model_size(synthetic_multioutput):
    X_train, Y_train, X_eval, Y_eval = synthetic_multioutput
    bl = RidgeBaseline()
    bl.fit(X_train, Y_train)
    assert bl.model_size_bytes() > 0


def test_ridge_predict_before_fit_raises():
    bl = RidgeBaseline()
    with pytest.raises(RuntimeError):
        bl.predict(np.zeros((5, 10)))


# ---------------------------------------------------------------------------
# PLS baseline
# ---------------------------------------------------------------------------


def test_pls_fits_predicts(synthetic_multioutput):
    X_train, Y_train, X_eval, Y_eval = synthetic_multioutput
    bl = PLSBaseline(n_components=3)
    bl.fit(X_train, Y_train)
    Y_hat = bl.predict(X_eval)
    assert Y_hat.shape == Y_eval.shape, "PLS predict shape mismatch"
    assert np.all(np.isfinite(Y_hat)), "PLS predict produced non-finite values"


def test_pls_model_size(synthetic_multioutput):
    X_train, Y_train, X_eval, Y_eval = synthetic_multioutput
    bl = PLSBaseline(n_components=3)
    bl.fit(X_train, Y_train)
    assert bl.model_size_bytes() > 0


def test_pls_predict_before_fit_raises():
    bl = PLSBaseline()
    with pytest.raises(RuntimeError):
        bl.predict(np.zeros((5, 10)))


# ---------------------------------------------------------------------------
# ElasticNet baseline
# ---------------------------------------------------------------------------


def test_elasticnet_fits_predicts(synthetic_multioutput):
    X_train, Y_train, X_eval, Y_eval = synthetic_multioutput
    bl = ElasticNetBaseline(alpha=0.01, l1_ratio=0.5)
    bl.fit(X_train, Y_train)
    Y_hat = bl.predict(X_eval)
    assert Y_hat.shape == Y_eval.shape, "ElasticNet predict shape mismatch"
    assert np.all(np.isfinite(Y_hat)), "ElasticNet predict produced non-finite values"


def test_elasticnet_model_size(synthetic_multioutput):
    X_train, Y_train, X_eval, Y_eval = synthetic_multioutput
    bl = ElasticNetBaseline()
    bl.fit(X_train, Y_train)
    assert bl.model_size_bytes() > 0


def test_elasticnet_predict_before_fit_raises():
    bl = ElasticNetBaseline()
    with pytest.raises(RuntimeError):
        bl.predict(np.zeros((5, 10)))


# ---------------------------------------------------------------------------
# PerStratum baseline
# ---------------------------------------------------------------------------


def test_perstratum_fits_predicts(synthetic_multioutput, strata_labels):
    X_train, Y_train, X_eval, Y_eval = synthetic_multioutput
    train_labels, eval_labels = strata_labels
    bl = PerStratumFirstOrderBaseline()
    bl.fit(X_train, Y_train, strata=train_labels)
    Y_hat = bl.predict(X_eval, strata=eval_labels)
    assert Y_hat.shape == Y_eval.shape, "PerStratum predict shape mismatch"
    assert np.all(np.isfinite(Y_hat)), "PerStratum predict produced non-finite values"


def test_perstratum_model_size(synthetic_multioutput, strata_labels):
    X_train, Y_train, X_eval, Y_eval = synthetic_multioutput
    train_labels, _ = strata_labels
    bl = PerStratumFirstOrderBaseline()
    bl.fit(X_train, Y_train, strata=train_labels)
    assert bl.model_size_bytes() > 0


def test_perstratum_fit_requires_strata():
    bl = PerStratumFirstOrderBaseline()
    with pytest.raises((ValueError, TypeError)):
        bl.fit(np.zeros((10, 5)), np.zeros((10, 2)))


def test_perstratum_unseen_stratum_falls_back(synthetic_multioutput, strata_labels):
    """Unseen strata at eval time should use fallback without error."""
    X_train, Y_train, X_eval, Y_eval = synthetic_multioutput
    train_labels, _ = strata_labels
    bl = PerStratumFirstOrderBaseline()
    bl.fit(X_train, Y_train, strata=train_labels)
    # Use a stratum label never seen in training
    unseen_labels = np.array(["Z"] * X_eval.shape[0])
    Y_hat = bl.predict(X_eval, strata=unseen_labels)
    assert Y_hat.shape == Y_eval.shape


# ---------------------------------------------------------------------------
# compare_baselines harness
# ---------------------------------------------------------------------------


def test_harness_schema(synthetic_multioutput):
    X_train, Y_train, X_eval, Y_eval = synthetic_multioutput
    baselines = [
        RidgeBaseline(alpha=1.0),
        PLSBaseline(n_components=3),
        ElasticNetBaseline(alpha=0.01),
    ]
    df = compare_baselines(baselines, X_train, Y_train, X_eval, Y_eval)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(baselines)
    missing = REQUIRED_COLUMNS - set(df.columns)
    assert not missing, f"Missing columns: {missing}"


def test_harness_values_finite(synthetic_multioutput):
    X_train, Y_train, X_eval, Y_eval = synthetic_multioutput
    baselines = [RidgeBaseline(), PLSBaseline(n_components=2), ElasticNetBaseline()]
    df = compare_baselines(baselines, X_train, Y_train, X_eval, Y_eval)
    assert (df["rmse"] >= 0).all(), "rmse must be non-negative"
    assert df["rmse"].apply(np.isfinite).all(), "rmse must be finite"
    assert (df["fit_time_s"] >= 0).all()
    assert (df["eval_time_s"] >= 0).all()
    assert (df["model_size_bytes"] > 0).all()
    assert (df["peak_memory_mb"] >= 0).all()


def test_harness_with_perstratum(synthetic_multioutput, strata_labels):
    """PerStratumFirstOrderBaseline integrates with the harness via fit/predict kwargs."""
    X_train, Y_train, X_eval, Y_eval = synthetic_multioutput
    train_labels, eval_labels = strata_labels
    bl = PerStratumFirstOrderBaseline()
    bname = bl.name
    df = compare_baselines(
        [bl],
        X_train,
        Y_train,
        X_eval,
        Y_eval,
        fit_kwargs={bname: {"strata": train_labels}},
        predict_kwargs={bname: {"strata": eval_labels}},
    )
    assert len(df) == 1
    missing = REQUIRED_COLUMNS - set(df.columns)
    assert not missing, f"Missing columns: {missing}"
    assert df["rmse"].iloc[0] >= 0


def test_harness_all_four_baselines(synthetic_multioutput, strata_labels):
    """All four baseline types are included and harness returns four rows."""
    X_train, Y_train, X_eval, Y_eval = synthetic_multioutput
    train_labels, eval_labels = strata_labels
    bl_ridge = RidgeBaseline()
    bl_pls = PLSBaseline(n_components=3)
    bl_en = ElasticNetBaseline()
    bl_ps = PerStratumFirstOrderBaseline()
    baselines = [bl_ridge, bl_pls, bl_en, bl_ps]
    df = compare_baselines(
        baselines,
        X_train,
        Y_train,
        X_eval,
        Y_eval,
        fit_kwargs={bl_ps.name: {"strata": train_labels}},
        predict_kwargs={bl_ps.name: {"strata": eval_labels}},
    )
    assert len(df) == 4
    assert set(df["name"]) == {bl_ridge.name, bl_pls.name, bl_en.name, bl_ps.name}
