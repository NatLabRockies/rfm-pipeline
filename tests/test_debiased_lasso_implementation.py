"""Unit tests for the de-biased LASSO helpers.

These tests exercise shape, numeric stability, and basic statistical
properties on small deterministic synthetic inputs. They are intentionally
lightweight to remain fast in CI while providing meaningful coverage.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Lasso

from bsm_rfm.debiased_lasso import (
    debias_coeffs_batched,
    nodewise_precision,
    ztests_from_debias_scores,
)


def test_nodewise_precision_shapes():
    rng = np.random.RandomState(123)
    X = rng.normal(size=(100, 8))
    Theta, tau2 = nodewise_precision(X, alpha_node=1e-3, max_iter=2000)
    assert Theta.shape == (8, 8)
    assert tau2.shape == (8,)
    assert np.isfinite(Theta).all()


def test_debias_coeffs_batched_shapes_and_difference():
    rng = np.random.RandomState(0)
    n, p, k = 80, 10, 3
    X = rng.normal(size=(n, p))
    trueB = np.zeros((p, k))
    trueB[0, 0] = 1.5
    trueB[2, 1] = -1.0
    Y = X @ trueB + 0.1 * rng.normal(size=(n, k))

    # Fit simple per-column Lasso as a stand-in for the LASSO path fit.
    B_hat = np.zeros((p, k))
    for j in range(k):
        model = Lasso(alpha=0.1, fit_intercept=False, max_iter=5000)
        model.fit(X, Y[:, j])
        B_hat[:, j] = model.coef_

    Theta, tau2 = nodewise_precision(X, alpha_node=1e-3)
    B_tilde, sigma2 = debias_coeffs_batched(X, Y, B_hat, Theta)

    assert B_tilde.shape == B_hat.shape
    assert sigma2.shape == (k,)
    assert not np.allclose(B_tilde, B_hat)


def test_ztests_from_debias_scores_properties():
    rng = np.random.RandomState(1)
    n, p, k = 60, 6, 2
    X = rng.normal(size=(n, p))
    trueB = np.zeros((p, k))
    trueB[1, 0] = 0.8
    Y = X @ trueB + 0.1 * rng.normal(size=(n, k))

    # obtain a simple B_hat
    B_hat = np.zeros((p, k))
    for j in range(k):
        model = Lasso(alpha=0.1, fit_intercept=False, max_iter=5000)
        model.fit(X, Y[:, j])
        B_hat[:, j] = model.coef_

    Theta, tau2 = nodewise_precision(X, alpha_node=1e-3)
    B_tilde, sigma2 = debias_coeffs_batched(X, Y, B_hat, Theta)

    Sigma_hat = (X.T @ X) / float(n)
    z, pvals = ztests_from_debias_scores(B_tilde, Theta, Sigma_hat, sigma2, n)

    assert z.shape == B_hat.shape
    assert pvals.shape == B_hat.shape
    assert ((pvals >= 0.0) & (pvals <= 1.0)).all()
