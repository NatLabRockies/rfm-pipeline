"""Additional deterministic tests for de-biased LASSO helpers.

These tests are intentionally small and deterministic to catch platform-sensitive
behavior (empty support, nondeterministic alpha selection) early in CI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bsm_rfm import debiased_lasso as dl


def _make_toy_data(n=200, p=50, q=3, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, p))
    B = np.zeros((p, q), dtype=float)
    # strong shared signal on first 5 features
    B[:5, :] = np.array([[3.0, 1.5, 0.0]] * 5)
    Y = X @ B + rng.normal(scale=0.1, size=(n, q))
    Xdf = pd.DataFrame(X, columns=[f"f{i}" for i in range(p)])
    Ydf = pd.DataFrame(Y, columns=[f"y{i}" for i in range(q)])
    return Xdf, Ydf


def test_compute_debiased_lasso_support_nonempty_and_ebic():
    X, Y = _make_toy_data()
    config = {"alpha_fracs": [0.5, 0.25, 0.1, 0.05], "ebic_gamma": 0.5}
    artifacts = dl.compute_debiased_lasso_artifacts(X, Y, config=config)

    assert "alpha_path" in artifacts
    assert "ebic_scores" in artifacts
    assert "alo_kkt" in artifacts
    assert "final_stable_support" in artifacts

    alpha_path = artifacts["alpha_path"]
    ebic = artifacts["ebic_scores"]
    alo = artifacts["alo_kkt"]["alo"]
    kkt = artifacts["alo_kkt"]["kkt_margin"]

    assert len(alpha_path) == len(ebic) == len(alo) == len(kkt)
    assert ebic.dtype == float or isinstance(ebic, np.ndarray)

    # final stable support should be non-empty on this strong-signal toy example
    final_support = artifacts["final_stable_support"]
    assert isinstance(final_support, list)
    assert len(final_support) >= 1


def test_nodewise_debias_ztests_basic_shapes_and_pvals():
    X, Y = _make_toy_data()
    n, p = X.shape
    q = Y.shape[1]

    # Use a simple B_hat (zeros) to exercise correction path
    B_hat = np.zeros((p, q), dtype=float)

    Theta, tau2 = dl.nodewise_precision(X, alpha_node=0.05, max_iter=500)
    assert Theta.shape == (p, p)
    assert tau2.shape == (p,)

    B_tilde, sigma2 = dl.debias_coeffs_batched(X, Y, B_hat, Theta)
    assert B_tilde.shape == (p, q)
    assert sigma2.shape == (q,)

    Sigma_hat = (pd.DataFrame(X).cov().to_numpy()) / float(max(1, n))
    z, pvals = dl.ztests_from_debias_scores(B_tilde, Theta, Sigma_hat, sigma2, n)
    assert z.shape == (p, q)
    assert pvals.shape == (p, q)
    # p-values must be in [0, 1] and finite
    assert (pvals >= 0).all() and (pvals <= 1).all()
    assert np.isfinite(pvals).all()
