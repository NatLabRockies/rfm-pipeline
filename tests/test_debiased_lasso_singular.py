"""Ensure de-biased LASSO handles singular / collinear feature matrices.

This test constructs a dataset with duplicated columns (singular X'TX) and
verifies the compute_debiased_lasso_artifacts pipeline and debias helpers run
without error and produce well-shaped numeric outputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rfm_pipeline import debiased_lasso as dl


def test_debiased_lasso_handles_singular_X() -> None:
    rng = np.random.RandomState(42)
    n = 100
    p = 20
    q = 2

    X = rng.normal(size=(n, p))
    # create collinearity: duplicate column 0 into column 1
    X[:, 1] = X[:, 0]

    B = np.zeros((p, q), dtype=float)
    B[:3, :] = np.array([[2.0, 1.0]] * 3)

    Y = X @ B + rng.normal(scale=0.1, size=(n, q))

    Xdf = pd.DataFrame(X, columns=[f"f{i}" for i in range(p)])
    Ydf = pd.DataFrame(Y, columns=[f"y{i}" for i in range(q)])

    config = {"alpha_fracs": [0.5, 0.25, 0.1], "ebic_gamma": 0.5}

    artifacts = dl.compute_debiased_lasso_artifacts(Xdf, Ydf, config=config)

    assert isinstance(artifacts, dict)
    assert "final_stable_support" in artifacts
    assert len(artifacts["final_stable_support"]) >= 1

    Theta, tau2 = dl.nodewise_precision(Xdf, alpha_node=0.05, max_iter=500)
    assert Theta.shape == (p, p)
    assert tau2.shape == (p,)

    B_hat = np.zeros((p, q), dtype=float)
    B_tilde, sigma2 = dl.debias_coeffs_batched(Xdf, Ydf, B_hat, Theta)
    assert B_tilde.shape == (p, q)
    assert sigma2.shape == (q,)

    Sigma_hat = (pd.DataFrame(Xdf).cov().to_numpy()) / float(max(1, n))
    z, pvals = dl.ztests_from_debias_scores(B_tilde, Theta, Sigma_hat, sigma2, n)
    assert z.shape == (p, q)
    assert pvals.shape == (p, q)
    assert np.isfinite(pvals).all()
    assert ((pvals >= 0) & (pvals <= 1)).all()
