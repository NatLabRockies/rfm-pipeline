"""Deterministic regression-style validation for de-biased LASSO artifacts.

This test checks reproducible shapes and basic numerical sanity across platforms.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rfm_pipeline import debiased_lasso as dl


def _make_toy_data(n=200, p=50, q=3, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, p))
    B = np.zeros((p, q), dtype=float)
    B[:5, :] = np.array([[3.0, 1.5, 0.0]] * 5)
    Y = X @ B + rng.normal(scale=0.1, size=(n, q))
    Xdf = pd.DataFrame(X, columns=[f"f{i}" for i in range(p)])
    Ydf = pd.DataFrame(Y, columns=[f"y{i}" for i in range(q)])
    return Xdf, Ydf


def test_debiased_lasso_regression_basic_sanity():
    X, Y = _make_toy_data()
    config = {"alpha_fracs": [0.5, 0.25, 0.1, 0.05], "ebic_gamma": 0.5}

    artifacts = dl.compute_debiased_lasso_artifacts(X, Y, config=config)

    # Keys and shapes
    assert "alpha_path" in artifacts
    assert "ebic_scores" in artifacts
    assert "final_stable_support" in artifacts
    assert "debiased_pvalues" in artifacts

    assert len(artifacts["alpha_path"]) == len(artifacts["ebic_scores"]) == 4
    assert np.isfinite(artifacts["ebic_scores"]).all()

    # Final support size should be stable on strong shared signal
    assert isinstance(artifacts["final_stable_support"], list)
    assert len(artifacts["final_stable_support"]) == 5

    # Debiased p-values shape and sanity
    pvals = np.asarray(artifacts["debiased_pvalues"])
    assert pvals.shape == (X.shape[1], Y.shape[1])
    assert np.isfinite(pvals).all()
    assert ((pvals >= 0.0) & (pvals <= 1.0)).all()
