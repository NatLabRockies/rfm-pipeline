"""Compare compute_debiased_lasso_artifacts output against golden fixtures.

This test ensures deterministic numeric outputs and final support across platforms.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from bsm_rfm import debiased_lasso as dl


def _make_toy(n=200, p=50, q=3, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, p))
    B = np.zeros((p, q), dtype=float)
    B[:5, :] = np.array([[3.0, 1.5, 0.0]] * 5)
    Y = X @ B + rng.normal(scale=0.1, size=(n, q))
    Xdf = pd.DataFrame(X, columns=[f"f{i}" for i in range(p)])
    Ydf = pd.DataFrame(Y, columns=[f"y{i}" for i in range(q)])
    return Xdf, Ydf


def test_debiased_lasso_matches_golden():
    X, Y = _make_toy()
    config = {"alpha_fracs": [0.5, 0.25, 0.1, 0.05], "ebic_gamma": 0.5}

    artifacts = dl.compute_debiased_lasso_artifacts(X, Y, config=config)

    gz = np.load("tests/golden/debiased_lasso_golden.npz")

    # Numeric comparisons
    np.testing.assert_allclose(artifacts["ebic_scores"], gz["ebic_scores"], rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(
        artifacts["debiased_coeffs"], gz["debiased_coeffs"], rtol=1e-8, atol=1e-12
    )
    np.testing.assert_allclose(
        artifacts["debiased_pvalues"], gz["debiased_pvalues"], rtol=1e-8, atol=1e-12
    )

    # alpha path
    np.testing.assert_allclose(
        np.array(artifacts["alpha_path"]), gz["alpha_path"], rtol=1e-12, atol=0
    )

    # final support list equality
    with open("tests/golden/debiased_lasso_golden.json") as f:
        golden_support = json.load(f)["final_stable_support"]

    assert artifacts["final_stable_support"] == golden_support
