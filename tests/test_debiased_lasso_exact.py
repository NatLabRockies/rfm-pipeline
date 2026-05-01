import importlib

try:
    dl = importlib.import_module("bsm_rfm.debiased_lasso")
except ModuleNotFoundError:
    dl = None
import numpy as np
import pandas as pd
import pytest

# Test-first contract expectations for deterministic de-biased LASSO artifacts


def make_toy_data(n=200, p=20, q=3, random_state=123):
    rs = np.random.RandomState(random_state)
    X = pd.DataFrame(rs.normal(size=(n, p)), columns=[f"x{i}" for i in range(p)])
    # Create low-rank Y with noise
    W = rs.normal(size=(p, q))
    Y = pd.DataFrame(
        X.values @ W + rs.normal(scale=0.1, size=(n, q)),
        columns=[f"y{j}" for j in range(q)],
    )
    return X, Y


def test_debiased_lasso_artifact_keys_and_shapes():
    """Contract test: the de-biased LASSO artifact generator must return a dict with
    specific keys and shapes. This test is expected to fail until the exact API is implemented.
    """
    X, Y = make_toy_data()

    # The public contract: compute_debiased_lasso_artifacts returns a dict containing:

    if dl is None:
        pytest.fail("debiased_lasso module missing; implement compute_debiased_lasso_artifacts")

    # The public contract: compute_debiased_lasso_artifacts returns a dict containing:
    res = dl.compute_debiased_lasso_artifacts(X, Y)
    assert isinstance(res, dict)
    assert "alpha_path" in res and len(res["alpha_path"]) > 1
    assert "ebic_scores" in res and len(res["ebic_scores"]) == len(res["alpha_path"])
    assert "alo_kkt" in res and set(["alo", "kkt_margin"]).issubset(set(res["alo_kkt"].keys()))
    assert "component_coeffs" in res and isinstance(res["component_coeffs"], list)
    assert "debiased_coeffs" in res and res["debiased_coeffs"].shape == (X.shape[1], Y.shape[1])
    assert "debiased_pvalues" in res and res["debiased_pvalues"].shape == (X.shape[1], Y.shape[1])
    assert "final_stable_support" in res and isinstance(res["final_stable_support"], list)
    assert len(res["final_stable_support"]) > 0
