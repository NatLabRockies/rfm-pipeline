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
    # - 'alpha_path': list or array of alpha fractions tested (len > 1)
    # - 'ebic_scores': array of same length as alpha_path
    # - 'alo_kkt': dict with keys 'alo' and 'kkt_margin' each arrays matching alpha_path
    # - 'component_coeffs': a list (length r) of 2D arrays (p x ) for per-component coefficients
    # - 'debiased_coeffs': 2D array (p x q)
    # - 'debiased_pvalues': 2D array (p x q)
    # - 'final_stable_support': list of selected predictor names (non-empty)

    with pytest.raises(NotImplementedError):
        dl.compute_debiased_lasso_artifacts(X, Y)
