import numpy as np
import pandas as pd
from sklearn.linear_model import MultiTaskLasso
from sklearn.preprocessing import StandardScaler

from rfm_pipeline.debiased_lasso import (
    debias_coeffs_batched,
    nodewise_precision,
    ztests_from_debias_scores,
)


def _make_toy(n=200, p=20, q=3, random_state=123):
    rs = np.random.RandomState(random_state)
    X = pd.DataFrame(rs.normal(size=(n, p)), columns=[f"x{i}" for i in range(p)])
    W = rs.normal(size=(p, q)) * 0.5
    Y = pd.DataFrame(
        X.values @ W + rs.normal(scale=0.1, size=(n, q)),
        columns=[f"y{j}" for j in range(q)],
    )
    return X, Y


def test_nodewise_precision_shapes_and_relations():
    X, _ = _make_toy()
    Theta, tau2 = nodewise_precision(X, alpha_node=0.1)
    p = X.shape[1]
    assert Theta.shape == (p, p)
    assert tau2.shape == (p,)
    assert np.all(np.isfinite(Theta))
    assert np.all(tau2 >= 0)
    # diagonal relation Theta[j,j] * tau2[j] ~= 1
    assert np.allclose(np.diag(Theta) * tau2, np.ones(p), atol=1e-6)


def test_debias_and_ztests_shapes():
    X, Y = _make_toy()
    p, q = X.shape[1], Y.shape[1]

    # quick B_hat via MultiTaskLasso on standardized X
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    Yc = Y.values - Y.values.mean(axis=0, keepdims=True)

    model = MultiTaskLasso(alpha=0.1, fit_intercept=False, max_iter=5000)
    model.fit(Xs, Yc)
    B_hat = model.coef_.T

    Theta, tau2 = nodewise_precision(pd.DataFrame(Xs, columns=X.columns), alpha_node=0.1)

    B_tilde, sigma2 = debias_coeffs_batched(
        pd.DataFrame(Xs, columns=X.columns),
        pd.DataFrame(Yc),
        B_hat,
        Theta,
    )
    assert B_tilde.shape == (p, q)
    assert sigma2.shape == (q,)

    Sigma_hat = (Xs.T @ Xs) / float(Xs.shape[0])
    z, pvals = ztests_from_debias_scores(B_tilde, Theta, Sigma_hat, sigma2, Xs.shape[0])
    assert z.shape == (p, q)
    assert pvals.shape == (p, q)
    assert np.all((pvals >= 0) & (pvals <= 1))
