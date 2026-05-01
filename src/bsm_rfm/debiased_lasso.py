"""Deterministic de-biased LASSO helpers recovered from the archived notebook.

This module implements a minimal, tested subset of the notebook routines used to
form nodewise precision estimates, compute de-biased coefficients, and produce
z-tests from the de-biased scores.

The implementation is intentionally small and numerically explicit so the
behavior can be unit-tested on small deterministic inputs before being
integrated into the manuscript-stage pipeline.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm
from sklearn.linear_model import Lasso


def nodewise_precision(
    X: np.ndarray, *, alpha_node: float = 1e-3, max_iter: int = 3000
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a nodewise precision matrix Theta and residual variances.

    Implements a standard nodewise-LASSO approximation to the precision
    matrix used by de-biased LASSO constructions. For each column j of X
    we regress X[:, j] on the remaining columns using Lasso without an
    intercept, collect the residual variance tau2_j and build the row j of
    Theta as [-coef / tau2_j] on the off-diagonals and 1/tau2_j on the
    diagonal.

    Parameters
    ----------
    X
        Numeric design matrix, shape (n_samples, n_features).
    alpha_node
        Lasso penalty used for nodewise regressions.
    max_iter
        Maximum iterations for the Lasso solver.

    Returns
    -------
    Theta
        Approximate precision matrix, shape (p, p).
    tau2
        Residual variances for each node regression, shape (p,).
    """
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    Theta = np.zeros((p, p), dtype=float)
    tau2 = np.empty(p, dtype=float)

    if p == 0:
        raise ValueError("X must have at least one column")

    for j in range(p):
        if p == 1:
            # Degenerate single-column case: precision is inverse variance.
            tau2_j = float(np.var(X[:, j], ddof=1))
            tau2[j] = tau2_j
            Theta[j, j] = 1.0 / tau2_j if tau2_j > 0.0 else 0.0
            continue

        mask = [k for k in range(p) if k != j]
        X_minus = X[:, mask]
        y = X[:, j]
        model = Lasso(alpha=float(alpha_node), fit_intercept=False, max_iter=int(max_iter))
        model.fit(X_minus, y)
        coef = np.asarray(model.coef_, dtype=float)
        resid = y - X_minus @ coef
        tau2_j = float(np.mean(resid**2))
        tau2[j] = tau2_j
        if tau2_j <= 0.0:
            Theta[j, j] = 0.0
            Theta[j, mask] = 0.0
        else:
            Theta[j, j] = 1.0 / tau2_j
            Theta[j, mask] = -coef / tau2_j

    return Theta, tau2


def debias_coeffs_batched(
    X: np.ndarray, Y: np.ndarray, B_hat: np.ndarray, Theta: np.ndarray, *, batch: int = 512
) -> tuple[np.ndarray, np.ndarray]:
    """Compute de-biased coefficients and per-response residual variances.

    This implements the standard de-biased-LASSO correction:
        B_tilde = B_hat + Theta @ (X^T (Y - X B_hat)) / n

    Parameters
    ----------
    X
        Design matrix, shape (n, p).
    Y
        Response matrix, shape (n, k) or (n,) for single-output.
    B_hat
        LASSO coefficient matrix, shape (p, k) or (p,) for single-output.
    Theta
        Nodewise precision approximation, shape (p, p).
    batch
        Batch size hint for large p (not used by this minimal impl).

    Returns
    -------
    B_tilde
        De-biased coefficient matrix, shape (p, k).
    sigma2
        Residual variances per response, shape (k,).
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    B_hat = np.asarray(B_hat, dtype=float)
    Theta = np.asarray(Theta, dtype=float)

    n = X.shape[0]
    if Y.ndim == 1:
        Y = Y[:, None]
    if B_hat.ndim == 1:
        B_hat = B_hat[:, None]

    if B_hat.shape[0] != X.shape[1]:
        raise ValueError("B_hat must have shape (p, k) where p == X.shape[1]")

    R = Y - X @ B_hat  # (n, k)
    S = X.T @ R  # (p, k)
    correction = Theta @ S / float(n)
    B_tilde = B_hat + correction

    sigma2 = np.var(R, axis=0, ddof=1)
    return B_tilde, sigma2


def ztests_from_debias_scores(
    B_tilde: np.ndarray, Theta: np.ndarray, Sigma_hat: np.ndarray, sigma2: np.ndarray, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute z-statistics and two-sided p-values for de-biased coefficients.

    Variance approximation per coefficient j,k is
        var_{j,k} ≈ (sigma2_k / n) * (Theta[j, :] @ Sigma_hat @ Theta[j, :].T)

    Parameters
    ----------
    B_tilde
        De-biased coefficients, shape (p, k).
    Theta
        Nodewise precision, shape (p, p).
    Sigma_hat
        Empirical feature covariance X^T X / n, shape (p, p).
    sigma2
        Residual variances per response, shape (k,).
    n
        Number of training rows.

    Returns
    -------
    z
        Z-statistics, shape (p, k).
    pvals
        Two-sided p-values, shape (p, k).
    """
    B_tilde = np.asarray(B_tilde, dtype=float)
    Theta = np.asarray(Theta, dtype=float)
    Sigma_hat = np.asarray(Sigma_hat, dtype=float)
    sigma2 = np.asarray(sigma2, dtype=float)

    if B_tilde.ndim == 1:
        B_tilde = B_tilde[:, None]
    p, k = B_tilde.shape

    v = np.empty(p, dtype=float)
    for j in range(p):
        th = Theta[j : j + 1, :].reshape(1, p)
        v_j = float(th @ Sigma_hat @ th.T)
        v[j] = max(v_j, 0.0)

    var = (v[:, None] * sigma2[None, :]) / float(n)
    sd = np.sqrt(var)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = B_tilde / sd
    pvals = 2.0 * (1.0 - norm.cdf(np.abs(z)))
    return z, pvals
