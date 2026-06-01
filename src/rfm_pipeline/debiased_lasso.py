"""De-biased LASSO helper utilities and pipeline.

This module implements a deterministic, test-focused de-biased-LASSO pipeline
sufficient for unit tests and artifact-schema contracts. It is not intended to
be a drop-in reproduction of the original notebook pipeline; the recovered
notebook remains the authoritative source when performing a final validation.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, MultiTaskLasso
from sklearn.preprocessing import StandardScaler


def _numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.shape[0] == 0 or frame.shape[1] == 0:
        raise ValueError("Input frames must be non-empty")
    return frame.apply(pd.to_numeric, errors="raise")


def _normal_cdf(x: np.ndarray) -> np.ndarray:
    # Use error function to compute standard normal CDF
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def _safe_log(x: float) -> float:
    return math.log(max(x, 1e-12))


def _ebic_score(rss: float, n: int, p: int, s: int, gamma: float = 0.5) -> float:
    # Approximate EBIC using combinatorial term computed via log-gamma for stability
    if rss <= 0:
        rss = 1e-8
    base = n * _safe_log(rss / float(n))
    penalty = s * math.log(n)
    if s <= 0:
        comb_term = 0.0
    else:
        # log choose(p, s)
        comb_term = math.lgamma(p + 1) - math.lgamma(s + 1) - math.lgamma(p - s + 1)
    return base + penalty + 2.0 * gamma * comb_term


def compute_debiased_lasso_artifacts(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    *,
    config: dict | None = None,
) -> dict[str, Any]:
    """Compute a lightweight de-biased-LASSO artifact bundle for testing.

    Returns a dict with keys:
      - 'alpha_path': list[float]
      - 'ebic_scores': np.ndarray
      - 'alo_kkt': dict with keys 'alo' and 'kkt_margin'
      - 'component_coeffs': list of per-component coefficient arrays (p x )
      - 'debiased_coeffs': np.ndarray (p x q)
      - 'debiased_pvalues': np.ndarray (p x q)
      - 'final_stable_support': list[str]

    The implementation below is intentionally conservative and deterministic.
    """
    if config is None:
        config = {}

    Xf = _numeric_frame(X)
    Yf = _numeric_frame(Y)

    n, p = Xf.shape
    _, q = Yf.shape

    X_cols = list(Xf.columns)

    # Standardize X and center Y
    scaler = StandardScaler()
    X_std = scaler.fit_transform(Xf.values)
    Y_center = Yf.values - np.mean(Yf.values, axis=0, keepdims=True)

    # default alpha fraction path (matches recovered notebook grid)
    alpha_fracs: list[float] = config.get("alpha_fracs", [0.75, 0.5, 0.25, 0.10, 0.05, 0.02, 0.01])
    gamma = float(config.get("ebic_gamma", 0.5))

    # compute a simple alpha_max (correlation-based)
    corr = np.abs(X_std.T @ Y_center)
    alpha_max = float(np.max(corr)) / max(1.0, float(n))

    ebic_scores: list[float] = []
    alo_vals: list[float] = []
    kkt_margins: list[float] = []
    coef_path: list[np.ndarray] = []

    for frac in alpha_fracs:
        alpha = float(frac) * (alpha_max + 1e-12)
        # fit multi-task LASSO on standardized X
        model = MultiTaskLasso(alpha=alpha, fit_intercept=False, max_iter=5000)
        model.fit(X_std, Y_center)
        # model.coef_ shape: (n_tasks, n_features)
        coefs = model.coef_.T  # shape (p, q)

        # residuals and rss
        preds = X_std @ coefs
        resid = Y_center - preds
        rss = float(np.sum(resid**2))

        # support size
        support_mask = np.any(np.abs(coefs) > 1e-8, axis=1)
        s = int(np.sum(support_mask))

        ebic = _ebic_score(rss, n, p, s, gamma=gamma)
        ebic_scores.append(float(ebic))

        # ALO: use a tiny proxy (rss normalized)
        alo_vals.append(float(rss / float(max(1, n * (1 + s)))))

        # KKT margin (proxy): alpha - max correlation of X^T residuals / n
        max_corr = float(np.max(np.abs(X_std.T @ resid))) / max(1.0, float(n))
        kkt_margins.append(float(max(0.0, alpha - max_corr)))

        coef_path.append(coefs)

    ebic_arr = np.array(ebic_scores)

    # choose best alpha by EBIC
    best_idx = int(np.nanargmin(ebic_arr))
    best_coefs = coef_path[best_idx]  # p x q

    # Compute final stable support
    support_mask_best = np.any(np.abs(best_coefs) > 1e-8, axis=1)
    if support_mask_best.sum() == 0:
        # fallback: choose top-k by L1 across outputs (k= max(1, int(p*0.05)))
        norms = np.sum(np.abs(best_coefs), axis=1)
        k = max(1, int(max(1, round(float(p) * 0.05))))
        top_idx = np.argsort(-norms)[:k]
        support_mask_best[top_idx] = True

    final_support = [X_cols[i] for i in np.where(support_mask_best)[0]]

    # Debias: OLS refit on selected support per-output
    debiased_coeffs = np.zeros((p, q), dtype=float)
    debiased_pvalues = np.ones((p, q), dtype=float)

    sel_idx = np.where(support_mask_best)[0]
    if sel_idx.size > 0:
        X_sel = Xf.values[:, sel_idx]
        # add intercept
        X_with_const = np.column_stack([np.ones(n), X_sel])
        xtx = X_with_const.T @ X_with_const
        try:
            xtx_inv = np.linalg.inv(xtx)
        except np.linalg.LinAlgError:
            xtx_inv = np.linalg.pinv(xtx)

        for j in range(q):
            yj = Yf.values[:, j]
            beta = np.linalg.lstsq(X_with_const, yj, rcond=None)[0]
            # store coefficients (skip intercept)
            debiased_coeffs[sel_idx, j] = beta[1:]

            residuals = yj - X_with_const @ beta
            dof = max(1, n - X_with_const.shape[1])
            sigma2 = float(np.sum(residuals**2) / dof)

            cov_beta = sigma2 * xtx_inv
            se = np.sqrt(np.maximum(0.0, np.diag(cov_beta)))
            # avoid zero se
            se[se == 0.0] = 1e-12

            t_stats = beta[1:] / se[1:]
            # normal approx p-values
            pvals = 2.0 * (1.0 - _normal_cdf(np.abs(t_stats)))
            debiased_pvalues[sel_idx, j] = pvals
    else:
        # No selected features -> keep zeros and pvals as ones
        pass

    artifacts: dict[str, Any] = {
        "alpha_path": list(alpha_fracs),
        "ebic_scores": np.array(ebic_scores),
        "alo_kkt": {"alo": np.array(alo_vals), "kkt_margin": np.array(kkt_margins)},
        "component_coeffs": [c for c in coef_path],
        "debiased_coeffs": debiased_coeffs,
        "debiased_pvalues": debiased_pvalues,
        "final_stable_support": final_support,
    }

    return artifacts


# ---------------------------------------------------------------------------
# Nodewise LASSO precision estimation, debias correction, and z-tests
# ---------------------------------------------------------------------------


def nodewise_precision(X, alpha_node: float = 0.1, max_iter: int = 1000):
    """Estimate approximate precision rows via nodewise Lasso.

    Returns
    -------
    Theta : np.ndarray
        Approximate precision rows (p x p) where row j stores theta_j.
    tau2 : np.ndarray
        Residual variances from nodewise regressions (p,).
    """
    Xf = pd.DataFrame(X).apply(pd.to_numeric, errors="raise")
    n, p = Xf.shape
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(Xf.values)

    Theta = np.zeros((p, p), dtype=float)
    tau2 = np.zeros(p, dtype=float)

    for j in range(p):
        mask = [k for k in range(p) if k != j]
        X_other = Xs[:, mask]
        y = Xs[:, j]

        model = Lasso(alpha=float(alpha_node), fit_intercept=False, max_iter=int(max_iter))
        model.fit(X_other, y)
        gamma = model.coef_

        resid = y - X_other @ gamma
        tau2_j = float(np.mean(resid**2))
        tau2[j] = tau2_j

        inv_tau = 1.0 / tau2_j if tau2_j > 0 else 0.0
        Theta[j, j] = inv_tau
        Theta[j, mask] = -gamma * inv_tau

    return Theta, tau2


def debias_coeffs_batched(X, Y, B_hat, Theta):
    """Compute debiased coefficients and per-response variance estimates.

    Parameters
    ----------
    X, Y
        Data frames or array-likes with shapes (n, p) and (n, q).
    B_hat : array-like
        Penalized coefficient estimate with shape (p, q).
    Theta : np.ndarray
        Approximate precision matrix rows (p x p).

    Returns
    -------
    B_tilde : np.ndarray
        Debiased coefficient estimates (p, q).
    sigma2 : np.ndarray
        Per-response residual variance estimates (q,).
    """
    Xf = pd.DataFrame(X).apply(pd.to_numeric, errors="raise")
    Yf = pd.DataFrame(Y).apply(pd.to_numeric, errors="raise")

    n = Xf.shape[0]
    U = Yf.values - Xf.values @ B_hat  # n x q

    correction = Theta @ (Xf.values.T @ U) / float(max(1, n))
    B_tilde = B_hat + correction

    sigma2 = np.sum(U**2, axis=0) / float(max(1, n))

    return B_tilde, sigma2


def ztests_from_debias_scores(B_tilde, Theta, Sigma_hat, sigma2, n):
    """Compute z-statistics and two-sided p-values for debiased coefficients.

    Sigma_hat is the empirical feature covariance scaled by 1/n (p x p).
    """
    p, q = B_tilde.shape
    z = np.zeros((p, q), dtype=float)
    pvals = np.ones((p, q), dtype=float)

    for j in range(p):
        v = float(Theta[j, :].dot(Sigma_hat).dot(Theta[j, :].T))
        for k in range(q):
            var = float(sigma2[k]) * v / float(max(1, n))
            sd = math.sqrt(var) if var > 0 else 1e-12
            zval = float(B_tilde[j, k]) / sd
            z[j, k] = zval
            pvals[j, k] = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(zval) / math.sqrt(2.0))))

    return z, pvals
