"""Structural measurements extracted from any (X, Y) dataset.

These measurements form the feature vector used by the measurement-based
meta-model that predicts pipeline performance. They are computable from any
dataset (synthetic or real) without running the pipeline, enabling
synthetic-to-real transfer for nRMSE prediction.

All measurements are scalar summaries chosen to capture structural properties
that affect pipeline performance:

* output-shape: how much variance the null mean predictor leaves on the table
  (std/range), whether output tails are heavier or thinner than Gaussian
  (kurtosis), and how compactly variance is distributed across outputs
  (PCA spectrum decay).
* input-structure: collinearity / redundancy in X (input correlation spectrum)
  and effective input dimensionality.
* signal-concentration: how marginal correlation between X_j and outputs is
  distributed (Pareto-like => few dominant inputs; uniform => spread). The
  pipeline can exploit concentrated signal more aggressively, so this is a
  primary driver of γ.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

__all__ = ["DGPMeasurements", "measure_dataset"]


@dataclass(frozen=True)
class DGPMeasurements:
    """Scalar structural measurements of an (X, Y) dataset."""

    n_rows: int
    n_inputs: int
    n_outputs: int

    output_std_over_range_mean: float
    output_std_over_range_std: float
    output_kurtosis_mean: float
    output_kurtosis_median: float
    output_skewness_abs_mean: float

    output_spectrum_top1_share: float
    output_spectrum_top5_share: float
    output_spectrum_effective_rank: float
    output_spectrum_decay_exponent: float

    input_correlation_offdiag_mean_abs: float
    input_correlation_effective_rank: float

    marginal_xy_corr_mean_abs: float
    marginal_xy_corr_top1pct_mean_abs: float
    marginal_xy_corr_pareto_alpha: float

    def to_dict(self) -> dict[str, float | int]:
        """Return measurements as a flat dict for serialization or DataFrame rows."""
        return asdict(self)


def measure_dataset(
    X: np.ndarray | pd.DataFrame,
    Y: np.ndarray | pd.DataFrame,
    *,
    max_outputs_for_pca: int = 2000,
    max_inputs_for_corr: int = 500,
    max_rows_for_corr: int = 20000,
    random_state: int = 0,
) -> DGPMeasurements:
    """Compute structural measurements from (X, Y).

    Parameters
    ----------
    X
        Inputs, shape ``(n_rows, n_inputs)``. ``sample_id`` column is dropped if present.
    Y
        Outputs, shape ``(n_rows, n_outputs)``. ``sample_id`` column is dropped if present.
    max_outputs_for_pca
        Cap output PCA to a random subset for tractability on wide-output datasets.
    max_inputs_for_corr
        Cap input correlation spectrum to a random subset on wide-input datasets.
    max_rows_for_corr
        Cap rows used for correlation computations.
    random_state
        Seed for subsampling.
    """
    X = _as_matrix(X)
    Y = _as_matrix(Y)
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"X rows={X.shape[0]} != Y rows={Y.shape[0]}")

    rng = np.random.default_rng(random_state)
    n_rows, n_inputs = X.shape
    _, n_outputs = Y.shape

    # Subsample for tractability.
    row_idx = (
        rng.choice(n_rows, size=max_rows_for_corr, replace=False)
        if n_rows > max_rows_for_corr
        else np.arange(n_rows)
    )
    out_idx = (
        rng.choice(n_outputs, size=max_outputs_for_pca, replace=False)
        if n_outputs > max_outputs_for_pca
        else np.arange(n_outputs)
    )
    in_idx = (
        rng.choice(n_inputs, size=max_inputs_for_corr, replace=False)
        if n_inputs > max_inputs_for_corr
        else np.arange(n_inputs)
    )

    Xs = X[np.ix_(row_idx, in_idx)]
    Ys = Y[np.ix_(row_idx, out_idx)]
    Yfull = Y[:, out_idx]

    # Output shape statistics (use full rows for accurate moments).
    std = Yfull.std(axis=0, ddof=1)
    rng_per_out = np.nanmax(Yfull, axis=0) - np.nanmin(Yfull, axis=0)
    mask = rng_per_out > 1e-12
    sr = std[mask] / rng_per_out[mask]

    kurt = _excess_kurtosis(Yfull[:, mask])
    skew = _skewness(Yfull[:, mask])

    # Output spectrum via centered Y SVD.
    Yc = Ys - Ys.mean(axis=0, keepdims=True)
    # Economy SVD; for n_rows >> n_outputs use rows x outputs orientation already.
    try:
        sv = np.linalg.svd(Yc, full_matrices=False, compute_uv=False)
    except np.linalg.LinAlgError:
        sv = np.array([float("nan")])
    ev = sv**2
    ev_total = ev.sum()
    top1 = float(ev[0] / ev_total) if ev_total > 0 else float("nan")
    top5 = float(ev[:5].sum() / ev_total) if ev_total > 0 else float("nan")
    eff_rank_out = _effective_rank(ev)
    decay = _spectrum_decay_exponent(ev)

    # Input correlation spectrum.
    Xc = Xs - Xs.mean(axis=0, keepdims=True)
    Xstd = Xc.std(axis=0, ddof=1)
    Xstd_safe = np.where(Xstd > 1e-12, Xstd, 1.0)
    Xn = Xc / Xstd_safe
    corrX = (Xn.T @ Xn) / max(Xn.shape[0] - 1, 1)
    n_in = corrX.shape[0]
    offdiag_mask = ~np.eye(n_in, dtype=bool)
    offdiag_mean_abs = float(np.mean(np.abs(corrX[offdiag_mask]))) if n_in > 1 else float("nan")
    eig_in = np.linalg.eigvalsh(corrX)
    eig_in = np.clip(eig_in, 0.0, None)
    eff_rank_in = _effective_rank(eig_in[::-1])

    # Marginal X_j vs output correlations.
    # Use mean-output Yhat to avoid n_inputs x n_outputs blowup.
    Ymean_centered = Ys.mean(axis=1)
    Ymean_centered = Ymean_centered - Ymean_centered.mean()
    Ymean_std = Ymean_centered.std(ddof=1) or 1.0
    Ymean_norm = Ymean_centered / Ymean_std
    xy_corrs = (Xn.T @ Ymean_norm) / max(Xn.shape[0] - 1, 1)
    xy_abs = np.abs(xy_corrs)
    xy_abs_sorted = np.sort(xy_abs)[::-1]
    top_n = max(1, int(np.ceil(0.01 * len(xy_abs))))
    pareto_alpha = _pareto_decay_exponent(xy_abs_sorted)

    return DGPMeasurements(
        n_rows=int(n_rows),
        n_inputs=int(n_inputs),
        n_outputs=int(n_outputs),
        output_std_over_range_mean=float(sr.mean()),
        output_std_over_range_std=float(sr.std(ddof=1)) if sr.size > 1 else 0.0,
        output_kurtosis_mean=float(np.mean(kurt)),
        output_kurtosis_median=float(np.median(kurt)),
        output_skewness_abs_mean=float(np.mean(np.abs(skew))),
        output_spectrum_top1_share=top1,
        output_spectrum_top5_share=top5,
        output_spectrum_effective_rank=eff_rank_out,
        output_spectrum_decay_exponent=decay,
        input_correlation_offdiag_mean_abs=offdiag_mean_abs,
        input_correlation_effective_rank=eff_rank_in,
        marginal_xy_corr_mean_abs=float(np.mean(xy_abs)),
        marginal_xy_corr_top1pct_mean_abs=float(np.mean(xy_abs_sorted[:top_n])),
        marginal_xy_corr_pareto_alpha=pareto_alpha,
    )


def _as_matrix(arr: np.ndarray | pd.DataFrame) -> np.ndarray:
    if isinstance(arr, pd.DataFrame):
        if "sample_id" in arr.columns:
            arr = arr.drop(columns=["sample_id"])
        return arr.to_numpy(dtype=np.float64)
    return np.asarray(arr, dtype=np.float64)


def _excess_kurtosis(Y: np.ndarray) -> np.ndarray:
    mu = Y.mean(axis=0)
    var = Y.var(axis=0, ddof=1)
    var_safe = np.where(var > 1e-24, var, 1.0)
    m4 = np.mean((Y - mu) ** 4, axis=0)
    return m4 / (var_safe**2) - 3.0


def _skewness(Y: np.ndarray) -> np.ndarray:
    mu = Y.mean(axis=0)
    var = Y.var(axis=0, ddof=1)
    sd = np.sqrt(np.where(var > 1e-24, var, 1.0))
    m3 = np.mean((Y - mu) ** 3, axis=0)
    return m3 / (sd**3)


def _effective_rank(eigenvalues: np.ndarray) -> float:
    """Roy & Vetterli effective rank: exp(H) of normalized spectrum entropy."""
    ev = np.asarray(eigenvalues, dtype=np.float64)
    ev = ev[np.isfinite(ev) & (ev > 1e-18)]
    if ev.size == 0:
        return float("nan")
    p = ev / ev.sum()
    p = p[p > 0]
    return float(np.exp(-np.sum(p * np.log(p))))


def _spectrum_decay_exponent(eigenvalues: np.ndarray) -> float:
    """Fit log(λ_i) = a - α·log(i) over top eigenvalues; return α."""
    ev = np.asarray(eigenvalues, dtype=np.float64)
    ev = ev[ev > 0]
    n = min(50, ev.size)
    if n < 3:
        return float("nan")
    y = np.log(ev[:n])
    x = np.log(np.arange(1, n + 1))
    slope, _ = np.polyfit(x, y, 1)
    return float(-slope)


def _pareto_decay_exponent(sorted_abs_corrs: np.ndarray) -> float:
    """Fit log(|corr_i|) = a - α·log(i) over top entries; return α."""
    v = sorted_abs_corrs[sorted_abs_corrs > 1e-12]
    n = min(100, v.size)
    if n < 3:
        return float("nan")
    y = np.log(v[:n])
    x = np.log(np.arange(1, n + 1))
    slope, _ = np.polyfit(x, y, 1)
    return float(-slope)
