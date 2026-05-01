"""Evaluation metrics for reduced-form model screening and final fits."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def macro_nrmse_with_ref(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    Y_ref: np.ndarray,
    *,
    min_range: float = 1e-6,
) -> tuple[float, dict[str, int], np.ndarray]:
    """Compute macro nRMSE using a fixed external normalization reference.

    Parameters
    ----------
    Y_true
        Observed outputs for the evaluation set, shape ``(n_rows, n_outputs)``.
    Y_pred
        Predicted outputs for the evaluation set, shape matching ``Y_true``.
    Y_ref
        Reference matrix whose columnwise ranges define the normalization denominator.
    min_range
        Minimum allowable range for an output to be included in the macro average.

    Returns
    -------
    tuple[float, dict[str, int], numpy.ndarray]
        Macro nRMSE, bookkeeping dictionary with the number of usable outputs, and the
        per-output RMSE vector before normalization.
    """
    Y_true = np.asarray(Y_true, dtype=np.float64)
    Y_pred = np.asarray(Y_pred, dtype=np.float64)
    Y_ref = np.asarray(Y_ref, dtype=np.float64)

    rmse = np.sqrt(np.mean((Y_true - Y_pred) ** 2, axis=0))
    ref_range = np.nanmax(Y_ref, axis=0) - np.nanmin(Y_ref, axis=0)
    mask = ref_range >= min_range
    used = int(np.sum(mask))
    if used == 0:
        return float("nan"), {"k_used": 0, "k_total": int(Y_true.shape[1])}, rmse
    nrmse_per_output = rmse[mask] / ref_range[mask]
    return (
        float(np.mean(nrmse_per_output)),
        {"k_used": used, "k_total": int(Y_true.shape[1])},
        rmse,
    )


def bootstrap_macro_nrmse_ci(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    Y_ref: np.ndarray,
    *,
    min_range: float = 1e-6,
    n_boot: int = 1000,
    alpha: float = 0.05,
    random_state: int = 123,
    sample_size: int | None = None,
) -> dict[str, Any]:
    """Estimate a percentile bootstrap interval for macro nRMSE.

    Parameters
    ----------
    Y_true
        Observed outputs for the evaluation set.
    Y_pred
        Predicted outputs for the evaluation set.
    Y_ref
        Reference matrix used to define the fixed normalization ranges.
    min_range
        Minimum allowable reference range for inclusion in the macro average.
    n_boot
        Number of row-bootstrap replicates.
    alpha
        Two-sided error level for the percentile interval.
    random_state
        Seed for bootstrap resampling.
    sample_size
        Optional bootstrap draw size. Defaults to the number of evaluation rows.

    Returns
    -------
    dict[str, Any]
        Summary payload containing the point estimate, percentile confidence interval,
        bootstrap settings, and bookkeeping fields.

    Raises
    ------
    ValueError
        Raised when the evaluation set is empty, the bootstrap draw size is invalid,
        or all bootstrap replicates are non-finite.
    """
    Y_true = np.asarray(Y_true, dtype=np.float64)
    Y_pred = np.asarray(Y_pred, dtype=np.float64)
    Y_ref = np.asarray(Y_ref, dtype=np.float64)

    point_estimate, info, rmse = macro_nrmse_with_ref(
        Y_true,
        Y_pred,
        Y_ref,
        min_range=min_range,
    )
    n_rows = Y_true.shape[0]
    if n_rows == 0:
        raise ValueError("Cannot bootstrap an empty evaluation set.")
    draw_size = int(sample_size or n_rows)
    if draw_size <= 0:
        raise ValueError("sample_size must be positive.")

    rng = np.random.RandomState(random_state)
    boot = np.empty(int(n_boot), dtype=np.float64)
    for boot_index in range(int(n_boot)):
        idx = rng.randint(0, n_rows, size=draw_size)
        boot[boot_index], _, _ = macro_nrmse_with_ref(
            Y_true[idx],
            Y_pred[idx],
            Y_ref,
            min_range=min_range,
        )

    valid = np.isfinite(boot)
    if not np.any(valid):
        raise ValueError("All bootstrap replicates were non-finite.")

    bootstrap_std = 0.0
    if np.sum(valid) > 1:
        bootstrap_std = float(np.nanstd(boot[valid], ddof=1))

    return {
        "point_estimate": float(point_estimate),
        "ci_lower": float(np.nanquantile(boot[valid], alpha / 2)),
        "ci_upper": float(np.nanquantile(boot[valid], 1 - alpha / 2)),
        "bootstrap_mean": float(np.nanmean(boot[valid])),
        "bootstrap_std": bootstrap_std,
        "n_boot": int(n_boot),
        "bootstrap_sample_size": int(draw_size),
        "alpha": float(alpha),
        "ci_type": "percentile",
        "normalization_reference": "fixed Y_ref range",
        "k_used": int(info["k_used"]),
        "k_total": int(info["k_total"]),
        "rmse_macro": float(np.nanmean(rmse)) if rmse.size else float("nan"),
    }


def make_null_mean_prediction(Y_train: np.ndarray, n_rows: int) -> np.ndarray:
    """Create a mean-only baseline prediction matrix.

    Parameters
    ----------
    Y_train
        Training response matrix used to estimate the per-output means.
    n_rows
        Number of prediction rows to generate.

    Returns
    -------
    numpy.ndarray
        Repeated row vector of training means with shape ``(n_rows, n_outputs)``.
    """
    Y_train = np.asarray(Y_train, dtype=np.float64)
    mu = np.nanmean(Y_train, axis=0)
    return np.repeat(mu[None, :], int(n_rows), axis=0)


def per_output_nrmse_frame(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    Y_ref: np.ndarray,
    output_names: list[str],
    *,
    min_range: float = 1e-6,
) -> pd.DataFrame:
    """Return per-output nRMSE as a DataFrame.

    Parameters
    ----------
    Y_true
        Observed outputs for the evaluation set, shape ``(n_rows, n_outputs)``.
    Y_pred
        Predicted outputs for the evaluation set, shape matching ``Y_true``.
    Y_ref
        Reference matrix whose columnwise ranges define the normalization denominator.
    output_names
        Names of the output columns, length must equal ``n_outputs``.
    min_range
        Minimum allowable reference range for a column to be included in the macro average.

    Returns
    -------
    pandas.DataFrame
        One row per output with columns ``output_name``, ``rmse``, ``ref_range``,
        ``nrmse``, and ``included_in_macro``.

    Raises
    ------
    ValueError
        Raised when ``output_names`` length does not match the number of output columns.
    """
    Y_true = np.asarray(Y_true, dtype=np.float64)
    Y_pred = np.asarray(Y_pred, dtype=np.float64)
    Y_ref = np.asarray(Y_ref, dtype=np.float64)

    n_outputs = Y_true.shape[1] if Y_true.ndim == 2 else 1
    if len(output_names) != n_outputs:
        raise ValueError(
            f"output_names length ({len(output_names)}) does not match n_outputs ({n_outputs})."
        )

    rmse = np.sqrt(np.mean((Y_true - Y_pred) ** 2, axis=0))
    ref_range = np.nanmax(Y_ref, axis=0) - np.nanmin(Y_ref, axis=0)
    included = ref_range >= min_range
    nrmse = np.where(included, rmse / np.where(included, ref_range, 1.0), float("nan"))

    return pd.DataFrame(
        {
            "output_name": output_names,
            "rmse": rmse.tolist(),
            "ref_range": ref_range.tolist(),
            "nrmse": nrmse.tolist(),
            "included_in_macro": included.tolist(),
        }
    )
