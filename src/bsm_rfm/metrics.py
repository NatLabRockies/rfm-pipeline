"""Evaluation metrics for reduced-form model screening and final fits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


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
    n_jobs: int = 1,
    checkpoint_dir: Path | None = None,
    active_bootstrap_indices: set[int] | None = None,
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
    n_jobs
        Number of parallel jobs for bootstrap replicates. ``1`` runs serially;
        ``-1`` uses all available CPUs. Row-index draws are pre-generated before
        parallelization so results are bit-identical regardless of ``n_jobs``.
    checkpoint_dir
        Optional checkpoint root. When provided, per-replicate bootstrap values are
        cached and reused across reruns so interrupted runs continue from remaining
        replicates.
    active_bootstrap_indices
        Optional subset of 0-based replicate indices to compute in this invocation.
        Used by distributed shard workers to warm checkpoint files without recomputing
        all replicates in every shard.

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

    # Pre-generate all row-index draws so results are deterministic regardless of n_jobs.
    rng = np.random.RandomState(random_state)
    idx_draws = rng.randint(0, n_rows, size=(int(n_boot), draw_size))

    checkpoint_run_dir: Path | None = None
    if checkpoint_dir is not None:
        signature_payload = {
            "n_rows": int(n_rows),
            "n_outputs": int(Y_true.shape[1]),
            "n_boot": int(n_boot),
            "draw_size": int(draw_size),
            "alpha": float(alpha),
            "random_state": int(random_state),
            "min_range": float(min_range),
            "y_true_shape": list(Y_true.shape),
            "y_pred_shape": list(Y_pred.shape),
            "y_ref_shape": list(Y_ref.shape),
            "y_true_mean": float(np.nanmean(Y_true)),
            "y_pred_mean": float(np.nanmean(Y_pred)),
            "y_ref_mean": float(np.nanmean(Y_ref)),
        }
        signature = hashlib.sha256(
            json.dumps(signature_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        checkpoint_run_dir = Path(checkpoint_dir) / signature
        checkpoint_run_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = checkpoint_run_dir / "checkpoint_metadata.json"
        if not metadata_path.exists():
            metadata_path.write_text(
                json.dumps(
                    signature_payload,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

    def _one_replicate(idx: np.ndarray) -> float:
        val, _, _ = macro_nrmse_with_ref(Y_true[idx], Y_pred[idx], Y_ref, min_range=min_range)
        return val

    def _replicate_path(rep_index: int) -> Path | None:
        if checkpoint_run_dir is None:
            return None
        return checkpoint_run_dir / f"replicate_{rep_index:06d}.npy"

    boot = np.full(int(n_boot), np.nan, dtype=np.float64)
    active_indices = (
        set(range(int(n_boot)))
        if active_bootstrap_indices is None
        else {int(index) for index in active_bootstrap_indices if 0 <= int(index) < int(n_boot)}
    )
    if not active_indices:
        raise ValueError("active_bootstrap_indices must include at least one valid index.")
    pending_indices: list[int] = []
    for rep_index in range(int(n_boot)):
        cached_path = _replicate_path(rep_index)
        if cached_path is None or not cached_path.exists():
            if rep_index in active_indices:
                pending_indices.append(rep_index)
            continue
        try:
            cached = np.load(cached_path, allow_pickle=False)
        except (OSError, ValueError):
            if rep_index in active_indices:
                pending_indices.append(rep_index)
            continue
        cached_arr = np.asarray(cached, dtype=np.float64).reshape(-1)
        if cached_arr.size != 1:
            if rep_index in active_indices:
                pending_indices.append(rep_index)
            continue
        boot[rep_index] = float(cached_arr[0])

    if pending_indices:
        if n_jobs == 1:
            computed_values = [_one_replicate(idx_draws[i]) for i in pending_indices]
        else:
            computed_values = Parallel(n_jobs=n_jobs)(
                delayed(_one_replicate)(idx_draws[i]) for i in pending_indices
            )
        for rep_index, value in zip(pending_indices, computed_values, strict=True):
            boot[rep_index] = float(value)
            cached_path = _replicate_path(rep_index)
            if cached_path is not None:
                temp_path = cached_path.with_suffix(".tmp.npy")
                np.save(temp_path, np.array([boot[rep_index]], dtype=np.float64))
                temp_path.replace(cached_path)

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
        "normalization_reference": "Y_train range",
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
