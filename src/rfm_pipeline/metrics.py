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


def stratified_bootstrap_macro_nrmse_ci(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    Y_ref: np.ndarray,
    strata: np.ndarray | list,
    *,
    min_range: float = 1e-6,
    n_boot: int = 1000,
    alpha: float = 0.05,
    random_state: int = 123,
    n_stability_batches: int = 10,
    round_digits: int | None = 4,
    n_jobs: int = 1,
) -> dict[str, Any]:
    """Percentile bootstrap CI for macro nRMSE with within-stratum resampling.

    For each replicate, rows are resampled independently within each stratum
    (i.e. stratified bootstrap), preserving stratum proportions in the sample.
    Monte-Carlo endpoint-stability diagnostics are computed by splitting the
    ``n_boot`` replicates into ``n_stability_batches`` equal groups and measuring
    the standard deviation of each batch's CI endpoint.

    Parameters
    ----------
    Y_true
        Observed outputs, shape ``(n_rows, n_outputs)``.
    Y_pred
        Predicted outputs, shape matching ``Y_true``.
    Y_ref
        Reference matrix for fixed normalization ranges.
    strata
        1-D array-like of length ``n_rows`` with stratum labels.
    min_range
        Minimum reference range for output inclusion.
    n_boot
        Number of bootstrap replicates (default 1000).
    alpha
        Two-sided error level.
    random_state
        Seed for reproducibility.
    n_stability_batches
        Number of equal-sized groups to use for MC endpoint-stability diagnostics.
        Must be ≥2 and divide ``n_boot`` evenly; if ``n_boot`` is not divisible,
        the last batch absorbs any remainder.
    round_digits
        Number of decimal places to round CI endpoints. ``None`` disables rounding.
    n_jobs
        Parallel jobs. ``1`` runs serially.

    Returns
    -------
    dict[str, Any]
        Keys: ``point_estimate``, ``ci_lower``, ``ci_upper``,
        ``bootstrap_mean``, ``bootstrap_std``, ``n_boot``, ``alpha``,
        ``ci_type``, ``k_used``, ``k_total``, ``round_digits``,
        ``mc_stability`` (dict with ``lower_std`` and ``upper_std``,
        the SD of CI endpoints across stability batches).

    Raises
    ------
    ValueError
        When the evaluation set is empty, strata length mismatches, or all
        replicates are non-finite.
    """
    Y_true = np.asarray(Y_true, dtype=np.float64)
    Y_pred = np.asarray(Y_pred, dtype=np.float64)
    Y_ref = np.asarray(Y_ref, dtype=np.float64)
    strata_arr = np.asarray(strata)

    n_rows = Y_true.shape[0]
    if n_rows == 0:
        raise ValueError("Cannot bootstrap an empty evaluation set.")
    if len(strata_arr) != n_rows:
        raise ValueError(f"strata length ({len(strata_arr)}) must match n_rows ({n_rows}).")

    point_estimate, info, rmse = macro_nrmse_with_ref(Y_true, Y_pred, Y_ref, min_range=min_range)

    unique_strata, strata_inv = np.unique(strata_arr, return_inverse=True)
    strata_indices: list[np.ndarray] = [
        np.where(strata_inv == i)[0] for i in range(len(unique_strata))
    ]

    rng = np.random.RandomState(random_state)
    # Pre-draw within-stratum indices for all replicates.
    idx_draws: list[list[np.ndarray]] = []
    for _ in range(int(n_boot)):
        draw_parts = [rng.choice(s_idx, size=len(s_idx), replace=True) for s_idx in strata_indices]
        idx_draws.append(draw_parts)

    def _one_replicate(draw_parts: list[np.ndarray]) -> float:
        idx = np.concatenate(draw_parts)
        val, _, _ = macro_nrmse_with_ref(Y_true[idx], Y_pred[idx], Y_ref, min_range=min_range)
        return val

    if n_jobs == 1:
        boot = np.array([_one_replicate(draw_parts) for draw_parts in idx_draws], dtype=np.float64)
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(_one_replicate)(draw_parts) for draw_parts in idx_draws
        )
        boot = np.array(results, dtype=np.float64)

    valid = np.isfinite(boot)
    if not np.any(valid):
        raise ValueError("All bootstrap replicates were non-finite.")

    boot_valid = boot[valid]
    ci_lower = float(np.nanquantile(boot_valid, alpha / 2))
    ci_upper = float(np.nanquantile(boot_valid, 1 - alpha / 2))
    if round_digits is not None:
        ci_lower = round(ci_lower, round_digits)
        ci_upper = round(ci_upper, round_digits)

    # Monte-Carlo endpoint stability: split replicates into batches.
    n_batches = max(2, int(n_stability_batches))
    base_size = len(boot_valid) // n_batches
    batch_lowers: list[float] = []
    batch_uppers: list[float] = []
    start = 0
    for b in range(n_batches):
        end = start + base_size + (1 if b < len(boot_valid) % n_batches else 0)
        batch = boot_valid[start:end]
        if len(batch) >= 2:
            batch_lowers.append(float(np.quantile(batch, alpha / 2)))
            batch_uppers.append(float(np.quantile(batch, 1 - alpha / 2)))
        start = end

    mc_stability = {
        "lower_std": float(np.std(batch_lowers, ddof=1))
        if len(batch_lowers) >= 2
        else float("nan"),
        "upper_std": float(np.std(batch_uppers, ddof=1))
        if len(batch_uppers) >= 2
        else float("nan"),
        "n_batches": len(batch_lowers),
    }

    bootstrap_std = float(np.std(boot_valid, ddof=1)) if len(boot_valid) > 1 else 0.0

    return {
        "point_estimate": float(point_estimate),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "bootstrap_mean": float(np.mean(boot_valid)),
        "bootstrap_std": bootstrap_std,
        "n_boot": int(n_boot),
        "n_valid_boot": int(np.sum(valid)),
        "alpha": float(alpha),
        "ci_type": "stratified_percentile",
        "k_used": int(info["k_used"]),
        "k_total": int(info["k_total"]),
        "round_digits": round_digits,
        "mc_stability": mc_stability,
        "strata_labels": [str(s) for s in unique_strata],
        "strata_counts": [int(len(s)) for s in strata_indices],
    }


def bootstrap_paired_difference_ci(
    Y_true: np.ndarray,
    Y_pred_a: np.ndarray,
    Y_pred_b: np.ndarray,
    Y_ref: np.ndarray,
    *,
    min_range: float = 1e-6,
    n_boot: int = 1000,
    alpha: float = 0.05,
    random_state: int = 123,
    strata: np.ndarray | list | None = None,
    round_digits: int | None = 4,
    n_jobs: int = 1,
) -> dict[str, Any]:
    """Percentile bootstrap CI for the paired difference nRMSE(A) − nRMSE(B).

    On each replicate, the same row indices (or within-stratum indices when
    ``strata`` is provided) are used for both models so the difference is
    computed on paired samples.

    Parameters
    ----------
    Y_true
        Observed outputs, shape ``(n_rows, n_outputs)``.
    Y_pred_a
        Predictions from model A.
    Y_pred_b
        Predictions from model B.
    Y_ref
        Reference matrix for fixed normalization ranges.
    min_range
        Minimum reference range for output inclusion.
    n_boot
        Number of bootstrap replicates (default 1000).
    alpha
        Two-sided error level.
    random_state
        Seed for reproducibility.
    strata
        Optional 1-D array of stratum labels for within-stratum resampling.
    round_digits
        Decimal places to round CI endpoints. ``None`` disables rounding.
    n_jobs
        Parallel jobs.

    Returns
    -------
    dict[str, Any]
        Keys: ``point_estimate`` (nRMSE_A − nRMSE_B), ``ci_lower``,
        ``ci_upper``, ``bootstrap_mean``, ``bootstrap_std``, ``n_boot``,
        ``alpha``, ``ci_type``, ``round_digits``.
        Negative point estimate means A outperforms B.

    Raises
    ------
    ValueError
        When arrays are empty or strata length mismatches.
    """
    Y_true = np.asarray(Y_true, dtype=np.float64)
    Y_pred_a = np.asarray(Y_pred_a, dtype=np.float64)
    Y_pred_b = np.asarray(Y_pred_b, dtype=np.float64)
    Y_ref = np.asarray(Y_ref, dtype=np.float64)

    n_rows = Y_true.shape[0]
    if n_rows == 0:
        raise ValueError("Cannot bootstrap an empty evaluation set.")

    point_a, _, _ = macro_nrmse_with_ref(Y_true, Y_pred_a, Y_ref, min_range=min_range)
    point_b, _, _ = macro_nrmse_with_ref(Y_true, Y_pred_b, Y_ref, min_range=min_range)
    point_diff = float(point_a) - float(point_b)

    # Build per-replicate index draws.
    rng = np.random.RandomState(random_state)
    if strata is not None:
        strata_arr = np.asarray(strata)
        if len(strata_arr) != n_rows:
            raise ValueError(f"strata length ({len(strata_arr)}) must match n_rows ({n_rows}).")
        unique_strata, strata_inv = np.unique(strata_arr, return_inverse=True)
        strata_indices: list[np.ndarray] = [
            np.where(strata_inv == i)[0] for i in range(len(unique_strata))
        ]
        idx_draws_list: list[np.ndarray] = []
        for _ in range(int(n_boot)):
            draw_parts = [
                rng.choice(s_idx, size=len(s_idx), replace=True) for s_idx in strata_indices
            ]
            idx_draws_list.append(np.concatenate(draw_parts))
        idx_draws_arr = idx_draws_list
    else:
        idx_matrix = rng.randint(0, n_rows, size=(int(n_boot), n_rows))
        idx_draws_arr = [idx_matrix[i] for i in range(int(n_boot))]

    def _one_replicate(idx: np.ndarray) -> float:
        a, _, _ = macro_nrmse_with_ref(Y_true[idx], Y_pred_a[idx], Y_ref, min_range=min_range)
        b, _, _ = macro_nrmse_with_ref(Y_true[idx], Y_pred_b[idx], Y_ref, min_range=min_range)
        return float(a) - float(b)

    if n_jobs == 1:
        boot = np.array([_one_replicate(idx) for idx in idx_draws_arr], dtype=np.float64)
    else:
        results = Parallel(n_jobs=n_jobs)(delayed(_one_replicate)(idx) for idx in idx_draws_arr)
        boot = np.array(results, dtype=np.float64)

    valid = np.isfinite(boot)
    if not np.any(valid):
        raise ValueError("All bootstrap replicates were non-finite.")

    boot_valid = boot[valid]
    ci_lower = float(np.nanquantile(boot_valid, alpha / 2))
    ci_upper = float(np.nanquantile(boot_valid, 1 - alpha / 2))
    if round_digits is not None:
        ci_lower = round(ci_lower, round_digits)
        ci_upper = round(ci_upper, round_digits)

    bootstrap_std = float(np.std(boot_valid, ddof=1)) if len(boot_valid) > 1 else 0.0

    return {
        "point_estimate": point_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "bootstrap_mean": float(np.mean(boot_valid)),
        "bootstrap_std": bootstrap_std,
        "n_boot": int(n_boot),
        "n_valid_boot": int(np.sum(valid)),
        "alpha": float(alpha),
        "ci_type": "paired_percentile",
        "round_digits": round_digits,
    }


def stratified_nrmse_summary(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    Y_ref: np.ndarray,
    strata: dict[str, np.ndarray | list],
    output_names: list[str],
    *,
    min_range: float = 1e-6,
    tail_quantiles: tuple[float, ...] = (0.95, 0.99),
    metric_population: str | None = None,
) -> dict[str, Any]:
    """Compute nRMSE summary stratified by arbitrary row-level keys.

    Produces an overall summary (macro nRMSE, per-output distribution, worst
    output) and a per-stratum breakdown for every key in ``strata``.  Every
    result carries an explicit ``metric_population`` label so the caller always
    knows which outputs and rows are covered.

    Parameters
    ----------
    Y_true
        Observed outputs, shape ``(n_rows, n_outputs)``.
    Y_pred
        Predicted outputs, shape matching ``Y_true``.
    Y_ref
        Reference matrix for fixed normalization ranges, shape
        ``(n_ref_rows, n_outputs)``.
    strata
        Mapping of strata key names (e.g. ``"scenario"``, ``"year"``) to
        1-D array-like of length ``n_rows`` with the corresponding label per
        row.  Each key produces a separate marginal breakdown.
    output_names
        Names of the output columns; length must equal ``n_outputs``.
    min_range
        Minimum reference range for output inclusion in the macro average.
    tail_quantiles
        Upper quantiles to compute over the per-output nRMSE distribution
        (e.g. ``(0.95, 0.99)`` for p95 and p99).
    metric_population
        Explicit label describing the population covered.  Defaults to
        ``"included_outputs_all_rows"`` when ``None``.

    Returns
    -------
    dict[str, Any]
        Keys:

        - ``metric_population``: label string
        - ``n_outputs_used``, ``n_outputs_total``
        - ``macro_nrmse``: mean of per-output nRMSE over included outputs
        - ``median_nrmse``: median of per-output nRMSE over included outputs
        - ``tail_nrmse``: mapping quantile → value
        - ``worst_output``: ``{"name": str, "nrmse": float}``
        - ``by_stratum``: mapping strata key → list of per-value dicts,
          each with ``value``, ``n_rows``, ``macro_nrmse``, ``median_nrmse``,
          ``worst_output``, and ``metric_population``.

    Raises
    ------
    ValueError
        When array shapes are inconsistent or a strata array has wrong length.
    """
    Y_true = np.asarray(Y_true, dtype=np.float64)
    Y_pred = np.asarray(Y_pred, dtype=np.float64)
    Y_ref = np.asarray(Y_ref, dtype=np.float64)

    n_rows, n_outputs = Y_true.shape
    if len(output_names) != n_outputs:
        raise ValueError(f"output_names length ({len(output_names)}) != n_outputs ({n_outputs}).")

    population_label = metric_population or "included_outputs_all_rows"

    def _output_stats(
        yt: np.ndarray,
        yp: np.ndarray,
        yr: np.ndarray,
        pop_label: str,
    ) -> dict[str, Any]:
        rmse = np.sqrt(np.mean((yt - yp) ** 2, axis=0))
        ref_range = np.nanmax(yr, axis=0) - np.nanmin(yr, axis=0)
        included = ref_range >= min_range
        nrmse_per = np.where(included, rmse / np.where(included, ref_range, 1.0), np.nan)
        valid_nrmse = nrmse_per[included]
        if valid_nrmse.size == 0:
            macro = float("nan")
            median = float("nan")
            tails = {q: float("nan") for q in tail_quantiles}
            worst = {"name": None, "nrmse": float("nan")}
        else:
            macro = float(np.nanmean(valid_nrmse))
            median = float(np.nanmedian(valid_nrmse))
            tails = {q: float(np.nanquantile(valid_nrmse, q)) for q in tail_quantiles}
            worst_idx = int(np.nanargmax(nrmse_per))
            worst = {"name": output_names[worst_idx], "nrmse": float(nrmse_per[worst_idx])}
        return {
            "metric_population": pop_label,
            "n_outputs_used": int(np.sum(included)),
            "n_outputs_total": int(n_outputs),
            "macro_nrmse": macro,
            "median_nrmse": median,
            "tail_nrmse": tails,
            "worst_output": worst,
        }

    overall = _output_stats(Y_true, Y_pred, Y_ref, population_label)

    by_stratum: dict[str, list[dict[str, Any]]] = {}
    for key, labels_raw in strata.items():
        labels = np.asarray(labels_raw)
        if len(labels) != n_rows:
            raise ValueError(
                f"strata['{key}'] length ({len(labels)}) must match n_rows ({n_rows})."
            )
        unique_vals = np.unique(labels)
        breakdown: list[dict[str, Any]] = []
        for val in unique_vals:
            mask = labels == val
            sub_pop = f"{population_label}|{key}={val}"
            sub = _output_stats(Y_true[mask], Y_pred[mask], Y_ref, sub_pop)
            sub["value"] = str(val)
            sub["n_rows"] = int(np.sum(mask))
            breakdown.append(sub)
        by_stratum[key] = breakdown

    overall["by_stratum"] = by_stratum
    return overall


def excluded_output_error_summary(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    excluded_mask: np.ndarray | list,
    output_names: list[str],
    *,
    domain_scales: np.ndarray | list | None = None,
) -> dict[str, Any]:
    """Compute absolute and domain-scaled error summary for excluded outputs.

    Excluded outputs are those filtered out of the headline macro nRMSE (e.g.
    because their reference range is below ``min_range``).  This helper
    surfaces their RMSE so they are not silently ignored.

    Parameters
    ----------
    Y_true
        Observed outputs, shape ``(n_rows, n_outputs)``.
    Y_pred
        Predicted outputs, shape matching ``Y_true``.
    excluded_mask
        Boolean array of length ``n_outputs``.  ``True`` marks an output as
        excluded from the headline metric.
    output_names
        Names of all outputs; length must equal ``n_outputs``.
    domain_scales
        Optional per-output scale factors (e.g. the known physical domain
        range) used to compute domain-scaled RMSE.  Length must equal
        ``n_outputs``.  When ``None``, ``domain_scaled_rmse`` is ``None`` for
        every output.

    Returns
    -------
    dict[str, Any]
        Keys:

        - ``metric_population``: ``"excluded_outputs"``
        - ``n_excluded``: number of excluded outputs
        - ``n_total``: total number of outputs
        - ``outputs``: list of per-output dicts with ``name``, ``rmse``,
          ``domain_scaled_rmse`` (float or ``None``), and ``excluded``.
        - ``median_rmse``: median absolute RMSE over excluded outputs
          (``nan`` when no excluded outputs).
        - ``mean_rmse``: mean absolute RMSE over excluded outputs.

    Raises
    ------
    ValueError
        When ``output_names`` or ``domain_scales`` lengths mismatch.
    """
    Y_true = np.asarray(Y_true, dtype=np.float64)
    Y_pred = np.asarray(Y_pred, dtype=np.float64)
    excl = np.asarray(excluded_mask, dtype=bool)

    n_outputs = Y_true.shape[1] if Y_true.ndim == 2 else 1
    if len(output_names) != n_outputs:
        raise ValueError(f"output_names length ({len(output_names)}) != n_outputs ({n_outputs}).")
    if len(excl) != n_outputs:
        raise ValueError(f"excluded_mask length ({len(excl)}) != n_outputs ({n_outputs}).")
    if domain_scales is not None:
        scales = np.asarray(domain_scales, dtype=np.float64)
        if len(scales) != n_outputs:
            raise ValueError(f"domain_scales length ({len(scales)}) != n_outputs ({n_outputs}).")
    else:
        scales = None

    rmse_all = np.sqrt(np.mean((Y_true - Y_pred) ** 2, axis=0))

    outputs: list[dict[str, Any]] = []
    excluded_rmses: list[float] = []
    for i, name in enumerate(output_names):
        rmse_val = float(rmse_all[i])
        domain_scaled = None
        if scales is not None and excl[i]:
            s = float(scales[i])
            domain_scaled = float(rmse_val / s) if s > 0 else float("nan")
        entry: dict[str, Any] = {
            "name": name,
            "rmse": rmse_val,
            "domain_scaled_rmse": domain_scaled,
            "excluded": bool(excl[i]),
        }
        outputs.append(entry)
        if excl[i]:
            excluded_rmses.append(rmse_val)

    excl_arr = (
        np.array(excluded_rmses, dtype=np.float64)
        if excluded_rmses
        else np.array([], dtype=np.float64)
    )
    return {
        "metric_population": "excluded_outputs",
        "n_excluded": int(np.sum(excl)),
        "n_total": n_outputs,
        "outputs": outputs,
        "median_rmse": float(np.median(excl_arr)) if excl_arr.size > 0 else float("nan"),
        "mean_rmse": float(np.mean(excl_arr)) if excl_arr.size > 0 else float("nan"),
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
