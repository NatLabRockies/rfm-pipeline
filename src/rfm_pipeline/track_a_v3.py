"""Helpers for Track A v3 BSM-transfer modeling."""

from __future__ import annotations

import numpy as np
import pandas as pd

LARGE_OUTPUT_THRESHOLD = 5_000.0


def _knob_columns(feature_cols: list[str], bsm_ops: dict[str, float]) -> list[str]:
    return [c for c in feature_cols if c in bsm_ops]


def _robust_scale(values: np.ndarray) -> np.ndarray:
    med = np.nanmedian(values, axis=0)
    mad = np.nanmedian(np.abs(values - med), axis=0)
    std = np.nanstd(values, axis=0)
    scale = np.where(mad > 1e-9, mad, std)
    scale = np.where(scale > 1e-9, scale, 1.0)
    return scale


def compute_knob_distances(
    frame: pd.DataFrame,
    knob_cols: list[str],
    bsm_ops: dict[str, float],
) -> np.ndarray:
    """Return robustly scaled Euclidean distances to BSM knob settings."""
    knobs = frame[knob_cols].to_numpy(dtype=float)
    bsm = np.array([bsm_ops[c] for c in knob_cols], dtype=float)
    scale = _robust_scale(knobs)
    z = (knobs - bsm) / scale
    return np.sqrt(np.sum(z**2, axis=1))


def estimate_bsm_measurements_knn(
    frame: pd.DataFrame,
    feature_cols: list[str],
    bsm_ops: dict[str, float],
    k_neighbors: int = 8,
) -> tuple[dict[str, float], dict[str, float]]:
    """Impute BSM-only measurement columns using knob-space nearest neighbors."""
    knob_cols = _knob_columns(feature_cols, bsm_ops)
    measurement_cols = [c for c in feature_cols if c not in knob_cols]
    if not knob_cols:
        raise ValueError("No BSM knob columns overlap with feature columns.")
    if not measurement_cols:
        return {}, {"nearest_knob_distance": 0.0, "farthest_knob_distance": 0.0}

    distances = compute_knob_distances(frame, knob_cols, bsm_ops)
    k_eff = int(max(1, min(k_neighbors, len(frame))))
    order = np.argsort(distances)
    nn_idx = order[:k_eff]
    nn_dist = distances[nn_idx]
    weights = 1.0 / np.maximum(nn_dist, 1e-6)
    weights = weights / weights.sum()

    out: dict[str, float] = {}
    for col in measurement_cols:
        vals = frame[col].to_numpy(dtype=float)[nn_idx]
        out[col] = float(np.sum(weights * vals))

    diagnostics = {
        "nearest_knob_distance": float(nn_dist.min()),
        "farthest_knob_distance": float(distances.max()),
        "knn_distance_mean": float(nn_dist.mean()),
    }
    return out, diagnostics


def compute_near_bsm_sample_weights(
    frame: pd.DataFrame,
    feature_cols: list[str],
    bsm_ops: dict[str, float],
    weight_max: float = 2.5,
    distance_bandwidth: float = 2.0,
) -> np.ndarray:
    """Upweight rows closer to BSM in knob space using an RBF kernel."""
    knob_cols = _knob_columns(feature_cols, bsm_ops)
    if not knob_cols:
        return np.ones(len(frame), dtype=float)
    distances = compute_knob_distances(frame, knob_cols, bsm_ops)
    kernel = np.exp(-np.square(distances / max(distance_bandwidth, 1e-6)))
    weights = 1.0 + (max(weight_max, 1.0) - 1.0) * kernel
    return weights.astype(float)


def build_runtime_model_frame(knob_frame: pd.DataFrame) -> pd.DataFrame:
    """Build runtime-oracle and stage-complexity features per (dgp_idx, config_idx)."""
    required_cols = ["dgp_idx", "config_idx", "n_runs", "n_inputs", "n_outputs"]
    for col in required_cols:
        if col not in knob_frame.columns:
            raise ValueError(f"Missing runtime column: {col}")

    data = knob_frame.copy()
    n_runs = data["n_runs"].to_numpy(dtype=float)
    n_inputs = data["n_inputs"].to_numpy(dtype=float)
    n_outputs = data["n_outputs"].to_numpy(dtype=float)

    default = pd.Series(np.nan, index=data.index)
    p_screen = (
        data.get("stages.empirical_null_screening.n_permutations", default).astype(float).to_numpy()
    )
    p_int = (
        data.get("stages.interaction_discovery.n_permutations", default).astype(float).to_numpy()
    )
    n_stab = (
        data.get("stages.sparse_selection.n_stability_subsamples", default).astype(float).to_numpy()
    )
    lasso = (
        data.get("stages.sparse_selection.lasso_alpha_grid_size", default).astype(float).to_numpy()
    )

    screen_term = p_screen * n_inputs * n_runs
    interaction_term = p_int * np.square(n_inputs) * n_runs
    sparse_term = n_stab * n_inputs * n_runs * lasso
    base_term = n_inputs * n_runs
    output_term = n_outputs * n_runs
    large_output_term = np.where(n_outputs >= LARGE_OUTPUT_THRESHOLD, output_term, 0.0)

    runtime_oracle = screen_term + interaction_term + sparse_term + base_term + output_term

    out = pd.DataFrame(
        {
            "dgp_idx": data["dgp_idx"].to_numpy(),
            "config_idx": data["config_idx"].to_numpy(),
            "runtime_oracle": runtime_oracle,
            "runtime_screen_term": screen_term,
            "runtime_interaction_term": interaction_term,
            "runtime_sparse_term": sparse_term,
            "runtime_base_term": base_term,
            "runtime_output_term": output_term,
            "runtime_large_output_term": large_output_term,
        }
    )
    return out
