from __future__ import annotations

import numpy as np
import pandas as pd

from rfm_pipeline.track_a_v3 import (
    build_runtime_model_frame,
    compute_near_bsm_sample_weights,
    estimate_bsm_measurements_knn,
)


def test_estimate_bsm_measurements_knn_prefers_nearby_rows() -> None:
    frame = pd.DataFrame(
        {
            "dgp_idx": [0, 1, 2],
            "config_idx": [0, 0, 0],
            "n_inputs": [130, 135, 40],
            "n_runs": [28000, 29000, 6000],
            "n_outputs": [9800, 10000, 500],
            "output_spectrum_top1_share": [0.65, 0.75, 0.05],
            "output_skewness_abs_mean": [5.2, 5.6, 0.2],
        }
    )
    feature_cols = [
        "n_inputs",
        "n_runs",
        "n_outputs",
        "output_spectrum_top1_share",
        "output_skewness_abs_mean",
    ]
    bsm_ops = {"n_inputs": 135, "n_runs": 28750, "n_outputs": 9954}

    imputed, diagnostics = estimate_bsm_measurements_knn(
        frame=frame,
        feature_cols=feature_cols,
        bsm_ops=bsm_ops,
        k_neighbors=2,
    )

    assert 0.65 < imputed["output_spectrum_top1_share"] < 0.75
    assert 5.2 < imputed["output_skewness_abs_mean"] < 5.6
    assert diagnostics["nearest_knob_distance"] < diagnostics["farthest_knob_distance"]


def test_compute_near_bsm_sample_weights_upweights_close_rows() -> None:
    frame = pd.DataFrame(
        {
            "n_inputs": [135, 134, 45],
            "n_runs": [28750, 28000, 6000],
            "n_outputs": [9954, 9800, 500],
        }
    )
    bsm_ops = {"n_inputs": 135, "n_runs": 28750, "n_outputs": 9954}
    feature_cols = ["n_inputs", "n_runs", "n_outputs"]

    weights = compute_near_bsm_sample_weights(
        frame=frame,
        feature_cols=feature_cols,
        bsm_ops=bsm_ops,
        weight_max=3.0,
        distance_bandwidth=1.5,
    )

    assert weights[0] > weights[1] > weights[2]
    assert np.all(weights >= 1.0)
    assert np.all(weights <= 3.0)


def test_build_runtime_model_frame_adds_large_output_regime_term() -> None:
    frame = pd.DataFrame(
        {
            "dgp_idx": [0, 1],
            "config_idx": [0, 0],
            "n_inputs": [120, 120],
            "n_runs": [10000, 10000],
            "n_outputs": [4000, 9000],
            "stages.empirical_null_screening.n_permutations": [101, 101],
            "stages.interaction_discovery.n_permutations": [31, 31],
            "stages.sparse_selection.n_stability_subsamples": [50, 50],
            "stages.sparse_selection.lasso_alpha_grid_size": [40, 40],
        }
    )

    runtime_frame = build_runtime_model_frame(frame)

    assert "runtime_oracle" in runtime_frame.columns
    assert "runtime_large_output_term" in runtime_frame.columns
    assert runtime_frame.loc[0, "runtime_large_output_term"] == 0.0
    assert runtime_frame.loc[1, "runtime_large_output_term"] > 0.0
    assert np.all(runtime_frame["runtime_oracle"] > 0)
