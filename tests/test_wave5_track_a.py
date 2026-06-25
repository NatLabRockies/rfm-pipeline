from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rfm_pipeline.wave5_track_a import (
    aggregate_wave5_replicates,
    compute_efficiency_targets,
    select_model_feature_columns,
)


def test_compute_efficiency_targets_adds_oracle_gap_and_eta() -> None:
    frame = pd.DataFrame(
        {
            "nrmse_null": [0.20, 0.0],
            "nrmse_final": [0.10, 0.0],
            "noise_snr": [24.0, 24.0],
        }
    )

    out = compute_efficiency_targets(frame)

    expected_oracle = 0.20 * np.sqrt(1.0 / 25.0)
    assert out.loc[0, "nrmse_oracle"] == expected_oracle
    assert out.loc[0, "oracle_gap"] == pytest.approx(expected_oracle * 4.0)
    assert out.loc[0, "eta_total"] == 0.625
    assert np.isnan(out.loc[1, "eta_total"])


def test_select_model_feature_columns_keeps_track_a_order() -> None:
    cols = [
        "variance_threshold",
        "n_inputs",
        "output_spectrum_top1_share",
        "stages.final_artifacts.delta_threshold_override",
    ]

    selected = select_model_feature_columns(cols)

    assert selected == [
        "n_inputs",
        "output_spectrum_top1_share",
        "variance_threshold",
        "stages.final_artifacts.delta_threshold_override",
    ]


def test_aggregate_wave5_replicates_averages_numeric_rows() -> None:
    frame = pd.DataFrame(
        {
            "dgp_idx": [0, 0, 0, 1],
            "config_idx": [2, 2, 3, 0],
            "replicate": [0, 1, 0, 0],
            "nrmse_final": [0.1, 0.3, 0.4, 0.2],
            "label": ["a", "a", "b", "c"],
        }
    )

    out = aggregate_wave5_replicates(frame)

    row = out[(out["dgp_idx"] == 0) & (out["config_idx"] == 2)].iloc[0]
    assert row["nrmse_final"] == 0.2
    assert row["replicate"] == 0.5
    assert row["label"] == "a"
