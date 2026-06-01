"""Tests for test data."""

from __future__ import annotations

import pandas as pd
import pytest

from rfm_pipeline.data import (
    add_scenario_flags,
    align_xy,
    ensure_id_columns,
    fit_standardizers,
    make_boolean_combination_labels,
    stratified_holdout_split,
    stratified_subset_by_boolean_combination,
)

SCENARIOS = [
    "AFSCoff_UAEOROoff",
    "AFSCon_UAEOROoff",
    "AFSCoff_UAEOROon",
    "AFSCon_UAEOROon",
]


def make_demo_xy() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    outputs = []
    run_id = 0
    for scenario in SCENARIOS:
        for _ in range(5):
            rows.append(
                {
                    "run_id": run_id,
                    "scenario": scenario,
                    "x1": float(run_id),
                    "x2": float(run_id + 10),
                }
            )
            outputs.append(
                {
                    "run_id": run_id,
                    "scenario": scenario,
                    "y1": float(run_id * 2),
                    "y2": float(run_id * 3),
                }
            )
            run_id += 1
    return pd.DataFrame(rows), pd.DataFrame(outputs)


def test_add_scenario_flags_from_column():
    X, _ = make_demo_xy()
    flagged = add_scenario_flags(X)
    assert flagged["AFSC"].tolist()[:4] == [0, 0, 0, 0]
    assert flagged["UAEORO"].tolist()[:4] == [0, 0, 0, 0]
    assert flagged["AFSC"].iloc[-1] == 1
    assert flagged["UAEORO"].iloc[-1] == 1


def test_make_boolean_combination_labels_uses_indicator_columns():
    frame = pd.DataFrame({"AFSC": [0, 0, 1, 1], "UAEORO": [0, 1, 0, 1]})
    labels = make_boolean_combination_labels(frame)
    assert labels.tolist() == [
        "AFSC0_UAEORO0",
        "AFSC0_UAEORO1",
        "AFSC1_UAEORO0",
        "AFSC1_UAEORO1",
    ]


def test_ensure_id_columns_promotes_index_names():
    X, _ = make_demo_xy()
    indexed = X.set_index(["run_id", "scenario"])
    exposed = ensure_id_columns(indexed, ["run_id", "scenario"])
    assert ["run_id", "scenario"] == exposed.columns[:2].tolist()


def test_align_xy_inner_aligns_row_index():
    X, Y = make_demo_xy()
    X = X.set_index(["run_id", "scenario"])
    Y = Y.iloc[2:].set_index(["run_id", "scenario"])
    X_aligned, Y_aligned = align_xy(X, Y)
    assert list(X_aligned.index) == list(Y_aligned.index)
    assert len(X_aligned) == len(Y)


def test_stratified_holdout_split_preserves_four_scenarios():
    X, Y = make_demo_xy()
    X = add_scenario_flags(X).set_index(["run_id", "scenario"])
    Y = Y.set_index(["run_id", "scenario"])
    X_train, X_holdout, Y_train, Y_holdout = stratified_holdout_split(
        X,
        Y,
        holdout_fraction=0.20,
        random_state=7,
    )
    assert len(X_train) == 16
    assert len(X_holdout) == 4
    assert set(X_holdout["AFSC"].astype(str) + X_holdout["UAEORO"].astype(str)) == {
        "00",
        "01",
        "10",
        "11",
    }
    assert X_train.index.equals(Y_train.index)
    assert X_holdout.index.equals(Y_holdout.index)


def test_fit_standardizers_uses_training_statistics_only():
    X, Y = make_demo_xy()
    X = add_scenario_flags(X)
    X_train = X.iloc[:12].set_index(["run_id", "scenario"])
    X_holdout = X.iloc[12:].set_index(["run_id", "scenario"])
    Y_train = Y.iloc[:12].set_index(["run_id", "scenario"])
    Y_holdout = Y.iloc[12:].set_index(["run_id", "scenario"])
    bundle = fit_standardizers(
        X_train[["x1", "x2", "AFSC", "UAEORO"]],
        X_holdout[["x1", "x2", "AFSC", "UAEORO"]],
        Y_train,
        Y_holdout,
    )
    assert bundle.x_train_scaled.shape == (12, 4)
    assert bundle.x_holdout_scaled.shape == (8, 4)
    assert bundle.y_train_scaled.shape == (12, 2)
    assert bundle.y_holdout_scaled.shape == (8, 2)


def test_stratified_subset_by_boolean_combination_draws_equal_counts_per_scenario():
    rows = []
    scenario_order = [
        "AFSCoff_UAEOROoff",
        "AFSCoff_UAEOROon",
        "AFSCon_UAEOROoff",
        "AFSCon_UAEOROon",
    ]
    for scenario in scenario_order:
        for run_id in range(6000):
            rows.append(
                {
                    "scenario": scenario,
                    "run_id": f"{scenario}-{run_id}",
                    "x": run_id,
                }
            )
    frame = add_scenario_flags(pd.DataFrame(rows))
    subset = stratified_subset_by_boolean_combination(
        frame,
        n_per_combination=5000,
        random_state=19,
    )
    counts = make_boolean_combination_labels(subset).value_counts().to_dict()
    assert counts == {
        "AFSC0_UAEORO0": 5000,
        "AFSC0_UAEORO1": 5000,
        "AFSC1_UAEORO0": 5000,
        "AFSC1_UAEORO1": 5000,
    }
    assert len(subset) == 20000
    assert subset["run_id"].is_unique


def test_stratified_subset_by_boolean_combination_requires_large_enough_strata():
    frame = pd.DataFrame({"AFSC": [0, 0, 1], "UAEORO": [0, 1, 1], "x": [1, 2, 3]})
    with pytest.raises(ValueError, match="Expected all four boolean combinations|cannot draw"):
        stratified_subset_by_boolean_combination(frame, n_per_combination=2)
