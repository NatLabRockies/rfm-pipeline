"""Tests for regularized feature screening."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rfm_pipeline.regularized_screening import (
    fit_multitask_elastic_net_screen,
    screening_selection_table,
)


def test_fit_multitask_elastic_net_screen_recovers_signal_features() -> None:
    rng = np.random.RandomState(7)
    X = pd.DataFrame(
        rng.normal(size=(80, 4)),
        columns=["x1", "x2", "x3", "x4"],
    )
    Y = pd.DataFrame(
        {
            "y1": 3.0 * X["x1"] - 2.0 * X["x3"],
            "y2": -1.5 * X["x1"] + 1.0 * X["x3"],
        }
    )

    result = fit_multitask_elastic_net_screen(
        X,
        Y,
        cv=4,
        l1_ratio=(0.9, 1.0),
        alphas=50,
        max_iter=10000,
        selection_tol=1e-4,
        random_state=11,
    )

    assert set(result.selected_features) == {"x1", "x3"}
    assert result.feature_names == ("x1", "x2", "x3", "x4")
    assert result.output_names == ("y1", "y2")
    assert result.cv_folds == 4


def test_screening_selection_table_preserves_feature_order_and_selection_metadata() -> None:
    rng = np.random.RandomState(5)
    X = pd.DataFrame(rng.normal(size=(40, 3)), columns=["a", "b", "c"])
    Y = pd.DataFrame({"y": 4.0 * X["b"]})

    result = fit_multitask_elastic_net_screen(
        X,
        Y,
        cv=3,
        l1_ratio=1.0,
        alphas=25,
        max_iter=10000,
        selection_tol=1e-5,
        random_state=13,
    )
    table = screening_selection_table(result)

    assert table["feature_name"].tolist() == ["a", "b", "c"]
    assert table["original_position"].tolist() == [0, 1, 2]
    selected_row = table.loc[table["feature_name"] == "b"].iloc[0]
    assert bool(selected_row["selected"])
    assert int(selected_row["nonzero_output_count"]) == 1


def test_screening_rejects_misaligned_rows() -> None:
    X = pd.DataFrame({"x": [1.0, 2.0, 3.0]}, index=["a", "b", "c"])
    Y = pd.DataFrame({"y": [1.0, 2.0, 3.0]}, index=["b", "c", "d"])

    with pytest.raises(ValueError, match="identical row indexes"):
        fit_multitask_elastic_net_screen(X, Y, cv=2)
