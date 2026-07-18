"""Tests for the generic data preparation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rfm_pipeline.data import (
    align_xy,
    combination_labels,
    ensure_id_columns,
    fit_standardizers,
    stratified_holdout_split,
    stratified_subset_by_combination,
)


def make_demo_xy() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a small synthetic (X, Y) pair with two boolean strata columns."""
    rows: list[dict[str, object]] = []
    outputs: list[dict[str, object]] = []
    run_id = 0
    for a in (0, 1):
        for b in (0, 1):
            for _ in range(5):
                rows.append(
                    {
                        "run_id": run_id,
                        "stratum_a": a,
                        "stratum_b": b,
                        "x1": float(run_id),
                        "x2": float(run_id + 10),
                    }
                )
                outputs.append(
                    {
                        "run_id": run_id,
                        "stratum_a": a,
                        "stratum_b": b,
                        "y1": float(run_id * 2),
                        "y2": float(run_id * 3),
                    }
                )
                run_id += 1
    return pd.DataFrame(rows), pd.DataFrame(outputs)


def test_combination_labels_joins_columns_in_order():
    frame = pd.DataFrame(
        {
            "stratum_a": [0, 0, 1, 1],
            "stratum_b": [0, 1, 0, 1],
        }
    )
    labels = combination_labels(frame, columns=["stratum_a", "stratum_b"])
    assert labels.tolist() == [
        "stratum_a=0|stratum_b=0",
        "stratum_a=0|stratum_b=1",
        "stratum_a=1|stratum_b=0",
        "stratum_a=1|stratum_b=1",
    ]


def test_combination_labels_missing_column_raises():
    frame = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(ValueError, match="Missing columns"):
        combination_labels(frame, columns=["a", "missing"])


def test_ensure_id_columns_promotes_index_names():
    X, _ = make_demo_xy()
    indexed = X.set_index(["run_id", "stratum_a"])
    exposed = ensure_id_columns(indexed, ["run_id", "stratum_a"])
    assert ["run_id", "stratum_a"] == exposed.columns[:2].tolist()


def test_align_xy_inner_aligns_row_index():
    X, Y = make_demo_xy()
    X = X.set_index(["run_id"])
    Y = Y.iloc[2:].set_index(["run_id"])
    X_aligned, Y_aligned = align_xy(X, Y)
    assert list(X_aligned.index) == list(Y_aligned.index)
    assert len(X_aligned) == len(Y)


def test_stratified_holdout_split_preserves_four_strata():
    X, Y = make_demo_xy()
    X = X.set_index(["run_id"])
    Y = Y.set_index(["run_id"])
    X_train, X_holdout, Y_train, Y_holdout = stratified_holdout_split(
        X,
        Y,
        holdout_fraction=0.20,
        random_state=7,
        stratify_columns=["stratum_a", "stratum_b"],
    )
    assert len(X_train) == 16
    assert len(X_holdout) == 4
    combos = combination_labels(X_holdout, columns=["stratum_a", "stratum_b"])
    assert set(combos.tolist()) == {
        "stratum_a=0|stratum_b=0",
        "stratum_a=0|stratum_b=1",
        "stratum_a=1|stratum_b=0",
        "stratum_a=1|stratum_b=1",
    }
    assert X_train.index.equals(Y_train.index)
    assert X_holdout.index.equals(Y_holdout.index)


def test_fit_standardizers_uses_training_statistics_only():
    X, Y = make_demo_xy()
    X_train = X.iloc[:12].set_index(["run_id"])
    X_holdout = X.iloc[12:].set_index(["run_id"])
    Y_train = Y.iloc[:12].set_index(["run_id"])
    Y_holdout = Y.iloc[12:].set_index(["run_id"])
    bundle = fit_standardizers(
        X_train[["x1", "x2", "stratum_a", "stratum_b"]],
        X_holdout[["x1", "x2", "stratum_a", "stratum_b"]],
        Y_train[["y1", "y2"]],
        Y_holdout[["y1", "y2"]],
    )
    assert bundle.x_train_scaled.shape == (12, 4)
    assert bundle.x_holdout_scaled.shape == (8, 4)
    assert bundle.y_train_scaled.shape == (12, 2)
    assert bundle.y_holdout_scaled.shape == (8, 2)


def test_stratified_subset_by_combination_draws_equal_counts_per_stratum():
    rows: list[dict[str, object]] = []
    strata = [(a, b, c) for a in (0, 1) for b in (0, 1) for c in ("x", "y")]
    for a, b, c in strata:
        for run_id in range(60):
            rows.append(
                {
                    "stratum_a": a,
                    "stratum_b": b,
                    "stratum_c": c,
                    "run_id": f"{a}-{b}-{c}-{run_id}",
                    "x": run_id,
                }
            )
    frame = pd.DataFrame(rows)
    subset = stratified_subset_by_combination(
        frame,
        columns=["stratum_a", "stratum_b", "stratum_c"],
        n_per_combination=50,
        random_state=19,
    )
    counts = (
        combination_labels(subset, columns=["stratum_a", "stratum_b", "stratum_c"])
        .value_counts()
        .to_dict()
    )
    assert len(counts) == 8
    assert set(counts.values()) == {50}
    assert len(subset) == 400
    assert subset["run_id"].is_unique


def test_stratified_subset_by_combination_requires_large_enough_strata():
    frame = pd.DataFrame(
        {
            "stratum_a": [0, 0, 1],
            "stratum_b": [0, 1, 1],
            "x": [1, 2, 3],
        }
    )
    with pytest.raises(ValueError, match="cannot draw"):
        stratified_subset_by_combination(
            frame,
            columns=["stratum_a", "stratum_b"],
            n_per_combination=2,
        )


def test_stratified_subset_by_combination_is_deterministic_under_fixed_seed():
    rng = np.random.RandomState(0)
    rows = []
    for a in (0, 1):
        for b in (0, 1):
            for run_id in range(200):
                rows.append(
                    {
                        "stratum_a": a,
                        "stratum_b": b,
                        "run_id": f"{a}-{b}-{run_id}",
                        "x": float(rng.rand()),
                    }
                )
    frame = pd.DataFrame(rows)
    first = stratified_subset_by_combination(
        frame,
        columns=["stratum_a", "stratum_b"],
        n_per_combination=25,
        random_state=42,
    )
    second = stratified_subset_by_combination(
        frame,
        columns=["stratum_a", "stratum_b"],
        n_per_combination=25,
        random_state=42,
    )
    pd.testing.assert_frame_equal(first, second)
