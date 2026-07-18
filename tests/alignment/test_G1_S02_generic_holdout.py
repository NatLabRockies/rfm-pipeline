"""Alignment tests for G1-S02: generic stratify_columns in stratified_holdout_split."""

from __future__ import annotations

import numpy as np
import pandas as pd

from rfm_pipeline.data import stratified_holdout_split


def _make_xy(
    n_per_stratum: int = 5,
    *,
    col_a: str = "cat_a",
    col_b: str = "cat_b",
    random_state: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthetic X/Y with generic categorical stratification columns."""
    rng = np.random.default_rng(random_state)
    rows_x = []
    rows_y = []
    run_id = 0
    for a in [0, 1]:
        for b in [0, 1]:
            for _ in range(n_per_stratum):
                rows_x.append(
                    {
                        "run_id": run_id,
                        col_a: a,
                        col_b: b,
                        "feat1": rng.standard_normal(),
                        "feat2": rng.standard_normal(),
                    }
                )
                rows_y.append({"run_id": run_id, "out1": rng.standard_normal()})
                run_id += 1
    X = pd.DataFrame(rows_x).set_index("run_id")
    Y = pd.DataFrame(rows_y).set_index("run_id")
    return X, Y


# ---------------------------------------------------------------------------
# G1-S02 acceptance criteria
# ---------------------------------------------------------------------------


def test_G1_S02_generic_stratify_columns_deterministic():
    """stratify_columns on arbitrary column names produces a deterministic split."""
    X, Y = _make_xy()
    X_train_a, X_h_a, Y_train_a, Y_h_a = stratified_holdout_split(
        X, Y, holdout_fraction=0.25, random_state=42, stratify_columns=["cat_a", "cat_b"]
    )
    X_train_b, X_h_b, Y_train_b, Y_h_b = stratified_holdout_split(
        X, Y, holdout_fraction=0.25, random_state=42, stratify_columns=["cat_a", "cat_b"]
    )
    assert X_train_a.index.tolist() == X_train_b.index.tolist()
    assert X_h_a.index.tolist() == X_h_b.index.tolist()


def test_G1_S02_all_strata_represented_in_holdout():
    """All four strata appear in the holdout when stratify_columns are provided."""
    X, Y = _make_xy(n_per_stratum=5)
    X_train, X_holdout, Y_train, Y_holdout = stratified_holdout_split(
        X, Y, holdout_fraction=0.25, random_state=99, stratify_columns=["cat_a", "cat_b"]
    )
    strata_in_holdout = set(
        (X_holdout["cat_a"].astype(str) + "_" + X_holdout["cat_b"].astype(str)).tolist()
    )
    assert strata_in_holdout == {"0_0", "0_1", "1_0", "1_1"}


def test_G1_S02_stratum_proportions_preserved():
    """Holdout stratum proportions are close to the full-data proportions."""
    X, Y = _make_xy(n_per_stratum=50)
    X_train, X_holdout, _, _ = stratified_holdout_split(
        X, Y, holdout_fraction=0.20, random_state=7, stratify_columns=["cat_a", "cat_b"]
    )
    full_label = X["cat_a"].astype(str) + "_" + X["cat_b"].astype(str)
    holdout_label = X_holdout["cat_a"].astype(str) + "_" + X_holdout["cat_b"].astype(str)
    full_props = full_label.value_counts(normalize=True).sort_index()
    holdout_props = holdout_label.value_counts(normalize=True).sort_index()
    for stratum in full_props.index:
        assert abs(full_props[stratum] - holdout_props.get(stratum, 0.0)) < 0.05


def test_G1_S02_scenario_column_fallback_when_stratify_columns_empty():
    """When stratify_columns=() the function falls back to scenario_column."""
    X, Y = _make_xy()
    X = X.assign(scenario=X["cat_a"].astype(str))
    X_train, X_holdout, Y_train, Y_holdout = stratified_holdout_split(
        X, Y, holdout_fraction=0.25, random_state=42, scenario_column="scenario"
    )
    # All scenario values should appear in holdout
    assert set(X_holdout["scenario"].unique()) == {"0", "1"}
    assert X_train.index.equals(Y_train.index)
    assert X_holdout.index.equals(Y_holdout.index)


def test_G1_S02_scenario_column_fallback_when_columns_absent():
    """When stratify_columns names are not in X, fall back to scenario_column."""
    X, Y = _make_xy()
    X = X.assign(scenario=X["cat_a"].astype(str))
    # Request columns that don't exist → fallback
    X_train, X_holdout, _, _ = stratified_holdout_split(
        X,
        Y,
        holdout_fraction=0.25,
        random_state=42,
        scenario_column="scenario",
        stratify_columns=["nonexistent_col"],
    )
    assert len(X_train) + len(X_holdout) == len(X)


def test_G1_S02_index_alignment_preserved():
    """X and Y share the same index after the split."""
    X, Y = _make_xy(n_per_stratum=10)
    X_train, X_holdout, Y_train, Y_holdout = stratified_holdout_split(
        X, Y, holdout_fraction=0.20, random_state=3, stratify_columns=["cat_a", "cat_b"]
    )
    assert X_train.index.equals(Y_train.index)
    assert X_holdout.index.equals(Y_holdout.index)


def test_G1_S02_no_afsc_uaeoro_in_signature():
    """stratified_holdout_split no longer accepts afsc_column or uaeoro_column."""
    import inspect

    sig = inspect.signature(stratified_holdout_split)
    assert "afsc_column" not in sig.parameters
    assert "uaeoro_column" not in sig.parameters
