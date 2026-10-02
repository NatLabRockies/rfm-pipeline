"""Tests for the feature-expansion specification boundary."""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

from rfm_pipeline.feature_expansion import (
    apply_feature_expansion,
    default_feature_expansion_spec,
    ordered_expanded_feature_names,
)
from rfm_pipeline.transforms import INVERSE, QUADRATIC


def test_ordered_expanded_feature_names_follow_expected_catalog_order():
    spec = default_feature_expansion_spec(
        base_features=["x1", "x2"],
        scenario_flags=("FLAG_A", "FLAG_B"),
        add_transforms={"x1": [QUADRATIC], "x2": [INVERSE]},
        interaction_pairs=(("x1", "FLAG_A"), ("x1", "x2")),
    )
    assert ordered_expanded_feature_names(spec) == (
        "x1",
        "x2",
        "FLAG_A",
        "FLAG_B",
        "x1_sq",
        "x2_inv",
        "x1*FLAG_A",
        "x1*x2",
    )


def test_default_spec_rejects_transform_for_unknown_base_feature() -> None:
    with pytest.raises(ValueError, match="unknown base features.*typo"):
        default_feature_expansion_spec(
            base_features=["x1"],
            add_transforms={"typo": [QUADRATIC]},
        )


def test_apply_feature_expansion_materializes_expected_columns_and_values():
    frame = pd.DataFrame(
        {
            "x1": [2.0, 3.0],
            "x2": [4.0, 5.0],
            "cat_a": [0, 1],
            "cat_b": [1, 0],
        }
    )
    spec = default_feature_expansion_spec(
        base_features=["x1", "x2"],
        add_transforms={"x1": [QUADRATIC], "x2": [INVERSE]},
        interaction_pairs=(("x1", "cat_a"), ("x1", "x2")),
    )
    result = apply_feature_expansion(frame, spec)
    assert result.ordered_columns == ordered_expanded_feature_names(spec)
    assert result.expanded_frame.columns.tolist() == list(result.ordered_columns)
    assert result.expanded_frame["x1_sq"].tolist() == [4.0, 9.0]
    assert result.expanded_frame["x2_inv"].round(4).tolist() == [0.25, 0.2]
    assert result.expanded_frame["x1*cat_a"].tolist() == [0.0, 3.0]
    assert result.expanded_frame["x1*x2"].tolist() == [8.0, 15.0]


def test_apply_feature_expansion_requires_all_referenced_columns():
    frame = pd.DataFrame({"x1": [1.0], "cat_a": [0], "cat_b": [1]})
    spec = default_feature_expansion_spec(
        base_features=["x1", "x2"],
        interaction_pairs=(("x1", "x2"),),
    )
    with pytest.raises(KeyError, match="missing required feature columns"):
        apply_feature_expansion(frame, spec)


def test_apply_feature_expansion_inverse_of_zero_warns_and_returns_nan():
    frame = pd.DataFrame({"x1": [0.0], "cat_a": [0], "cat_b": [1]})
    spec = default_feature_expansion_spec(
        base_features=["x1"],
        add_transforms={"x1": [INVERSE]},
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = apply_feature_expansion(frame, spec)
    import math

    assert math.isnan(result.expanded_frame["x1_inv"].iloc[0])
    assert any("x1" in str(w.message) for w in caught)
