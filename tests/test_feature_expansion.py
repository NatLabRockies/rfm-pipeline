"""Tests for the feature-expansion specification boundary."""

from __future__ import annotations

import pandas as pd
import pytest

from bsm_rfm.feature_expansion import (
    apply_feature_expansion,
    default_feature_expansion_spec,
    ordered_expanded_feature_names,
)


def test_ordered_expanded_feature_names_follow_expected_catalog_order():
    spec = default_feature_expansion_spec(
        base_features=["x1", "x2"],
        add_quadratic_for=("x1",),
        add_inverse_for=("x2",),
        interaction_pairs=(("x1", "AFSC"), ("x1", "x2")),
    )
    assert ordered_expanded_feature_names(spec) == (
        "x1",
        "x2",
        "AFSC",
        "UAEORO",
        "x1_quadratic",
        "x2_inverse",
        "x1*AFSC",
        "x1*x2",
    )


def test_apply_feature_expansion_materializes_expected_columns_and_values():
    frame = pd.DataFrame(
        {
            "x1": [2.0, 3.0],
            "x2": [4.0, 5.0],
            "AFSC": [0, 1],
            "UAEORO": [1, 0],
        }
    )
    spec = default_feature_expansion_spec(
        base_features=["x1", "x2"],
        add_quadratic_for=("x1",),
        add_inverse_for=("x2",),
        interaction_pairs=(("x1", "AFSC"), ("x1", "x2")),
    )
    result = apply_feature_expansion(frame, spec)
    assert result.ordered_columns == ordered_expanded_feature_names(spec)
    assert result.expanded_frame.columns.tolist() == list(result.ordered_columns)
    assert result.expanded_frame["x1_quadratic"].tolist() == [4.0, 9.0]
    assert result.expanded_frame["x2_inverse"].round(4).tolist() == [0.25, 0.2]
    assert result.expanded_frame["x1*AFSC"].tolist() == [0.0, 3.0]
    assert result.expanded_frame["x1*x2"].tolist() == [8.0, 15.0]


def test_apply_feature_expansion_requires_all_referenced_columns():
    frame = pd.DataFrame({"x1": [1.0], "AFSC": [0], "UAEORO": [1]})
    spec = default_feature_expansion_spec(
        base_features=["x1", "x2"],
        interaction_pairs=(("x1", "x2"),),
    )
    with pytest.raises(KeyError, match="missing required feature columns"):
        apply_feature_expansion(frame, spec)


def test_apply_feature_expansion_rejects_inverse_of_zero():
    frame = pd.DataFrame({"x1": [0.0], "AFSC": [0], "UAEORO": [1]})
    spec = default_feature_expansion_spec(
        base_features=["x1"],
        add_inverse_for=("x1",),
    )
    with pytest.raises(ValueError, match="inverse-transformed"):
        apply_feature_expansion(frame, spec)
