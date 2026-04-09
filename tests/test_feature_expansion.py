"""Tests for explicit feature-expansion contracts."""

from __future__ import annotations

import pandas as pd
import pytest

from bsm_rfm.feature_expansion import (
    FeatureExpansionSpec,
    apply_feature_expansion,
    expanded_feature_names,
    make_feature_expansion_spec,
)


def test_make_feature_expansion_spec_preserves_order_and_declared_terms():
    spec = make_feature_expansion_spec(
        ["x2", "x1", "x2"],
        add_quadratic_for=["x1"],
        add_inverse_for=["x2"],
        interaction_pairs=[("x1", "x2"), ("x1", "x2")],
    )
    assert spec.base_features == ("x2", "x1")
    assert spec.transforms == {"x2": ("inverse",), "x1": ("quadratic",)}
    assert spec.interaction_pairs == (("x1", "x2"),)


def test_make_feature_expansion_spec_rejects_unknown_transform_target():
    with pytest.raises(ValueError, match="unknown base feature"):
        make_feature_expansion_spec(["x1"], add_quadratic_for=["x2"])


def test_expanded_feature_names_follow_canonical_order():
    spec = make_feature_expansion_spec(
        ["x1", "x2"],
        add_quadratic_for=["x1"],
        add_inverse_for=["x2"],
        interaction_pairs=[("x1", "x2")],
    )
    assert expanded_feature_names(spec) == (
        "x1",
        "x2",
        "x1_quadratic",
        "x2_inverse",
        "[x1 * x2]",
        "AFSC",
        "UAEORO",
    )


def test_apply_feature_expansion_materializes_expected_columns_and_values():
    frame = pd.DataFrame(
        {
            "x1": [2.0, 3.0],
            "x2": [4.0, 5.0],
            "AFSC": [1, 0],
            "UAEORO": [0, 1],
        }
    )
    spec = make_feature_expansion_spec(
        ["x1", "x2"],
        add_quadratic_for=["x1"],
        add_inverse_for=["x2"],
        interaction_pairs=[("x1", "x2")],
    )
    result = apply_feature_expansion(frame, spec)
    assert result.feature_order == (
        "x1",
        "x2",
        "x1_quadratic",
        "x2_inverse",
        "[x1 * x2]",
        "AFSC",
        "UAEORO",
    )
    assert result.expanded_frame["x1_quadratic"].tolist() == [4.0, 9.0]
    assert result.expanded_frame["[x1 * x2]"].tolist() == [8.0, 15.0]
    assert result.expanded_frame["x2_inverse"].round(6).tolist() == [0.25, 0.2]
    assert set(result.feature_metadata["feature_type"]) == {
        "first_order",
        "nonlinear_transformation",
        "second_order",
        "scenario_flag",
    }


def test_apply_feature_expansion_rejects_missing_required_columns():
    spec = make_feature_expansion_spec(["x1"])
    frame = pd.DataFrame({"x1": [1.0], "AFSC": [1]})
    with pytest.raises(KeyError, match="UAEORO"):
        apply_feature_expansion(frame, spec)


def test_apply_feature_expansion_rejects_inverse_of_zero():
    frame = pd.DataFrame({"x1": [0.0], "AFSC": [1], "UAEORO": [0]})
    spec = make_feature_expansion_spec(["x1"], add_inverse_for=["x1"])
    with pytest.raises(ZeroDivisionError, match="inverse term"):
        apply_feature_expansion(frame, spec)


def test_apply_feature_expansion_rejects_unsupported_transform():
    frame = pd.DataFrame({"x1": [2.0], "AFSC": [1], "UAEORO": [0]})
    spec = FeatureExpansionSpec(base_features=("x1",), transforms={"x1": ("cube",)})
    with pytest.raises(ValueError, match="Unsupported transform"):
        apply_feature_expansion(frame, spec)
