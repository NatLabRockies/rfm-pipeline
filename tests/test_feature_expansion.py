"""Tests for explicit notebook-derived feature expansion."""

from __future__ import annotations

import pandas as pd
import pytest

from bsm_rfm.feature_expansion import (
    apply_feature_expansion,
    feature_catalog_from_spec,
    make_feature_expansion_spec,
)


def test_make_feature_expansion_spec_adds_scenario_flag_interactions_without_duplicates():
    spec = make_feature_expansion_spec(
        ["x1", "x2"],
        add_quadratic_for=["x1"],
        add_inverse_for=["x2"],
        interaction_pairs=[("x1", "x2"), ("x2", "x1")],
        interaction_with_scenario_flags=True,
    )
    assert spec.base_features == ("x1", "x2", "AFSC", "UAEORO")
    assert spec.transforms == {"x1": ("quadratic",), "x2": ("inverse",)}
    assert spec.interactions == (
        ("x1", "x2"),
        ("AFSC", "x1"),
        ("UAEORO", "x1"),
        ("AFSC", "x2"),
        ("UAEORO", "x2"),
    )


def test_feature_catalog_from_spec_labels_terms_as_notebook_derived():
    spec = make_feature_expansion_spec(
        ["feedstock"],
        add_quadratic_for=["feedstock"],
        interaction_with_scenario_flags=False,
    )
    catalog = feature_catalog_from_spec(spec)
    assert catalog["name"].tolist() == ["feedstock", "AFSC", "UAEORO", "feedstock_quadratic"]
    assert set(catalog["provenance"]) == {"notebook-derived"}
    assert catalog.iloc[-1]["term_type"] == "nonlinear_transformation"


def test_apply_feature_expansion_materializes_first_order_transform_and_interaction_terms():
    frame = pd.DataFrame(
        {
            "x1": [2.0, 4.0],
            "x2": [5.0, 10.0],
            "AFSC": [0, 1],
            "UAEORO": [1, 0],
        }
    )
    spec = make_feature_expansion_spec(
        ["x1", "x2"],
        add_quadratic_for=["x1"],
        add_inverse_for=["x2"],
        interaction_pairs=[("x1", "x2")],
        interaction_with_scenario_flags=False,
    )
    result = apply_feature_expansion(frame, spec)
    assert result.expanded_frame.columns.tolist() == [
        "x1",
        "x2",
        "AFSC",
        "UAEORO",
        "x1_quadratic",
        "x2_inverse",
        "x1*x2",
    ]
    assert result.expanded_frame["x1_quadratic"].tolist() == [4.0, 16.0]
    assert result.expanded_frame["x2_inverse"].tolist() == [0.2, 0.1]
    assert result.expanded_frame["x1*x2"].tolist() == [10.0, 40.0]
    assert result.catalog["name"].tolist() == result.expanded_frame.columns.tolist()


def test_apply_feature_expansion_rejects_zero_for_inverse_transform():
    frame = pd.DataFrame({"x1": [0.0], "AFSC": [0], "UAEORO": [1]})
    spec = make_feature_expansion_spec(
        ["x1"],
        add_inverse_for=["x1"],
        interaction_with_scenario_flags=False,
    )
    with pytest.raises(ValueError, match="undefined for zero-valued inputs"):
        apply_feature_expansion(frame, spec)


def test_make_feature_expansion_spec_rejects_unknown_interaction_columns():
    with pytest.raises(ValueError, match="references unknown base features"):
        make_feature_expansion_spec(["x1"], interaction_pairs=[("x1", "x2")])
