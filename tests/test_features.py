"""Tests for test features."""

from __future__ import annotations

from rfm_pipeline.features import parse_selected_input_structure


def test_parse_selected_input_structure_classifies_recovered_name_patterns():
    features = [
        "AFSC",
        "WW.progress ratios commercial[SludgeToHTL]*AFSC",
        "OHC.Retirement Frac[TransEster]_quadratic",
    ]
    structure, type_summary, transformation_summary = parse_selected_input_structure(features)
    assert structure.loc[0, "input_type"] == "first_order"
    assert structure.loc[1, "input_type"] == "second_order"
    assert structure.loc[2, "input_type"] == "nonlinear_transformation"
    assert set(type_summary["input_type"]) == {
        "first_order",
        "second_order",
        "nonlinear_transformation",
    }
    assert transformation_summary.iloc[0]["transformation"] == "quadratic"
