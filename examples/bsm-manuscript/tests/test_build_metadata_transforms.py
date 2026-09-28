"""Focused tests for build_metadata transform-suffix classification.

The production pipeline emits nonlinear transforms as *suffixes*
(`x_sqrt`, `x_sq`, `x_log1p`, `x_inv`), while an earlier convention used
prefixes (`sqrt_x`). ``cross_reference_inputs`` / ``_strip_transform`` must
recognize both so every coefficient column maps back to a real base input
instead of being mislabeled ``identity`` and self-referencing.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_metadata.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("_build_metadata", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_build_metadata"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load_module()


@pytest.mark.parametrize(
    "fname,expected_base",
    [
        ("Input A_sqrt", "Input A"),
        ("Input A_sq", "Input A"),
        ("Input A_log1p", "Input A"),
        ("Input A_inv", "Input A"),
        ("Input A_log", "Input A"),
        ("Input A_squared", "Input A"),
        # prefix convention still handled
        ("sqrt_Input A", "Input A"),
        ("inverse_Input A", "Input A"),
        # untransformed identity passes through unchanged
        ("Input A", "Input A"),
    ],
)
def test_strip_transform_handles_suffix_and_prefix(mod, fname, expected_base):
    assert mod._strip_transform(fname) == expected_base


def test_cross_reference_classifies_suffix_transforms(mod):
    inputs = [
        {
            "name": "Input A",
            "units": "unitless",
            "min_sample_value": 0.0,
            "max_sample_value": 1.0,
            "pathway_impacted": "P1",
        },
        {
            "name": "Input B",
            "units": "unitless",
            "min_sample_value": 0.0,
            "max_sample_value": 1.0,
            "pathway_impacted": "P2",
        },
    ]
    coef = pd.DataFrame(
        [
            {"final_support_position": 0, "feature_name": "Input A", "feature_type": "numeric"},
            {"final_support_position": 1, "feature_name": "Input A_sqrt", "feature_type": "transformation"},
            {"final_support_position": 2, "feature_name": "Input A_sq", "feature_type": "transformation"},
            {"final_support_position": 3, "feature_name": "Input B_log1p", "feature_type": "transformation"},
            {"final_support_position": 4, "feature_name": "Input B_inv", "feature_type": "transformation"},
            {"final_support_position": 5, "feature_name": "Input A_sqrt:Input B", "feature_type": "interaction"},
        ]
    )
    coef["n_outputs_excluding_zero"] = 1
    coef["stability_selection_frequency"] = 1.0
    coef["hc3_retained_after_filter"] = True
    coef["delta_nrmse_when_feature_removed"] = 0.0
    out = mod.cross_reference_inputs(coef, inputs).set_index("feature_name")

    assert out.loc["Input A", "transformation"] == "identity"
    assert out.loc["Input A_sqrt", "transformation"] == "sqrt"
    assert out.loc["Input A_sq", "transformation"] == "quadratic"
    assert out.loc["Input B_log1p", "transformation"] == "log1p"
    assert out.loc["Input B_inv", "transformation"] == "inverse"

    # transformed features resolve to their real base input, never self-reference
    assert out.loc["Input A_sqrt", "base_input_1"] == "Input A"
    assert out.loc["Input A_sq", "base_input_1"] == "Input A"
    assert out.loc["Input B_log1p", "base_input_1"] == "Input B"
    assert out.loc["Input B_inv", "base_input_1"] == "Input B"

    # interaction side carrying a suffix strips back to the base input
    row = out.loc["Input A_sqrt:Input B"]
    assert row["transformation"] == "interaction"
    assert row["base_input_1"] == "Input A"
    assert row["base_input_2"] == "Input B"
