"""Unit tests for TransformDef and the default transform library."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from rfm_pipeline.transforms import (
    DEFAULT_TRANSFORM_LIBRARY,
    EXPONENTIAL,
    INVERSE,
    LOGARITHMIC,
    QUADRATIC,
    SQRT,
    TransformDef,
    warn_nan_transforms,
)

# ---------------------------------------------------------------------------
# TransformDef construction
# ---------------------------------------------------------------------------


def test_transform_def_column_name():
    assert QUADRATIC.column_name("height") == "height_sq"
    assert LOGARITHMIC.column_name("income") == "income_log1p"
    assert INVERSE.column_name("price") == "price_inv"


def test_transform_def_from_config_roundtrip():
    original = TransformDef(expr="x**3", label="cube", name="cubic")
    config = original.to_config()
    restored = TransformDef.from_config(config)
    assert restored == original


def test_transform_def_from_config_minimal():
    td = TransformDef.from_config({"expr": "x**2", "label": "sq"})
    assert td.expr == "x**2"
    assert td.label == "sq"
    assert td.name is None


def test_transform_def_requires_expr_and_label():
    with pytest.raises((KeyError, TypeError)):
        TransformDef.from_config({"expr": "x**2"})


# ---------------------------------------------------------------------------
# TransformDef.apply — valid domain
# ---------------------------------------------------------------------------


def test_quadratic_apply():
    x = np.array([2.0, 3.0, 4.0])
    result = QUADRATIC.apply(x)
    np.testing.assert_allclose(result, [4.0, 9.0, 16.0])


def test_logarithmic_apply():
    x = np.array([0.0, 1.0, math.e - 1])
    result = LOGARITHMIC.apply(x)
    assert result[0] == pytest.approx(0.0)
    assert result[1] == pytest.approx(math.log(2))
    assert result[2] == pytest.approx(1.0)


def test_inverse_apply():
    x = np.array([2.0, 4.0])
    result = INVERSE.apply(x)
    np.testing.assert_allclose(result, [0.5, 0.25])


def test_sqrt_apply():
    x = np.array([0.0, 4.0, 9.0])
    result = SQRT.apply(x)
    np.testing.assert_allclose(result, [0.0, 2.0, 3.0])


def test_exponential_apply():
    x = np.array([0.0, 1.0])
    result = EXPONENTIAL.apply(x)
    np.testing.assert_allclose(result, [1.0, math.e])


# ---------------------------------------------------------------------------
# TransformDef.apply — invalid domain → NaN (no exception)
# ---------------------------------------------------------------------------


def test_inverse_of_zero_returns_nan():
    result = INVERSE.apply(np.array([0.0, 1.0]))
    assert math.isnan(result[0])
    assert result[1] == pytest.approx(1.0)


def test_sqrt_of_negative_returns_nan():
    result = SQRT.apply(np.array([-1.0, 4.0]))
    assert math.isnan(result[0])
    assert result[1] == pytest.approx(2.0)


def test_log_of_negative_returns_nan():
    result = LOGARITHMIC.apply(np.array([-2.0, 0.0]))
    assert math.isnan(result[0])
    assert result[1] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TransformDef.valid_fraction
# ---------------------------------------------------------------------------


def test_valid_fraction_all_valid():
    assert QUADRATIC.valid_fraction(np.array([1.0, 2.0, 3.0])) == pytest.approx(1.0)


def test_valid_fraction_partial():
    frac = INVERSE.valid_fraction(np.array([0.0, 1.0, 2.0]))
    assert frac == pytest.approx(2 / 3)


def test_valid_fraction_all_invalid():
    assert SQRT.valid_fraction(np.array([-1.0, -2.0])) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# warn_nan_transforms
# ---------------------------------------------------------------------------


def test_warn_nan_transforms_emits_warnings():
    nan_report = {"height_inv": ["a", "b", "c"]}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_nan_transforms(nan_report)
    messages = [str(w.message) for w in caught]
    assert any("height_inv" in m for m in messages)


def test_warn_nan_transforms_skips_empty_feature_list():
    nan_report = {"height_inv": [], "income_inv": []}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_nan_transforms(nan_report)
    assert len(caught) == 0


# ---------------------------------------------------------------------------
# DEFAULT_TRANSFORM_LIBRARY
# ---------------------------------------------------------------------------


def test_default_transform_library_has_four_entries():
    # The exponential transform is available explicitly but omitted by default.
    assert len(DEFAULT_TRANSFORM_LIBRARY) == 4


def test_default_transform_library_labels():
    labels = {td.label for td in DEFAULT_TRANSFORM_LIBRARY}
    assert labels == {"sq", "log1p", "inv", "sqrt"}


# ---------------------------------------------------------------------------
# Custom SymPy expression
# ---------------------------------------------------------------------------


def test_custom_sympy_expression():
    cubic = TransformDef(expr="x**3", label="cube")
    result = cubic.apply(np.array([2.0, -1.0]))
    np.testing.assert_allclose(result, [8.0, -1.0])
