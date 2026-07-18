"""P0-S02: Design matrix includes categorical main effects + optional interactions.

Tests:
- categorical main-effect columns appear in the design matrix
- requested categorical×scalar interaction columns appear
- empty-declaration (no categoricals) output is byte-for-byte identical to input
- column names are stable and introspectable
- drop_first=False emits all levels; drop_first=True (default) drops reference
- KeyError on missing column; ValueError on scalar-also-declared-categorical
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rfm_pipeline.config import CategoricalInputDecl
from rfm_pipeline.features import (
    DesignMatrixSpec,
    build_design_matrix,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_X() -> pd.DataFrame:
    """Small synthetic X: one scalar + one binary block column."""
    rng = np.random.default_rng(42)
    n = 12
    return pd.DataFrame(
        {
            "score": rng.standard_normal(n),
            "block": (["A"] * 6) + (["B"] * 6),
        }
    )


@pytest.fixture()
def multi_level_X() -> pd.DataFrame:
    """Synthetic X with scalar and a 3-level categorical."""
    rng = np.random.default_rng(7)
    n = 15
    return pd.DataFrame(
        {
            "value": rng.standard_normal(n),
            "group": (["low"] * 5) + (["mid"] * 5) + (["high"] * 5),
        }
    )


# ---------------------------------------------------------------------------
# Empty-declaration parity with baseline (regression guard)
# ---------------------------------------------------------------------------


class TestEmptyDeclarationParity:
    def test_P0_S02_no_categoricals_returns_identical_frame(self, simple_X):
        result = build_design_matrix(simple_X)
        pd.testing.assert_frame_equal(result.matrix, simple_X)

    def test_P0_S02_empty_spec_returns_identical_frame(self, simple_X):
        spec = DesignMatrixSpec()
        result = build_design_matrix(simple_X, spec)
        pd.testing.assert_frame_equal(result.matrix, simple_X)

    def test_P0_S02_empty_spec_column_order_preserved(self, simple_X):
        result = build_design_matrix(simple_X)
        assert result.ordered_columns == tuple(simple_X.columns)

    def test_P0_S02_empty_spec_no_categorical_or_interaction_columns(self, simple_X):
        result = build_design_matrix(simple_X)
        assert result.categorical_columns == ()
        assert result.interaction_columns == ()

    def test_P0_S02_empty_spec_dtypes_preserved(self, simple_X):
        result = build_design_matrix(simple_X)
        for col in simple_X.columns:
            assert result.matrix[col].dtype == simple_X[col].dtype


# ---------------------------------------------------------------------------
# Categorical main-effect columns
# ---------------------------------------------------------------------------


class TestCategoricalMainEffects:
    def test_P0_S02_binary_categorical_produces_indicator_column(self, simple_X):
        decl = CategoricalInputDecl(name="block")
        spec = DesignMatrixSpec(categorical_inputs=(decl,))
        result = build_design_matrix(simple_X, spec)
        # drop_first=True (default): 2 levels → 1 indicator
        assert len(result.categorical_columns) == 1
        assert result.categorical_columns[0].startswith("block__")

    def test_P0_S02_indicator_column_values_are_zero_or_one(self, simple_X):
        decl = CategoricalInputDecl(name="block")
        spec = DesignMatrixSpec(categorical_inputs=(decl,))
        result = build_design_matrix(simple_X, spec)
        for col in result.categorical_columns:
            assert set(result.matrix[col].unique()).issubset({0, 1})

    def test_P0_S02_scalar_column_still_present(self, simple_X):
        decl = CategoricalInputDecl(name="block")
        spec = DesignMatrixSpec(categorical_inputs=(decl,))
        result = build_design_matrix(simple_X, spec)
        assert "score" in result.matrix.columns

    def test_P0_S02_original_categorical_column_dropped(self, simple_X):
        decl = CategoricalInputDecl(name="block")
        spec = DesignMatrixSpec(categorical_inputs=(decl,))
        result = build_design_matrix(simple_X, spec)
        # Raw "block" column replaced by encoded indicators
        assert "block" not in result.matrix.columns

    def test_P0_S02_drop_first_false_emits_all_levels(self, simple_X):
        decl = CategoricalInputDecl(name="block")
        spec = DesignMatrixSpec(categorical_inputs=(decl,), drop_first=False)
        result = build_design_matrix(simple_X, spec)
        # 2 levels → 2 indicator columns
        assert len(result.categorical_columns) == 2

    def test_P0_S02_three_level_categorical_drop_first(self, multi_level_X):
        decl = CategoricalInputDecl(name="group")
        spec = DesignMatrixSpec(categorical_inputs=(decl,))
        result = build_design_matrix(multi_level_X, spec)
        # 3 levels → 2 indicator columns (drop_first=True)
        assert len(result.categorical_columns) == 2

    def test_P0_S02_three_level_categorical_all_levels(self, multi_level_X):
        decl = CategoricalInputDecl(name="group")
        spec = DesignMatrixSpec(categorical_inputs=(decl,), drop_first=False)
        result = build_design_matrix(multi_level_X, spec)
        assert len(result.categorical_columns) == 3

    def test_P0_S02_explicit_levels_determine_column_names(self, simple_X):
        decl = CategoricalInputDecl(name="block", levels=["A", "B"])
        spec = DesignMatrixSpec(categorical_inputs=(decl,), drop_first=False)
        result = build_design_matrix(simple_X, spec)
        assert "block__A" in result.matrix.columns
        assert "block__B" in result.matrix.columns

    def test_P0_S02_column_names_are_stable(self, simple_X):
        """Same spec applied twice yields the same column order."""
        decl = CategoricalInputDecl(name="block")
        spec = DesignMatrixSpec(categorical_inputs=(decl,))
        r1 = build_design_matrix(simple_X, spec)
        r2 = build_design_matrix(simple_X, spec)
        assert r1.ordered_columns == r2.ordered_columns

    def test_P0_S02_indicator_dtype_is_uint8(self, simple_X):
        decl = CategoricalInputDecl(name="block")
        spec = DesignMatrixSpec(categorical_inputs=(decl,), drop_first=False)
        result = build_design_matrix(simple_X, spec)
        for col in result.categorical_columns:
            assert result.matrix[col].dtype == "uint8"

    def test_P0_S02_ordered_columns_matches_matrix_columns(self, simple_X):
        decl = CategoricalInputDecl(name="block")
        spec = DesignMatrixSpec(categorical_inputs=(decl,))
        result = build_design_matrix(simple_X, spec)
        assert result.ordered_columns == tuple(result.matrix.columns)


# ---------------------------------------------------------------------------
# Categorical × scalar interaction columns
# ---------------------------------------------------------------------------


class TestInteractionColumns:
    def test_P0_S02_interaction_columns_appear(self, simple_X):
        decl = CategoricalInputDecl(name="block")
        spec = DesignMatrixSpec(
            categorical_inputs=(decl,),
            interaction_pairs=(("block", "score"),),
        )
        result = build_design_matrix(simple_X, spec)
        assert len(result.interaction_columns) > 0

    def test_P0_S02_interaction_column_name_contains_both_parts(self, simple_X):
        decl = CategoricalInputDecl(name="block")
        spec = DesignMatrixSpec(
            categorical_inputs=(decl,),
            interaction_pairs=(("block", "score"),),
        )
        result = build_design_matrix(simple_X, spec)
        for col in result.interaction_columns:
            assert "block" in col
            assert "score" in col

    def test_P0_S02_interaction_column_values_are_float(self, simple_X):
        decl = CategoricalInputDecl(name="block")
        spec = DesignMatrixSpec(
            categorical_inputs=(decl,),
            interaction_pairs=(("block", "score"),),
        )
        result = build_design_matrix(simple_X, spec)
        for col in result.interaction_columns:
            assert result.matrix[col].dtype == np.float64

    def test_P0_S02_interaction_values_correct(self, simple_X):
        """Interaction column = indicator × scalar."""
        decl = CategoricalInputDecl(name="block", levels=["A", "B"])
        spec = DesignMatrixSpec(
            categorical_inputs=(decl,),
            interaction_pairs=(("block", "score"),),
            drop_first=False,
        )
        result = build_design_matrix(simple_X, spec)
        # The block__B x score column should equal block__B * score
        ix_col = "block__B_x_score"
        expected = (simple_X["block"] == "B").astype(float) * simple_X["score"]
        pd.testing.assert_series_equal(
            result.matrix[ix_col].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_P0_S02_no_interaction_pairs_no_interaction_columns(self, simple_X):
        decl = CategoricalInputDecl(name="block")
        spec = DesignMatrixSpec(categorical_inputs=(decl,))
        result = build_design_matrix(simple_X, spec)
        assert result.interaction_columns == ()

    def test_P0_S02_interaction_columns_in_ordered_columns(self, simple_X):
        decl = CategoricalInputDecl(name="block")
        spec = DesignMatrixSpec(
            categorical_inputs=(decl,),
            interaction_pairs=(("block", "score"),),
        )
        result = build_design_matrix(simple_X, spec)
        for col in result.interaction_columns:
            assert col in result.ordered_columns


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_P0_S02_missing_categorical_column_raises_key_error(self, simple_X):
        decl = CategoricalInputDecl(name="nonexistent")
        spec = DesignMatrixSpec(categorical_inputs=(decl,))
        with pytest.raises(KeyError, match="nonexistent"):
            build_design_matrix(simple_X, spec)

    def test_P0_S02_missing_scalar_column_in_interaction_raises_key_error(self, simple_X):
        decl = CategoricalInputDecl(name="block")
        spec = DesignMatrixSpec(
            categorical_inputs=(decl,),
            interaction_pairs=(("block", "missing_scalar"),),
        )
        with pytest.raises(KeyError, match="missing_scalar"):
            build_design_matrix(simple_X, spec)

    def test_P0_S02_scalar_also_categorical_raises_value_error(self, simple_X):
        decl_block = CategoricalInputDecl(name="block")
        decl_score = CategoricalInputDecl(name="score")
        spec = DesignMatrixSpec(
            categorical_inputs=(decl_block, decl_score),
            interaction_pairs=(("block", "score"),),
        )
        with pytest.raises(ValueError, match="score"):
            build_design_matrix(simple_X, spec)
