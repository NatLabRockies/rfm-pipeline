"""P0-S03: Exported evaluator honors categorical inputs — scenario-contrast test.

Tests:
- Fitting with a non-zero categorical main effect: changing only the categorical
  value in a record changes the prediction (scenario-contrast).
- Fitting with a zero categorical main effect: predictions are invariant to the
  categorical value.
- Export artifact records the categorical columns/levels used.
- FinalOLSFitResult stores the design spec and categorical feature columns.
- predict_from_raw_records falls back gracefully when no design spec is stored.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rfm_pipeline.config import CategoricalInputDecl
from rfm_pipeline.features import DesignMatrixSpec
from rfm_pipeline.final_ols import (
    FinalOLSFitResult,
    build_postfit_artifacts,
    fit_final_ols,
    fit_final_ols_with_design,
    predict_from_raw_records,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def nonzero_block_fit() -> FinalOLSFitResult:
    """Synthetic fit where the block column has a known non-zero coefficient.

    Y = 2 * score + 5 * (block == "B") + small_noise
    """
    rng = np.random.default_rng(42)
    n = 60
    block = ["A"] * 30 + ["B"] * 30
    score = rng.standard_normal(n)
    y = 2.0 * score + 5.0 * (np.array(block) == "B").astype(float) + rng.standard_normal(n) * 0.05
    X = pd.DataFrame({"score": score, "block": block})
    Y = pd.DataFrame({"outcome": y})

    decl = CategoricalInputDecl(name="block", levels=["A", "B"])
    spec = DesignMatrixSpec(categorical_inputs=(decl,), drop_first=False)
    return fit_final_ols_with_design(X, Y, spec)


@pytest.fixture()
def zero_block_fit() -> FinalOLSFitResult:
    """Synthetic fit where the block column has a coefficient of exactly zero.

    Y = 3 * score with no noise; block is uncorrelated with the residual
    so OLS assigns it a coefficient of zero.
    """
    # Identical score values for both blocks → block indicator orthogonal to score.
    scores_half = np.linspace(-1.0, 1.0, 30)
    score = np.concatenate([scores_half, scores_half])
    block = ["A"] * 30 + ["B"] * 30
    y = 3.0 * score  # no noise; block has zero effect by construction
    X = pd.DataFrame({"score": score, "block": block})
    Y = pd.DataFrame({"outcome": y})

    decl = CategoricalInputDecl(name="block", levels=["A", "B"])
    spec = DesignMatrixSpec(categorical_inputs=(decl,), drop_first=False)
    return fit_final_ols_with_design(X, Y, spec)


# ---------------------------------------------------------------------------
# Scenario-contrast: non-zero block effect
# ---------------------------------------------------------------------------


class TestNonzeroBlockScenarioContrast:
    def test_P0_S03_changing_block_changes_prediction(self, nonzero_block_fit):
        """Flipping the block column must change the prediction."""
        rec_a = pd.DataFrame({"score": [0.0], "block": ["A"]})
        rec_b = pd.DataFrame({"score": [0.0], "block": ["B"]})
        pred_a = predict_from_raw_records(nonzero_block_fit, rec_a)["outcome"].iloc[0]
        pred_b = predict_from_raw_records(nonzero_block_fit, rec_b)["outcome"].iloc[0]
        assert pred_a != pred_b, "Non-zero block effect must change prediction."

    def test_P0_S03_block_contrast_magnitude_is_large(self, nonzero_block_fit):
        """Contrast magnitude should be close to the planted 5-unit effect."""
        rec_a = pd.DataFrame({"score": [0.0], "block": ["A"]})
        rec_b = pd.DataFrame({"score": [0.0], "block": ["B"]})
        pred_a = predict_from_raw_records(nonzero_block_fit, rec_a)["outcome"].iloc[0]
        pred_b = predict_from_raw_records(nonzero_block_fit, rec_b)["outcome"].iloc[0]
        contrast = abs(pred_b - pred_a)
        assert contrast > 1.0, f"Expected large contrast from planted effect; got {contrast:.4f}"

    def test_P0_S03_scalar_held_fixed_only_categorical_varies(self, nonzero_block_fit):
        """Predictions differ across multiple scalar values when only block varies."""
        rng = np.random.default_rng(7)
        scores = rng.standard_normal(5)
        for s in scores:
            rec_a = pd.DataFrame({"score": [s], "block": ["A"]})
            rec_b = pd.DataFrame({"score": [s], "block": ["B"]})
            p_a = predict_from_raw_records(nonzero_block_fit, rec_a)["outcome"].iloc[0]
            p_b = predict_from_raw_records(nonzero_block_fit, rec_b)["outcome"].iloc[0]
            assert p_a != p_b


# ---------------------------------------------------------------------------
# Scenario-contrast: zero block effect → predictions invariant
# ---------------------------------------------------------------------------


class TestZeroBlockPredictionInvariant:
    def test_P0_S03_zero_block_predictions_are_equal(self, zero_block_fit):
        """With a zero block coefficient, predictions must not change with block."""
        rec_a = pd.DataFrame({"score": [1.0], "block": ["A"]})
        rec_b = pd.DataFrame({"score": [1.0], "block": ["B"]})
        pred_a = predict_from_raw_records(zero_block_fit, rec_a)["outcome"].iloc[0]
        pred_b = predict_from_raw_records(zero_block_fit, rec_b)["outcome"].iloc[0]
        assert abs(pred_a - pred_b) < 1e-9, (
            f"Zero-effect block must not affect predictions; contrast={pred_a - pred_b:.2e}"
        )

    def test_P0_S03_zero_block_multiple_scores_invariant(self, zero_block_fit):
        """Invariance holds across multiple scalar values."""
        rng = np.random.default_rng(13)
        scores = rng.standard_normal(5)
        for s in scores:
            rec_a = pd.DataFrame({"score": [s], "block": ["A"]})
            rec_b = pd.DataFrame({"score": [s], "block": ["B"]})
            p_a = predict_from_raw_records(zero_block_fit, rec_a)["outcome"].iloc[0]
            p_b = predict_from_raw_records(zero_block_fit, rec_b)["outcome"].iloc[0]
            assert abs(p_a - p_b) < 1e-9, (
                f"Prediction changed for score={s:.3f}: diff={p_a - p_b:.2e}"
            )


# ---------------------------------------------------------------------------
# FinalOLSFitResult stores design spec and categorical columns
# ---------------------------------------------------------------------------


class TestFitResultStoresDesignMetadata:
    def test_P0_S03_fit_result_has_design_spec(self, nonzero_block_fit):
        assert nonzero_block_fit.design_spec is not None

    def test_P0_S03_fit_result_design_spec_matches_input(self, nonzero_block_fit):
        assert len(nonzero_block_fit.design_spec.categorical_inputs) == 1
        assert nonzero_block_fit.design_spec.categorical_inputs[0].name == "block"

    def test_P0_S03_fit_result_has_categorical_feature_columns(self, nonzero_block_fit):
        assert len(nonzero_block_fit.categorical_feature_columns) > 0

    def test_P0_S03_categorical_feature_columns_prefixed_correctly(self, nonzero_block_fit):
        for col in nonzero_block_fit.categorical_feature_columns:
            assert col.startswith("block__"), f"Unexpected column name: {col}"

    def test_P0_S03_no_design_spec_on_plain_fit_result(self):
        """Plain fit_final_ols returns a result with no design_spec."""
        rng = np.random.default_rng(0)
        X = pd.DataFrame({"x": rng.standard_normal(20)})
        Y = pd.DataFrame({"y": rng.standard_normal(20)})
        result = fit_final_ols(X, Y)
        assert result.design_spec is None
        assert result.categorical_feature_columns == ()


# ---------------------------------------------------------------------------
# predict_from_raw_records fallback (no design spec)
# ---------------------------------------------------------------------------


class TestPredictFromRawRecordsFallback:
    def test_P0_S03_fallback_matches_predict_final_ols(self):
        """Without a design spec, predict_from_raw_records == predict_final_ols."""
        from rfm_pipeline.final_ols import predict_final_ols

        rng = np.random.default_rng(5)
        X = pd.DataFrame({"a": rng.standard_normal(30), "b": rng.standard_normal(30)})
        Y = pd.DataFrame({"out": rng.standard_normal(30)})
        result = fit_final_ols(X, Y)

        X_new = pd.DataFrame({"a": [1.0, -1.0], "b": [0.5, -0.5]})
        expected = predict_final_ols(result, X_new)
        actual = predict_from_raw_records(result, X_new)
        pd.testing.assert_frame_equal(actual, expected)


# ---------------------------------------------------------------------------
# Export artifact records categorical metadata
# ---------------------------------------------------------------------------


class TestExportArtifactRecordsCategorical:
    def test_P0_S03_manifest_has_categorical_inputs_key(self, nonzero_block_fit):
        bundle = build_postfit_artifacts(
            nonzero_block_fit,
            dataset_tag="test",
            all_input_features=["score", "block"],
        )
        assert "categorical_inputs" in bundle["manifest"]

    def test_P0_S03_manifest_categorical_inputs_records_block_column(self, nonzero_block_fit):
        bundle = build_postfit_artifacts(
            nonzero_block_fit,
            dataset_tag="test",
            all_input_features=["score", "block"],
        )
        cat_inputs = bundle["manifest"]["categorical_inputs"]
        names = [entry["name"] for entry in cat_inputs]
        assert "block" in names

    def test_P0_S03_manifest_categorical_feature_columns_present(self, nonzero_block_fit):
        bundle = build_postfit_artifacts(
            nonzero_block_fit,
            dataset_tag="test",
            all_input_features=["score", "block"],
        )
        assert "categorical_feature_columns" in bundle["manifest"]
        assert len(bundle["manifest"]["categorical_feature_columns"]) > 0

    def test_P0_S03_manifest_no_categorical_keys_without_spec(self):
        """A result from plain fit_final_ols must not add categorical keys to manifest."""
        rng = np.random.default_rng(1)
        X = pd.DataFrame({"x": rng.standard_normal(20)})
        Y = pd.DataFrame({"y": rng.standard_normal(20)})
        result = fit_final_ols(X, Y)
        bundle = build_postfit_artifacts(
            result,
            dataset_tag="test",
            all_input_features=["x"],
        )
        assert "categorical_inputs" not in bundle["manifest"]
        assert "categorical_feature_columns" not in bundle["manifest"]

    def test_P0_S03_manifest_categorical_inputs_records_levels(self, nonzero_block_fit):
        bundle = build_postfit_artifacts(
            nonzero_block_fit,
            dataset_tag="test",
            all_input_features=["score", "block"],
        )
        cat_inputs = bundle["manifest"]["categorical_inputs"]
        block_entry = next(e for e in cat_inputs if e["name"] == "block")
        assert block_entry["levels"] == ["A", "B"]
