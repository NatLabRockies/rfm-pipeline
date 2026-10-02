"""Tests for reproducible categorical design-matrix prediction.

Tests:
- Fitting with levels=None stores explicit levels in the design spec.
- Predicting on a 2-row scenario-contrast batch raises no missing-column error.
- Flipping the categorical value changes the prediction when its coefficient is nonzero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rfm_pipeline.features import CategoricalInputDecl, DesignMatrixSpec
from rfm_pipeline.final_ols import (
    FinalOLSFitResult,
    fit_final_ols_with_design,
    predict_from_raw_records,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def levels_none_nonzero_fit() -> FinalOLSFitResult:
    """Synthetic fit with levels=None and a planted non-zero categorical effect.

    Y = 2*score + 5*(group == "B") + noise
    Levels are intentionally not declared so they must be inferred at fit time.
    """
    rng = np.random.default_rng(99)
    n = 60
    group = ["A"] * 30 + ["B"] * 30
    score = rng.standard_normal(n)
    y = 2.0 * score + 5.0 * (np.array(group) == "B").astype(float) + rng.standard_normal(n) * 0.05
    X = pd.DataFrame({"score": score, "group": group})
    Y = pd.DataFrame({"outcome": y})

    # levels=None — must be resolved from training data
    decl = CategoricalInputDecl(name="group", levels=None)
    spec = DesignMatrixSpec(categorical_inputs=(decl,), drop_first=True)
    return fit_final_ols_with_design(X, Y, spec)


# ---------------------------------------------------------------------------
# R3-S04 tests
# ---------------------------------------------------------------------------


class TestR3S04CategoricalLevelsPersist:
    def test_R3_S04_stored_spec_has_explicit_levels(self, levels_none_nonzero_fit):
        """The stored design spec must carry resolved levels, not None."""
        stored_spec = levels_none_nonzero_fit.design_spec
        assert stored_spec is not None
        for decl in stored_spec.categorical_inputs:
            assert decl.levels is not None, (
                f"levels for {decl.name!r} must be resolved at fit time, not None"
            )

    def test_R3_S04_stored_levels_match_training_data(self, levels_none_nonzero_fit):
        """The stored levels must be ['A', 'B'] (sorted unique from training data)."""
        stored_spec = levels_none_nonzero_fit.design_spec
        group_decl = next(d for d in stored_spec.categorical_inputs if d.name == "group")
        assert group_decl.levels == ["A", "B"]

    def test_R3_S04_two_row_predict_no_missing_columns(self, levels_none_nonzero_fit):
        """Predicting on a 2-row batch must not raise missing-column errors."""
        batch = pd.DataFrame({"score": [0.0, 0.0], "group": ["A", "B"]})
        # Must not raise
        result = predict_from_raw_records(levels_none_nonzero_fit, batch)
        assert result.shape == (2, 1)

    def test_R3_S04_single_row_each_category_no_error(self, levels_none_nonzero_fit):
        """Predicting on a single-row batch (one category only) must not raise."""
        rec_a = pd.DataFrame({"score": [1.0], "group": ["A"]})
        rec_b = pd.DataFrame({"score": [1.0], "group": ["B"]})
        pred_a = predict_from_raw_records(levels_none_nonzero_fit, rec_a)
        pred_b = predict_from_raw_records(levels_none_nonzero_fit, rec_b)
        assert pred_a.shape == (1, 1)
        assert pred_b.shape == (1, 1)

    def test_R3_S04_flipping_categorical_changes_prediction(self, levels_none_nonzero_fit):
        """Flipping the categorical must change the prediction (nonzero effect)."""
        rec_a = pd.DataFrame({"score": [0.0], "group": ["A"]})
        rec_b = pd.DataFrame({"score": [0.0], "group": ["B"]})
        pred_a = predict_from_raw_records(levels_none_nonzero_fit, rec_a)["outcome"].iloc[0]
        pred_b = predict_from_raw_records(levels_none_nonzero_fit, rec_b)["outcome"].iloc[0]
        assert pred_a != pred_b, "Non-zero categorical effect must produce different predictions."

    def test_R3_S04_contrast_magnitude_reflects_planted_effect(self, levels_none_nonzero_fit):
        """Contrast between A and B should be close to the planted 5-unit effect."""
        rec_a = pd.DataFrame({"score": [0.0], "group": ["A"]})
        rec_b = pd.DataFrame({"score": [0.0], "group": ["B"]})
        pred_a = predict_from_raw_records(levels_none_nonzero_fit, rec_a)["outcome"].iloc[0]
        pred_b = predict_from_raw_records(levels_none_nonzero_fit, rec_b)["outcome"].iloc[0]
        contrast = abs(pred_b - pred_a)
        assert contrast > 1.0, f"Expected large contrast from planted effect; got {contrast:.4f}"

    def test_R3_S04_scenario_contrast_two_row_batch(self, levels_none_nonzero_fit):
        """Two-row batch with both categories returns two distinct predictions."""
        batch = pd.DataFrame({"score": [0.0, 0.0], "group": ["A", "B"]})
        preds = predict_from_raw_records(levels_none_nonzero_fit, batch)["outcome"]
        assert preds.iloc[0] != preds.iloc[1], (
            "Two-row batch predictions must differ when categorical effect is nonzero."
        )
