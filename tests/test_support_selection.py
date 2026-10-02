"""P0-S11: Validated support-selection rule via internal validation (F6).

Tests cover:
- signal_term_retention: true signal features are in the selected support.
- null_term_removal: null features (zero true coefficient) are removed.
- internal_only_threshold_selection: sealed test partition is never accessed
  during threshold selection.
- threshold_sensitivity: sensitivity table is populated for all candidates.
- result_structure: SupportSelectionResult fields are consistent and correct.
- screening_importance_helper: compute_enriched_coef_magnitudes works correctly.

Fixture:
- Synthetic design + response with a known sparse coefficient vector; fixed seed.
  First N_SIGNAL_FEATURES features have true coefficient 2.0; remaining are 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rfm_pipeline.data import SealedTestAccessError, make_sealed_split
from rfm_pipeline.final_ols import SupportSelectionResult, select_support_via_refit
from rfm_pipeline.regularized_screening import (
    ScreeningImportanceResult,
    compute_enriched_coef_magnitudes,
)

# ---------------------------------------------------------------------------
# Synthetic DGP parameters
# ---------------------------------------------------------------------------

_SEED = 42
_N_TOTAL = 300
_N_SIGNAL = 5
_N_NULL = 10
_P = _N_SIGNAL + _N_NULL  # 15 features total
_NOISE_STD = 0.5
_TRUE_COEF_VALUE = 2.0
_THRESHOLDS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sparse_dgp() -> dict:
    """Synthetic sparse DGP with known support.

    Returns a dict with:
      X_df, y_df, df (combined), feature_names, signal_features, null_features.
    """
    rng = np.random.default_rng(_SEED)
    X = rng.standard_normal((_N_TOTAL, _P))
    true_coef = np.zeros(_P)
    true_coef[:_N_SIGNAL] = _TRUE_COEF_VALUE
    y = X @ true_coef + _NOISE_STD * rng.standard_normal(_N_TOTAL)

    feature_names = [f"x{i}" for i in range(_P)]
    signal_features = [f"x{i}" for i in range(_N_SIGNAL)]
    null_features = [f"x{i}" for i in range(_N_SIGNAL, _P)]

    X_df = pd.DataFrame(X, columns=feature_names)
    y_df = pd.DataFrame({"y": y})
    df = pd.concat([X_df, y_df], axis=1)
    df["stratum"] = "A"

    return {
        "X_df": X_df,
        "y_df": y_df,
        "df": df,
        "feature_names": feature_names,
        "signal_features": signal_features,
        "null_features": null_features,
    }


@pytest.fixture(scope="module")
def sealed_split_fixture(sparse_dgp: dict):
    """Sealed split from the sparse DGP combined frame."""
    return make_sealed_split(
        sparse_dgp["df"],
        strata_column="stratum",
        val_fraction=0.20,
        test_fraction=0.20,
        random_state=_SEED,
    )


@pytest.fixture(scope="module")
def support_result(sparse_dgp: dict, sealed_split_fixture) -> SupportSelectionResult:
    """Run select_support_via_refit on the sparse DGP."""
    return select_support_via_refit(
        feature_cols=sparse_dgp["feature_names"],
        response_cols=["y"],
        sealed_split=sealed_split_fixture,
        thresholds=_THRESHOLDS,
        seed=_SEED,
    )


# ---------------------------------------------------------------------------
# Signal-term retention
# ---------------------------------------------------------------------------


class TestSignalTermRetention:
    def test_P0_S11_all_signal_features_retained(
        self, support_result: SupportSelectionResult, sparse_dgp: dict
    ) -> None:
        """All true signal features appear in the selected support."""
        selected = set(support_result.selected_features)
        missing = [f for f in sparse_dgp["signal_features"] if f not in selected]
        assert missing == [], f"Signal features not retained: {missing}"

    def test_P0_S11_selected_features_are_subset_of_candidates(
        self, support_result: SupportSelectionResult, sparse_dgp: dict
    ) -> None:
        """Every selected feature is among the enriched candidates."""
        candidate_set = set(sparse_dgp["feature_names"])
        for f in support_result.selected_features:
            assert f in candidate_set, f"{f!r} not in enriched candidate set"

    def test_P0_S11_support_is_strictly_reduced(
        self, support_result: SupportSelectionResult, sparse_dgp: dict
    ) -> None:
        """Selected support is strictly smaller than the enriched candidate set."""
        assert len(support_result.selected_features) < len(sparse_dgp["feature_names"])


# ---------------------------------------------------------------------------
# Null-term removal
# ---------------------------------------------------------------------------


class TestNullTermRemoval:
    def test_P0_S11_some_null_features_removed(
        self, support_result: SupportSelectionResult, sparse_dgp: dict
    ) -> None:
        """At least one null feature is excluded from the selected support."""
        selected = set(support_result.selected_features)
        removed_nulls = [f for f in sparse_dgp["null_features"] if f not in selected]
        assert len(removed_nulls) > 0, f"No null features removed; selected={sorted(selected)}"

    def test_P0_S11_majority_null_features_removed(
        self, support_result: SupportSelectionResult, sparse_dgp: dict
    ) -> None:
        """More than half of null features are excluded from the selected support."""
        selected = set(support_result.selected_features)
        n_removed = sum(1 for f in sparse_dgp["null_features"] if f not in selected)
        assert n_removed > len(sparse_dgp["null_features"]) // 2, (
            f"Only {n_removed}/{len(sparse_dgp['null_features'])} null features removed"
        )


# ---------------------------------------------------------------------------
# Internal-only threshold selection
# ---------------------------------------------------------------------------


class TestInternalOnlyThresholdSelection:
    def test_P0_S11_sealed_split_remains_sealed_after_selection(self, sparse_dgp: dict) -> None:
        """After select_support_via_refit, the sealed split is still sealed."""
        split = make_sealed_split(
            sparse_dgp["df"],
            strata_column="stratum",
            val_fraction=0.20,
            test_fraction=0.20,
            random_state=99,
        )
        select_support_via_refit(
            feature_cols=sparse_dgp["feature_names"],
            response_cols=["y"],
            sealed_split=split,
            thresholds=[0.1, 0.3, 0.5],
            seed=0,
        )
        with pytest.raises(SealedTestAccessError):
            _ = split.test

    def test_P0_S11_unsealed_split_raises_sealed_test_access_error(self, sparse_dgp: dict) -> None:
        """Passing an already-unsealed split raises SealedTestAccessError."""
        split = make_sealed_split(
            sparse_dgp["df"],
            strata_column="stratum",
            val_fraction=0.20,
            test_fraction=0.20,
            random_state=11,
        )
        split.unseal(reason="test: premature unseal")
        with pytest.raises(SealedTestAccessError):
            select_support_via_refit(
                feature_cols=sparse_dgp["feature_names"],
                response_cols=["y"],
                sealed_split=split,
                thresholds=[0.1, 0.5],
                seed=0,
            )

    def test_P0_S11_unsealed_error_is_runtime_error_subtype(self, sparse_dgp: dict) -> None:
        """SealedTestAccessError is a RuntimeError subtype."""
        split = make_sealed_split(
            sparse_dgp["df"],
            strata_column="stratum",
            val_fraction=0.20,
            test_fraction=0.20,
            random_state=22,
        )
        split.unseal(reason="subtype check")
        with pytest.raises(SealedTestAccessError) as exc_info:
            select_support_via_refit(
                feature_cols=sparse_dgp["feature_names"],
                response_cols=["y"],
                sealed_split=split,
                thresholds=[0.1],
                seed=0,
            )
        assert isinstance(exc_info.value, RuntimeError)


# ---------------------------------------------------------------------------
# Threshold sensitivity
# ---------------------------------------------------------------------------


class TestThresholdSensitivity:
    def test_P0_S11_sensitivity_table_has_all_thresholds(
        self, support_result: SupportSelectionResult
    ) -> None:
        """Sensitivity table has one row per candidate threshold."""
        assert len(support_result.threshold_sensitivity) == len(_THRESHOLDS)

    def test_P0_S11_sensitivity_table_has_required_columns(
        self, support_result: SupportSelectionResult
    ) -> None:
        """Sensitivity table contains threshold, n_selected, val_score columns."""
        required = {"threshold", "n_selected", "val_score"}
        assert required.issubset(set(support_result.threshold_sensitivity.columns))

    def test_P0_S11_n_selected_is_nonincreasing_with_threshold(
        self, support_result: SupportSelectionResult
    ) -> None:
        """Higher thresholds select at most as many features as lower thresholds."""
        df = support_result.threshold_sensitivity.sort_values("threshold").reset_index(drop=True)
        n_sel = df["n_selected"].to_numpy()
        for i in range(len(n_sel) - 1):
            assert n_sel[i] >= n_sel[i + 1], (
                f"n_selected not non-increasing: {n_sel[i]} < {n_sel[i + 1]} at index {i}"
            )

    def test_P0_S11_best_threshold_is_from_candidates(
        self, support_result: SupportSelectionResult
    ) -> None:
        """Best threshold is one of the supplied candidate thresholds."""
        assert support_result.best_threshold in _THRESHOLDS


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------


class TestResultStructure:
    def test_P0_S11_return_type_is_support_selection_result(
        self, support_result: SupportSelectionResult
    ) -> None:
        """Return type is SupportSelectionResult."""
        assert isinstance(support_result, SupportSelectionResult)

    def test_P0_S11_selected_mask_consistent_with_selected_features(
        self, support_result: SupportSelectionResult
    ) -> None:
        """selected_mask and selected_features agree with each other."""
        expected = tuple(
            f
            for f, m in zip(support_result.feature_names, support_result.selected_mask, strict=True)
            if m
        )
        assert support_result.selected_features == expected

    def test_P0_S11_coef_magnitudes_length_matches_candidates(
        self, support_result: SupportSelectionResult, sparse_dgp: dict
    ) -> None:
        """initial_coef_magnitudes has one value per enriched candidate."""
        assert len(support_result.initial_coef_magnitudes) == len(sparse_dgp["feature_names"])

    def test_P0_S11_coef_magnitudes_all_nonnegative(
        self, support_result: SupportSelectionResult
    ) -> None:
        """Coefficient magnitudes are all non-negative."""
        assert np.all(support_result.initial_coef_magnitudes >= 0)

    def test_P0_S11_empty_thresholds_raises_value_error(
        self, sparse_dgp: dict, sealed_split_fixture
    ) -> None:
        """Empty threshold list raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            select_support_via_refit(
                feature_cols=sparse_dgp["feature_names"],
                response_cols=["y"],
                sealed_split=sealed_split_fixture,
                thresholds=[],
                seed=0,
            )


# ---------------------------------------------------------------------------
# Screening importance helper (regularized_screening.py)
# ---------------------------------------------------------------------------


class TestScreeningImportanceHelper:
    def test_P0_S11_compute_enriched_coef_magnitudes_returns_correct_type(
        self, sparse_dgp: dict
    ) -> None:
        """compute_enriched_coef_magnitudes returns ScreeningImportanceResult."""
        result = compute_enriched_coef_magnitudes(
            sparse_dgp["X_df"],
            sparse_dgp["y_df"],
        )
        assert isinstance(result, ScreeningImportanceResult)

    def test_P0_S11_signal_coef_magnitudes_larger_than_null(self, sparse_dgp: dict) -> None:
        """Signal features have strictly larger coefficient magnitudes than null features."""
        result = compute_enriched_coef_magnitudes(
            sparse_dgp["X_df"],
            sparse_dgp["y_df"],
        )
        mag_map = dict(zip(result.feature_names, result.coef_magnitudes, strict=True))
        signal_mags = [mag_map[f] for f in sparse_dgp["signal_features"]]
        null_mags = [mag_map[f] for f in sparse_dgp["null_features"]]
        assert min(signal_mags) > max(null_mags), (
            f"Signal min={min(signal_mags):.4f} not > null max={max(null_mags):.4f}"
        )

    def test_P0_S11_feature_names_preserved_in_importance_result(self, sparse_dgp: dict) -> None:
        """Feature names in ScreeningImportanceResult match the input DataFrame columns."""
        result = compute_enriched_coef_magnitudes(
            sparse_dgp["X_df"],
            sparse_dgp["y_df"],
        )
        assert result.feature_names == tuple(sparse_dgp["feature_names"])

    def test_P0_S11_coef_magnitudes_length_in_importance_result(self, sparse_dgp: dict) -> None:
        """coef_magnitudes has one value per feature."""
        result = compute_enriched_coef_magnitudes(
            sparse_dgp["X_df"],
            sparse_dgp["y_df"],
        )
        assert len(result.coef_magnitudes) == len(sparse_dgp["feature_names"])
