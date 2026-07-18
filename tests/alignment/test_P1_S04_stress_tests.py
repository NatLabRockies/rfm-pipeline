"""P1-S04: Synthetic stress-test harness for blind spots (M6).

Tests cover:
- Each DGP generator produces the intended structural signature.
- The harness returns false-exclusion metrics in every result.
- A planted signal is detectable (false_exclusion_rate low) at high strength
  and undetectable (false_exclusion_rate high) at zero strength.
- The pca_threshold and range_threshold DGPs demonstrate their respective
  structural blind spots.
"""

from __future__ import annotations

import numpy as np
import pytest

from rfm_pipeline.stress_tests import (
    VALID_DGP_TYPES,
    StressTestResult,
    StressTestSpec,
    make_pca_threshold_dgp,
    make_pure_interaction_dgp,
    make_range_threshold_dgp,
    make_rare_output_signal_dgp,
    make_symmetric_nonlinearity_dgp,
    run_stress_test,
)

# ---------------------------------------------------------------------------
# Shared parameters for fast, deterministic test runs
# ---------------------------------------------------------------------------

_SEED = 7
_N_RUNS = 200
_N_INPUTS = 6
_N_OUTPUTS = 4
_N_OUTPUTS_PCA = 8  # pca_threshold needs ≥ 5 outputs
_CV = 2
_L1_RATIOS = (0.8, 1.0)


def _spec(
    dgp_type: str,
    *,
    signal_strength: float = 0.0,
    seed: int = _SEED,
    n_outputs: int = _N_OUTPUTS,
    n_runs: int = _N_RUNS,
    n_inputs: int = _N_INPUTS,
    rare_output_fraction: float = 0.25,
    pca_variance_threshold: float = 0.95,
    range_threshold: float = 0.3,
) -> StressTestSpec:
    return StressTestSpec(
        dgp_type=dgp_type,
        n_runs=n_runs,
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        signal_strength=signal_strength,
        seed=seed,
        screening_cv=_CV,
        screening_l1_ratios=_L1_RATIOS,
        rare_output_fraction=rare_output_fraction,
        pca_variance_threshold=pca_variance_threshold,
        range_threshold=range_threshold,
    )


# ---------------------------------------------------------------------------
# DGP structure tests
# ---------------------------------------------------------------------------


class TestPureInteractionDGPStructure:
    """pure_interaction DGP: signal is entirely in the x0×x1 term."""

    def test_P1_S04_pure_interaction_zero_main_effect_linear_correlation(self) -> None:
        """Linear correlation of x0 and x1 with Y is near zero at signal_strength=5."""
        s = _spec("pure_interaction", signal_strength=5.0)
        data = make_pure_interaction_dgp(s)

        X = data.X_train.to_numpy(dtype=float)
        Y = data.Y_train.to_numpy(dtype=float)
        y_mean = Y.mean(axis=1) - Y.mean()  # flatten to 1D

        for col_idx in (0, 1):
            x_col = X[:, col_idx] - X[:, col_idx].mean()
            corr = float(np.corrcoef(x_col, y_mean)[0, 1])
            assert abs(corr) < 0.3, (
                f"Column x{col_idx} has unexpectedly large linear correlation {corr:.3f} "
                "with Y; pure interaction DGP should have near-zero main effects."
            )

    def test_P1_S04_pure_interaction_true_signal_feature_named_correctly(self) -> None:
        s = _spec("pure_interaction", signal_strength=3.0)
        data = make_pure_interaction_dgp(s)
        assert data.true_signal_features == ("x0:x1",)

    def test_P1_S04_pure_interaction_interaction_term_correlates_with_Y(self) -> None:
        """The interaction term x0×x1 has non-trivial correlation with Y at high strength."""
        s = _spec("pure_interaction", signal_strength=5.0)
        data = make_pure_interaction_dgp(s)

        X = data.X_train.to_numpy(dtype=float)
        Y = data.Y_train.to_numpy(dtype=float)
        y_mean = Y.mean(axis=1) - Y.mean()

        interaction = (X[:, 0] - 0.5) * (X[:, 1] - 0.5)
        corr = float(np.corrcoef(interaction, y_mean)[0, 1])
        assert abs(corr) > 0.5, (
            f"Interaction term has low correlation with Y ({corr:.3f}); "
            "expected strong correlation at signal_strength=5."
        )

    def test_P1_S04_pure_interaction_dgp_type_field(self) -> None:
        s = _spec("pure_interaction")
        data = make_pure_interaction_dgp(s)
        assert data.dgp_type == "pure_interaction"

    def test_P1_S04_pure_interaction_train_holdout_sizes(self) -> None:
        s = _spec("pure_interaction")
        data = make_pure_interaction_dgp(s)
        assert len(data.X_train) + len(data.X_holdout) == s.n_runs
        assert len(data.X_holdout) >= 1


class TestSymmetricNonlinearityDGPStructure:
    """symmetric_nonlinearity DGP: signal is (x0 − 0.5)²."""

    def test_P1_S04_symmetric_nonlinearity_zero_linear_correlation(self) -> None:
        """x0 has near-zero linear correlation with Y at high signal_strength."""
        s = _spec("symmetric_nonlinearity", signal_strength=5.0)
        data = make_symmetric_nonlinearity_dgp(s)

        X = data.X_train.to_numpy(dtype=float)
        Y = data.Y_train.to_numpy(dtype=float)
        y_mean = Y.mean(axis=1) - Y.mean()

        corr = float(np.corrcoef(X[:, 0], y_mean)[0, 1])
        assert abs(corr) < 0.3, (
            f"x0 has unexpectedly large linear correlation {corr:.3f}; "
            "symmetric nonlinearity DGP should be invisible to linear screening."
        )

    def test_P1_S04_symmetric_nonlinearity_sq_term_correlates_with_Y(self) -> None:
        """(x0 − 0.5)² has non-trivial correlation with Y at high strength."""
        s = _spec("symmetric_nonlinearity", signal_strength=5.0)
        data = make_symmetric_nonlinearity_dgp(s)

        X = data.X_train.to_numpy(dtype=float)
        Y = data.Y_train.to_numpy(dtype=float)
        y_mean = Y.mean(axis=1) - Y.mean()

        sq_x0 = (X[:, 0] - 0.5) ** 2
        corr = float(np.corrcoef(sq_x0, y_mean)[0, 1])
        assert abs(corr) > 0.5, f"sq_x0 correlation {corr:.3f} is too low; expected strong signal."

    def test_P1_S04_symmetric_nonlinearity_true_feature_name(self) -> None:
        s = _spec("symmetric_nonlinearity", signal_strength=3.0)
        data = make_symmetric_nonlinearity_dgp(s)
        assert data.true_signal_features == ("sq_x0",)

    def test_P1_S04_symmetric_nonlinearity_shapes(self) -> None:
        s = _spec("symmetric_nonlinearity")
        data = make_symmetric_nonlinearity_dgp(s)
        assert data.X_train.shape[1] == s.n_inputs
        assert data.Y_train.shape[1] == s.n_outputs


class TestRareOutputSignalDGPStructure:
    """rare_output_signal DGP: x0 affects only a small fraction of outputs."""

    def test_P1_S04_rare_output_most_outputs_uncorrelated_with_x0(self) -> None:
        """The majority of outputs should have near-zero correlation with x0."""
        rare_frac = 0.25
        s = _spec("rare_output_signal", signal_strength=5.0, rare_output_fraction=rare_frac)
        data = make_rare_output_signal_dgp(s)

        X = data.X_train.to_numpy(dtype=float)
        Y = data.Y_train.to_numpy(dtype=float)
        x0 = X[:, 0]

        n_signal = max(1, round(s.n_outputs * rare_frac))
        # Outputs beyond n_signal are pure noise.
        null_corrs = [
            abs(float(np.corrcoef(x0, Y[:, j])[0, 1])) for j in range(n_signal, s.n_outputs)
        ]
        if null_corrs:
            assert max(null_corrs) < 0.4, (
                f"Null outputs have unexpectedly large correlation with x0: {null_corrs}"
            )

    def test_P1_S04_rare_output_signal_outputs_correlated_with_x0(self) -> None:
        """The signal outputs should have strong correlation with x0."""
        rare_frac = 0.25
        s = _spec("rare_output_signal", signal_strength=5.0, rare_output_fraction=rare_frac)
        data = make_rare_output_signal_dgp(s)

        X = data.X_train.to_numpy(dtype=float)
        Y = data.Y_train.to_numpy(dtype=float)
        x0 = X[:, 0]

        n_signal = max(1, round(s.n_outputs * rare_frac))
        signal_corrs = [abs(float(np.corrcoef(x0, Y[:, j])[0, 1])) for j in range(n_signal)]
        assert all(c > 0.5 for c in signal_corrs), (
            f"Signal outputs have low correlation with x0: {signal_corrs}"
        )

    def test_P1_S04_rare_output_true_feature_is_x0(self) -> None:
        s = _spec("rare_output_signal")
        data = make_rare_output_signal_dgp(s)
        assert data.true_signal_features == ("x0",)


class TestPCAThresholdDGPStructure:
    """pca_threshold DGP: signal lives in a low-variance output PC."""

    def test_P1_S04_pca_threshold_signal_in_last_eigenvalue_at_zero_strength(
        self,
    ) -> None:
        """At zero signal_strength, the signal output's variance fraction is small."""
        from sklearn.decomposition import PCA

        s = _spec("pca_threshold", signal_strength=0.0, n_outputs=_N_OUTPUTS_PCA)
        data = make_pca_threshold_dgp(s)

        Y_arr = data.Y_train.to_numpy(dtype=float)
        pca = PCA(n_components=Y_arr.shape[1])
        pca.fit(Y_arr)
        # The smallest explained variance ratio should be much smaller than the bulk.
        evr = pca.explained_variance_ratio_
        assert evr[-1] < evr[0] * 0.3, (
            f"Smallest PC variance fraction {evr[-1]:.4f} not much smaller than "
            f"bulk {evr[0]:.4f}; DGP structure not as intended."
        )

    def test_P1_S04_pca_threshold_signal_pc_grows_with_signal_strength(self) -> None:
        """Increasing signal_strength enlarges the variance of the signal output."""
        s_low = _spec("pca_threshold", signal_strength=0.0, n_outputs=_N_OUTPUTS_PCA)
        s_high = _spec("pca_threshold", signal_strength=8.0, n_outputs=_N_OUTPUTS_PCA)

        data_low = make_pca_threshold_dgp(s_low)
        data_high = make_pca_threshold_dgp(s_high)

        # Last output column variance should be larger at high strength.
        var_low = float(data_low.Y_train.iloc[:, -1].var(ddof=1))
        var_high = float(data_high.Y_train.iloc[:, -1].var(ddof=1))
        assert var_high > var_low * 5, (
            f"Signal output variance at high strength ({var_high:.3f}) is not "
            f"substantially larger than at zero strength ({var_low:.3f})."
        )

    def test_P1_S04_pca_threshold_requires_n_outputs_ge_5(self) -> None:
        s = _spec("pca_threshold", signal_strength=1.0, n_outputs=4)
        with pytest.raises(ValueError, match="n_outputs >= 5"):
            make_pca_threshold_dgp(s)

    def test_P1_S04_pca_threshold_true_feature_is_x0(self) -> None:
        s = _spec("pca_threshold", n_outputs=_N_OUTPUTS_PCA)
        data = make_pca_threshold_dgp(s)
        assert data.true_signal_features == ("x0",)


class TestRangeThresholdDGPStructure:
    """range_threshold DGP: signal input has small empirical range."""

    def test_P1_S04_range_threshold_x0_range_below_threshold(self) -> None:
        """x0 empirical range should be well below the range_threshold."""
        threshold = 0.3
        s = _spec("range_threshold", signal_strength=3.0, range_threshold=threshold)
        data = make_range_threshold_dgp(s)

        x0 = data.X_train["x0"].to_numpy(dtype=float)
        empirical_range = float(x0.max() - x0.min())
        assert empirical_range < threshold, (
            f"x0 range {empirical_range:.4f} is not below threshold {threshold}; "
            "range_threshold DGP structure not as intended."
        )

    def test_P1_S04_range_threshold_other_inputs_have_normal_range(self) -> None:
        """Inputs x1, x2, ... should have range >> range_threshold."""
        threshold = 0.3
        s = _spec("range_threshold", signal_strength=3.0, range_threshold=threshold)
        data = make_range_threshold_dgp(s)

        for col in data.X_train.columns[1:]:
            col_range = float(data.X_train[col].max() - data.X_train[col].min())
            assert col_range > threshold * 2, f"{col} range {col_range:.4f} is unexpectedly small."

    def test_P1_S04_range_threshold_true_feature_is_x0(self) -> None:
        s = _spec("range_threshold")
        data = make_range_threshold_dgp(s)
        assert data.true_signal_features == ("x0",)


# ---------------------------------------------------------------------------
# Harness false-exclusion metrics
# ---------------------------------------------------------------------------


class TestHarnessFalseExclusionMetrics:
    """The harness always returns a StressTestResult with the required fields."""

    @pytest.mark.parametrize("dgp_type", sorted(VALID_DGP_TYPES - {"pca_threshold"}))
    def test_P1_S04_harness_result_has_required_fields(self, dgp_type: str) -> None:
        """StressTestResult has all required false-exclusion and error fields."""
        s = _spec(dgp_type, signal_strength=0.0, n_runs=100)
        result = run_stress_test(s)

        assert isinstance(result, StressTestResult)
        assert result.dgp_type == dgp_type
        assert isinstance(result.n_true_signals, int) and result.n_true_signals >= 0
        assert isinstance(result.n_false_exclusions, int) and result.n_false_exclusions >= 0
        assert 0.0 <= result.false_exclusion_rate <= 1.0
        assert isinstance(result.false_excluded_features, tuple)
        assert isinstance(result.holdout_nrmse, float) and result.holdout_nrmse >= 0.0
        assert isinstance(result.selected_features, tuple)

    def test_P1_S04_harness_pca_threshold_result_has_required_fields(self) -> None:
        s = _spec("pca_threshold", signal_strength=0.0, n_outputs=_N_OUTPUTS_PCA, n_runs=100)
        result = run_stress_test(s)
        assert isinstance(result, StressTestResult)
        assert 0.0 <= result.false_exclusion_rate <= 1.0
        assert result.holdout_nrmse >= 0.0

    def test_P1_S04_false_exclusion_rate_consistent_with_counts(self) -> None:
        """false_exclusion_rate = n_false_exclusions / n_true_signals."""
        s = _spec("pure_interaction", signal_strength=0.0, n_runs=120)
        result = run_stress_test(s)
        expected_rate = result.n_false_exclusions / max(result.n_true_signals, 1)
        assert abs(result.false_exclusion_rate - expected_rate) < 1e-9

    def test_P1_S04_false_excluded_features_consistent_with_selected(self) -> None:
        """Every false-excluded feature is absent from selected_features."""
        s = _spec("pure_interaction", signal_strength=0.0, n_runs=120)
        result = run_stress_test(s)
        selected_set = set(result.selected_features)
        for feat in result.false_excluded_features:
            assert feat not in selected_set, (
                f"Feature {feat!r} is both in false_excluded and selected."
            )

    def test_P1_S04_harness_n_true_signals_equals_one(self) -> None:
        """All DGP types plant exactly one signal feature."""
        for dgp_type in sorted(VALID_DGP_TYPES - {"pca_threshold"}):
            s = _spec(dgp_type, signal_strength=1.0, n_runs=120)
            result = run_stress_test(s)
            assert result.n_true_signals == 1, (
                f"{dgp_type}: expected 1 true signal, got {result.n_true_signals}"
            )


# ---------------------------------------------------------------------------
# Signal detectability: high strength detected, zero strength not detected
# ---------------------------------------------------------------------------


class TestSignalDetectability:
    """A planted signal is detectable at high strength and absent at zero strength."""

    def test_P1_S04_pure_interaction_high_signal_detected(self) -> None:
        """At high signal_strength the interaction feature should be selected."""
        s = _spec("pure_interaction", signal_strength=5.0)
        result = run_stress_test(s)
        assert result.false_exclusion_rate < 0.5, (
            f"pure_interaction with signal_strength=5.0: "
            f"false_exclusion_rate={result.false_exclusion_rate:.3f} is too high; "
            "the interaction signal should be detected."
        )

    def test_P1_S04_pure_interaction_zero_signal_not_detected(self) -> None:
        """At zero signal_strength the interaction feature should not be selected."""
        s = _spec("pure_interaction", signal_strength=0.0)
        result = run_stress_test(s)
        assert result.false_exclusion_rate >= 0.5, (
            f"pure_interaction with signal_strength=0.0: "
            f"false_exclusion_rate={result.false_exclusion_rate:.3f}; "
            "a zero-strength feature should not survive screening."
        )

    def test_P1_S04_symmetric_nonlinearity_high_signal_detected(self) -> None:
        """At high signal_strength the squared feature should be selected."""
        s = _spec("symmetric_nonlinearity", signal_strength=5.0)
        result = run_stress_test(s)
        assert result.false_exclusion_rate < 0.5, (
            f"symmetric_nonlinearity with signal_strength=5.0: "
            f"false_exclusion_rate={result.false_exclusion_rate:.3f} is too high."
        )

    def test_P1_S04_symmetric_nonlinearity_zero_signal_not_detected(self) -> None:
        """At zero signal_strength the squared feature should not be selected."""
        s = _spec("symmetric_nonlinearity", signal_strength=0.0)
        result = run_stress_test(s)
        assert result.false_exclusion_rate >= 0.5


# ---------------------------------------------------------------------------
# Blind-spot tests for structural filters
# ---------------------------------------------------------------------------


class TestStructuralBlindSpots:
    """PCA threshold and range threshold produce false exclusions structurally."""

    def test_P1_S04_range_threshold_always_excludes_signal_input(self) -> None:
        """x0 is always a false exclusion because its range is below the filter threshold."""
        s = _spec("range_threshold", signal_strength=5.0, range_threshold=0.3)
        result = run_stress_test(s)
        assert result.false_exclusion_rate == 1.0, (
            f"range_threshold DGP with signal_strength=5.0: expected false_exclusion_rate=1.0 "
            f"(x0 range < threshold), got {result.false_exclusion_rate:.3f}."
        )
        assert "x0" in result.false_excluded_features

    def test_P1_S04_pca_threshold_excludes_signal_at_zero_strength(self) -> None:
        """At zero signal_strength, x0's variance component is discarded by PCA."""
        s = _spec(
            "pca_threshold",
            signal_strength=0.0,
            n_outputs=_N_OUTPUTS_PCA,
            pca_variance_threshold=0.95,
        )
        result = run_stress_test(s)
        assert result.false_exclusion_rate == 1.0, (
            f"pca_threshold at zero strength: expected false_exclusion_rate=1.0, "
            f"got {result.false_exclusion_rate:.3f}."
        )

    def test_P1_S04_pca_threshold_retains_signal_at_high_strength(self) -> None:
        """At high signal_strength, x0's variance is large enough to survive PCA."""
        s = _spec(
            "pca_threshold",
            signal_strength=10.0,
            n_outputs=_N_OUTPUTS_PCA,
            pca_variance_threshold=0.95,
        )
        result = run_stress_test(s)
        assert result.false_exclusion_rate < 1.0, (
            f"pca_threshold at signal_strength=10.0: x0 should be retained, "
            f"got false_exclusion_rate={result.false_exclusion_rate:.3f}."
        )


# ---------------------------------------------------------------------------
# Spec validation
# ---------------------------------------------------------------------------


class TestStressTestSpecValidation:
    def test_P1_S04_invalid_dgp_type_raises(self) -> None:
        with pytest.raises(ValueError, match="dgp_type"):
            StressTestSpec(
                dgp_type="bad_type",
                n_runs=100,
                n_inputs=6,
                n_outputs=3,
                signal_strength=1.0,
                seed=0,
            )

    def test_P1_S04_n_runs_too_small_raises(self) -> None:
        with pytest.raises(ValueError, match="n_runs"):
            StressTestSpec(
                dgp_type="pure_interaction",
                n_runs=5,
                n_inputs=6,
                n_outputs=3,
                signal_strength=1.0,
                seed=0,
            )

    def test_P1_S04_all_valid_dgp_types_accepted(self) -> None:
        from rfm_pipeline.stress_tests import VALID_DGP_TYPES

        for dgp_type in VALID_DGP_TYPES - {"pca_threshold"}:
            s = StressTestSpec(
                dgp_type=dgp_type,
                n_runs=50,
                n_inputs=4,
                n_outputs=2,
                signal_strength=1.0,
                seed=0,
            )
            assert s.dgp_type == dgp_type
