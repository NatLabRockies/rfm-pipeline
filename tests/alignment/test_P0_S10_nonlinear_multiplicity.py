"""P0-S10: Nonlinear discovery multiplicity correction + discovery/choice separation (F5).

Tests cover:

- nonlinear_multiplicity_corrected_alpha: correct Bonferroni computation, boundary handling,
  error handling.
- choose_transform_by_cv: k-fold CV selects the planted transform family; error handling.
- nonlinear_discovery_with_multiplicity_correction:
  - null_control: under a complete null (all features/responses independent), false-selection
    is controlled at the configured alpha level.
  - planted_nonlinearity_recovery: under a planted log-transform signal, the signal feature
    is detected and the log transform family is selected by CV.

Acceptance criteria (P0-S10):
- Nonlinear discovery applies a documented multiplicity correction accounting for the full
  search (inputs × components × transforms), and separates discovery from transform choice
  via nested/independent (k-fold CV) validation.
- Under a complete null, false-selection is controlled at the configured level.
- Under a planted nonlinearity, the correct transform family is recovered.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from rfm_pipeline.manuscript_stages import (
    TransformDef,
    choose_transform_by_cv,
    nonlinear_discovery_with_multiplicity_correction,
    nonlinear_multiplicity_corrected_alpha,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_SEED = 42
_N_TRAIN = 300
_N_NULL_FEATURES = 6
_N_COMPONENTS = 3
_ALPHA = 0.05

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _std(v: np.ndarray) -> np.ndarray:
    """Zero-mean unit-scale; return zeros for constant input."""
    mean, scale = float(v.mean()), float(v.std(ddof=0))
    if scale <= 0.0:
        return np.zeros_like(v, dtype=float)
    return (v - mean) / scale


def _make_null_fixture(
    *,
    n_train: int = _N_TRAIN,
    n_features: int = _N_NULL_FEATURES,
    n_components: int = _N_COMPONENTS,
    seed: int = _SEED,
) -> tuple[
    dict[str, np.ndarray],
    np.ndarray,
    dict[str, list[tuple[str, str, TransformDef]]],
    list[str],
    list[int],
]:
    """Return (x_by_base, y_scaled, candidates_by_base, component_names, active_comp_indices)
    for a pure null: all features are independent of all components.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_train, n_features))
    Y_raw = rng.standard_normal((n_train, n_components))
    Y = (Y_raw - Y_raw.mean(axis=0)) / (Y_raw.std(axis=0, ddof=0) + 1e-12)

    td_log = TransformDef(expr="log(x)", label="log", name="log")
    td_sqrt = TransformDef(expr="sqrt(x)", label="sqrt", name="sqrt")

    x_by_base: dict[str, np.ndarray] = {}
    candidates_by_base: dict[str, list[tuple[str, str, TransformDef]]] = {}
    for i in range(n_features):
        base = f"x{i}"
        # Shift to positive so log/sqrt are valid
        x_vals = np.abs(X[:, i]) + 0.1
        x_by_base[base] = x_vals
        candidates_by_base[base] = [
            (f"{base}_log", base, td_log),
            (f"{base}_sqrt", base, td_sqrt),
        ]

    component_names = [f"PC{j + 1}" for j in range(n_components)]
    active_comp_indices = list(range(n_components))
    return x_by_base, Y, candidates_by_base, component_names, active_comp_indices


def _make_planted_fixture(
    *,
    n_train: int = _N_TRAIN,
    n_null_features: int = 4,
    n_components: int = 2,
    signal_strength: float = 4.0,
    noise_scale: float = 0.05,
    seed: int = _SEED,
) -> tuple[
    dict[str, np.ndarray],
    np.ndarray,
    dict[str, list[tuple[str, str, TransformDef]]],
    list[str],
    list[int],
    str,  # signal base feature name
    str,  # expected best transform name
]:
    """Return fixture with one planted log-transform signal in component 0.

    The signal feature ``x_signal`` has x ~ Uniform(0.1, 5), and
    ``y[:, 0] = signal_strength * standardize(log(x_signal)) + noise``.

    The correct transform family to recover is ``log``.

    Returns
    -------
    (x_by_base, y_scaled, candidates_by_base, component_names, active_comp_indices,
     signal_base_name, expected_best_transform_name)
    """
    rng = np.random.default_rng(seed)

    # Signal feature: positive uniform so log is valid and strong
    x_signal = rng.uniform(0.1, 5.0, size=n_train)
    log_x = np.log(x_signal)

    # Null features: standard normal (shifted positive for domain validity)
    X_null = rng.standard_normal((n_train, n_null_features))

    # Response: signal in component 0, pure noise in others
    y_signal_raw = signal_strength * _std(log_x) + noise_scale * rng.standard_normal(n_train)
    Y_raw = rng.standard_normal((n_train, n_components))
    Y_raw[:, 0] = y_signal_raw

    Y = (Y_raw - Y_raw.mean(axis=0)) / (Y_raw.std(axis=0, ddof=0) + 1e-12)

    td_log = TransformDef(expr="log(x)", label="log", name="log")
    td_sqrt = TransformDef(expr="sqrt(x)", label="sqrt", name="sqrt")

    x_by_base: dict[str, np.ndarray] = {}
    candidates_by_base: dict[str, list[tuple[str, str, TransformDef]]] = {}

    # Signal feature first
    signal_base = "x_signal"
    x_by_base[signal_base] = x_signal
    candidates_by_base[signal_base] = [
        (f"{signal_base}_log", signal_base, td_log),
        (f"{signal_base}_sqrt", signal_base, td_sqrt),
    ]

    # Null features (shifted positive)
    for i in range(n_null_features):
        base = f"x_null{i}"
        x_by_base[base] = np.abs(X_null[:, i]) + 0.1
        candidates_by_base[base] = [
            (f"{base}_log", base, td_log),
            (f"{base}_sqrt", base, td_sqrt),
        ]

    component_names = [f"PC{j + 1}" for j in range(n_components)]
    active_comp_indices = list(range(n_components))
    return (
        x_by_base,
        Y,
        candidates_by_base,
        component_names,
        active_comp_indices,
        signal_base,
        f"{signal_base}_log",
    )


# ---------------------------------------------------------------------------
# nonlinear_multiplicity_corrected_alpha — unit tests
# ---------------------------------------------------------------------------


class TestNonlinearMultiplicityCorrectedAlpha:
    def test_P0_S10_returns_float(self):
        result = nonlinear_multiplicity_corrected_alpha(0.05, 10, 5)
        assert isinstance(result, float)

    def test_P0_S10_correct_bonferroni_computation(self):
        """alpha / (n_inputs * n_components) must equal the result."""
        alpha, n_inputs, n_components = 0.05, 10, 5
        expected = alpha / (n_inputs * n_components)
        assert nonlinear_multiplicity_corrected_alpha(
            alpha, n_inputs, n_components
        ) == pytest.approx(expected)

    def test_P0_S10_family_size_1_returns_alpha(self):
        """With n_inputs=1, n_components=1, corrected alpha equals alpha."""
        assert nonlinear_multiplicity_corrected_alpha(0.05, 1, 1) == pytest.approx(0.05)

    def test_P0_S10_manuscript_scale_276_transforms_20_components(self):
        """Manuscript scale: 276 inputs × 20 components → family_size=5520."""
        result = nonlinear_multiplicity_corrected_alpha(0.05, 276, 20)
        expected = 0.05 / (276 * 20)
        assert result == pytest.approx(expected)
        assert result < 0.05

    def test_P0_S10_larger_family_gives_smaller_alpha(self):
        small_fam = nonlinear_multiplicity_corrected_alpha(0.05, 5, 3)
        large_fam = nonlinear_multiplicity_corrected_alpha(0.05, 20, 10)
        assert large_fam < small_fam

    def test_P0_S10_alpha_zero_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            nonlinear_multiplicity_corrected_alpha(0.0, 10, 5)

    def test_P0_S10_alpha_one_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            nonlinear_multiplicity_corrected_alpha(1.0, 10, 5)

    def test_P0_S10_n_inputs_zero_raises(self):
        with pytest.raises(ValueError, match="n_inputs"):
            nonlinear_multiplicity_corrected_alpha(0.05, 0, 5)

    def test_P0_S10_n_components_zero_raises(self):
        with pytest.raises(ValueError, match="n_components"):
            nonlinear_multiplicity_corrected_alpha(0.05, 10, 0)

    def test_P0_S10_negative_n_inputs_raises(self):
        with pytest.raises(ValueError, match="n_inputs"):
            nonlinear_multiplicity_corrected_alpha(0.05, -1, 5)

    def test_P0_S10_documented_family_size_in_docstring(self):
        """Docstring must mention n_inputs × n_components as family size."""
        doc = nonlinear_multiplicity_corrected_alpha.__doc__ or ""
        assert "n_inputs" in doc
        assert "n_components" in doc


# ---------------------------------------------------------------------------
# choose_transform_by_cv — unit tests
# ---------------------------------------------------------------------------


class TestChooseTransformByCv:
    def _make_log_signal(
        self,
        n: int = 200,
        signal_strength: float = 3.0,
        noise: float = 0.05,
        seed: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        x = rng.uniform(0.1, 5.0, size=n)
        log_x = _std(np.log(x))
        y = signal_strength * log_x + noise * rng.standard_normal(n)
        return x, y

    def test_P0_S10_returns_tuple_of_str_and_float(self):
        x, y = self._make_log_signal()
        td_log = TransformDef(expr="log(x)", label="log", name="log")
        td_sqrt = TransformDef(expr="sqrt(x)", label="sqrt", name="sqrt")
        result = choose_transform_by_cv(x, y, [("log_x", td_log), ("sqrt_x", td_sqrt)])
        assert isinstance(result, tuple)
        name, rmse = result
        assert isinstance(name, str)
        assert isinstance(rmse, float)

    def test_P0_S10_selects_log_for_log_planted_signal(self):
        """Under a planted log signal, log transform must be selected over sqrt."""
        x, y = self._make_log_signal(n=300, signal_strength=4.0, noise=0.05, seed=_SEED)
        td_log = TransformDef(expr="log(x)", label="log", name="log")
        td_sqrt = TransformDef(expr="sqrt(x)", label="sqrt", name="sqrt")
        name, rmse = choose_transform_by_cv(
            x, y, [("log_x", td_log), ("sqrt_x", td_sqrt)], n_splits=5, seed=_SEED
        )
        assert name == "log_x", f"Expected log_x but got {name!r}"
        assert math.isfinite(rmse)
        assert rmse >= 0.0

    def test_P0_S10_cv_rmse_is_finite_and_non_negative(self):
        x, y = self._make_log_signal()
        td_log = TransformDef(expr="log(x)", label="log", name="log")
        _, rmse = choose_transform_by_cv(x, y, [("log_x", td_log)])
        assert math.isfinite(rmse)
        assert rmse >= 0.0

    def test_P0_S10_empty_candidates_raises(self):
        x, y = self._make_log_signal()
        with pytest.raises(ValueError, match="candidates"):
            choose_transform_by_cv(x, y, [])

    def test_P0_S10_2d_x_vals_raises(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((50, 2))
        y = rng.standard_normal(50)
        td = TransformDef(expr="log(x)", label="log", name="log")
        with pytest.raises(ValueError, match="1-D"):
            choose_transform_by_cv(x, y, [("log_x", td)])

    def test_P0_S10_n_splits_less_than_2_raises(self):
        x, y = self._make_log_signal(n=50)
        td = TransformDef(expr="log(x)", label="log", name="log")
        with pytest.raises(ValueError, match="n_splits"):
            choose_transform_by_cv(x, y, [("log_x", td)], n_splits=1)

    def test_P0_S10_deterministic_with_fixed_seed(self):
        """Same seed should produce the same result every time."""
        x, y = self._make_log_signal(n=100, seed=7)
        td_log = TransformDef(expr="log(x)", label="log", name="log")
        td_sqrt = TransformDef(expr="sqrt(x)", label="sqrt", name="sqrt")
        r1 = choose_transform_by_cv(x, y, [("log_x", td_log), ("sqrt_x", td_sqrt)], seed=0)
        r2 = choose_transform_by_cv(x, y, [("log_x", td_log), ("sqrt_x", td_sqrt)], seed=0)
        assert r1 == r2

    def test_P0_S10_2d_y_vals_accepted(self):
        """2-D y_vals (n, k) should be accepted and RMSE should be finite."""
        rng = np.random.default_rng(3)
        x = rng.uniform(0.1, 5.0, size=120)
        y = np.column_stack([_std(np.log(x)), rng.standard_normal(120) * 0.1])
        td_log = TransformDef(expr="log(x)", label="log", name="log")
        td_sqrt = TransformDef(expr="sqrt(x)", label="sqrt", name="sqrt")
        name, rmse = choose_transform_by_cv(
            x, y, [("log_x", td_log), ("sqrt_x", td_sqrt)], seed=_SEED
        )
        assert isinstance(name, str)
        assert math.isfinite(rmse)


# ---------------------------------------------------------------------------
# nonlinear_discovery_with_multiplicity_correction — null control
# ---------------------------------------------------------------------------


class TestNullFalseSelectionControl:
    """Under a complete null (all features independent of response), false-selection
    must be controlled at the configured alpha level."""

    def test_P0_S10_null_zero_detections_fixed_seed(self):
        """With a fixed seed under complete null, no features should be selected."""
        x_by_base, Y, candidates_by_base, component_names, active_comp_indices = _make_null_fixture(
            n_train=_N_TRAIN,
            n_features=_N_NULL_FEATURES,
            n_components=_N_COMPONENTS,
            seed=_SEED,
        )
        results = nonlinear_discovery_with_multiplicity_correction(
            x_by_base=x_by_base,
            y_scaled=Y,
            candidates_by_base=candidates_by_base,
            component_names=component_names,
            active_comp_indices=active_comp_indices,
            alpha=_ALPHA,
            n_cv_splits=5,
            seed=_SEED,
        )

        n_detected = sum(1 for r in results.values() if r["nonlinear"])
        assert n_detected == 0, (
            f"Expected 0 false detections under complete null, got {n_detected}. "
            "Multiplicity correction may be missing or incorrect."
        )

    def test_P0_S10_null_all_results_present(self):
        """Results must contain one entry per base feature."""
        x_by_base, Y, candidates_by_base, component_names, active_comp_indices = (
            _make_null_fixture()
        )
        results = nonlinear_discovery_with_multiplicity_correction(
            x_by_base=x_by_base,
            y_scaled=Y,
            candidates_by_base=candidates_by_base,
            component_names=component_names,
            active_comp_indices=active_comp_indices,
            alpha=_ALPHA,
        )
        assert set(results.keys()) == set(x_by_base.keys())

    def test_P0_S10_null_result_schema(self):
        """Each result dict must contain all documented keys."""
        x_by_base, Y, candidates_by_base, component_names, active_comp_indices = _make_null_fixture(
            n_features=2, n_components=2
        )
        results = nonlinear_discovery_with_multiplicity_correction(
            x_by_base=x_by_base,
            y_scaled=Y,
            candidates_by_base=candidates_by_base,
            component_names=component_names,
            active_comp_indices=active_comp_indices,
            alpha=_ALPHA,
        )
        required_keys = {
            "nonlinear",
            "best_transform_name",
            "best_edf",
            "best_p",
            "best_comp_name",
            "best_cv_rmse",
            "corrected_alpha",
            "family_size",
        }
        for base_feat, result in results.items():
            missing = required_keys - set(result.keys())
            assert not missing, f"Missing keys for {base_feat}: {missing}"

    def test_P0_S10_corrected_alpha_recorded_correctly(self):
        """corrected_alpha must equal alpha / (n_inputs * n_components)."""
        n_features, n_components = 5, 3
        x_by_base, Y, candidates_by_base, component_names, active_comp_indices = _make_null_fixture(
            n_features=n_features, n_components=n_components
        )
        results = nonlinear_discovery_with_multiplicity_correction(
            x_by_base=x_by_base,
            y_scaled=Y,
            candidates_by_base=candidates_by_base,
            component_names=component_names,
            active_comp_indices=active_comp_indices,
            alpha=_ALPHA,
        )
        expected_corrected = _ALPHA / (n_features * n_components)
        for result in results.values():
            assert result["corrected_alpha"] == pytest.approx(expected_corrected)
            assert result["family_size"] == n_features * n_components

    def test_P0_S10_null_nonlinear_flag_is_bool(self):
        """nonlinear flag must be a bool, not a numpy bool or int."""
        x_by_base, Y, candidates_by_base, component_names, active_comp_indices = _make_null_fixture(
            n_features=2, n_components=2
        )
        results = nonlinear_discovery_with_multiplicity_correction(
            x_by_base=x_by_base,
            y_scaled=Y,
            candidates_by_base=candidates_by_base,
            component_names=component_names,
            active_comp_indices=active_comp_indices,
            alpha=_ALPHA,
        )
        for result in results.values():
            assert isinstance(result["nonlinear"], bool)

    def test_P0_S10_uncorrected_would_inflate_false_positives(self):
        """Regression: without correction, the raw p<0.01 threshold inflates FP.

        With n_inputs=6, n_components=3, family=18 tests, the uncorrected
        p<0.01 threshold is ~18× less strict than the corrected threshold.
        Verify that the corrected alpha is strictly smaller than the raw threshold.
        """
        n_features, n_components, raw_threshold = 6, 3, 0.01
        corrected = nonlinear_multiplicity_corrected_alpha(0.05, n_features, n_components)
        assert corrected < raw_threshold, (
            f"Corrected alpha {corrected} should be < raw threshold {raw_threshold} "
            "to control family-wise error rate."
        )


# ---------------------------------------------------------------------------
# nonlinear_discovery_with_multiplicity_correction — planted signal recovery
# ---------------------------------------------------------------------------


class TestPlantedNonlinearityRecovery:
    """Under a planted log-transform nonlinearity, the signal feature must be
    detected and the log transform family must be selected by CV."""

    def test_P0_S10_signal_feature_detected(self):
        """The planted log-signal feature must be detected as nonlinear."""
        (
            x_by_base,
            Y,
            candidates_by_base,
            component_names,
            active_comp_indices,
            signal_base,
            _expected_transform,
        ) = _make_planted_fixture(
            n_train=_N_TRAIN,
            signal_strength=5.0,
            noise_scale=0.02,
            seed=_SEED,
        )
        results = nonlinear_discovery_with_multiplicity_correction(
            x_by_base=x_by_base,
            y_scaled=Y,
            candidates_by_base=candidates_by_base,
            component_names=component_names,
            active_comp_indices=active_comp_indices,
            alpha=_ALPHA,
            n_cv_splits=5,
            seed=_SEED,
        )
        assert signal_base in results, f"Signal feature {signal_base!r} missing from results."
        assert results[signal_base]["nonlinear"], (
            f"Planted log signal in {signal_base!r} not detected. "
            f"best_p={results[signal_base]['best_p']:.4g}, "
            f"corrected_alpha={results[signal_base]['corrected_alpha']:.4g}"
        )

    def test_P0_S10_correct_transform_recovered_by_cv(self):
        """The selected transform for the signal feature must be log (not sqrt)."""
        (
            x_by_base,
            Y,
            candidates_by_base,
            component_names,
            active_comp_indices,
            signal_base,
            expected_transform,
        ) = _make_planted_fixture(
            n_train=_N_TRAIN,
            signal_strength=5.0,
            noise_scale=0.02,
            seed=_SEED,
        )
        results = nonlinear_discovery_with_multiplicity_correction(
            x_by_base=x_by_base,
            y_scaled=Y,
            candidates_by_base=candidates_by_base,
            component_names=component_names,
            active_comp_indices=active_comp_indices,
            alpha=_ALPHA,
            n_cv_splits=5,
            seed=_SEED,
        )
        result = results[signal_base]
        assert result["nonlinear"], "Signal feature must be detected first."
        assert result["best_transform_name"] == expected_transform, (
            f"Expected transform {expected_transform!r} but got "
            f"{result['best_transform_name']!r}. "
            "CV-based transform selection failed to recover the planted log family."
        )

    def test_P0_S10_null_features_not_detected(self):
        """Null features in the planted fixture must NOT be detected."""
        (
            x_by_base,
            Y,
            candidates_by_base,
            component_names,
            active_comp_indices,
            signal_base,
            _,
        ) = _make_planted_fixture(seed=_SEED)
        results = nonlinear_discovery_with_multiplicity_correction(
            x_by_base=x_by_base,
            y_scaled=Y,
            candidates_by_base=candidates_by_base,
            component_names=component_names,
            active_comp_indices=active_comp_indices,
            alpha=_ALPHA,
            n_cv_splits=5,
            seed=_SEED,
        )
        null_bases = [b for b in x_by_base if b != signal_base]
        false_detections = [b for b in null_bases if results[b]["nonlinear"]]
        assert not false_detections, (
            f"Null features incorrectly detected as nonlinear: {false_detections}. "
            "Multiplicity correction may be insufficient."
        )

    def test_P0_S10_signal_cv_rmse_finite(self):
        """When the signal feature is detected, best_cv_rmse must be finite."""
        (
            x_by_base,
            Y,
            candidates_by_base,
            component_names,
            active_comp_indices,
            signal_base,
            _,
        ) = _make_planted_fixture(signal_strength=5.0, noise_scale=0.02, seed=_SEED)
        results = nonlinear_discovery_with_multiplicity_correction(
            x_by_base=x_by_base,
            y_scaled=Y,
            candidates_by_base=candidates_by_base,
            component_names=component_names,
            active_comp_indices=active_comp_indices,
            alpha=_ALPHA,
        )
        result = results[signal_base]
        if result["nonlinear"]:
            assert math.isfinite(result["best_cv_rmse"]), (
                "best_cv_rmse must be finite when feature is declared nonlinear."
            )

    def test_P0_S10_non_nonlinear_cv_rmse_is_nan(self):
        """Features not declared nonlinear must have best_cv_rmse == nan."""
        x_by_base, Y, candidates_by_base, component_names, active_comp_indices = _make_null_fixture(
            n_features=3, n_components=2
        )
        results = nonlinear_discovery_with_multiplicity_correction(
            x_by_base=x_by_base,
            y_scaled=Y,
            candidates_by_base=candidates_by_base,
            component_names=component_names,
            active_comp_indices=active_comp_indices,
            alpha=_ALPHA,
        )
        for result in results.values():
            if not result["nonlinear"]:
                assert math.isnan(result["best_cv_rmse"]), (
                    "best_cv_rmse must be nan when feature is not nonlinear."
                )

    def test_P0_S10_discovery_uses_corrected_not_raw_threshold(self):
        """Planted signal is detected with corrected alpha; confirm corrected_alpha < 0.01.

        The manuscript raw threshold was p < 0.01 without correction.
        The corrected threshold must be smaller to control FWER.
        """
        (
            x_by_base,
            Y,
            candidates_by_base,
            component_names,
            active_comp_indices,
            signal_base,
            _,
        ) = _make_planted_fixture(signal_strength=5.0, noise_scale=0.02, seed=_SEED)
        results = nonlinear_discovery_with_multiplicity_correction(
            x_by_base=x_by_base,
            y_scaled=Y,
            candidates_by_base=candidates_by_base,
            component_names=component_names,
            active_comp_indices=active_comp_indices,
            alpha=_ALPHA,
        )
        # All results share the same corrected_alpha and family_size
        for result in results.values():
            # The corrected alpha must be strictly smaller than the raw 0.01 threshold
            # (n_inputs * n_components >= 10 in the planted fixture → corrected < 0.005)
            assert result["corrected_alpha"] < 0.01, (
                f"corrected_alpha={result['corrected_alpha']:.6g} must be < raw 0.01 threshold. "
                "Multiplicity correction is not being applied."
            )
            break  # All results have the same corrected_alpha
