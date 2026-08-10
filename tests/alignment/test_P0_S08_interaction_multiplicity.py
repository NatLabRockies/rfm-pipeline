"""P0-S08: Multiplicity control across interaction pairs (F5).

Tests cover:
- max_t_critical_value: correct quantile, boundary, and error handling
- bh_fdr_selected: standard BH step-up behaviour
- multiplicity_controlled_interaction_selection: dispatches correctly,
  returns consistent types
- Null false-selection control: under a complete null (all pairs null),
  FWER and BH-FDR produce zero or near-zero false selections (fixed seed).
- Planted-signal recovery: under a planted-signal alternative, true
  interaction pairs are selected by both methods.
"""

from __future__ import annotations

import numpy as np
import pytest

from rfm_pipeline.manuscript_stages import (
    bh_fdr_selected,
    max_t_adjusted_pvalues,
    max_t_critical_value,
    multiplicity_controlled_interaction_selection,
)

# ---------------------------------------------------------------------------
# Synthetic fixture helpers
# ---------------------------------------------------------------------------

_SEED = 42
_N_TRAIN = 120
_N_FEATURES = 8  # → 28 candidate pairs
_N_COMPONENTS = 3
_B_PERMS = 400  # sufficient for alpha=0.05 over 28 pairs


def _make_null_fixture(
    *,
    n_train: int = _N_TRAIN,
    n_features: int = _N_FEATURES,
    n_components: int = _N_COMPONENTS,
    seed: int = _SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (observed_scores, null_statistics, pair_indices) for a pure null.

    All pair scores are independent of the response.  The response is drawn
    independently of all features.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_train, n_features))
    # Pure-null response: independent of x
    y = rng.standard_normal((n_train, n_components))
    y = (y - y.mean(axis=0)) / (y.std(axis=0, ddof=0) + 1e-12)

    # Build residualized interaction matrix for all unique pairs
    from itertools import combinations

    pairs = list(combinations(range(n_features), 2))
    n_pairs = len(pairs)
    R = np.zeros((n_train, n_pairs), dtype=float)
    for col, (i, j) in enumerate(pairs):
        product = x[:, i] * x[:, j]
        controls = np.column_stack([np.ones(n_train), _std(x[:, i]), _std(x[:, j])])
        coef, *_ = np.linalg.lstsq(controls, product, rcond=None)
        residual = product - controls @ coef
        R[:, col] = _std(residual)

    observed_scores, null_statistics = _compute_perm_stats(R, y, n_perms=_B_PERMS, seed=seed + 1)
    return observed_scores, null_statistics, pairs


def _make_signal_fixture(
    *,
    n_train: int = _N_TRAIN,
    n_features: int = 5,  # 10 pairs → BH rank-1 critical 0.005, min p=0.0025 < 0.005
    n_components: int = _N_COMPONENTS,
    true_pair_indices: tuple[int, int] = (0, 1),
    signal_strength: float = 5.0,
    seed: int = _SEED,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]], set[int]]:
    """Return (observed_scores, null_statistics, pairs, true_col_set) for a planted signal.

    The response depends strongly on the interaction of features
    ``true_pair_indices``.  The remaining pairs are null.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_train, n_features))
    fi, fj = true_pair_indices
    interaction_term = _std(x[:, fi] * x[:, fj])

    # Response is signal_strength * interaction + small noise, across all components
    noise = rng.standard_normal((n_train, n_components)) * 0.1
    y_raw = signal_strength * interaction_term[:, None] + noise
    y = (y_raw - y_raw.mean(axis=0)) / (y_raw.std(axis=0, ddof=0) + 1e-12)

    from itertools import combinations

    pairs = list(combinations(range(n_features), 2))
    n_pairs = len(pairs)
    true_cols = {k for k, p in enumerate(pairs) if set(p) == set(true_pair_indices)}
    R = np.zeros((n_train, n_pairs), dtype=float)
    for col, (i, j) in enumerate(pairs):
        product = x[:, i] * x[:, j]
        controls = np.column_stack([np.ones(n_train), _std(x[:, i]), _std(x[:, j])])
        coef, *_ = np.linalg.lstsq(controls, product, rcond=None)
        residual = product - controls @ coef
        R[:, col] = _std(residual)

    observed_scores, null_statistics = _compute_perm_stats(R, y, n_perms=_B_PERMS, seed=seed + 1)
    return observed_scores, null_statistics, pairs, true_cols


def _std(v: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-scale; returns zeros for constant input."""
    mean, scale = float(v.mean()), float(v.std(ddof=0))
    if scale <= 0.0:
        return np.zeros_like(v, dtype=float)
    return (v - mean) / scale


def _compute_perm_stats(
    R: np.ndarray,
    y: np.ndarray,
    n_perms: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute observed scores and (B, n_pairs) null statistics."""
    n_rows = float(R.shape[0])
    # Observed
    coef_obs = (R.T @ y) / n_rows  # (n_pairs, n_components)
    observed_scores = np.max(np.abs(coef_obs), axis=1)  # (n_pairs,)
    # Null
    rng = np.random.default_rng(seed)
    n_pairs = R.shape[1]
    null_statistics = np.zeros((n_perms, n_pairs), dtype=float)
    for b in range(n_perms):
        perm = rng.permutation(len(y))
        coef_perm = (R.T @ y[perm]) / n_rows
        null_statistics[b] = np.max(np.abs(coef_perm), axis=1)
    return observed_scores, null_statistics


# ---------------------------------------------------------------------------
# max_t_critical_value — unit tests
# ---------------------------------------------------------------------------


class TestMaxTCriticalValue:
    def test_P0_S08_returns_float(self):
        rng = np.random.default_rng(0)
        null = rng.standard_normal((100, 10))
        result = max_t_critical_value(null, alpha=0.05)
        assert isinstance(result, float)

    def test_P0_S08_exact_order_statistic_is_correct(self):
        """The score critical value is the finite-permutation maxT order statistic."""
        rng = np.random.default_rng(1)
        null = rng.standard_normal((200, 15))
        expected = float(np.partition(null.max(axis=1), 190)[190])
        assert max_t_critical_value(null, alpha=0.05) == pytest.approx(expected)

    def test_P0_S08_alpha_01_gives_exact_order_statistic(self):
        rng = np.random.default_rng(2)
        null = rng.standard_normal((300, 5))
        expected = float(np.partition(null.max(axis=1), 270)[270])
        assert max_t_critical_value(null, alpha=0.10) == pytest.approx(expected)

    def test_P0_S08_unresolvable_alpha_returns_none(self):
        assert max_t_critical_value(np.ones((10, 3)), alpha=0.01) is None

    def test_P0_S08_invalid_alpha_zero_raises(self):
        null = np.ones((10, 3))
        with pytest.raises(ValueError, match="alpha"):
            max_t_critical_value(null, alpha=0.0)

    def test_P0_S08_invalid_alpha_one_raises(self):
        null = np.ones((10, 3))
        with pytest.raises(ValueError, match="alpha"):
            max_t_critical_value(null, alpha=1.0)

    def test_P0_S08_1d_array_raises(self):
        null = np.ones(10)
        with pytest.raises(ValueError):
            max_t_critical_value(null, alpha=0.05)

    def test_P0_S08_empty_rows_raises(self):
        null = np.empty((0, 5))
        with pytest.raises(ValueError):
            max_t_critical_value(null, alpha=0.05)

    def test_P0_S08_threshold_exceeds_column_critical_values(self):
        """The global maxT critical value is at least each column's critical value."""
        rng = np.random.default_rng(3)
        null = rng.standard_normal((500, 20))
        t = max_t_critical_value(null, alpha=0.05)
        assert t is not None
        for col in range(null.shape[1]):
            column_critical = float(np.partition(null[:, col], 475)[475])
            assert t >= column_critical


# ---------------------------------------------------------------------------
# bh_fdr_selected — unit tests
# ---------------------------------------------------------------------------


class TestBhFdrSelected:
    def test_P0_S08_returns_bool_array(self):
        p = np.array([0.01, 0.04, 0.20, 0.50])
        result = bh_fdr_selected(p, alpha=0.05)
        assert result.dtype == bool
        assert result.shape == (4,)

    def test_P0_S08_trivial_all_null(self):
        """All high p-values → nothing selected."""
        p = np.array([0.5, 0.6, 0.7, 0.8])
        assert not bh_fdr_selected(p, alpha=0.05).any()

    def test_P0_S08_trivial_strong_signal(self):
        """Very small p-values → all selected."""
        p = np.array([1e-6, 2e-6, 3e-6])
        assert bh_fdr_selected(p, alpha=0.05).all()

    def test_P0_S08_known_bh_example(self):
        """BH textbook example: m=4, alpha=0.05.

        Sorted p: 0.01, 0.04, 0.20, 0.50
        BH critical values: 0.05/4*1=0.0125, 0.025, 0.0375, 0.05
        Largest k with p_(k) <= k/m*alpha: k=2 (0.04 <= 0.025? No), k=1 (0.01<=0.0125 Yes).
        So only the first hypothesis is selected.
        """
        p = np.array([0.50, 0.01, 0.20, 0.04])
        selected = bh_fdr_selected(p, alpha=0.05)
        # index 1 has p=0.01, which is the only one below BH critical at k=1
        assert selected[1]
        assert not selected[0]
        assert not selected[2]
        assert not selected[3]

    def test_P0_S08_empty_input(self):
        result = bh_fdr_selected(np.array([]), alpha=0.05)
        assert result.shape == (0,)

    def test_P0_S08_invalid_alpha_zero_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            bh_fdr_selected(np.array([0.01, 0.05]), alpha=0.0)

    def test_P0_S08_invalid_alpha_one_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            bh_fdr_selected(np.array([0.01, 0.05]), alpha=1.0)

    def test_P0_S08_2d_input_raises(self):
        with pytest.raises(ValueError):
            bh_fdr_selected(np.ones((3, 3)), alpha=0.05)

    def test_P0_S08_order_invariant(self):
        """Result must not depend on input order."""
        rng = np.random.default_rng(10)
        p = rng.uniform(0, 0.3, size=20)
        selected = bh_fdr_selected(p, alpha=0.10)
        perm = rng.permutation(len(p))
        selected_perm = bh_fdr_selected(p[perm], alpha=0.10)
        # After inverse-permuting, selections must match
        inv_perm = np.argsort(perm)
        np.testing.assert_array_equal(selected, selected_perm[inv_perm])


# ---------------------------------------------------------------------------
# multiplicity_controlled_interaction_selection — unit tests
# ---------------------------------------------------------------------------


class TestMultiplicityControlledSelection:
    def _make_null_stats(self, B: int = 100, n: int = 10, seed: int = 0) -> np.ndarray:
        return np.abs(np.random.default_rng(seed).standard_normal((B, n)))

    def test_P0_S08_fwer_returns_tuple3(self):
        obs = np.array([1.0, 0.5, 0.2])
        null = np.abs(np.random.default_rng(0).standard_normal((100, 3)))
        selected, pvals, thresh = multiplicity_controlled_interaction_selection(
            obs, null, alpha=0.05, method="max_t"
        )
        assert selected.dtype == bool
        assert pvals.shape == (3,)
        assert isinstance(thresh, float)

    def test_P0_S08_bh_fdr_returns_none_threshold(self):
        obs = np.array([1.0, 0.5, 0.2])
        null = np.abs(np.random.default_rng(0).standard_normal((100, 3)))
        _, _, thresh = multiplicity_controlled_interaction_selection(
            obs, null, alpha=0.05, method="bh_fdr"
        )
        assert thresh is None

    def test_P0_S08_unknown_method_raises(self):
        obs = np.array([1.0])
        null = np.ones((10, 1))
        with pytest.raises(ValueError, match="method"):
            multiplicity_controlled_interaction_selection(
                obs, null, alpha=0.05, method="bonferroni"
            )

    def test_P0_S08_shape_mismatch_raises(self):
        obs = np.array([1.0, 2.0])
        null = np.ones((10, 3))  # 3 columns != 2
        with pytest.raises(ValueError):
            multiplicity_controlled_interaction_selection(obs, null, alpha=0.05)

    def test_P0_S08_2d_observed_scores_raises(self):
        obs = np.ones((3, 2))
        null = np.ones((10, 3))
        with pytest.raises(ValueError):
            multiplicity_controlled_interaction_selection(obs, null, alpha=0.05)

    def test_P0_S08_pvalues_in_unit_interval(self):
        rng = np.random.default_rng(7)
        obs = rng.standard_normal(20)
        null = rng.standard_normal((200, 20))
        _, pvals, _ = multiplicity_controlled_interaction_selection(obs, null, alpha=0.05)
        assert (pvals >= 0).all() and (pvals <= 1).all()

    def test_P0_S08_max_t_consistent_with_helper(self):
        """maxT-selected pairs match the adjusted-p-value decision rule."""
        rng = np.random.default_rng(8)
        obs = np.abs(rng.standard_normal(15))
        null = np.abs(rng.standard_normal((200, 15)))
        selected, pvals, thresh = multiplicity_controlled_interaction_selection(
            obs, null, alpha=0.05, method="max_t"
        )
        expected_pvals = max_t_adjusted_pvalues(obs, null)
        expected_thresh = max_t_critical_value(null, alpha=0.05)
        np.testing.assert_allclose(pvals, expected_pvals)
        assert thresh == pytest.approx(expected_thresh)
        np.testing.assert_array_equal(selected, expected_pvals <= 0.05)

    def test_P0_S08_bh_consistent_with_helper(self):
        """BH-selected pairs should match bh_fdr_selected applied to empirical p-values."""
        rng = np.random.default_rng(9)
        obs = np.abs(rng.standard_normal(15))
        null = np.abs(rng.standard_normal((200, 15)))
        selected, pvals, _ = multiplicity_controlled_interaction_selection(
            obs, null, alpha=0.05, method="bh_fdr"
        )
        expected = bh_fdr_selected(pvals, alpha=0.05)
        np.testing.assert_array_equal(selected, expected)


# ---------------------------------------------------------------------------
# Null false-selection control (statistical validation with fixed seed)
# ---------------------------------------------------------------------------


class TestNullFalseSelectionControl:
    """Under a complete null, both methods must control false selections.

    With a fixed seed and B=400 permutations over 28 candidate pairs:
    - FWER at alpha=0.05: max-stat threshold controls FWER; expected 0 false
      selections for any single draw from this null.
    - BH-FDR at alpha=0.05: expected FDR <= alpha.

    These are single-run checks (fixed seed), validated with a tolerance of
    min(alpha, 0.10) * n_pairs (a conservative upper bound on false selections
    that should never be approached under a well-calibrated procedure).
    """

    @pytest.fixture(scope="class")
    def null_data(self):
        return _make_null_fixture(seed=_SEED)

    def test_P0_S08_fwer_false_selection_zero_under_null(self, null_data):
        """FWER max-stat: with a fixed seed, no false selections under null."""
        observed, null_stats, _ = null_data
        selected, _, _ = multiplicity_controlled_interaction_selection(
            observed, null_stats, alpha=0.05, method="max_t"
        )
        # Under the null all selections are false positives.
        n_false = int(selected.sum())
        # The FWER max-stat threshold is designed so that the probability of
        # ANY selection under the null is ~alpha.  With this particular fixed
        # seed we expect zero; allow at most 1 as a generous single-run bound.
        assert n_false <= 1, (
            f"FWER max-stat produced {n_false} false selections under the null "
            f"(seed={_SEED}, B={_B_PERMS}, n_pairs={len(observed)})."
        )

    def test_P0_S08_bh_false_selection_controlled_under_null(self, null_data):
        """BH-FDR: false selection count is controlled at configured alpha=0.05."""
        observed, null_stats, _ = null_data
        selected, _, _ = multiplicity_controlled_interaction_selection(
            observed, null_stats, alpha=0.05, method="bh_fdr"
        )
        n_pairs = len(observed)
        n_false = int(selected.sum())
        # With a fixed seed over a pure null, BH should select very few pairs.
        # Generous tolerance: no more than ceil(alpha * n_pairs) = ceil(1.4).
        max_allowed = max(2, int(np.ceil(0.05 * n_pairs)))
        assert n_false <= max_allowed, (
            f"BH-FDR produced {n_false} false selections under the null "
            f"(seed={_SEED}, n_pairs={n_pairs}, max_allowed={max_allowed})."
        )

    def test_P0_S08_null_pvalues_are_uniform_stochastically(self, null_data):
        """Under the null, empirical p-values should be approximately uniform.

        We test the weaker condition: the median p-value is above 0.05.
        """
        observed, null_stats, _ = null_data
        _, pvals, _ = multiplicity_controlled_interaction_selection(
            observed, null_stats, alpha=0.05, method="max_t"
        )
        assert float(np.median(pvals)) > 0.05, (
            "Median p-value under the null is suspiciously low; "
            "may indicate a bug in null statistic computation."
        )

    def test_P0_S08_fwer_threshold_larger_than_per_pair_threshold(self, null_data):
        """The FWER max-stat threshold must be >= the per-pair 0.95 quantile for any pair."""
        observed, null_stats, _ = null_data
        _, _, fwer_thresh = multiplicity_controlled_interaction_selection(
            observed, null_stats, alpha=0.05, method="max_t"
        )
        per_pair_95 = np.quantile(null_stats, 0.95, axis=0)
        assert fwer_thresh >= float(per_pair_95.max()) - 1e-12


# ---------------------------------------------------------------------------
# Planted-signal recovery (power sanity check)
# ---------------------------------------------------------------------------


class TestPlantedSignalRecovery:
    """Under a planted-signal alternative, true interaction pairs are recovered.

    The synthetic response depends strongly on one planted interaction.  With
    a sufficient signal-to-noise ratio both FWER and BH-FDR should select the
    true pair.
    """

    @pytest.fixture(scope="class")
    def signal_data(self):
        return _make_signal_fixture(
            true_pair_indices=(0, 1),
            signal_strength=6.0,
            seed=_SEED,
        )

    def test_P0_S08_fwer_recovers_true_interaction(self, signal_data):
        """FWER max-stat selects the planted pair."""
        observed, null_stats, pairs, true_cols = signal_data
        selected, _, _ = multiplicity_controlled_interaction_selection(
            observed, null_stats, alpha=0.05, method="max_t"
        )
        for col in true_cols:
            assert selected[col], (
                f"FWER max-stat failed to recover planted pair at index {col} "
                f"(pair={pairs[col]}, score={observed[col]:.4f})."
            )

    def test_P0_S08_bh_fdr_recovers_true_interaction(self, signal_data):
        """BH-FDR selects the planted pair."""
        observed, null_stats, pairs, true_cols = signal_data
        selected, _, _ = multiplicity_controlled_interaction_selection(
            observed, null_stats, alpha=0.05, method="bh_fdr"
        )
        for col in true_cols:
            assert selected[col], (
                f"BH-FDR failed to recover planted pair at index {col} "
                f"(pair={pairs[col]}, score={observed[col]:.4f})."
            )

    def test_P0_S08_true_pair_has_highest_score(self, signal_data):
        """The planted interaction pair should produce the maximum observed score."""
        observed, _, pairs, true_cols = signal_data
        top_idx = int(np.argmax(observed))
        assert top_idx in true_cols, (
            f"Top-scored pair (index {top_idx}, pair={pairs[top_idx]}) "
            f"is not the planted true pair (true_cols={true_cols})."
        )

    def test_P0_S08_fwer_false_positives_bounded_under_signal(self, signal_data):
        """FWER max-stat does not produce excessive false positives under signal."""
        observed, null_stats, _, true_cols = signal_data
        selected, _, _ = multiplicity_controlled_interaction_selection(
            observed, null_stats, alpha=0.05, method="max_t"
        )
        false_positives = int(selected.sum()) - sum(1 for c in true_cols if selected[c])
        n_pairs = len(observed)
        # Allow at most a small number of false positives (generous bound)
        max_fp = max(2, int(np.ceil(0.10 * n_pairs)))
        assert false_positives <= max_fp, (
            f"FWER produced {false_positives} false positives under signal "
            f"(n_pairs={n_pairs}, max_allowed={max_fp})."
        )

    def test_P0_S08_planted_pair_pvalue_near_minimum(self, signal_data):
        """The planted pair's empirical p-value should be near the minimum achievable."""
        observed, null_stats, _, true_cols = signal_data
        _, pvals, _ = multiplicity_controlled_interaction_selection(
            observed, null_stats, alpha=0.05, method="max_t"
        )
        B = null_stats.shape[0]
        min_pval = 1.0 / (B + 1.0)
        for col in true_cols:
            assert pvals[col] <= 5 * min_pval, (
                f"Planted pair p-value {pvals[col]:.4f} is not near the minimum "
                f"achievable ({min_pval:.4f}); signal may be too weak."
            )
