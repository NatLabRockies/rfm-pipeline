"""P0-S05: Threshold/pruning selection uses internal validation only.

Tests:
- Selection runs using only train/val data from a SealedSplitResult.
- Attempting to pass an unsealed split raises SealedTestAccessError.
- Selection is deterministic given the same seed.
- The known-optimum threshold is recovered by the trivial scorer.
- Empty threshold list raises ValueError.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rfm_pipeline.data import SealedTestAccessError, make_sealed_split
from rfm_pipeline.manuscript_pipeline_helpers import select_threshold_on_internal_validation

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_data() -> pd.DataFrame:
    """Minimal synthetic dataset with a 'stratum' column."""
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame(
        {
            "stratum": (["A"] * (n // 2)) + (["B"] * (n // 2)),
            "x": rng.standard_normal(n),
            "y": rng.standard_normal(n),
        }
    )
    return df


@pytest.fixture()
def sealed_split(simple_data: pd.DataFrame):
    """Sealed split from simple_data."""
    return make_sealed_split(
        simple_data,
        strata_column="stratum",
        val_fraction=0.20,
        test_fraction=0.20,
        random_state=7,
    )


def make_trivial_scorer(optimum: float):
    """Return a scorer whose minimum is at *optimum*.

    score(t, train, val) = |t - optimum|.  The scorer ignores the data frames
    (it is only a structural test of the selection machinery).
    """

    def scorer(threshold: float, train_df: pd.DataFrame, val_df: pd.DataFrame) -> float:
        # Intentionally does NOT access any external test partition.
        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(val_df, pd.DataFrame)
        return abs(threshold - optimum)

    return scorer


# ---------------------------------------------------------------------------
# Core selection behaviour
# ---------------------------------------------------------------------------


class TestSelectionUsesInternalDataOnly:
    def test_P0_S05_selects_known_optimum(self, sealed_split):
        """The trivial scorer's known optimum is correctly selected."""
        optimum = 0.3
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
        best = select_threshold_on_internal_validation(
            sealed_split,
            thresholds,
            make_trivial_scorer(optimum),
            seed=0,
        )
        assert best == pytest.approx(optimum)

    def test_P0_S05_split_remains_sealed_after_selection(self, sealed_split):
        """Calling selection must not unseal the split."""
        thresholds = [0.1, 0.5, 0.9]
        select_threshold_on_internal_validation(
            sealed_split,
            thresholds,
            make_trivial_scorer(0.5),
            seed=0,
        )
        # split._sealed must still be True — test access must still raise
        with pytest.raises(SealedTestAccessError):
            _ = sealed_split.test

    def test_P0_S05_scorer_receives_train_and_val_frames(self, sealed_split):
        """Scorer is called with DataFrames matching train/val sizes."""
        expected_train_len = len(sealed_split.train)
        expected_val_len = len(sealed_split.val)
        observed: list[tuple[int, int]] = []

        def recording_scorer(threshold, train_df, val_df):
            observed.append((len(train_df), len(val_df)))
            return abs(threshold - 0.5)

        thresholds = [0.3, 0.5, 0.7]
        select_threshold_on_internal_validation(sealed_split, thresholds, recording_scorer, seed=0)
        for train_len, val_len in observed:
            assert train_len == expected_train_len
            assert val_len == expected_val_len

    def test_P0_S05_scorer_not_called_with_test_data(self, sealed_split):
        """Scorer must never receive the sealed test partition.

        We verify this structurally: after selection the split is still sealed,
        so any access to split.test would have raised.  We additionally confirm
        that the data frames passed to the scorer match train/val shapes, not
        the test shape.
        """
        test_like_size: list[int] = []

        # We cannot know the test size without unsealing — so we unseal a
        # *separate* split built from the same data to obtain the test size for
        # comparison, without touching the split under test.
        import numpy as np

        from rfm_pipeline.data import make_sealed_split as _make

        rng = np.random.default_rng(42)
        n = 200
        ref_data = pd.DataFrame(
            {
                "stratum": (["A"] * 100) + (["B"] * 100),
                "x": rng.standard_normal(n),
                "y": rng.standard_normal(n),
            }
        )
        ref_split = _make(
            ref_data,
            strata_column="stratum",
            val_fraction=0.20,
            test_fraction=0.20,
            random_state=7,
        )
        ref_split.unseal(reason="reference size check")
        ref_test_len = len(ref_split.test)

        def size_recording_scorer(threshold, train_df, val_df):
            test_like_size.append(len(train_df))
            return abs(threshold - 0.5)

        select_threshold_on_internal_validation(sealed_split, [0.5], size_recording_scorer, seed=0)
        # The train frame passed to the scorer must NOT equal the test size.
        assert all(s != ref_test_len for s in test_like_size)


# ---------------------------------------------------------------------------
# Sealed-test guard enforcement
# ---------------------------------------------------------------------------


class TestSealedTestGuard:
    def test_P0_S05_unsealed_split_raises_sealed_test_error(self, sealed_split):
        """Passing an already-unsealed split must raise SealedTestAccessError."""
        sealed_split.unseal(reason="premature unseal for test")
        thresholds = [0.1, 0.5, 0.9]
        with pytest.raises(SealedTestAccessError, match="already been unsealed"):
            select_threshold_on_internal_validation(
                sealed_split,
                thresholds,
                make_trivial_scorer(0.5),
                seed=0,
            )

    def test_P0_S05_error_is_sealed_test_access_error_subtype(self, sealed_split):
        """The raised exception must be SealedTestAccessError (a RuntimeError)."""
        sealed_split.unseal(reason="subtype check")
        with pytest.raises(SealedTestAccessError) as exc_info:
            select_threshold_on_internal_validation(
                sealed_split, [0.1], make_trivial_scorer(0.1), seed=0
            )
        assert isinstance(exc_info.value, RuntimeError)

    def test_P0_S05_raises_before_scorer_is_called(self, sealed_split):
        """Guard must fire before the scorer is invoked."""
        sealed_split.unseal(reason="guard order check")
        scorer_called = []

        def tracking_scorer(t, train_df, val_df):
            scorer_called.append(t)
            return 0.0

        with pytest.raises(SealedTestAccessError):
            select_threshold_on_internal_validation(
                sealed_split, [0.1, 0.5], tracking_scorer, seed=0
            )
        assert scorer_called == [], "scorer was called despite unsealed split"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_P0_S05_same_seed_gives_same_result(self, sealed_split, simple_data):
        """Two calls with the same seed on the same split return the same threshold."""
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]

        def const_scorer(t, train_df, val_df):
            # Constant score: all thresholds tie; tie-break is seed-driven.
            return 1.0

        split_a = make_sealed_split(
            simple_data,
            strata_column="stratum",
            val_fraction=0.20,
            test_fraction=0.20,
            random_state=7,
        )
        split_b = make_sealed_split(
            simple_data,
            strata_column="stratum",
            val_fraction=0.20,
            test_fraction=0.20,
            random_state=7,
        )
        result_a = select_threshold_on_internal_validation(
            split_a, thresholds, const_scorer, seed=99
        )
        result_b = select_threshold_on_internal_validation(
            split_b, thresholds, const_scorer, seed=99
        )
        assert result_a == result_b

    def test_P0_S05_different_seeds_may_differ_on_ties(self, simple_data):
        """Different seeds can produce different results when all scores tie."""
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]

        def const_scorer(t, train_df, val_df):
            return 1.0

        results = set()
        for seed_val in range(50):
            sp = make_sealed_split(
                simple_data,
                strata_column="stratum",
                val_fraction=0.20,
                test_fraction=0.20,
                random_state=7,
            )
            results.add(
                select_threshold_on_internal_validation(sp, thresholds, const_scorer, seed=seed_val)
            )
        # With 50 different seeds and 5 candidates, at least 2 distinct outcomes.
        assert len(results) > 1

    def test_P0_S05_no_tie_is_always_deterministic_without_seed(self, sealed_split):
        """When there is a unique best threshold, no seed is needed for determinism."""
        optimum = 0.35
        thresholds = [0.1, 0.2, 0.35, 0.5, 0.9]
        result_1 = select_threshold_on_internal_validation(
            sealed_split, thresholds, make_trivial_scorer(optimum), seed=None
        )
        # Rebuild to get a fresh sealed split
        import numpy as np

        from rfm_pipeline.data import make_sealed_split as _make

        rng = np.random.default_rng(42)
        n = 200
        data2 = pd.DataFrame(
            {
                "stratum": (["A"] * 100) + (["B"] * 100),
                "x": rng.standard_normal(n),
                "y": rng.standard_normal(n),
            }
        )
        sp2 = _make(
            data2, strata_column="stratum", val_fraction=0.20, test_fraction=0.20, random_state=7
        )
        result_2 = select_threshold_on_internal_validation(
            sp2, thresholds, make_trivial_scorer(optimum), seed=None
        )
        assert result_1 == result_2 == pytest.approx(optimum)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_P0_S05_empty_thresholds_raises_value_error(self, sealed_split):
        with pytest.raises(ValueError, match="non-empty"):
            select_threshold_on_internal_validation(
                sealed_split, [], make_trivial_scorer(0.5), seed=0
            )

    def test_P0_S05_single_threshold_returned_directly(self, sealed_split):
        result = select_threshold_on_internal_validation(
            sealed_split, [0.42], make_trivial_scorer(0.42), seed=0
        )
        assert result == pytest.approx(0.42)
