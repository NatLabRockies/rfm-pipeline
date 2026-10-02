"""Tests for the sealed train/validation/test split protocol.

Tests:
- Stratification balance is preserved across all three partitions.
- Accessing the sealed test partition before unsealing raises SealedTestAccessError.
- Explicit unseal succeeds and is recorded in the unseal log.
- Strata balance DataFrame is introspectable.
- Combined partition sizes equal the total input size.
- Validation partition is always accessible without unsealing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rfm_pipeline.data import SealedSplitResult, SealedTestAccessError, make_sealed_split

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def stratified_data() -> pd.DataFrame:
    """Synthetic dataset with a 'stratum' key column and four equal strata."""
    rng = np.random.default_rng(0)
    n_per_stratum = 100
    strata = ["A", "B", "C", "D"]
    records = []
    for s in strata:
        records.append(
            pd.DataFrame(
                {
                    "stratum": s,
                    "x1": rng.standard_normal(n_per_stratum),
                    "x2": rng.standard_normal(n_per_stratum),
                    "y": rng.standard_normal(n_per_stratum),
                }
            )
        )
    return pd.concat(records, ignore_index=True)


@pytest.fixture()
def sealed_split(stratified_data: pd.DataFrame) -> SealedSplitResult:
    """Default sealed split from the stratified_data fixture."""
    return make_sealed_split(
        stratified_data,
        strata_column="stratum",
        val_fraction=0.15,
        test_fraction=0.15,
        random_state=42,
    )


# ---------------------------------------------------------------------------
# Partition size coverage
# ---------------------------------------------------------------------------


class TestPartitionSizes:
    def test_total_size_equals_input(self, sealed_split, stratified_data):
        """Train + val + sealed-test must cover all input rows."""
        n_train = len(sealed_split.train)
        n_val = len(sealed_split.val)
        # Unseal just to count rows for this verification.
        sealed_split.unseal(reason="test: size verification")
        n_test = len(sealed_split.test)
        assert n_train + n_val + n_test == len(stratified_data)

    def test_train_is_largest_partition(self, sealed_split, stratified_data):
        """Train partition should be larger than val (no access to test needed)."""
        assert len(sealed_split.train) > len(sealed_split.val)

    def test_val_is_nonempty(self, sealed_split):
        assert len(sealed_split.val) > 0

    def test_train_is_nonempty(self, sealed_split):
        assert len(sealed_split.train) > 0


# ---------------------------------------------------------------------------
# Strata balance
# ---------------------------------------------------------------------------


class TestStrataBalance:
    def test_strata_balance_has_correct_columns(self, sealed_split):
        balance = sealed_split.strata_balance
        assert set(balance.columns) == {"train", "val", "test"}

    def test_strata_balance_index_contains_all_strata(self, sealed_split):
        balance = sealed_split.strata_balance
        assert set(balance.index.tolist()) == {"A", "B", "C", "D"}

    def test_all_strata_represented_in_train(self, sealed_split):
        """Every stratum must appear at least once in the training partition."""
        assert (sealed_split.strata_balance["train"] > 0).all()

    def test_all_strata_represented_in_val(self, sealed_split):
        assert (sealed_split.strata_balance["val"] > 0).all()

    def test_all_strata_represented_in_test(self, sealed_split):
        assert (sealed_split.strata_balance["test"] > 0).all()

    def test_strata_proportions_roughly_equal_in_train(self, sealed_split):
        """With 4 equal strata each partition should be ~25% each stratum."""
        train_counts = sealed_split.strata_balance["train"]
        proportions = train_counts / train_counts.sum()
        assert (proportions > 0.15).all(), f"Stratum underrepresented in train: {proportions}"
        assert (proportions < 0.40).all(), f"Stratum overrepresented in train: {proportions}"

    def test_strata_proportions_roughly_equal_in_val(self, sealed_split):
        val_counts = sealed_split.strata_balance["val"]
        proportions = val_counts / val_counts.sum()
        assert (proportions > 0.15).all(), f"Stratum underrepresented in val: {proportions}"
        assert (proportions < 0.40).all(), f"Stratum overrepresented in val: {proportions}"

    def test_balance_row_sums_match_partition_lengths(self, sealed_split, stratified_data):
        """Strata balance counts must add up to actual partition sizes."""
        balance = sealed_split.strata_balance
        assert balance["train"].sum() == len(sealed_split.train)
        assert balance["val"].sum() == len(sealed_split.val)
        sealed_split.unseal(reason="test: balance row sums verification")
        assert balance["test"].sum() == len(sealed_split.test)


# ---------------------------------------------------------------------------
# Sealed-test access guard
# ---------------------------------------------------------------------------


class TestSealedTestAccessGuard:
    def test_accessing_test_before_unseal_raises(self, sealed_split):
        """Accessing .test on a sealed result must raise SealedTestAccessError."""
        with pytest.raises(SealedTestAccessError):
            _ = sealed_split.test

    def test_error_is_runtime_subtype(self, sealed_split):
        """SealedTestAccessError must be a RuntimeError."""
        assert issubclass(SealedTestAccessError, RuntimeError)

    def test_train_accessible_while_sealed(self, sealed_split):
        """train is always accessible without unsealing."""
        df = sealed_split.train
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_val_accessible_while_sealed(self, sealed_split):
        """val is always accessible without unsealing."""
        df = sealed_split.val
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_second_access_still_raises_before_unseal(self, sealed_split):
        """Repeated access before unseal must keep raising."""
        with pytest.raises(SealedTestAccessError):
            _ = sealed_split.test
        with pytest.raises(SealedTestAccessError):
            _ = sealed_split.test


# ---------------------------------------------------------------------------
# Explicit unseal
# ---------------------------------------------------------------------------


class TestExplicitUnseal:
    def test_unseal_allows_test_access(self, sealed_split):
        """After unsealing, .test must return a non-empty DataFrame."""
        sealed_split.unseal(reason="final evaluation")
        df = sealed_split.test
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_unseal_log_records_reason(self, sealed_split):
        """unseal_log must capture the supplied reason string."""
        sealed_split.unseal(reason="final evaluation phase")
        log = sealed_split.unseal_log
        assert len(log) == 1
        assert log[0]["reason"] == "final evaluation phase"

    def test_unseal_log_has_timestamp(self, sealed_split):
        """Each unseal entry must contain a timestamp key."""
        sealed_split.unseal(reason="check timestamp")
        log = sealed_split.unseal_log
        assert "timestamp" in log[0]
        assert isinstance(log[0]["timestamp"], str)
        assert len(log[0]["timestamp"]) > 0

    def test_unseal_log_is_snapshot(self, sealed_split):
        """Mutating the returned unseal_log must not affect the internal log."""
        sealed_split.unseal(reason="snapshot test")
        log_copy = sealed_split.unseal_log
        log_copy.append({"reason": "injected", "timestamp": "fake"})
        assert len(sealed_split.unseal_log) == 1

    def test_unseal_log_empty_before_unsealing(self, sealed_split):
        assert sealed_split.unseal_log == []

    def test_multiple_unseal_calls_all_logged(self, sealed_split):
        sealed_split.unseal(reason="first")
        sealed_split.unseal(reason="second")
        log = sealed_split.unseal_log
        assert len(log) == 2
        assert log[0]["reason"] == "first"
        assert log[1]["reason"] == "second"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_missing_strata_column_raises(self, stratified_data):
        with pytest.raises(ValueError, match="strata_column"):
            make_sealed_split(stratified_data, strata_column="nonexistent")

    def test_bad_fractions_raise(self, stratified_data):
        with pytest.raises(ValueError):
            make_sealed_split(
                stratified_data,
                strata_column="stratum",
                val_fraction=0.6,
                test_fraction=0.6,
            )
