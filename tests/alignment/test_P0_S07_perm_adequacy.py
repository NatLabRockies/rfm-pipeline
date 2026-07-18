"""P0-S07: Interaction permutation-adequacy guard (F5).

With B permutations the smallest resolvable per-pair p-value is 1/(B+1).
After Bonferroni correction across *family_size* pairs the per-pair alpha is
(1 - quantile) / family_size.  The guard must raise PermutationAdequacyError
when B is too small to resolve the requested tail.

Tests:
- insufficient count for a 0.995 tail raises PermutationAdequacyError
- sufficient count passes without error
- boundary case: exactly the minimum required count passes
- family_size scales the required count correctly
- error message is informative
- min_permutations_required is correct for known values
- invalid quantile / family_size arguments raise ValueError
"""

from __future__ import annotations

import pytest

from rfm_pipeline.manuscript_stages import (
    PermutationAdequacyError,
    check_permutation_adequacy,
    min_permutations_required,
)

# ---------------------------------------------------------------------------
# min_permutations_required — unit tests
# ---------------------------------------------------------------------------


class TestMinPermutationsRequired:
    def test_P0_S07_known_value_0995_family1(self):
        """q=0.995, family=1 → need B >= 199."""
        # alpha_per_pair = 0.005; 1/alpha = 200; min_B = 200 - 1 = 199
        assert min_permutations_required(0.995, family_size=1) == 199

    def test_P0_S07_known_value_0995_family2(self):
        """q=0.995, family=2 → need B >= 399 (Bonferroni doubles requirement)."""
        # alpha_per_pair = 0.0025; 1/alpha = 400; min_B = 399
        assert min_permutations_required(0.995, family_size=2) == 399

    def test_P0_S07_known_value_09_family1(self):
        """q=0.9, family=1 → need B >= 9."""
        # alpha=0.1; 1/alpha=10; min_B=10-1=9
        assert min_permutations_required(0.9, family_size=1) == 9

    def test_P0_S07_family_size_scales_requirement(self):
        """min_permutations_required scales correctly with family_size."""
        # q=0.995, fs=1: ceil(1/0.005)-1 = 200-1 = 199
        # q=0.995, fs=10: ceil(10/0.005)-1 = 2000-1 = 1999
        b1 = min_permutations_required(0.995, family_size=1)
        b10 = min_permutations_required(0.995, family_size=10)
        assert b1 == 199
        assert b10 == 1999

    def test_P0_S07_invalid_quantile_zero_raises(self):
        """quantile=0 is invalid."""
        with pytest.raises(ValueError, match="quantile"):
            min_permutations_required(0.0)

    def test_P0_S07_invalid_quantile_one_raises(self):
        """quantile=1 is invalid."""
        with pytest.raises(ValueError, match="quantile"):
            min_permutations_required(1.0)

    def test_P0_S07_invalid_family_size_zero_raises(self):
        """family_size=0 is invalid."""
        with pytest.raises(ValueError, match="family_size"):
            min_permutations_required(0.995, family_size=0)

    def test_P0_S07_invalid_family_size_negative_raises(self):
        """family_size=-1 is invalid."""
        with pytest.raises(ValueError, match="family_size"):
            min_permutations_required(0.995, family_size=-1)


# ---------------------------------------------------------------------------
# check_permutation_adequacy — guard behaviour
# ---------------------------------------------------------------------------


class TestCheckPermutationAdequacyInsufficient:
    def test_P0_S07_insufficient_count_0995_raises(self):
        """B=30 is insufficient for q=0.995 (needs 199); must raise."""
        with pytest.raises(PermutationAdequacyError):
            check_permutation_adequacy(30, 0.995, family_size=1)

    def test_P0_S07_error_message_mentions_B_and_quantile(self):
        """The error message should reference the configured B and the quantile."""
        with pytest.raises(PermutationAdequacyError, match="B=30"):
            check_permutation_adequacy(30, 0.995, family_size=1)

    def test_P0_S07_error_message_mentions_minimum_required(self):
        """The error message should state the minimum required count."""
        with pytest.raises(PermutationAdequacyError, match="199"):
            check_permutation_adequacy(30, 0.995, family_size=1)

    def test_P0_S07_zero_perms_raises(self):
        """B=0 is always insufficient."""
        with pytest.raises((PermutationAdequacyError, ValueError)):
            check_permutation_adequacy(0, 0.995, family_size=1)

    def test_P0_S07_one_below_boundary_raises(self):
        """B = min_required - 1 must raise."""
        min_B = min_permutations_required(0.995, family_size=1)
        with pytest.raises(PermutationAdequacyError):
            check_permutation_adequacy(min_B - 1, 0.995, family_size=1)


class TestCheckPermutationAdequacySufficient:
    def test_P0_S07_sufficient_count_passes(self):
        """B=200 is sufficient for q=0.995 (needs 199); must not raise."""
        check_permutation_adequacy(200, 0.995, family_size=1)

    def test_P0_S07_large_count_passes(self):
        """Generous B passes for a modest quantile and family_size."""
        # q=0.99, fs=1 → min_B=99; B=1000 >> 99
        check_permutation_adequacy(1000, 0.99, family_size=1)

    def test_P0_S07_boundary_exact_minimum_passes(self):
        """B = min_required exactly should pass (boundary case)."""
        min_B = min_permutations_required(0.995, family_size=1)
        check_permutation_adequacy(min_B, 0.995, family_size=1)  # must not raise

    def test_P0_S07_family_size_boundary_passes(self):
        """B at the exact family-corrected minimum passes."""
        min_B = min_permutations_required(0.995, family_size=5)
        check_permutation_adequacy(min_B, 0.995, family_size=5)

    def test_P0_S07_family_size_below_boundary_raises(self):
        """B one less than the family-corrected minimum raises."""
        min_B = min_permutations_required(0.995, family_size=5)
        with pytest.raises(PermutationAdequacyError):
            check_permutation_adequacy(min_B - 1, 0.995, family_size=5)
