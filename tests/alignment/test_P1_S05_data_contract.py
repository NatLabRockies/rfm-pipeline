"""P1-S05: Manuscript data-contract claim↔artifact check (F3).

Tests cover:
- A matching claim passes without error.
- A mismatched numeric claim fails with a useful diff message.
- A mismatched string claim fails with a useful diff message.
- A missing source artifact raises MissingSourceError.
- Float tolerance is respected (near-match within tol passes; outside fails).
- Multi-claim manifest: all pass → empty list; one bad → exactly one message.
"""

from __future__ import annotations

import pandas as pd
import pytest

from rfm_pipeline.manuscript_data_contract import (
    ClaimMismatchError,
    MissingSourceError,
    assert_claims,
    validate_claims,
)

# ---------------------------------------------------------------------------
# Fixtures — tiny synthetic source table written to tmp_path
# ---------------------------------------------------------------------------


@pytest.fixture()
def source_csv(tmp_path):
    """Write a tiny synthetic source table and return its path."""
    df = pd.DataFrame(
        {
            "metric": ["accuracy", "f1", "auc"],
            "model": ["rfm", "rfm", "rfm"],
            "value": [0.853, 0.741, 0.912],
            "n_features": [12, 12, 12],
        }
    )
    path = tmp_path / "results_table.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture()
def matching_float_claim(source_csv):
    return [
        {
            "label": "Table 2 accuracy",
            "expected": 0.853,
            "source_path": str(source_csv),
            "column": "value",
            "row_filter": {"metric": "accuracy", "model": "rfm"},
        }
    ]


@pytest.fixture()
def mismatched_float_claim(source_csv):
    return [
        {
            "label": "Table 2 accuracy",
            "expected": 0.999,  # wrong
            "source_path": str(source_csv),
            "column": "value",
            "row_filter": {"metric": "accuracy", "model": "rfm"},
        }
    ]


@pytest.fixture()
def matching_int_claim(source_csv):
    return [
        {
            "label": "n_features",
            "expected": 12,
            "source_path": str(source_csv),
            "column": "n_features",
            "row_filter": {"metric": "accuracy"},
        }
    ]


@pytest.fixture()
def mismatched_int_claim(source_csv):
    return [
        {
            "label": "n_features",
            "expected": 99,  # wrong
            "source_path": str(source_csv),
            "column": "n_features",
            "row_filter": {"metric": "accuracy"},
        }
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestValidateClaims:
    def test_matching_float_claim_returns_empty(self, matching_float_claim):
        result = validate_claims(matching_float_claim)
        assert result == []

    def test_mismatched_float_returns_one_message(self, mismatched_float_claim):
        result = validate_claims(mismatched_float_claim)
        assert len(result) == 1
        msg = result[0]
        assert "Table 2 accuracy" in msg
        assert "0.999" in msg or "0.853" in msg

    def test_mismatch_message_includes_diff(self, mismatched_float_claim):
        result = validate_claims(mismatched_float_claim)
        assert "diff=" in result[0]

    def test_matching_int_claim_returns_empty(self, matching_int_claim):
        result = validate_claims(matching_int_claim)
        assert result == []

    def test_mismatched_int_returns_message(self, mismatched_int_claim):
        result = validate_claims(mismatched_int_claim)
        assert len(result) == 1
        assert "n_features" in result[0]

    def test_missing_source_raises(self, tmp_path):
        claim = [
            {
                "label": "ghost claim",
                "expected": 1.0,
                "source_path": str(tmp_path / "nonexistent.csv"),
                "column": "value",
            }
        ]
        with pytest.raises(MissingSourceError, match="nonexistent.csv"):
            validate_claims(claim)

    def test_float_within_tolerance_passes(self, source_csv):
        claim = [
            {
                "label": "near match",
                "expected": 0.853 + 1e-9,  # within default 1e-6 tol
                "source_path": str(source_csv),
                "column": "value",
                "row_filter": {"metric": "accuracy"},
            }
        ]
        assert validate_claims(claim) == []

    def test_float_outside_tolerance_fails(self, source_csv):
        claim = [
            {
                "label": "just outside",
                "expected": 0.853 + 0.01,  # outside 1e-6 tol
                "source_path": str(source_csv),
                "column": "value",
                "row_filter": {"metric": "accuracy"},
            }
        ]
        result = validate_claims(claim)
        assert len(result) == 1

    def test_multi_claim_all_pass(self, source_csv):
        claims = [
            {
                "label": "accuracy",
                "expected": 0.853,
                "source_path": str(source_csv),
                "column": "value",
                "row_filter": {"metric": "accuracy"},
            },
            {
                "label": "f1",
                "expected": 0.741,
                "source_path": str(source_csv),
                "column": "value",
                "row_filter": {"metric": "f1"},
            },
        ]
        assert validate_claims(claims) == []

    def test_multi_claim_one_bad_returns_one_message(self, source_csv):
        claims = [
            {
                "label": "accuracy",
                "expected": 0.853,
                "source_path": str(source_csv),
                "column": "value",
                "row_filter": {"metric": "accuracy"},
            },
            {
                "label": "f1 wrong",
                "expected": 0.999,  # wrong
                "source_path": str(source_csv),
                "column": "value",
                "row_filter": {"metric": "f1"},
            },
        ]
        result = validate_claims(claims)
        assert len(result) == 1
        assert "f1 wrong" in result[0]

    def test_no_row_filter_uses_first_row(self, source_csv):
        claim = [
            {
                "label": "first row value",
                "expected": 0.853,
                "source_path": str(source_csv),
                "column": "value",
            }
        ]
        assert validate_claims(claim) == []


class TestAssertClaims:
    def test_passes_silently_on_match(self, matching_float_claim):
        assert_claims(matching_float_claim)  # no exception

    def test_raises_claim_mismatch_error(self, mismatched_float_claim):
        with pytest.raises(ClaimMismatchError) as exc_info:
            assert_claims(mismatched_float_claim)
        msg = str(exc_info.value)
        assert "Table 2 accuracy" in msg
        assert "1 claim(s) failed" in msg

    def test_raises_missing_source_error(self, tmp_path):
        claim = [
            {
                "label": "ghost",
                "expected": 1.0,
                "source_path": str(tmp_path / "missing.csv"),
                "column": "value",
            }
        ]
        with pytest.raises(MissingSourceError):
            assert_claims(claim)

    def test_custom_tolerance_respected(self, source_csv):
        claim = [
            {
                "label": "loose tol",
                "expected": 0.853 + 0.05,
                "source_path": str(source_csv),
                "column": "value",
                "row_filter": {"metric": "accuracy"},
            }
        ]
        # passes with loose tolerance
        assert_claims(claim, tolerance=0.1)
        # fails with tight tolerance
        with pytest.raises(ClaimMismatchError):
            assert_claims(claim, tolerance=1e-6)
