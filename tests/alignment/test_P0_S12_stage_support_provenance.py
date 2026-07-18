"""P0-S12: Per-stage support-composition provenance (F6).

Tests cover:
- correct_per_stage_records: synthetic trace where one stage removes terms and
  another removes none yields correct per-stage in/out/removed sets.
- noop_stage_explicit: a stage that removes nothing reports is_noop=True and
  empty removed_terms — making zero removals machine-checkable.
- removing_stage_correct: a stage that removes terms reports the exact removed set.
- record_count: five records (one per stage) are always returned.
- stage_names_and_order: stages appear in pipeline order.
- n_removed_property: n_removed == len(removed_terms).
- consistency_invariant: n_in - n_out == n_removed for every record.
- output_not_subset_raises: build_stage_support_provenance raises ValueError when
  a stage output contains terms not present in its input.
- invalid_record_direct: StageSupportRecord raises ValueError on inconsistent args.

Fixture:
- Synthetic multi-stage selection trace with explicit known removals.
"""

from __future__ import annotations

import pytest

from rfm_pipeline.manuscript_stages import (
    StageSupportRecord,
    build_stage_support_provenance,
)

# ---------------------------------------------------------------------------
# Synthetic multi-stage selection trace
# ---------------------------------------------------------------------------
# Universe: 20 terms (t00 … t19).
# screen removes t15, t16, t17, t18, t19  (5 removed)
# lasso  removes t10, t11, t12, t13, t14  (5 removed)
# stability removes nothing               (0 removed — explicit no-op)
# hc3    removes t07, t08, t09             (3 removed)
# final  removes nothing                  (0 removed — explicit no-op)

_ALL = frozenset(f"t{i:02d}" for i in range(20))
_AFTER_SCREEN = frozenset(f"t{i:02d}" for i in range(15))  # 15 survive
_AFTER_LASSO = frozenset(f"t{i:02d}" for i in range(10))  # 10 survive
_AFTER_STABILITY = frozenset(f"t{i:02d}" for i in range(10))  # no-op
_AFTER_HC3 = frozenset(f"t{i:02d}" for i in range(7))  # 7 survive
_AFTER_FINAL = frozenset(f"t{i:02d}" for i in range(7))  # no-op


@pytest.fixture(scope="module")
def provenance() -> list[StageSupportRecord]:
    """Build provenance from the synthetic multi-stage selection trace."""
    return build_stage_support_provenance(
        candidates_before_screen=_ALL,
        candidates_after_screen=_AFTER_SCREEN,
        candidates_after_lasso=_AFTER_LASSO,
        candidates_after_stability=_AFTER_STABILITY,
        candidates_after_hc3=_AFTER_HC3,
        candidates_after_final_selection=_AFTER_FINAL,
    )


# ---------------------------------------------------------------------------
# Record count and structure
# ---------------------------------------------------------------------------


class TestRecordCount:
    def test_P0_S12_returns_five_records(self, provenance: list[StageSupportRecord]) -> None:
        """build_stage_support_provenance always returns exactly 5 records."""
        assert len(provenance) == 5

    def test_P0_S12_all_records_are_stage_support_records(
        self, provenance: list[StageSupportRecord]
    ) -> None:
        """Every returned object is a StageSupportRecord."""
        for rec in provenance:
            assert isinstance(rec, StageSupportRecord)


class TestStageNamesAndOrder:
    def test_P0_S12_stage_names_in_pipeline_order(
        self, provenance: list[StageSupportRecord]
    ) -> None:
        """Stages appear in pipeline order: screen, lasso, stability, hc3, final_selection."""
        names = [r.stage for r in provenance]
        assert names == ["screen", "lasso", "stability", "hc3", "final_selection"]


# ---------------------------------------------------------------------------
# Per-stage in/out/removed values
# ---------------------------------------------------------------------------


class TestScreenStage:
    def test_P0_S12_screen_n_in(self, provenance: list[StageSupportRecord]) -> None:
        assert provenance[0].n_in == 20

    def test_P0_S12_screen_n_out(self, provenance: list[StageSupportRecord]) -> None:
        assert provenance[0].n_out == 15

    def test_P0_S12_screen_removed_terms(self, provenance: list[StageSupportRecord]) -> None:
        expected = frozenset(f"t{i:02d}" for i in range(15, 20))
        assert provenance[0].removed_terms == expected

    def test_P0_S12_screen_is_not_noop(self, provenance: list[StageSupportRecord]) -> None:
        assert not provenance[0].is_noop


class TestLassoStage:
    def test_P0_S12_lasso_n_in(self, provenance: list[StageSupportRecord]) -> None:
        assert provenance[1].n_in == 15

    def test_P0_S12_lasso_n_out(self, provenance: list[StageSupportRecord]) -> None:
        assert provenance[1].n_out == 10

    def test_P0_S12_lasso_removed_terms(self, provenance: list[StageSupportRecord]) -> None:
        expected = frozenset(f"t{i:02d}" for i in range(10, 15))
        assert provenance[1].removed_terms == expected

    def test_P0_S12_lasso_is_not_noop(self, provenance: list[StageSupportRecord]) -> None:
        assert not provenance[1].is_noop


class TestStabilityStageNoop:
    def test_P0_S12_stability_n_in(self, provenance: list[StageSupportRecord]) -> None:
        assert provenance[2].n_in == 10

    def test_P0_S12_stability_n_out(self, provenance: list[StageSupportRecord]) -> None:
        assert provenance[2].n_out == 10

    def test_P0_S12_stability_removed_terms_empty(
        self, provenance: list[StageSupportRecord]
    ) -> None:
        """Stability stage removed nothing — removed_terms is an empty frozenset."""
        assert provenance[2].removed_terms == frozenset()

    def test_P0_S12_stability_is_noop(self, provenance: list[StageSupportRecord]) -> None:
        """is_noop is True when removed_terms is empty."""
        assert provenance[2].is_noop


class TestHc3Stage:
    def test_P0_S12_hc3_n_in(self, provenance: list[StageSupportRecord]) -> None:
        assert provenance[3].n_in == 10

    def test_P0_S12_hc3_n_out(self, provenance: list[StageSupportRecord]) -> None:
        assert provenance[3].n_out == 7

    def test_P0_S12_hc3_removed_terms(self, provenance: list[StageSupportRecord]) -> None:
        expected = frozenset({"t07", "t08", "t09"})
        assert provenance[3].removed_terms == expected

    def test_P0_S12_hc3_is_not_noop(self, provenance: list[StageSupportRecord]) -> None:
        assert not provenance[3].is_noop


class TestFinalSelectionStageNoop:
    def test_P0_S12_final_n_in(self, provenance: list[StageSupportRecord]) -> None:
        assert provenance[4].n_in == 7

    def test_P0_S12_final_n_out(self, provenance: list[StageSupportRecord]) -> None:
        assert provenance[4].n_out == 7

    def test_P0_S12_final_removed_terms_empty(self, provenance: list[StageSupportRecord]) -> None:
        """Final-selection stage removed nothing — removed_terms is an empty frozenset."""
        assert provenance[4].removed_terms == frozenset()

    def test_P0_S12_final_is_noop(self, provenance: list[StageSupportRecord]) -> None:
        assert provenance[4].is_noop


# ---------------------------------------------------------------------------
# Properties and invariants
# ---------------------------------------------------------------------------


class TestPropertiesAndInvariants:
    def test_P0_S12_n_removed_equals_len_removed_terms(
        self, provenance: list[StageSupportRecord]
    ) -> None:
        """n_removed == len(removed_terms) for every record."""
        for rec in provenance:
            assert rec.n_removed == len(rec.removed_terms)

    def test_P0_S12_consistency_invariant(self, provenance: list[StageSupportRecord]) -> None:
        """n_in - n_out == n_removed for every record."""
        for rec in provenance:
            assert rec.n_in - rec.n_out == rec.n_removed, (
                f"stage={rec.stage}: n_in={rec.n_in}, n_out={rec.n_out}, n_removed={rec.n_removed}"
            )


# ---------------------------------------------------------------------------
# Validation / error paths
# ---------------------------------------------------------------------------


class TestValidationErrors:
    def test_P0_S12_output_not_subset_raises_value_error(self) -> None:
        """build_stage_support_provenance raises ValueError when output contains
        terms absent from its input (stage output is not a subset of input)."""
        with pytest.raises(ValueError, match="screen"):
            build_stage_support_provenance(
                candidates_before_screen={"a", "b"},
                candidates_after_screen={"a", "b", "c"},  # "c" is new — invalid
                candidates_after_lasso={"a"},
                candidates_after_stability={"a"},
                candidates_after_hc3={"a"},
                candidates_after_final_selection={"a"},
            )

    def test_P0_S12_inconsistent_record_raises_value_error(self) -> None:
        """StageSupportRecord raises ValueError when removed_terms size != n_in - n_out."""
        with pytest.raises(ValueError):
            StageSupportRecord(
                stage="screen",
                n_in=10,
                n_out=8,
                removed_terms=frozenset({"x"}),  # only 1 term but n_in - n_out == 2
            )

    def test_P0_S12_negative_n_in_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            StageSupportRecord(
                stage="screen",
                n_in=-1,
                n_out=0,
                removed_terms=frozenset(),
            )

    def test_P0_S12_n_out_greater_than_n_in_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            StageSupportRecord(
                stage="screen",
                n_in=5,
                n_out=10,
                removed_terms=frozenset(),
            )
