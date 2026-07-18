"""G1-S01: Generic combination labels + stratified subset in data.py.

Acceptance criteria exercised here:

- ``combination_labels`` yields deterministic ``col=val|...`` labels for arbitrary
  categorical columns (not tied to AFSC/UAEORO or a fixed number of columns).
- ``stratified_subset_by_combination`` returns ``n_per_combination`` rows per
  observed combination, is deterministic under a fixed seed, and raises when a
  combination is too small under ``require_all_combinations=True``.
- ``add_scenario_flags`` / ``make_boolean_combination_labels`` /
  ``stratified_subset_by_boolean_combination`` are removed from
  ``rfm_pipeline.data``.
- No ``AFSC`` / ``UAEORO`` / ``add_scenario_flags`` token appears in
  ``src/rfm_pipeline/data.py`` under the definitions/labels that G1-S01 replaces.
  (``stratified_holdout_split`` retains ``AFSC`` / ``UAEORO`` as its default
  column names until G1-S02; those tokens are intentionally out of scope for
  this slice's denylist.)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from rfm_pipeline import data as data_module
from rfm_pipeline.data import (
    combination_labels,
    stratified_subset_by_combination,
)

_DATA_PY = Path(data_module.__file__)


def _make_three_way_frame(rows_per_combo: int, *, prefix: str = "row") -> pd.DataFrame:
    """Build a frame with 3 categorical strata columns and unique row identifiers."""
    records: list[dict[str, object]] = []
    row_id = 0
    for region in ("north", "south"):
        for tier in ("A", "B"):
            for regime in (0, 1):
                for _ in range(rows_per_combo):
                    records.append(
                        {
                            "region": region,
                            "tier": tier,
                            "regime": regime,
                            "row_id": f"{prefix}-{row_id}",
                            "value": float(row_id),
                        }
                    )
                    row_id += 1
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# combination_labels
# ---------------------------------------------------------------------------


def test_G1_S01_combination_labels_three_categorical_columns() -> None:
    frame = pd.DataFrame(
        {
            "region": ["north", "north", "south"],
            "tier": ["A", "B", "A"],
            "regime": [0, 1, 1],
        }
    )
    labels = combination_labels(frame, columns=["region", "tier", "regime"])
    assert labels.tolist() == [
        "region=north|tier=A|regime=0",
        "region=north|tier=B|regime=1",
        "region=south|tier=A|regime=1",
    ]


def test_G1_S01_combination_labels_output_column_name_and_index() -> None:
    frame = pd.DataFrame(
        {"a": [1, 2], "b": ["x", "y"]},
        index=pd.Index([10, 11], name="obs"),
    )
    labels = combination_labels(frame, columns=["a", "b"], output_column="combo_key")
    assert labels.name == "combo_key"
    assert list(labels.index) == [10, 11]
    assert labels.tolist() == ["a=1|b=x", "a=2|b=y"]


def test_G1_S01_combination_labels_column_order_preserved() -> None:
    frame = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
    forward = combination_labels(frame, columns=["a", "b", "c"]).iloc[0]
    reversed_ = combination_labels(frame, columns=["c", "b", "a"]).iloc[0]
    assert forward == "a=1|b=2|c=3"
    assert reversed_ == "c=3|b=2|a=1"


# ---------------------------------------------------------------------------
# stratified_subset_by_combination
# ---------------------------------------------------------------------------


def test_G1_S01_stratified_subset_returns_n_per_observed_combination() -> None:
    frame = _make_three_way_frame(rows_per_combo=40)
    subset = stratified_subset_by_combination(
        frame,
        columns=["region", "tier", "regime"],
        n_per_combination=25,
        random_state=123,
    )
    labels = combination_labels(subset, columns=["region", "tier", "regime"])
    counts = labels.value_counts().to_dict()
    # 2 * 2 * 2 = 8 observed combinations, 25 each.
    assert len(counts) == 8
    assert set(counts.values()) == {25}
    assert len(subset) == 8 * 25
    assert subset["row_id"].is_unique


def test_G1_S01_stratified_subset_is_deterministic_under_fixed_seed() -> None:
    frame = _make_three_way_frame(rows_per_combo=30)
    first = stratified_subset_by_combination(
        frame,
        columns=["region", "tier", "regime"],
        n_per_combination=10,
        random_state=2024,
    )
    second = stratified_subset_by_combination(
        frame,
        columns=["region", "tier", "regime"],
        n_per_combination=10,
        random_state=2024,
    )
    pd.testing.assert_frame_equal(first, second)


def test_G1_S01_stratified_subset_different_seeds_differ() -> None:
    frame = _make_three_way_frame(rows_per_combo=30)
    a = stratified_subset_by_combination(
        frame,
        columns=["region", "tier", "regime"],
        n_per_combination=10,
        random_state=1,
    )
    b = stratified_subset_by_combination(
        frame,
        columns=["region", "tier", "regime"],
        n_per_combination=10,
        random_state=2,
    )
    assert not a.reset_index(drop=True).equals(b.reset_index(drop=True))


def test_G1_S01_stratified_subset_raises_when_stratum_too_small() -> None:
    frame = _make_three_way_frame(rows_per_combo=5)
    with pytest.raises(ValueError, match="cannot draw"):
        stratified_subset_by_combination(
            frame,
            columns=["region", "tier", "regime"],
            n_per_combination=10,
            random_state=0,
            require_all_combinations=True,
        )


def test_G1_S01_stratified_subset_skips_small_strata_when_not_required() -> None:
    # Make one combination too small.
    frame = _make_three_way_frame(rows_per_combo=20)
    small = frame[
        (frame["region"] == "north") & (frame["tier"] == "A") & (frame["regime"] == 0)
    ].iloc[:3]
    rest = frame[~((frame["region"] == "north") & (frame["tier"] == "A") & (frame["regime"] == 0))]
    frame = pd.concat([small, rest], ignore_index=True)

    subset = stratified_subset_by_combination(
        frame,
        columns=["region", "tier", "regime"],
        n_per_combination=15,
        random_state=0,
        require_all_combinations=False,
    )
    labels = combination_labels(subset, columns=["region", "tier", "regime"])
    counts = labels.value_counts().to_dict()
    # 7 remaining combinations reach 15 rows; the short combination is skipped.
    assert set(counts.values()) == {15}
    assert len(counts) == 7


# ---------------------------------------------------------------------------
# Removal of the case-study-specific API
# ---------------------------------------------------------------------------


def test_G1_S01_removed_functions_absent_from_data_module() -> None:
    for name in (
        "add_scenario_flags",
        "_parse_on_off_flag",
        "make_boolean_combination_labels",
        "stratified_subset_by_boolean_combination",
    ):
        assert not hasattr(data_module, name), f"data module still exposes removed symbol {name!r}"


def test_G1_S01_no_afsc_uaeoro_scenario_flag_tokens_in_data_py_replaced_scope() -> None:
    """The G1-S01 replaced scope must not carry AFSC/UAEORO/add_scenario_flags.

    Reads ``data.py`` and asserts that the removed helper's tokens are gone.
    ``stratified_holdout_split`` still uses ``AFSC``/``UAEORO`` as default
    column names until G1-S02; those tokens are excluded here since they are
    that later slice's scope.
    """
    source = _DATA_PY.read_text(encoding="utf-8")
    # These tokens are unconditionally banned by G1-S01 acceptance.
    for token in (
        "add_scenario_flags",
        "_parse_on_off_flag",
        "make_boolean_combination_labels",
        "stratified_subset_by_boolean_combination",
    ):
        assert token not in source, f"Removed token {token!r} still present in data.py"
