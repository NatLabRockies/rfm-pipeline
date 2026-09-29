"""Tests for deterministic CI workload sharding."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest

from tools.ci_shards import select_shard, select_weighted_shard
from tools.run_test_shard import (
    build_pytest_command,
    collect_test_ids,
    estimated_test_weight,
    main,
)


def test_select_shard_partitions_every_item_exactly_once() -> None:
    items = [f"item-{index}" for index in range(17)]

    shards = [select_shard(items, index=index, count=4) for index in range(4)]

    selected = [item for shard in shards for item in shard]
    assert Counter(selected) == Counter(items)
    assert len(selected) == len(set(selected))


@pytest.mark.parametrize(
    ("index", "count"),
    [(-1, 4), (4, 4), (0, 0)],
)
def test_select_shard_rejects_invalid_coordinates(index: int, count: int) -> None:
    with pytest.raises(ValueError, match="shard"):
        select_shard(["item"], index=index, count=count)


def test_weighted_shards_separate_expensive_items_and_preserve_coverage() -> None:
    items = ["slow-a", "slow-b", "slow-c", *[f"fast-{index}" for index in range(9)]]
    weights = {"slow-a": 10, "slow-b": 10, "slow-c": 10}

    shards = [
        select_weighted_shard(
            items,
            index=index,
            count=3,
            weight=lambda item: weights.get(item, 1),
        )
        for index in range(3)
    ]

    assert Counter(item for shard in shards for item in shard) == Counter(items)
    assert all(sum(item.startswith("slow-") for item in shard) == 1 for shard in shards)


def test_test_weight_estimates_prioritize_full_stage_integrations() -> None:
    wrapper = (
        "tests/test_phase8b_chunked_io_integration.py::"
        "TestStageIntegrationEquivalence::"
        "test_sparse_selection_wrapper_matches_unwrapped_stage_result"
    )
    final_stage = (
        "tests/test_manuscript_final_artifacts.py::test_per_output_nrmse_schema_and_values"
    )

    assert estimated_test_weight(wrapper) == 240
    assert estimated_test_weight(final_stage) == 120
    assert estimated_test_weight("tests/test_public_api.py::test_import") == 1


def test_collect_test_ids_discovers_individual_items(tmp_path: Path) -> None:
    test_file = tmp_path / "test_sample.py"
    test_file.write_text(
        "def test_one():\n    pass\n\ndef test_two():\n    pass\n",
        encoding="utf-8",
    )

    nodeids = collect_test_ids(tmp_path)

    assert len(nodeids) == 2
    assert nodeids[0].endswith("test_sample.py::test_one")
    assert nodeids[1].endswith("test_sample.py::test_two")


def test_build_pytest_command_reports_slowest_tests(tmp_path: Path) -> None:
    test_id = f"{tmp_path}/tests/test_public.py::test_contract"

    command = build_pytest_command([test_id])

    assert command[:4] == [sys.executable, "-m", "pytest", "-q"]
    assert "--durations=20" in command
    assert command[-1] == test_id


def test_test_shard_cli_executes_only_selected_files(tmp_path: Path, monkeypatch) -> None:
    test_root = tmp_path / "tests"
    test_ids = [f"tests/test_sample.py::test_{index}" for index in range(5)]
    observed: dict[str, object] = {}

    def fake_run(command: list[str], *, check: bool) -> None:
        observed["command"] = command
        observed["check"] = check

    monkeypatch.setattr("tools.run_test_shard.subprocess.run", fake_run)
    monkeypatch.setattr("tools.run_test_shard.collect_test_ids", lambda _: test_ids)

    result = main(
        [
            "--shard-index",
            "1",
            "--shard-count",
            "2",
            "--test-root",
            str(test_root),
        ]
    )

    command = observed["command"]
    assert isinstance(command, list)
    assert command[-2:] == [test_ids[1], test_ids[3]]
    assert observed["check"] is True
    assert result == 0
