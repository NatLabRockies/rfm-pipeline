"""Tests for the GitHub Actions CI contract."""

from __future__ import annotations

from pathlib import Path

from tools.run_repository_gate import VALIDATION_TASKS


def test_ci_runs_full_validation_in_parallel_jobs() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "matrix:" in workflow
    assert "Run test shard" in workflow
    assert "Run notebook shard" in workflow
    assert "Run workflow smoke checks" in workflow
    assert "Build package and documentation" in workflow


def test_ci_cancels_superseded_runs_and_preserves_stable_check_name() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "cancel-in-progress: true" in workflow
    assert "name: validate (locked Python 3.12)" in workflow
    assert "if: ${{ always() }}" in workflow


def test_ci_builds_docs_once_before_upload() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert workflow.count("pixi run docs") == 1
    assert "Upload built docs" in workflow
    assert "path: docs/_build/html" in workflow
    assert "if-no-files-found: error" in workflow


def test_ci_shards_cover_tests_and_notebooks_without_skipping() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "-m tools.run_test_shard" in workflow
    assert "-m tools.execute_notebooks" in workflow
    assert "--shard-index" in workflow
    assert "--shard-count" in workflow
    assert "continue-on-error" not in workflow

    directly_invoked_tasks = set(VALIDATION_TASKS) - {"unit-tests", "notebook-tests"}
    for task in directly_invoked_tasks:
        assert f"pixi run {task}" in workflow


def test_ci_uses_thirteen_item_level_test_shards() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "shard: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]" in workflow
    assert "--shard-count 13" in workflow
