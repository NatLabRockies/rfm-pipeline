"""Tests for the GitHub Actions CI contract."""

from __future__ import annotations

from pathlib import Path


def test_ci_runs_one_fast_test_job_and_one_build_job() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "quality_tests:" in workflow
    assert "docs_package:" in workflow
    assert "pixi run unit-tests" in workflow
    assert "matrix:" not in workflow
    assert "notebook" not in workflow.lower()


def test_ci_cancels_superseded_runs_and_preserves_stable_check_name() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "cancel-in-progress: true" in workflow
    assert "name: validate (locked Python 3.12)" in workflow
    assert "if: ${{ always() }}" in workflow


def test_ci_builds_docs_and_validates_the_wheel_once() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert workflow.count("pixi run docs") == 1
    assert workflow.count("pixi run package-build") == 1
    assert workflow.count("pixi run package-smoke") == 1
    assert "path: docs/_build/html" in workflow
    assert "if-no-files-found: error" in workflow
