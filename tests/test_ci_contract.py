"""Tests for the GitHub Actions CI contract."""

from __future__ import annotations

from pathlib import Path


def test_ci_runs_repo_gate() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "Run repository gate" in workflow
    assert "pixi run ci" in workflow


def test_ci_runs_locked_pixi_environment_once() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "matrix:" not in workflow
    assert "python-version" not in workflow
    assert "name: validate (locked Python 3.12)" in workflow
    assert workflow.count("Run repository gate") == 1


def test_ci_rebuilds_docs_before_upload() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "Rebuild docs artifact" in workflow
    assert "pixi run docs" in workflow
    assert "Upload built docs" in workflow
    assert "path: docs/_build/html" in workflow
    assert "if-no-files-found: error" in workflow
