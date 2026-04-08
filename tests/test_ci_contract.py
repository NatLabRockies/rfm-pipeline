"""Tests for the GitHub Actions CI contract."""

from __future__ import annotations

from pathlib import Path


def test_ci_runs_repo_gate() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "Run repository gate" in workflow
    assert "./test_repo.sh --ci" in workflow


def test_gate_ci_mode_uses_locked_install() -> None:
    script = Path("test_repo.sh").read_text(encoding="utf-8")
    assert 'echo ">>> $PIXI_BIN install --locked"' in script
    assert '"$PIXI_BIN" install --locked' in script
    assert "--check|--ci)" in script


def test_ci_rebuilds_docs_before_upload() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "Rebuild docs artifact" in workflow
    assert "pixi run docs" in workflow
    assert "Upload built docs" in workflow
    assert "path: docs/_build/html" in workflow
    assert "if-no-files-found: error" in workflow
