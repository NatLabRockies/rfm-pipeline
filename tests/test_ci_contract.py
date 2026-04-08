"""Tests for the CI workflow contract."""

from __future__ import annotations

from pathlib import Path


def test_ci_syncs_locked_environment_before_gate() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "pixi install --locked" in workflow
    assert "pixi run build-import-smoke" in workflow
    assert "./test_repo.sh --ci" in workflow
