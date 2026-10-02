"""Tests for repository utility tools."""

from __future__ import annotations

from pathlib import Path

from tools.check_repo import scan_generated_python_artifacts, scan_text_hygiene


def test_scan_text_hygiene_detects_tabs_trailing_whitespace_and_missing_newline(tmp_path: Path):
    bad = tmp_path / "bad.py"
    bad.write_text("\tvalue = 1  ", encoding="utf-8")
    failures = scan_text_hygiene([bad])
    assert any("tab character found" in item for item in failures)
    assert any("trailing whitespace" in item for item in failures)
    assert any("missing terminal newline" in item for item in failures)


def test_scan_generated_python_artifacts_detects_pyc_and_pycache(tmp_path: Path):
    pycache = tmp_path / "pkg" / "__pycache__"
    pycache.mkdir(parents=True)
    pyc = tmp_path / "pkg" / "module.pyc"
    pyc.write_bytes(b"x")
    failures = scan_generated_python_artifacts(tmp_path)
    assert any("generated directory present" in item for item in failures)
    assert any("generated file present" in item for item in failures)
