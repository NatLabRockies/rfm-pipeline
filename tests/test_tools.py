"""Tests for repository utility tools."""

from __future__ import annotations

from pathlib import Path

import nbformat

from tools.check_repo import scan_generated_python_artifacts, scan_text_hygiene
from tools.notebook_hygiene import check_notebooks, sanitize_notebook, try_normalize_code_source


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


def test_sanitize_notebook_clears_outputs_and_execution_counts(tmp_path: Path):
    path = tmp_path / "demo.ipynb"
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell(
                "x = 1",
                execution_count=3,
                outputs=[nbformat.v4.new_output(output_type="stream", name="stdout", text="1\n")],
            )
        ]
    )
    nbformat.write(notebook, path)
    assert sanitize_notebook(path, write=True) is True
    assert check_notebooks(tmp_path) == []


def test_sanitize_notebook_repairs_literal_escaped_newlines(tmp_path: Path):
    path = tmp_path / "broken.ipynb"
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell(
                r"from notebooks.helpers import get_repo_root\n"
                r"root = get_repo_root()\n"
                r"print(root)\n"
            )
        ]
    )
    nbformat.write(notebook, path)

    assert any("invalid code cell syntax" in item for item in check_notebooks(tmp_path))
    assert sanitize_notebook(path, write=True) is True
    assert check_notebooks(tmp_path) == []


def test_try_normalize_code_source_is_noop_for_valid_python() -> None:
    source = 'pattern = r"\\n"\nprint(pattern)\n'
    normalized, changed = try_normalize_code_source(source)

    assert normalized == source
    assert changed is False
