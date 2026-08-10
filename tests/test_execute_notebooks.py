"""Tests for test execute notebooks."""

from __future__ import annotations

from pathlib import Path

from tools.execute_notebooks import build_nbconvert_command, discover_notebooks


def test_discover_notebooks_selects_expected_directories(tmp_path: Path) -> None:
    notebook_dir = tmp_path / "notebooks"
    notebook_dir.mkdir()
    expected = notebook_dir / "example.ipynb"
    expected.write_text("{}", encoding="utf-8")

    excluded = tmp_path / "docs" / "example.ipynb"
    excluded.parent.mkdir(parents=True)
    excluded.write_text("{}", encoding="utf-8")

    paths = discover_notebooks(tmp_path)

    assert paths == [expected]


def test_build_nbconvert_command_uses_temp_output_dir(tmp_path: Path) -> None:
    notebook = tmp_path / "notebooks" / "demo.ipynb"
    output_dir = tmp_path / "tmp-out"
    command = build_nbconvert_command(notebook, output_dir, timeout=900)

    assert str(notebook) in command
    assert str(output_dir) in command
    assert "--execute" in command
    assert "--to" in command
    assert "notebook" in command
    assert "--ExecutePreprocessor.timeout=900" in command
    assert "--ExecutePreprocessor.kernel_name=pixi-kernel-python3" in command
