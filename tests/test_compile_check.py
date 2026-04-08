"""Tests for test compile check."""

from __future__ import annotations

from pathlib import Path

from tools.compile_check import compile_python_files, iter_python_files


def test_iter_python_files_excludes_build_trees(tmp_path: Path) -> None:
    src_file = tmp_path / "src" / "demo.py"
    src_file.parent.mkdir(parents=True)
    src_file.write_text("x = 1\n", encoding="utf-8")

    excluded = tmp_path / "build" / "generated.py"
    excluded.parent.mkdir(parents=True)
    excluded.write_text("x = 2\n", encoding="utf-8")

    files = iter_python_files(tmp_path)

    assert files == [src_file]


def test_compile_python_files_compiles_sources_without_writing_cache(tmp_path: Path) -> None:
    src_file = tmp_path / "module.py"
    src_file.write_text("value = 1 + 1\n", encoding="utf-8")

    compiled = compile_python_files(tmp_path)

    assert compiled == 1
    assert not list(tmp_path.rglob("__pycache__"))
