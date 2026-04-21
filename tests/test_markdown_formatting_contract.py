"""Regression tests for Markdown formatting discovery."""

from __future__ import annotations

import subprocess
from pathlib import Path

from tools.markdown_files import list_markdown_files


def test_markdown_discovery_targets_tracked_and_untracked_files(tmp_path: Path) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    tracked = tmp_path / "README.md"
    tracked.write_text("# Title\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    untracked = tmp_path / "docs" / "new_doc.md"
    untracked.parent.mkdir()
    untracked.write_text("# Notes\n", encoding="utf-8")

    files = [path.relative_to(tmp_path).as_posix() for path in list_markdown_files(tmp_path)]

    assert files == ["README.md", "docs/new_doc.md"]


def test_markdown_discovery_excludes_generated_and_non_markdown_paths(tmp_path: Path) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)

    keep = tmp_path / "guide.markdown"
    keep.write_text("ok\n", encoding="utf-8")
    excluded = tmp_path / "docs" / "_build" / "generated.md"
    excluded.parent.mkdir(parents=True)
    excluded.write_text("skip\n", encoding="utf-8")
    macos = tmp_path / "__MACOSX" / "notes.md"
    macos.parent.mkdir(parents=True)
    macos.write_text("skip\n", encoding="utf-8")
    non_markdown = tmp_path / "README.txt"
    non_markdown.write_text("skip\n", encoding="utf-8")

    files = [path.relative_to(tmp_path).as_posix() for path in list_markdown_files(tmp_path)]

    assert files == ["guide.markdown"]


def test_markdown_tool_modules_import_through_package_namespace() -> None:
    format_module = __import__("tools.format_markdown", fromlist=["main"])
    check_module = __import__("tools.check_markdown", fromlist=["main"])

    assert callable(format_module.main)
    assert callable(check_module.main)
