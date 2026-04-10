#!/usr/bin/env python3
"""Check Markdown formatting with mdformat."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

EXCLUDED_DIR_NAMES = {
    ".git",
    ".pixi",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".ipynb_checkpoints",
    "__pycache__",
    "__MACOSX",
    "build",
    "dist",
}

EXCLUDED_PATH_PARTS = {
    "docs/_build",
}


def _is_markdown_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() == ".md"


def _is_excluded(path: Path, repo_root: Path) -> bool:
    rel = path.relative_to(repo_root)
    rel_str = rel.as_posix()

    if any(part in EXCLUDED_DIR_NAMES for part in rel.parts):
        return True

    if any(excluded in rel_str for excluded in EXCLUDED_PATH_PARTS):
        return True

    if rel_str.endswith(".egg-info"):
        return True

    return False


def _find_markdown_files(repo_root: Path) -> list[str]:
    files: list[str] = []
    for path in sorted(repo_root.rglob("*.md")):
        if _is_excluded(path, repo_root):
            continue
        if _is_markdown_file(path):
            files.append(str(path))
    return files


def main() -> int:
    """Run mdformat in check mode over repository Markdown files."""
    repo_root = Path(__file__).resolve().parents[1]
    markdown_files = _find_markdown_files(repo_root)

    if not markdown_files:
        print("No Markdown files found.")
        return 0

    cmd = [sys.executable, "-m", "mdformat", "--check", *markdown_files]
    result = subprocess.run(cmd, cwd=repo_root)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
