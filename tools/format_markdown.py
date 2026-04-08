"""Format tracked Markdown files while excluding generated/transient paths."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

EXCLUDED_PARTS = {
    ".git",
    ".pixi",
    ".pytest_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
    ".venv",
    "build",
    "dist",
    "_build",
    "docs/_build",
}

MARKDOWN_SUFFIXES = {".md", ".mdx", ".markdown"}


def _is_excluded(path: Path) -> bool:
    """Return True when the path is under an excluded directory."""
    parts = set(path.parts)
    return bool(parts & EXCLUDED_PARTS)


def _tracked_markdown_files() -> list[Path]:
    """Return tracked Markdown files in the repository."""
    result = subprocess.run(
        ["git", "ls-files"],
        check=True,
        capture_output=True,
        text=True,
    )
    files: list[Path] = []
    for line in result.stdout.splitlines():
        path = Path(line)
        if path.suffix.lower() not in MARKDOWN_SUFFIXES:
            continue
        if _is_excluded(path):
            continue
        files.append(path)
    return sorted(files)


def main() -> int:
    """Run mdformat over tracked Markdown files."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    files = _tracked_markdown_files()
    if not files:
        return 0

    command = [sys.executable, "-m", "mdformat"]
    if args.check:
        command.append("--check")
    command.extend(str(path) for path in files)

    completed = subprocess.run(command)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
