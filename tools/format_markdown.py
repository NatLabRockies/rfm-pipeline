"""Format repository Markdown files while excluding generated/transient paths."""

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
    "__MACOSX",
}
MARKDOWN_SUFFIXES = {".md", ".mdx", ".markdown"}


def _is_excluded(path: Path) -> bool:
    """Return whether a path is inside an excluded directory tree."""
    return any(part in EXCLUDED_PARTS for part in path.parts)


def _repo_markdown_files() -> list[Path]:
    """Return tracked and untracked Markdown files that are not ignored."""
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        check=True,
        capture_output=True,
        text=True,
    )

    files: set[Path] = set()
    for line in result.stdout.splitlines():
        path = Path(line)
        if path.suffix.lower() not in MARKDOWN_SUFFIXES:
            continue
        if _is_excluded(path):
            continue
        if path.is_file():
            files.add(path)

    return sorted(files)


def main() -> int:
    """Run mdformat over repository Markdown files."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    files = _repo_markdown_files()
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
