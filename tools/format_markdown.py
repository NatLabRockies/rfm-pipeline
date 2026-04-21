"""Format repository Markdown files while excluding generated/transient paths."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from tools.markdown_files import list_markdown_files


def main() -> int:
    """Run mdformat over repository Markdown files."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    files = list_markdown_files(repo_root)
    if not files:
        return 0

    command = [sys.executable, "-m", "mdformat"]
    if args.check:
        command.append("--check")
    command.extend(str(path) for path in files)

    completed = subprocess.run(command, cwd=repo_root)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
