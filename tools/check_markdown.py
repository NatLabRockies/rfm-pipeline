#!/usr/bin/env python3
"""Check Markdown formatting with mdformat."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tools.markdown_files import list_markdown_files


def main() -> int:
    """Run mdformat in check mode over repository Markdown files."""
    repo_root = Path(__file__).resolve().parents[1]
    markdown_files = [str(path) for path in list_markdown_files(repo_root)]

    if not markdown_files:
        print("No Markdown files found.")
        return 0

    cmd = [sys.executable, "-m", "mdformat", "--check", *markdown_files]
    result = subprocess.run(cmd, cwd=repo_root)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
