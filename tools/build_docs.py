"""Build Sphinx documentation into a temporary directory for gate validation."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path


def build_docs(root: Path) -> None:
    """Build Sphinx HTML docs into a temporary directory.

    Parameters
    ----------
    root
        Repository root.
    """
    docs_dir = root / "docs"
    with tempfile.TemporaryDirectory(prefix="bsm_docs_build_") as tmpdir:
        subprocess.run(
            [sys.executable, "-m", "sphinx", "-W", "-b", "html", str(docs_dir), tmpdir],
            cwd=root,
            check=True,
        )
        print(f"Docs build succeeded: {tmpdir}")


def main() -> int:
    """Run the docs-build command-line interface."""
    root = Path(__file__).resolve().parents[1]
    build_docs(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
