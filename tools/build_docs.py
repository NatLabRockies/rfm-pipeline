"""Build Sphinx documentation into a deterministic output directory."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def build_docs(root: Path, output_dir: Path | None = None) -> Path:
    """Build Sphinx HTML docs.

    Parameters
    ----------
    root
        Repository root.
    output_dir
        Optional output directory. Defaults to ``docs/_build/html``.

    Returns
    -------
    pathlib.Path
        The HTML output directory.
    """
    docs_dir = root / "docs"
    html_dir = output_dir if output_dir is not None else docs_dir / "_build" / "html"
    html_dir = html_dir.resolve()

    if html_dir.exists():
        shutil.rmtree(html_dir)

    html_dir.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [sys.executable, "-m", "sphinx", "-W", "-b", "html", str(docs_dir), str(html_dir)],
        cwd=root,
        check=True,
    )
    print(f"Docs build succeeded: {html_dir}")
    return html_dir


def main() -> int:
    """Run the docs-build command-line interface."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional explicit Sphinx HTML output directory.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    build_docs(root, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
