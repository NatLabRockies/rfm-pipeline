"""Notebook execution utilities for the repository gate."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

NOTEBOOK_DIR_CANDIDATES = (
    "notebooks",
    "examples",
    "reports",
    "analysis",
)
EXCLUDED_PARTS = {
    ".git",
    ".pixi",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    ".ipynb_checkpoints",
    "__MACOSX",
    "docs",
    "_build",
    "build",
    "dist",
}


def is_excluded(path: Path) -> bool:
    """Return whether a notebook path should be excluded from execution discovery.

    Parameters
    ----------
    path
        Candidate notebook path.

    Returns
    -------
    bool
        ``True`` when the path belongs to an excluded directory tree.
    """
    return any(part in EXCLUDED_PARTS for part in path.parts)


def discover_notebooks(root: Path) -> list[Path]:
    """Return notebooks that should be execution-tested.

    Parameters
    ----------
    root
        Repository root.

    Returns
    -------
    list[pathlib.Path]
        Sorted notebook paths relative to the repository selection rules.
    """
    notebooks: list[Path] = []
    for directory_name in NOTEBOOK_DIR_CANDIDATES:
        directory = root / directory_name
        if not directory.exists():
            continue
        for path in directory.rglob("*.ipynb"):
            if is_excluded(path):
                continue
            notebooks.append(path)
    return sorted(set(notebooks))


def build_nbconvert_command(
    notebook: Path,
    output_dir: Path,
    *,
    timeout: int,
) -> list[str]:
    """Build the Jupyter execution command for a single notebook.

    Parameters
    ----------
    notebook
        Notebook to execute.
    output_dir
        Temporary output directory receiving the executed notebook copy.
    timeout
        Per-cell timeout in seconds.

    Returns
    -------
    list[str]
        Command tokens suitable for ``subprocess.run``.
    """
    return [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        str(notebook),
        "--output-dir",
        str(output_dir),
        f"--ExecutePreprocessor.timeout={timeout}",
        "--ExecutePreprocessor.kernel_name=python3",
    ]


def execute_notebook(notebook: Path, repo_root: Path, *, timeout: int) -> None:
    """Execute one notebook in a temporary directory.

    Parameters
    ----------
    notebook
        Notebook path to execute.
    repo_root
        Repository root used as the execution working directory.
    timeout
        Per-cell timeout in seconds.
    """
    with tempfile.TemporaryDirectory(prefix="bsm_notebook_exec_") as tmpdir:
        output_dir = Path(tmpdir)
        command = build_nbconvert_command(notebook, output_dir, timeout=timeout)
        subprocess.run(command, cwd=repo_root, check=True)


def execute_all_notebooks(root: Path, *, timeout: int) -> int:
    """Execute all discovered repository notebooks.

    Parameters
    ----------
    root
        Repository root.
    timeout
        Per-cell timeout in seconds.

    Returns
    -------
    int
        Number of notebooks executed.
    """
    if shutil.which("jupyter") is None:
        raise RuntimeError("jupyter is required to execute notebooks in the repository gate.")

    notebooks = discover_notebooks(root)
    if not notebooks:
        print("No notebooks discovered for execution.")
        return 0

    for notebook in notebooks:
        relative = notebook.relative_to(root)
        print(f"Executing notebook: {relative}")
        execute_notebook(notebook, root, timeout=timeout)
    print(f"Executed {len(notebooks)} notebook(s).")
    return len(notebooks)


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=int, default=1200)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the notebook execution command-line interface."""
    args = parse_args(list(sys.argv[1:] if argv is None else argv))
    root = Path(__file__).resolve().parents[1]
    execute_all_notebooks(root, timeout=args.timeout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
