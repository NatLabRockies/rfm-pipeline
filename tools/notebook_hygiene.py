"""Notebook hygiene utilities.

The fixer clears execution counts and cell outputs so the repository gate can enforce a
clean, reviewable notebook state before commits.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat

EXCLUDED_PARTS = {
    ".git",
    ".pixi",
    ".pytest_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
    "docs",
    "_build",
}


def iter_notebook_paths(root: Path) -> list[Path]:
    """Return repository notebook paths excluding generated directories."""
    notebooks: list[Path] = []
    for path in root.rglob("*.ipynb"):
        if any(part in EXCLUDED_PARTS for part in path.parts):
            continue
        notebooks.append(path)
    return sorted(notebooks)


def notebook_has_outputs(path: Path) -> bool:
    """Return whether a notebook contains execution counts or stored outputs."""
    notebook = nbformat.read(path, as_version=4)
    for cell in notebook.cells:
        if cell.get("cell_type") != "code":
            continue
        if cell.get("execution_count") is not None:
            return True
        if cell.get("outputs"):
            return True
    return False


def sanitize_notebook(path: Path, *, write: bool) -> bool:
    """Strip outputs and execution counts from a notebook.

    Parameters
    ----------
    path
        Notebook path to inspect.
    write
        Whether to write changes back to disk.

    Returns
    -------
    bool
        ``True`` when the notebook required modification.
    """
    notebook = nbformat.read(path, as_version=4)
    changed = False
    for cell in notebook.cells:
        if cell.get("cell_type") != "code":
            continue
        if cell.get("execution_count") is not None:
            cell["execution_count"] = None
            changed = True
        if cell.get("outputs"):
            cell["outputs"] = []
            changed = True
    if changed and write:
        nbformat.write(notebook, path)
    return changed


def check_notebooks(root: Path) -> list[str]:
    """Return notebook-hygiene failures under ``root``."""
    failures: list[str] = []
    for path in iter_notebook_paths(root):
        if notebook_has_outputs(path):
            failures.append(f"notebook has stored outputs or execution counts: {path}")
    return failures


def fix_notebooks(root: Path) -> list[Path]:
    """Strip outputs from all notebooks requiring cleanup."""
    changed: list[Path] = []
    for path in iter_notebook_paths(root):
        if sanitize_notebook(path, write=True):
            changed.append(path)
    return changed


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["check", "fix"])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the notebook hygiene command-line interface."""
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))
    root = Path(__file__).resolve().parents[1]
    if args.mode == "fix":
        changed = fix_notebooks(root)
        print(f"Notebook hygiene fix completed. Cleaned {len(changed)} notebook(s).")
        return 0

    failures = check_notebooks(root)
    if failures:
        print("Notebook hygiene checks failed:")
        for failure in failures:
            print(f" - {failure}")
        return 1
    print("Notebook hygiene checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
