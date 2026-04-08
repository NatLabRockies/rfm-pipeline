"""Remove transient artifacts so the repository gate leaves a clean tree."""

from __future__ import annotations

import shutil
from pathlib import Path

DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".ipynb_checkpoints",
    "build",
    "dist",
    "htmlcov",
    ".coverage_html",
    "__MACOSX",
}
FILE_PATTERNS = ("*.pyc", "*.pyo", ".coverage", ".DS_Store")
DIR_PATTERNS = ("*.egg-info",)
RELATIVE_DIRS = (Path("docs") / "_build",)
EXCLUDED_PARTS = {
    ".git",
    ".pixi",
    ".venv",
}


def _is_excluded(path: Path) -> bool:
    """Return whether a path is inside an excluded directory tree."""
    return any(part in EXCLUDED_PARTS for part in path.parts)


def remove_transients(root: Path) -> list[Path]:
    """Remove common transient build, cache, and bytecode artifacts.

    Parameters
    ----------
    root
        Repository root.

    Returns
    -------
    list[pathlib.Path]
        Paths removed relative to ``root`` when possible.
    """
    removed: list[Path] = []

    for path in root.rglob("*"):
        if _is_excluded(path):
            continue
        if path.is_dir() and path.name in DIR_NAMES:
            shutil.rmtree(path, ignore_errors=True)
            removed.append(path)

    for pattern in DIR_PATTERNS:
        for path in root.rglob(pattern):
            if _is_excluded(path):
                continue
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
                removed.append(path)

    for rel_path in RELATIVE_DIRS:
        abs_path = root / rel_path
        if abs_path.exists():
            shutil.rmtree(abs_path, ignore_errors=True)
            removed.append(abs_path)

    for pattern in FILE_PATTERNS:
        for path in root.rglob(pattern):
            if _is_excluded(path):
                continue
            if path.exists():
                path.unlink()
                removed.append(path)

    seen: set[str] = set()
    ordered: list[Path] = []
    for path in removed:
        try:
            display = path.relative_to(root)
        except ValueError:
            display = path
        key = str(display)
        if key not in seen:
            seen.add(key)
            ordered.append(display)
    return sorted(ordered)


def main() -> int:
    """Run the transient-cleaning command-line interface."""
    root = Path(__file__).resolve().parents[1]
    removed = remove_transients(root)
    if removed:
        print("Removed transient artifacts:")
        for path in removed:
            print(f" - {path}")
    else:
        print("No transient artifacts found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
