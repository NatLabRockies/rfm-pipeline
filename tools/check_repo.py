"""Repository hygiene checks used by the local and CI gates."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

TEXT_SUFFIXES = {".py", ".md", ".rst", ".toml", ".yml", ".yaml", ".sh"}
EXCLUDED_PARTS = {
    ".git",
    ".pixi",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "docs",
    "_build",
    "build",
    "dist",
    "__MACOSX",
}
EXCLUDED_FRAGMENTS = {".egg-info/", "docs/_build/", ".ipynb_checkpoints/"}


def _is_excluded(path: Path) -> bool:
    """Return whether a path belongs to an excluded directory tree."""
    path_str = str(path).replace("\\", "/")
    if any(part in EXCLUDED_PARTS for part in path.parts):
        return True
    return any(fragment in path_str for fragment in EXCLUDED_FRAGMENTS)


def iter_text_files(root: Path) -> Iterable[Path]:
    """Yield repository text files relevant to the hygiene gate."""
    for path in root.rglob("*"):
        if path.is_dir() or _is_excluded(path):
            continue
        if path.suffix in TEXT_SUFFIXES:
            yield path


def scan_text_hygiene(paths: Iterable[Path]) -> list[str]:
    """Return text-formatting failures for a collection of files."""
    failures: list[str] = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            failures.append(f"non-UTF-8 text file: {path}")
            continue
        if "\t" in text:
            failures.append(f"tab character found: {path}")
        trailing = [
            index for index, line in enumerate(text.splitlines(), start=1) if line.rstrip() != line
        ]
        if trailing:
            failures.append(f"trailing whitespace in {path}: lines {trailing[:10]}")
        if text and not text.endswith("\n"):
            failures.append(f"missing terminal newline: {path}")
    return failures


def scan_generated_python_artifacts(root: Path) -> list[str]:
    """Return failures caused by committed Python-generated artifacts."""
    failures: list[str] = []
    for path in root.rglob("__pycache__"):
        if _is_excluded(path):
            continue
        failures.append(f"generated directory present: {path}")
    for path in root.rglob("*.pyc"):
        if _is_excluded(path):
            continue
        failures.append(f"generated file present: {path}")
    for path in root.rglob("*.egg-info"):
        if _is_excluded(path):
            continue
        failures.append(f"generated directory present: {path}")
    return failures


def main() -> int:
    """Run the repository hygiene command-line interface."""
    root = Path(__file__).resolve().parents[1]
    failures: list[str] = []
    failures.extend(scan_generated_python_artifacts(root))
    failures.extend(scan_text_hygiene(iter_text_files(root)))
    if failures:
        print("Repository checks failed:")
        for failure in failures:
            print(f" - {failure}")
        return 1
    print("Repository hygiene checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
