"""Syntax compilation checks that do not write ``__pycache__`` artifacts."""

from __future__ import annotations

from pathlib import Path

EXCLUDED_PARTS = {
    ".git",
    ".pixi",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__MACOSX",
    "build",
    "dist",
}


def iter_python_files(root: Path) -> list[Path]:
    """Return repository Python files that should pass syntax compilation.

    Parameters
    ----------
    root
        Repository root.

    Returns
    -------
    list[pathlib.Path]
        Sorted Python source files.
    """
    files: list[Path] = []
    for path in root.rglob("*.py"):
        if any(part in EXCLUDED_PARTS for part in path.parts):
            continue
        files.append(path)
    return sorted(files)


def compile_python_files(root: Path) -> int:
    """Compile repository Python files in memory.

    Parameters
    ----------
    root
        Repository root.

    Returns
    -------
    int
        Number of files compiled.
    """
    count = 0
    for path in iter_python_files(root):
        try:
            source = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise UnicodeDecodeError(
                exc.encoding,
                exc.object,
                exc.start,
                exc.end,
                f"{exc.reason} in {path}",
            ) from exc
        compile(source, str(path), "exec")
        count += 1
    print(f"Compiled {count} Python file(s) in memory.")
    return count


def main() -> int:
    """Run the compile-check command-line interface."""
    root = Path(__file__).resolve().parents[1]
    compile_python_files(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
