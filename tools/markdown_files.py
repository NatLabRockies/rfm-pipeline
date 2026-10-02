"""Shared Markdown file discovery for repository formatting and checks."""

from __future__ import annotations

import subprocess
from pathlib import Path

EXCLUDED_PARTS = {
    ".git",
    ".pixi",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".ipynb_checkpoints",
    "__pycache__",
    ".venv",
    "build",
    "dist",
    "_build",
    "__MACOSX",
}
MARKDOWN_SUFFIXES = {".md", ".mdx", ".markdown"}

EXCLUDED_FILES = {
    Path(".github/ISSUE_TEMPLATE/bug_report.md"),
    Path(".github/ISSUE_TEMPLATE/feature_request.md"),
}


def is_excluded(path: Path, repo_root: Path | None = None) -> bool:
    """Return whether a path is inside an excluded directory tree or listed in EXCLUDED_FILES."""
    rel = path.relative_to(repo_root) if repo_root is not None and path.is_absolute() else path
    if any(part in EXCLUDED_PARTS for part in rel.parts):
        return True
    return rel in EXCLUDED_FILES


def _git_markdown_files(repo_root: Path) -> list[Path]:
    """Return tracked and untracked Markdown files that are not ignored."""
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        check=True,
        capture_output=True,
        text=True,
        cwd=repo_root,
    )

    files: set[Path] = set()
    for line in result.stdout.splitlines():
        path = repo_root / line
        if path.suffix.lower() not in MARKDOWN_SUFFIXES:
            continue
        if is_excluded(path, repo_root):
            continue
        if path.is_file():
            files.add(path)

    return sorted(files)


def _walk_markdown_files(repo_root: Path) -> list[Path]:
    """Return Markdown files discovered by walking the repository tree."""
    files: list[Path] = []
    for path in sorted(repo_root.rglob("*")):
        if path.suffix.lower() not in MARKDOWN_SUFFIXES:
            continue
        if is_excluded(path, repo_root):
            continue
        if path.is_file():
            files.append(path)
    return files


def list_markdown_files(repo_root: Path) -> list[Path]:
    """Return repository Markdown files using git-aware discovery when available."""
    try:
        return _git_markdown_files(repo_root)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return _walk_markdown_files(repo_root)
