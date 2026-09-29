"""PA-C2: article-source resolution must recognize pixi-based study repos.

``normalize_manuscript_repo_root`` previously required a ``pyproject.toml`` marker,
which excludes fully pixi-managed study repositories (``pixi.toml`` only). The
publication run-of-record study directory is pixi-managed, so root resolution
must accept either a ``pyproject.toml`` or a ``pixi.toml`` alongside ``configs/``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rfm_pipeline.manuscript_runtime import normalize_manuscript_repo_root


def _make_repo(root: Path, marker: str) -> None:
    (root / "configs").mkdir(parents=True, exist_ok=True)
    (root / marker).write_text("# marker\n", encoding="utf-8")


def test_resolves_pixi_only_repo(tmp_path: Path) -> None:
    _make_repo(tmp_path, "pixi.toml")
    assert normalize_manuscript_repo_root(tmp_path) == tmp_path.resolve()


def test_resolves_pyproject_repo(tmp_path: Path) -> None:
    _make_repo(tmp_path, "pyproject.toml")
    assert normalize_manuscript_repo_root(tmp_path) == tmp_path.resolve()


def test_resolves_from_nested_path_pixi(tmp_path: Path) -> None:
    _make_repo(tmp_path, "pixi.toml")
    nested = tmp_path / "scripts" / "publication-run"
    nested.mkdir(parents=True, exist_ok=True)
    assert normalize_manuscript_repo_root(nested) == tmp_path.resolve()


def test_requires_configs_dir(tmp_path: Path) -> None:
    # Marker present but no configs/ -> not a study repo root.
    (tmp_path / "pixi.toml").write_text("# marker\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        normalize_manuscript_repo_root(tmp_path)


def test_raises_without_marker(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    with pytest.raises(FileNotFoundError):
        normalize_manuscript_repo_root(tmp_path)
