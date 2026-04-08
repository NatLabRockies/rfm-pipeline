"""Tests for the repository gate contract."""

from __future__ import annotations

from pathlib import Path

from tools.clean_transients import remove_transients


def test_remove_transients_cleans_common_cache_and_build_artifacts(tmp_path: Path) -> None:
    (tmp_path / "pkg" / "__pycache__").mkdir(parents=True)
    (tmp_path / "pkg" / "__pycache__" / "x.pyc").write_bytes(b"x")
    (tmp_path / ".ruff_cache").mkdir()
    (tmp_path / "docs" / "_build").mkdir(parents=True)
    (tmp_path / "build").mkdir()
    (tmp_path / "src" / "pkg.egg-info").mkdir(parents=True)
    (tmp_path / ".DS_Store").write_text("metadata", encoding="utf-8")

    removed = {str(path) for path in remove_transients(tmp_path)}

    assert "pkg/__pycache__" in removed
    assert ".ruff_cache" in removed
    assert "docs/_build" in removed
    assert "build" in removed
    assert "src/pkg.egg-info" in removed
    assert ".DS_Store" in removed
    assert not (tmp_path / "pkg" / "__pycache__").exists()
    assert not (tmp_path / ".ruff_cache").exists()
    assert not (tmp_path / "docs" / "_build").exists()
    assert not (tmp_path / "build").exists()
    assert not (tmp_path / "src" / "pkg.egg-info").exists()
    assert not (tmp_path / ".DS_Store").exists()


def test_remove_transients_preserves_pixi_environment_contents(tmp_path: Path) -> None:
    pixi_build = (
        tmp_path / ".pixi" / "envs" / "default" / "lib" / "python3.12" / "site-packages" / "build"
    )
    pixi_build.mkdir(parents=True)
    (pixi_build / "__init__.py").write_text("x = 1\n", encoding="utf-8")

    removed = {str(path) for path in remove_transients(tmp_path)}

    assert ".pixi/envs/default/lib/python3.12/site-packages/build" not in removed
    assert pixi_build.exists()


def test_test_repo_script_syncs_env_and_runs_full_gate() -> None:
    script = Path("test_repo.sh").read_text(encoding="utf-8")
    assert 'echo ">>> $PIXI_BIN install"' in script
    assert "run_python_smoke" in script
    assert script.index("run_task build-import-smoke") < script.index("run_task clean-transients")
    assert "run_task docs" in script
    assert "run_task package-build" in script
    assert "run_task notebook-tests" in script
    assert "run_task git-diff-check" in script


def test_pixi_build_environment_declares_no_isolation_build_requirements() -> None:
    pixi = Path("pixi.toml").read_text(encoding="utf-8")
    assert "[pypi-dependencies]" in pixi
    assert 'build = ">=1.2"' in pixi
    assert 'setuptools = ">=68"' in pixi
    assert 'wheel = ">=0.45"' in pixi


def test_pixi_package_build_task_uses_python_build() -> None:
    pixi = Path("pixi.toml").read_text(encoding="utf-8")
    assert 'package-build = "python -m build --no-isolation --sdist --wheel"' in pixi
