"""Tests for the repository gate contract."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from tools.clean_transients import remove_transients

PREP_TASKS = [
    "clean-transients",
    "format-python",
    "format-markdown",
    "fix-notebooks",
]

VALIDATION_TASKS = [
    "build-import-smoke",
    "clean-transients",
    "repo-hygiene",
    "lint",
    "format-check",
    "markdown-check",
    "notebook-check",
    "notebook-workflow-check",
    "compile-check",
    "unit-tests",
    "workflow-tests",
    "manuscript-reproduction-smoke",
    "notebook-tests",
    "docs",
    "package-build",
    "clean-transients",
    "repo-hygiene",
    "git-diff-check",
]


def _script_text() -> str:
    return Path("test_repo.sh").read_text(encoding="utf-8")


def _parse_array(script: str, name: str) -> list[str]:
    match = re.search(rf"{name}=\((.*?)\)", script, flags=re.DOTALL)
    assert match is not None
    return re.findall(r"\n\s*([a-z0-9-]+)", match.group(1))


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


def test_test_repo_script_declares_prepare_and_validation_task_sets() -> None:
    script = _script_text()

    assert _parse_array(script, "PREP_TASKS") == PREP_TASKS
    assert _parse_array(script, "VALIDATION_TASKS") == VALIDATION_TASKS


def test_test_repo_fix_and_check_modes_use_declared_task_sets() -> None:
    script = _script_text()

    assert '--fix|"")' in script
    assert 'run_task_list "${PREP_TASKS[@]}"' in script
    assert script.count('run_task_list "${VALIDATION_TASKS[@]}"') == 2
    assert "--check|--ci)" in script
    check_block = re.search(r"--check\|--ci\)(.*?)--clean\)", script, flags=re.DOTALL)
    assert check_block is not None
    assert 'run_task_list "${PREP_TASKS[@]}"' not in check_block.group(1)


def test_test_repo_clean_mode_reenters_through_script_path() -> None:
    script = _script_text()

    assert 'SCRIPT_PATH="$REPO_ROOT/$(basename "${BASH_SOURCE[0]}")"' in script
    assert 'bash "$SCRIPT_PATH" --fix' in script


def test_test_repo_reports_missing_pixi_cleanly(tmp_path: Path) -> None:
    result = subprocess.run(
        ["bash", "test_repo.sh", "--check"],
        check=False,
        cwd=Path(__file__).resolve().parents[1],
        env={"PATH": "/usr/bin:/bin", "PIXI_BIN": str(tmp_path / "missing-pixi")},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "error: pixi executable not found" in result.stderr


def test_pixi_declares_required_gate_tasks_and_build_dependencies() -> None:
    pixi = Path("pixi.toml").read_text(encoding="utf-8")
    required_snippets = [
        "[pypi-dependencies]",
        'build = ">=1.2"',
        'setuptools = ">=68"',
        'wheel = ">=0.45"',
        'package-build = "python -m build --no-isolation --sdist --wheel"',
        'format-markdown = "python -m tools.format_markdown"',
        'markdown-check = "python -m tools.check_markdown"',
        'manuscript-reproduction-smoke = "python tools/check_manuscript_reproduction.py"',
    ]

    for snippet in required_snippets:
        assert snippet in pixi
