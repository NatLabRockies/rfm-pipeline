"""Tests for the repository gate contract."""

from __future__ import annotations

from pathlib import Path

from tools.clean_transients import remove_transients
from tools.run_repository_gate import (
    FAST_VALIDATION_TASKS,
    PREP_TASKS,
    VALIDATION_TASKS,
    task_sequence,
)

EXPECTED_PREP_TASKS = [
    "clean-transients",
    "format-python",
    "format-markdown",
    "fix-notebooks",
]

EXPECTED_VALIDATION_TASKS = [
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
    "bsm-manuscript-example-tests",
    "workflow-tests",
    "manuscript-reproduction-smoke",
    "notebook-tests",
    "docs",
    "package-build",
    "package-smoke",
    "clean-transients",
    "repo-hygiene",
    "git-diff-check",
]

EXPECTED_FAST_VALIDATION_TASKS = [
    "build-import-smoke",
    "clean-transients",
    "repo-hygiene",
    "lint",
    "format-check",
    "markdown-check",
    "notebook-check",
    "notebook-workflow-check",
    "compile-check",
    "pre-push-tests",
    "bsm-manuscript-example-tests",
    "workflow-tests",
    "docs",
    "package-build",
    "package-smoke",
    "clean-transients",
    "repo-hygiene",
    "git-diff-check",
]


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


def test_repository_gate_declares_prepare_and_validation_task_sets() -> None:
    assert list(PREP_TASKS) == EXPECTED_PREP_TASKS
    assert list(VALIDATION_TASKS) == EXPECTED_VALIDATION_TASKS
    assert list(FAST_VALIDATION_TASKS) == EXPECTED_FAST_VALIDATION_TASKS


def test_repository_gate_modes_use_declared_task_sets() -> None:
    assert task_sequence("check") == VALIDATION_TASKS
    assert task_sequence("fast") == FAST_VALIDATION_TASKS
    assert task_sequence("fix") == PREP_TASKS + VALIDATION_TASKS
    assert task_sequence("clean") == PREP_TASKS + VALIDATION_TASKS


def test_internal_policy_files_are_not_part_of_the_public_repository() -> None:
    forbidden = ["AGENTS.md", "CODE_OF_CONDUCT.md", "SECURITY.md", "test_repo.sh"]
    assert not [name for name in forbidden if Path(name).exists()]


def test_pixi_declares_required_gate_tasks_and_build_dependencies() -> None:
    pixi = Path("pixi.toml").read_text(encoding="utf-8")
    required_snippets = [
        "[pypi-dependencies]",
        'build = ">=1.2"',
        'setuptools = ">=68"',
        'wheel = ">=0.45"',
        'package-build = "python -m build --no-isolation --sdist --wheel"',
        'package-smoke = "python tools/check_wheel.py"',
        'format-markdown = "python -m tools.format_markdown"',
        'markdown-check = "python -m tools.check_markdown"',
        'manuscript-reproduction-smoke = "python tools/check_manuscript_reproduction.py"',
        'notebook-tests = "python -m tools.execute_notebooks --timeout 1200"',
        'gate-fast = "python tools/run_repository_gate.py fast"',
        'gate = "python tools/run_repository_gate.py check"',
    ]

    for snippet in required_snippets:
        assert snippet in pixi


def test_pre_push_hook_uses_fast_gate() -> None:
    hooks = Path(".pre-commit-config.yaml").read_text(encoding="utf-8")

    assert "stages: [pre-push]" in hooks
    assert "pixi run gate-fast" in hooks
    assert "pixi run gate'" not in hooks


def test_bsm_example_suite_is_owned_by_dedicated_gate_task() -> None:
    pixi = Path("pixi.toml").read_text(encoding="utf-8")
    bridge = Path("tests/test_bsm_publication_example.py").read_text(encoding="utf-8")

    assert "bsm-manuscript-example-tests" in VALIDATION_TASKS
    assert "cd examples/bsm-manuscript" in pixi
    assert "python -m pytest -q tests" in pixi
    assert '"pytest"' not in bridge
