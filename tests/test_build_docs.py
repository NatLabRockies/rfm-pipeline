"""Tests for docs build configuration and command wiring."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from unittest.mock import patch

from tools.build_docs import build_docs


def _load_docs_conf_module():
    repo_root = Path(__file__).resolve().parents[1]
    conf_path = repo_root / "docs" / "conf.py"
    spec = importlib.util.spec_from_file_location("bsm_docs_conf", conf_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pyproject_version(repo_root: Path) -> str:
    pyproject = (repo_root / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"$', pyproject, flags=re.MULTILINE)
    assert match is not None
    return match.group(1)


def test_docs_conf_uses_ivar_for_attribute_sections() -> None:
    module = _load_docs_conf_module()
    assert module.napoleon_use_ivar is True


def test_docs_conf_release_matches_pyproject_version() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_docs_conf_module()
    version = _pyproject_version(repo_root)
    assert module.release == version
    assert module.version == version


def test_docs_index_includes_user_guides() -> None:
    index_text = Path("docs/index.md").read_text(encoding="utf-8")
    assert "quickstart" in index_text
    assert "export_bundle" in index_text
    assert "reproducibility_example" in index_text


def test_docs_index_includes_scope_boundary_guide() -> None:
    index_text = Path("docs/index.md").read_text(encoding="utf-8")
    assert "scope_boundary" in index_text


def test_module_plan_uses_live_module_names() -> None:
    module_plan = Path("docs/module_plan.md").read_text(encoding="utf-8")
    assert "`bsm_rfm.feature_expansion`" in module_plan
    assert "`bsm_rfm.regularized_screening`" in module_plan
    assert "`bsm_rfm.workflow`" in module_plan
    assert "`bsm_rfm.regularized_screen`" not in module_plan
    assert "`bsm_rfm.screening_null`" not in module_plan
    assert "`bsm_rfm.feature_engineering`" not in module_plan


def test_build_docs_uses_repo_build_directory_by_default(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True)

    with patch("tools.build_docs.subprocess.run") as run_mock:
        output_dir = build_docs(tmp_path)

    expected = (tmp_path / "docs" / "_build" / "html").resolve()
    assert output_dir == expected

    assert run_mock.call_count == 1
    args, kwargs = run_mock.call_args
    command = args[0]
    assert command[:4] == [command[0], "-m", "sphinx", "-W"]
    assert "html" in command
    assert str(docs_dir) in command
    assert str(expected) in command
    assert kwargs["cwd"] == tmp_path
    assert kwargs["check"] is True


def test_build_docs_respects_explicit_output_dir(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True)
    explicit = tmp_path / "artifacts" / "docs-html"

    with patch("tools.build_docs.subprocess.run") as run_mock:
        output_dir = build_docs(tmp_path, explicit)

    assert output_dir == explicit.resolve()

    assert run_mock.call_count == 1
    args, _kwargs = run_mock.call_args
    command = args[0]
    assert str(explicit.resolve()) in command


def test_docs_index_includes_manuscript_data_contract_guide() -> None:
    index_text = Path("docs/index.md").read_text(encoding="utf-8")
    assert "manuscript_data_contract" in index_text


def test_docs_index_includes_manuscript_runtime_guide() -> None:
    index_text = Path("docs/index.md").read_text(encoding="utf-8")
    assert "manuscript_runtime" in index_text
