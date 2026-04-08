"""Tests for docs build configuration and command wiring."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import patch

from tools.build_docs import build_docs


def test_docs_conf_uses_ivar_for_attribute_sections() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    conf_path = repo_root / "docs" / "conf.py"
    spec = importlib.util.spec_from_file_location("bsm_docs_conf", conf_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.napoleon_use_ivar is True


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
