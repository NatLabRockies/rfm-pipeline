"""Tests for test build docs."""

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


def test_build_docs_uses_temporary_directory(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    with patch("tools.build_docs.subprocess.run") as run_mock:
        build_docs(tmp_path)

    assert run_mock.call_count == 1
    args, kwargs = run_mock.call_args
    command = args[0]
    assert command[:4] == [command[0], "-m", "sphinx", "-W"]
    assert "html" in command
    assert str(docs_dir) in command
    assert kwargs["cwd"] == tmp_path
    assert kwargs["check"] is True
