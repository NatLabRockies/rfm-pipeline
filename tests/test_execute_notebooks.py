"""Tests for test execute notebooks."""

from __future__ import annotations

from pathlib import Path

from tools.execute_notebooks import (
    build_nbconvert_command,
    discover_notebooks,
    execute_notebook,
    select_notebooks,
)


def test_discover_notebooks_selects_expected_directories(tmp_path: Path) -> None:
    notebook_dir = tmp_path / "notebooks"
    notebook_dir.mkdir()
    expected = notebook_dir / "example.ipynb"
    expected.write_text("{}", encoding="utf-8")

    excluded = tmp_path / "docs" / "example.ipynb"
    excluded.parent.mkdir(parents=True)
    excluded.write_text("{}", encoding="utf-8")

    paths = discover_notebooks(tmp_path)

    assert paths == [expected]


def test_build_nbconvert_command_uses_temp_output_dir(tmp_path: Path) -> None:
    notebook = tmp_path / "notebooks" / "demo.ipynb"
    output_dir = tmp_path / "tmp-out"
    command = build_nbconvert_command(notebook, output_dir, timeout=900)

    assert str(notebook) in command
    assert str(output_dir) in command
    assert "--execute" in command
    assert "--to" in command
    assert "notebook" in command
    assert "--ExecutePreprocessor.timeout=900" in command
    assert "--ExecutePreprocessor.kernel_name=pixi-kernel-python3" in command


def test_select_notebooks_partitions_discovered_notebooks() -> None:
    notebooks = [Path(f"notebooks/{index}.ipynb") for index in range(7)]

    shards = [select_notebooks(notebooks, shard_index=index, shard_count=3) for index in range(3)]

    assert sorted(path for shard in shards for path in shard) == notebooks


def test_execute_notebook_bounds_kernel_demo_storage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    notebook = tmp_path / "notebooks" / "demo.ipynb"
    notebook.parent.mkdir()
    notebook.write_text("{}", encoding="utf-8")
    observed: dict[str, Path] = {}

    def fake_run(command, *, cwd, check, env):
        del command, cwd, check
        demo_parent = Path(env["RFM_MANUSCRIPT_DEMO_PARENT"])
        demo_parent.mkdir(parents=True)
        leaked_kernel_directory = demo_parent / "rfm_pipeline_demo_interrupted_kernel"
        leaked_kernel_directory.mkdir()
        (leaked_kernel_directory / "artifact.bin").write_bytes(b"test")
        observed["demo_parent"] = demo_parent

    monkeypatch.setattr("tools.execute_notebooks.subprocess.run", fake_run)

    execute_notebook(notebook, tmp_path, timeout=900)

    assert not observed["demo_parent"].exists()
