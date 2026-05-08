"""Tests for Phase 2 manuscript runtime and notebook skeletons."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bsm_rfm import (
    build_manuscript_notebook_context,
    load_manuscript_artifact_tables,
    manuscript_notebook_order,
    manuscript_runtime_summary_table,
    resolve_manuscript_runtime,
    validate_manuscript_artifact_tables,
    write_demo_manuscript_artifacts,
)
from bsm_rfm.manuscript_runtime import ManuscriptRuntimeContext


def test_resolve_manuscript_runtime_uses_real_data_when_configured() -> None:
    context = resolve_manuscript_runtime(Path.cwd())
    # With local config in place, real mode should be used
    assert context.mode == "real"
    assert context.runtime_dir is None
    assert set(context.artifact_paths) == {
        "input_metadata",
        "output_metadata",
        "case_study_input_matrix",
        "case_study_output_matrix",
        "manuscript_feature_catalog",
        "fixed_holdout_assignments",
    }


def test_demo_artifacts_validate_and_load(tmp_path: Path) -> None:
    paths = write_demo_manuscript_artifacts(tmp_path)
    tables = load_manuscript_artifact_tables(paths)
    inputs = tables["case_study_input_matrix"]

    assert validate_manuscript_artifact_tables(tables) == []
    assert inputs.shape[0] == 80
    assert tables["fixed_holdout_assignments"]["split"].tolist().count("holdout") == 16
    assert (inputs["x2"] > -1.0).all()
    assert abs(inputs["x1"].corr(inputs["x2"])) < 0.95


def test_notebook_context_and_summary_table_are_executable() -> None:
    context = build_manuscript_notebook_context(Path.cwd(), manuscript_notebook_order()[0])
    summary = manuscript_runtime_summary_table(context)
    assert context.runtime.mode in {"demo", "real"}
    assert summary["field"].tolist() == [
        "notebook_name",
        "mode",
        "local_override_used",
        "output_root",
        "unresolved_placeholders",
    ]


def test_notebook_context_accepts_nested_notebook_directory_path() -> None:
    nested = Path.cwd() / "notebooks" / "manuscript"
    context = build_manuscript_notebook_context(nested, manuscript_notebook_order()[0])
    assert context.runtime.repo_root == Path.cwd()


def test_manuscript_notebook_files_exist_in_frozen_order() -> None:
    notebook_root = Path("notebooks/manuscript")
    for notebook_name in manuscript_notebook_order():
        assert (notebook_root / notebook_name).exists()


def test_build_notebook_context_falls_back_to_demo_for_incompatible_real_sample_ids(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bad_paths = write_demo_manuscript_artifacts(tmp_path / "bad-real")
    holdout = pd.read_csv(bad_paths["fixed_holdout_assignments"])
    holdout["sample_id"] = holdout["sample_id"].astype(int) + 100_000
    holdout.to_csv(bad_paths["fixed_holdout_assignments"], index=False)

    runtime = ManuscriptRuntimeContext(
        mode="real",
        repo_root=Path.cwd(),
        artifact_paths=bad_paths,
        output_root=tmp_path / "bad-real-output",
        unresolved_placeholders=(),
        local_override_used=True,
        runtime_dir=None,
    )
    monkeypatch.setattr("bsm_rfm.manuscript_runtime.resolve_manuscript_runtime", lambda _: runtime)

    context = build_manuscript_notebook_context(Path.cwd(), manuscript_notebook_order()[0])

    assert context.runtime.mode == "demo"
    assert context.runtime.runtime_dir is not None
    assert len(context.tables["case_study_input_matrix"]) == 80
