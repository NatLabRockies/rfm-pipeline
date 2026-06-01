"""Tests for the end-to-end manuscript reproduction stage chain."""

from __future__ import annotations

from pathlib import Path

from rfm_pipeline import (
    build_manuscript_notebook_context,
    run_manuscript_reproduction_stage_chain,
)


def test_run_manuscript_reproduction_chain_writes_all_phase3_artifact_families() -> None:
    context = build_manuscript_notebook_context(
        Path.cwd(),
        "08_manuscript_tables_and_figures.ipynb",
    )

    result = run_manuscript_reproduction_stage_chain(context)

    assert set(result.artifact_paths) == {
        "output_conditioning",
        "empirical_null_screen",
        "interaction_discovery",
        "nonlinear_discovery",
        "sparse_selection",
        "final_manuscript_artifacts",
    }
    assert all(
        path.exists()
        for stage_paths in result.artifact_paths.values()
        for path in stage_paths.values()
    )
    assert result.output_conditioning.conditioning.summary.loc[0, "stage"] == (
        "output_conditioning"
    )
    assert result.empirical_null_screening.screening.summary.loc[0, "stage"] == (
        "empirical_null_screening"
    )
    assert result.interaction_discovery.interactions.summary.loc[0, "stage"] == (
        "interaction_discovery"
    )
    assert result.nonlinear_discovery.nonlinear.summary.loc[0, "stage"] == ("nonlinear_discovery")
    assert result.sparse_selection_stability.sparse_selection.summary.loc[0, "stage"] == (
        "sparse_selection_and_stability"
    )
    assert result.final_manuscript_artifacts.final_artifacts.summary.loc[0, "stage"] == (
        "final_manuscript_tables_and_figures"
    )
