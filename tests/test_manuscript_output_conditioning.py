"""Tests for Phase 3 manuscript output-conditioning stage."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from rfm_pipeline import (
    OutputConditioningSpec,
    build_manuscript_notebook_context,
    condition_manuscript_outputs,
    run_output_conditioning_stage,
)


def test_condition_manuscript_outputs_filters_and_scores_from_train_split() -> None:
    output_matrix = pd.DataFrame(
        {
            "sample_id": [1, 2, 3, 4, 5, 6],
            "constant": [5.0, 5.0, 5.0, 5.0, 9.0, 10.0],
            "tiny_relative_range": [100.0, 100.1, 100.0, 100.1, 102.0, 103.0],
            "signal_a": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "signal_b": [0.0, 2.0, 4.0, 6.0, 8.0, 10.0],
        }
    )
    holdout_assignments = pd.DataFrame(
        {
            "sample_id": [1, 2, 3, 4, 5, 6],
            "split": ["train", "train", "train", "train", "holdout", "holdout"],
        }
    )
    spec = OutputConditioningSpec(
        epsilon_var=1.0e-12,
        epsilon_snr=1.0e-2,
        snr_delta=1.0e-12,
        method="pca",
        retained_components=5,
        retained_variance_fraction=0.90,
    )

    result = condition_manuscript_outputs(output_matrix, holdout_assignments, spec)

    assert result.retained_output_names == ("signal_a", "signal_b")
    assert set(result.culled_output_names) == {"constant", "tiny_relative_range"}
    assert result.pca_scores.columns.tolist() == ["sample_id", "PC1"]
    assert result.pca_scores.shape[0] == len(output_matrix)
    assert result.pca_loadings["output_name"].tolist() == ["signal_a", "signal_b"]
    assert result.summary.loc[0, "n_outputs_culled"] == 2
    assert result.summary.loc[0, "n_components_retained"] == 1


def test_run_output_conditioning_stage_writes_notebook_artifacts() -> None:
    context = build_manuscript_notebook_context(Path.cwd(), "02_output_conditioning.ipynb")
    result = run_output_conditioning_stage(context)

    assert result.conditioning.summary.loc[0, "stage"] == "output_conditioning"
    assert set(result.artifact_paths) == {
        "output_filter_diagnostics",
        "pca_scores",
        "pca_loadings",
        "pca_explained_variance",
        "output_conditioning_summary",
    }
    assert all(path.exists() for path in result.artifact_paths.values())
    assert result.artifact_paths["pca_scores"].parent == (
        context.runtime.output_root / "output_conditioning"
    )
