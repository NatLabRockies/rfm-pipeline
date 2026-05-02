"""Test that sparse-selection stage writes a consolidated diagnostics artifact.

This is a test-first addition to ensure CI/local parity issues expose diagnostics
early in the pipeline.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bsm_rfm.manuscript_runtime import build_manuscript_notebook_context
from bsm_rfm.manuscript_stages import (
    run_sparse_selection_stability_stage,
    write_sparse_selection_stability_artifacts,
)


def test_sparse_selection_writes_diagnostics(tmp_path: Path) -> None:
    context = build_manuscript_notebook_context(
        Path.cwd(), "06_sparse_selection_and_stability.ipynb"
    )

    stage_result = run_sparse_selection_stability_stage(context)
    sparse_result = stage_result.sparse_selection

    paths = write_sparse_selection_stability_artifacts(sparse_result, tmp_path)
    assert "sparse_selection_diagnostics" in paths

    df = pd.read_csv(paths["sparse_selection_diagnostics"])
    assert df.shape[0] == 1

    expected_cols = {
        "n_candidate_terms",
        "n_full_support_terms",
        "n_final_stable_support_terms",
        "mean_resample_jaccard",
        "mean_resample_spearman",
        "passes_jaccard_threshold",
        "passes_spearman_threshold",
        "final_stable_support_nonempty",
    }
    assert expected_cols.issubset(set(df.columns))

    assert int(df.loc[0, "n_candidate_terms"]) > 0
    assert int(df.loc[0, "n_full_support_terms"]) > 0
    assert int(df.loc[0, "n_final_stable_support_terms"]) > 0
    assert bool(df.loc[0, "final_stable_support_nonempty"]) is True
