"""Check per-output included count matches per-output summary.

Ensures the per-output included flags and the computed summary counts are consistent.
"""

from __future__ import annotations

from pathlib import Path

from rfm_pipeline.manuscript_runtime import build_manuscript_notebook_context
from rfm_pipeline.manuscript_stages import run_final_manuscript_artifacts_stage


def test_per_output_n_included_matches_summary() -> None:
    context = build_manuscript_notebook_context(
        Path.cwd(), "08_manuscript_tables_and_figures.ipynb"
    )
    result = run_final_manuscript_artifacts_stage(context)
    per_out = result.final_artifacts.per_output_nrmse
    summary = result.final_artifacts.per_output_nrmse_summary

    assert len(summary) == 1
    row = summary.loc[0]
    n_included_summary = int(row["n_included"])
    n_total_summary = int(row["n_total"])

    # counts must match the per-output table
    assert n_total_summary == len(per_out)
    actual_included = int(per_out["included_in_macro"].sum())
    assert actual_included == n_included_summary
