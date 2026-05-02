"""Checks monotonicity of per-output nRMSE summary quantiles.

This guards against platform-sensitive bootstrap ordering or aggregation bugs.
"""

from __future__ import annotations

from pathlib import Path

from bsm_rfm.manuscript_runtime import build_manuscript_notebook_context
from bsm_rfm.manuscript_stages import run_final_manuscript_artifacts_stage


def test_per_output_nrmse_quantiles_monotonic() -> None:
    context = build_manuscript_notebook_context(
        Path.cwd(), "08_manuscript_tables_and_figures.ipynb"
    )
    result = run_final_manuscript_artifacts_stage(context)
    summary = result.final_artifacts.per_output_nrmse_summary

    assert len(summary) == 1
    row = summary.loc[0]
    p10 = float(row["p10"])
    p25 = float(row["p25"])
    p50 = float(row["p50"])
    p75 = float(row["p75"])
    p90 = float(row["p90"])

    assert p10 <= p25 <= p50 <= p75 <= p90

    # n_included cannot exceed n_total and must be at least 1
    n_included = int(row["n_included"])
    n_total = int(row["n_total"])
    assert 1 <= n_included <= n_total
