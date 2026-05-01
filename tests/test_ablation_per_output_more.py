"""Additional ablation and per-output nRMSE consistency tests.

These tests assert that the per-output nRMSE summary quantiles match the
per-output frame and that the worst-output names correspond to the largest
per-output nRMSE values. They also validate the ablation table models and
that nRMSE values are finite.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from bsm_rfm.manuscript_runtime import build_manuscript_notebook_context
from bsm_rfm.manuscript_stages import run_final_manuscript_artifacts_stage


def test_per_output_summary_quantiles_and_worst_outputs() -> None:
    nb = "08_manuscript_tables_and_figures.ipynb"
    context = build_manuscript_notebook_context(Path.cwd(), nb)
    result = run_final_manuscript_artifacts_stage(context)

    per_out = result.final_artifacts.per_output_nrmse
    summary = result.final_artifacts.per_output_nrmse_summary.loc[0]

    included = per_out.loc[per_out["included_in_macro"]].copy()
    assert len(included) >= 1

    nrmse_vals = included["nrmse"].to_numpy(dtype=float)
    quantiles = np.nanquantile(nrmse_vals, [0.10, 0.25, 0.50, 0.75, 0.90])

    # Quantiles in the summary should match direct computation
    assert math.isfinite(float(summary["p50"]))
    assert abs(float(summary["p10"]) - float(quantiles[0])) < 1e-12
    assert abs(float(summary["p25"]) - float(quantiles[1])) < 1e-12
    assert abs(float(summary["p50"]) - float(quantiles[2])) < 1e-12
    assert abs(float(summary["p75"]) - float(quantiles[3])) < 1e-12
    assert abs(float(summary["p90"]) - float(quantiles[4])) < 1e-12

    # Worst-output names and nRMSEs should correspond to the largest nRMSEs
    worst = included.nlargest(min(3, len(included)), "nrmse").reset_index(drop=True)
    for rank in range(1, 4):
        name_col = f"worst_{rank}_name"
        val_col = f"worst_{rank}_nrmse"
        if rank - 1 < len(worst):
            assert summary[name_col] == worst.loc[rank - 1, "output_name"]
            assert math.isfinite(float(summary[val_col]))
            assert abs(float(summary[val_col]) - float(worst.loc[rank - 1, "nrmse"])) < 1e-12
        else:
            # Missing entries should be None / NaN
            assert summary[name_col] is None or pd.isna(summary[name_col])
            assert math.isnan(float(summary[val_col]))


def test_ablation_table_models_and_nrmse_finite() -> None:
    nb = "08_manuscript_tables_and_figures.ipynb"
    context = build_manuscript_notebook_context(Path.cwd(), nb)
    result = run_final_manuscript_artifacts_stage(context)

    ablation = result.final_artifacts.ablation_table
    expected = {"null_mean", "main_effects_ols", "screened_ols", "penalized_ols", "final_ols"}

    assert expected.issubset(set(ablation["model_name"]))
    assert "nrmse" in ablation.columns
    assert all([math.isfinite(float(x)) for x in ablation["nrmse"].tolist()])
