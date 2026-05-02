"""Bootstrap CI sanity checks for the ablation table.

Ensure bootstrapped CIs are finite and contain the point estimate for each
ablation model. This helps detect platform-sensitive resampling or CI bugs.
"""

from __future__ import annotations

import math
from pathlib import Path

from bsm_rfm.manuscript_runtime import build_manuscript_notebook_context
from bsm_rfm.manuscript_stages import run_final_manuscript_artifacts_stage


def test_ablation_bootstrap_ci_contains_point_estimate() -> None:
    context = build_manuscript_notebook_context(
        Path.cwd(), "08_manuscript_tables_and_figures.ipynb"
    )
    result = run_final_manuscript_artifacts_stage(context)
    ablation = result.final_artifacts.ablation_table

    assert not ablation.empty
    for _, row in ablation.iterrows():
        nrmse = float(row["nrmse"])
        ci_lower = float(row["ci_lower"])
        ci_upper = float(row["ci_upper"])
        assert math.isfinite(nrmse)
        assert math.isfinite(ci_lower)
        assert math.isfinite(ci_upper)
        assert ci_lower <= nrmse <= ci_upper, (
            f"{row['model_name']} nRMSE {nrmse} not inside CI [{ci_lower}, {ci_upper}]"
        )
        assert ci_upper >= ci_lower
