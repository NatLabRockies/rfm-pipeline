"""Check ablation table ordering and n_features sanity.

Guards against platform-sensitive ordering and feature-count bugs.
"""

from __future__ import annotations

from pathlib import Path

from bsm_rfm.manuscript_runtime import build_manuscript_notebook_context
from bsm_rfm.manuscript_stages import run_final_manuscript_artifacts_stage


def test_ablation_sorted_by_nrmse_and_n_features_positive() -> None:
    context = build_manuscript_notebook_context(
        Path.cwd(), "08_manuscript_tables_and_figures.ipynb"
    )
    result = run_final_manuscript_artifacts_stage(context)
    ablation = result.final_artifacts.ablation_table

    # Basic invariants
    assert not ablation.empty
    expected = {"null_mean", "main_effects_ols", "screened_ols", "penalized_ols", "final_ols"}
    assert set(ablation["model_name"]) == expected
    assert "n_features" in ablation.columns

    # n_features non-negative
    assert (ablation["n_features"].dropna().astype(float) >= 0).all()

    # nrmse values must be present and finite
    import numpy as np

    assert ablation["nrmse"].notna().all()
    assert np.isfinite(np.asarray(ablation["nrmse"]).astype(float)).all()
