"""Tests for final manuscript table and figure artifact regeneration."""

from __future__ import annotations

import math
from pathlib import Path

from bsm_rfm.manuscript_runtime import (
    build_manuscript_notebook_context,
    load_manuscript_case_study_config,
)
from bsm_rfm.manuscript_stages import (
    final_manuscript_artifacts_spec_from_case_study_config,
    run_final_manuscript_artifacts_stage,
)

_ABLATION_MODEL_NAMES = {
    "null_mean",
    "main_effects_ols",
    "screened_ols",
    "penalized_ols",
    "final_ols",
}


def test_final_artifact_spec_matches_frozen_case_study_contract() -> None:
    config = load_manuscript_case_study_config(Path.cwd())
    spec = final_manuscript_artifacts_spec_from_case_study_config(config)

    assert spec.final_predictor_count_reference == 340
    assert spec.final_first_order_input_count_reference == 62
    assert spec.intermediate_penalized_holdout_nrmse_reference == 0.0859
    assert spec.final_ols_holdout_nrmse_reference == 0.0445
    assert spec.nrmse_denominator_definition == (
        "macro_average_rmse_divided_by_training_response_range"
    )
    assert spec.nrmse_min_range == 1.0e-6
    assert spec.nrmse_reference_matrix == "Y_train"
    assert spec.bootstrap_count == 200
    assert spec.bootstrap_alpha == 0.05
    assert spec.random_seed == 123
    assert spec.inferential_filter_interval_method == (
        "hc3_wald_95_percent_drop_if_zero_compatible_for_all_outputs"
    )
    assert spec.inferential_filter_alpha == 0.05


def test_run_final_manuscript_artifacts_stage_executes_demo_context() -> None:
    context = build_manuscript_notebook_context(
        Path.cwd(),
        "08_manuscript_tables_and_figures.ipynb",
    )

    result = run_final_manuscript_artifacts_stage(context)

    final_artifacts = result.final_artifacts
    assert final_artifacts.summary.loc[0, "stage"] == "final_manuscript_tables_and_figures"
    assert final_artifacts.final_ols_summary.loc[0, "stage"] == (
        "final_ols_and_manuscript_artifacts"
    )
    assert (
        final_artifacts.final_ols_summary.loc[0, "n_prefilter_features"]
        >= (final_artifacts.final_ols_summary.loc[0, "n_final_features"])
    )
    assert final_artifacts.final_ols_summary.loc[0, "n_final_features"] >= 1
    assert final_artifacts.hc3_wald_intervals["feature_name"].nunique() == len(
        final_artifacts.prefilter_support_features
    )
    assert set(final_artifacts.hc3_wald_intervals["zero_compatible"].unique()) <= {
        True,
        False,
    }
    assert final_artifacts.hc3_inferential_filter_summary["hc3_retained_after_filter"].any()
    assert final_artifacts.final_support_features["hc3_retained_after_filter"].all()
    assert set(final_artifacts.model_performance["model_name"]) == {
        "final_ols_demo",
        "final_ols_reference",
        "intermediate_penalized_reference",
        "null_mean_baseline_demo",
    }
    assert final_artifacts.figure_specs["figure_name"].tolist() == [
        "figure_model_performance",
        "figure_support_composition",
    ]
    assert result.artifact_paths["prefilter_support_features"].exists()
    assert result.artifact_paths["hc3_wald_intervals"].exists()
    assert result.artifact_paths["hc3_inferential_filter_summary"].exists()
    assert result.artifact_paths["model_performance"].exists()
    assert result.artifact_paths["workflow_stage_summary"].exists()
    assert result.artifact_paths["figure_model_performance_svg"].suffix == ".svg"
    assert result.artifact_paths["figure_model_performance_svg"].exists()
    assert result.artifact_paths["ablation_table"].exists()
    assert result.artifact_paths["per_output_nrmse"].exists()
    assert result.artifact_paths["per_output_nrmse_summary"].exists()


def test_ablation_table_contains_all_five_models() -> None:
    context = build_manuscript_notebook_context(
        Path.cwd(), "08_manuscript_tables_and_figures.ipynb"
    )
    result = run_final_manuscript_artifacts_stage(context)
    ablation = result.final_artifacts.ablation_table

    assert set(ablation["model_name"]) == _ABLATION_MODEL_NAMES
    assert "n_features" in ablation.columns
    assert "nrmse" in ablation.columns
    assert "ci_lower" in ablation.columns
    assert "ci_upper" in ablation.columns
    assert ablation["nrmse"].notna().all()


def test_ablation_table_final_ols_not_worse_than_null_mean() -> None:
    context = build_manuscript_notebook_context(
        Path.cwd(), "08_manuscript_tables_and_figures.ipynb"
    )
    result = run_final_manuscript_artifacts_stage(context)
    ablation = result.final_artifacts.ablation_table

    null_nrmse = float(ablation.loc[ablation["model_name"] == "null_mean", "nrmse"].iloc[0])
    final_nrmse = float(ablation.loc[ablation["model_name"] == "final_ols", "nrmse"].iloc[0])
    assert final_nrmse <= null_nrmse, (
        f"final_ols nRMSE ({final_nrmse:.4f}) is worse than null_mean ({null_nrmse:.4f})"
    )


def test_per_output_nrmse_schema_and_values() -> None:
    context = build_manuscript_notebook_context(
        Path.cwd(), "08_manuscript_tables_and_figures.ipynb"
    )
    result = run_final_manuscript_artifacts_stage(context)
    per_out = result.final_artifacts.per_output_nrmse

    assert "output_name" in per_out.columns
    assert "nrmse" in per_out.columns
    assert "included_in_macro" in per_out.columns
    assert len(per_out) >= 1
    included = per_out[per_out["included_in_macro"]]
    assert len(included) >= 1
    assert (included["nrmse"] > 0).all()
    assert included["nrmse"].isna().sum() == 0


def test_per_output_nrmse_mean_matches_macro_point_estimate() -> None:
    """Mean of included per-output nRMSE must equal the macro point estimate."""
    context = build_manuscript_notebook_context(
        Path.cwd(), "08_manuscript_tables_and_figures.ipynb"
    )
    result = run_final_manuscript_artifacts_stage(context)
    per_out = result.final_artifacts.per_output_nrmse
    final_ols_summary = result.final_artifacts.final_ols_summary

    macro_point = float(final_ols_summary.loc[0, "final_ols_holdout_nrmse"])
    included_mean = float(per_out.loc[per_out["included_in_macro"], "nrmse"].mean())
    assert math.isfinite(macro_point)
    assert math.isfinite(included_mean)
    assert abs(included_mean - macro_point) < 1e-9, (
        f"per-output mean nRMSE ({included_mean:.6f}) != macro point estimate ({macro_point:.6f})"
    )


def test_per_output_nrmse_summary_schema() -> None:
    context = build_manuscript_notebook_context(
        Path.cwd(), "08_manuscript_tables_and_figures.ipynb"
    )
    result = run_final_manuscript_artifacts_stage(context)
    summary = result.final_artifacts.per_output_nrmse_summary

    assert len(summary) == 1
    for col in ("p10", "p25", "p50", "p75", "p90", "n_included", "n_total"):
        assert col in summary.columns, f"Missing column: {col}"
    assert summary.loc[0, "n_included"] >= 1
    assert math.isfinite(float(summary.loc[0, "p50"]))
