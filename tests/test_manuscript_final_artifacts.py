"""Tests for final manuscript table and figure artifact regeneration."""

from __future__ import annotations

from pathlib import Path

from bsm_rfm.manuscript_runtime import (
    build_manuscript_notebook_context,
    load_manuscript_case_study_config,
)
from bsm_rfm.manuscript_stages import (
    final_manuscript_artifacts_spec_from_case_study_config,
    run_final_manuscript_artifacts_stage,
)


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
