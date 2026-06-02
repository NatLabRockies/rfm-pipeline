"""Tests for final manuscript table and figure artifact regeneration."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from rfm_pipeline.manuscript_runtime import (
    build_manuscript_notebook_context,
)
from rfm_pipeline.manuscript_stages import (
    FinalManuscriptArtifactsSpec,
    _ablation_main_effect_feature_names,
    _build_feature_pruning_diagnostics,
    _build_hc3_inferential_filter_tables,
    _build_legacy_feature_type_counts,
    _build_legacy_interaction_counts_by_module_pair,
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


def test_final_artifact_spec_accepts_optional_runtime_overrides() -> None:
    config = {
        "case_study": {
            "interface": {"holdout_random_seed": 456},
            "final_model": {
                "final_predictor_count": 10,
                "final_first_order_input_count": 4,
                "intermediate_penalized_holdout_nrmse": 0.1,
                "final_ols_holdout_nrmse": 0.2,
                "nrmse_denominator_definition": "macro",
                "nrmse_min_range": 1.0e-6,
                "nrmse_reference_matrix": "Y_train",
                "bootstrap_count": 12,
                "bootstrap_alpha": 0.1,
            },
            "final_inferential_filter": {
                "interval_method": "hc3_wald_95_percent_drop_if_zero_compatible_for_all_outputs",
                "alpha": 0.1,
                "output_subset_mode": "random_fraction",
                "output_fraction": 0.25,
                "output_names": ["y1", "y2"],
                "max_outputs": 4,
                "random_seed": 321,
                "subset_metric": "variance",
                "feature_pruning": {
                    "error_scale_quantile": 0.9,
                    "delta_threshold_override": 0.002,
                },
            },
        }
    }

    spec = final_manuscript_artifacts_spec_from_case_study_config(config)

    assert spec.bootstrap_count == 12
    assert spec.bootstrap_alpha == 0.1
    assert spec.inferential_filter_alpha == 0.1
    assert spec.hc3_output_subset_mode == "random_fraction"
    assert spec.hc3_output_fraction == pytest.approx(0.25)
    assert spec.hc3_output_names == ("y1", "y2")
    assert spec.hc3_output_max_outputs == 4
    assert spec.hc3_output_random_seed == 321
    assert spec.hc3_output_subset_metric == "variance"
    assert spec.pruning_error_scale_quantile == pytest.approx(0.9)
    assert spec.pruning_delta_threshold_override == pytest.approx(0.002)
    assert spec.pruning_remove_count_override is None
    assert spec.random_seed == 456


def test_hc3_filter_target_list_uses_only_requested_outputs() -> None:
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(123)
    x_train = pd.DataFrame(rng.normal(size=(40, 3)), columns=["x1", "x2", "x3"])
    y_train = pd.DataFrame(
        {
            "y1": rng.normal(size=40),
            "y2": 2.0 * x_train["x1"] + rng.normal(scale=0.05, size=40),
            "y3": rng.normal(size=40),
            "y4": -1.5 * x_train["x2"] + rng.normal(scale=0.05, size=40),
            "y5": rng.normal(size=40),
        }
    )

    intervals, summary = _build_hc3_inferential_filter_tables(
        x_train,
        y_train,
        alpha=0.05,
        interval_method="hc3_wald_95_percent_drop_if_zero_compatible_for_all_outputs",
        output_subset_mode="target_list",
        output_fraction=None,
        output_names=("y2", "y4"),
        max_outputs=None,
        random_seed=123,
        subset_metric="variance",
    )

    assert set(intervals["output_name"].unique()) == {"y2", "y4"}
    assert summary["n_outputs"].nunique() == 1
    assert int(summary["n_outputs"].iloc[0]) == 2


def test_hc3_filter_random_fraction_respects_cap() -> None:
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(456)
    x_train = pd.DataFrame(rng.normal(size=(30, 2)), columns=["x1", "x2"])
    y_train = pd.DataFrame(
        {
            "y0": 1.3 * x_train["x1"] + rng.normal(scale=0.08, size=30),
            "y1": -0.9 * x_train["x2"] + rng.normal(scale=0.08, size=30),
            "y2": 0.7 * x_train["x1"] + rng.normal(scale=0.08, size=30),
            "y3": rng.normal(size=30),
            "y4": rng.normal(size=30),
            "y5": rng.normal(size=30),
            "y6": rng.normal(size=30),
            "y7": rng.normal(size=30),
            "y8": rng.normal(size=30),
            "y9": rng.normal(size=30),
        }
    )

    intervals, summary = _build_hc3_inferential_filter_tables(
        x_train,
        y_train,
        alpha=0.05,
        interval_method="hc3_wald_95_percent_drop_if_zero_compatible_for_all_outputs",
        output_subset_mode="random_fraction",
        output_fraction=0.5,
        output_names=(),
        max_outputs=3,
        random_seed=789,
        subset_metric="variance",
    )

    assert len(intervals["output_name"].unique()) == 3
    assert int(summary["n_outputs"].iloc[0]) == 3


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
    assert (
        final_artifacts.final_ols_summary.loc[0, "n_hc3_features"]
        >= final_artifacts.final_ols_summary.loc[0, "n_final_features"]
    )
    assert final_artifacts.final_ols_summary.loc[0, "n_pruning_removed_features"] >= 0
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
        "null_mean_baseline_demo",
    }
    assert final_artifacts.figure_specs["figure_name"].tolist() == [
        "figure_model_performance",
        "figure_support_composition",
        "figure_selected_by_module_count",
        "figure_selected_by_module_share",
        "figure_nrmse_bootstrap_summary",
        "fig_feature_type_distribution",
        "fig_influential_by_module",
        "fig_module_pair_heatmap",
        "fig_module_total_interactions",
        "figure_feature_pruning_curve",
        "figure_per_output_nrmse_distribution",
    ]
    assert result.artifact_paths["prefilter_support_features"].exists()
    assert result.artifact_paths["hc3_wald_intervals"].exists()
    assert result.artifact_paths["hc3_inferential_filter_summary"].exists()
    assert result.artifact_paths["model_performance"].exists()
    assert result.artifact_paths["workflow_stage_summary"].exists()
    assert result.artifact_paths["figure_model_performance_svg"].suffix == ".svg"
    assert result.artifact_paths["figure_model_performance_svg"].exists()
    assert result.artifact_paths["figure_selected_by_module_count_svg"].exists()
    assert result.artifact_paths["figure_selected_by_module_share_svg"].exists()
    assert result.artifact_paths["figure_nrmse_bootstrap_summary_svg"].exists()
    assert result.artifact_paths["fig_feature_type_distribution_svg"].exists()
    assert result.artifact_paths["fig_influential_by_module_svg"].exists()
    assert result.artifact_paths["fig_module_pair_heatmap_svg"].exists()
    assert result.artifact_paths["fig_module_total_interactions_svg"].exists()
    assert result.artifact_paths["figure_per_output_nrmse_distribution_svg"].exists()
    assert result.artifact_paths["feature_type_counts"].exists()
    assert result.artifact_paths["influential_counts_by_module"].exists()
    assert result.artifact_paths["interaction_counts_by_module_pair"].exists()
    assert result.artifact_paths["interaction_density_module_matrix"].exists()
    assert result.artifact_paths["module_total_interactions"].exists()
    assert result.artifact_paths["ablation_table"].exists()
    assert result.artifact_paths["per_output_nrmse"].exists()
    assert result.artifact_paths["per_output_nrmse_summary"].exists()
    assert result.artifact_paths["feature_pruning_impact"].exists()
    assert result.artifact_paths["feature_pruning_summary"].exists()
    assert result.artifact_paths["figure_feature_pruning_curve_data"].exists()
    assert result.artifact_paths["figure_feature_pruning_curve_svg"].exists()
    assert int(final_artifacts.feature_pruning_summary.loc[0, "final_refit_n_features"]) == int(
        final_artifacts.final_ols_summary.loc[0, "n_final_features"]
    )


def test_feature_pruning_diagnostics_respect_remove_count_override() -> None:
    import numpy as np
    import pandas as pd

    from rfm_pipeline.final_ols import FinalOLSFitResult

    x_train = pd.DataFrame(
        {
            "f1": [0.1, 0.2, 0.3, 0.4, 0.5],
            "f2": [1.0, 1.1, 0.9, 1.2, 1.0],
        }
    )
    y_train = pd.DataFrame(
        {
            "y1": [0.15, 0.28, 0.44, 0.61, 0.83],
            "y2": [0.05, 0.10, 0.13, 0.18, 0.24],
        }
    )
    fit = FinalOLSFitResult(
        feature_names=("f1", "f2"),
        output_names=("y1", "y2"),
        coef_raw_scale=np.array([[2.0, 0.2], [0.4, 0.1]], dtype=float),
        intercept_raw_scale=np.array([0.0, 0.0], dtype=float),
        coef_standardized=np.zeros((2, 2), dtype=float),
        intercept_standardized=np.zeros(2, dtype=float),
        x_means=np.zeros(2, dtype=float),
        x_scales=np.ones(2, dtype=float),
        y_means=np.zeros(2, dtype=float),
        y_scales=np.ones(2, dtype=float),
        n_training_rows=5,
    )
    final_support_features = pd.DataFrame(
        {
            "feature_name": ["f1", "f2"],
            "feature_type": ["numeric", "numeric"],
            "origin": ["model_factors", "model_factors"],
        }
    )
    spec = FinalManuscriptArtifactsSpec(
        final_predictor_count_reference=2,
        final_first_order_input_count_reference=2,
        intermediate_penalized_holdout_nrmse_reference=0.1,
        final_ols_holdout_nrmse_reference=0.1,
        nrmse_denominator_definition="macro",
        nrmse_min_range=1.0e-6,
        nrmse_reference_matrix="Y_train",
        pruning_remove_count_override=1,
    )

    impact, curve, summary = _build_feature_pruning_diagnostics(
        final_fit=fit,
        x_train=x_train,
        y_train=y_train,
        final_support_features=final_support_features,
        spec=spec,
    )

    assert len(impact) == 2
    assert len(curve) == 2
    assert int(summary.loc[0, "effective_remove_count"]) == 1
    assert int(summary.loc[0, "effective_retained_features"]) == 1
    assert curve["selected_by_effective_cutoff"].sum() == 1


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


def test_legacy_feature_type_counts_classifies_colon_interactions() -> None:
    import pandas as pd

    final_support_features = pd.DataFrame(
        {
            "feature_name": [
                "AHC.foo:WW.bar",
                "sqrt_OHC.baz",
                "CHC.qux",
            ]
        }
    )
    counts = _build_legacy_feature_type_counts(final_support_features)
    observed = {str(row["feature_type"]): int(row["count"]) for _, row in counts.iterrows()}

    assert observed["Second Order"] == 1
    assert observed["Non-Linear"] == 1
    assert observed["First Order"] == 1


def test_legacy_interaction_counts_detects_colon_delimited_pairs() -> None:
    import pandas as pd

    final_support_features = pd.DataFrame(
        {
            "feature_name": [
                "AHC.foo:WW.bar",
                "AHC.foo:WW.baz",
                "CHC.qux",
            ]
        }
    )

    pair_counts_df, matrix = _build_legacy_interaction_counts_by_module_pair(final_support_features)

    assert not pair_counts_df.empty
    row = pair_counts_df.loc[
        (pair_counts_df["module_a"] == "Algal Hydrocarbons")
        & (pair_counts_df["module_b"] == "Wet Waste Hydrocarbons")
    ]
    assert not row.empty
    assert int(row.iloc[0]["count"]) == 2
    assert int(matrix.loc["Algal Hydrocarbons", "Wet Waste Hydrocarbons"]) == 2
    assert int(matrix.loc["Wet Waste Hydrocarbons", "Algal Hydrocarbons"]) == 2


def test_ablation_main_effect_feature_names_accepts_numeric_labels() -> None:
    import pandas as pd

    feature_catalog = pd.DataFrame(
        {
            "feature_name": ["x1", "x2", "x1:x2", "sqrt_x1"],
            "feature_type": ["numeric", "numeric", "interaction", "transformation"],
        }
    )
    input_matrix = pd.DataFrame({"sample_id": [1, 2], "x1": [0.1, 0.2], "x2": [0.3, 0.4]})

    names = _ablation_main_effect_feature_names(feature_catalog, input_matrix)

    assert names == ["x1", "x2"]
