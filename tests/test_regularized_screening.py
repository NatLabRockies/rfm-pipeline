"""Tests for recovered regularized-screening workflow contracts."""

from __future__ import annotations

from bsm_rfm.regularized_screening import (
    archived_multitask_elastic_net_spec,
    divergence_table,
    notebook_regularized_screening_spec,
    recovered_regularized_screening_comparison,
    screening_spec_table,
)


def test_archived_script_screening_spec_matches_recovered_audit_values() -> None:
    spec = archived_multitask_elastic_net_spec()

    assert spec.provenance == "archived_script"
    assert spec.model_family == "MultiTaskElasticNetCV"
    assert spec.holdout_fraction == 0.05
    assert spec.candidate_input_count == 352
    assert spec.full_output_count == 23495
    assert spec.tuning_subset_file == "tune_vars.csv"
    assert spec.selected_feature_count is None


def test_notebook_screening_spec_matches_recovered_audit_values() -> None:
    spec = notebook_regularized_screening_spec()

    assert spec.provenance == "notebook_derived"
    assert spec.holdout_fraction == 0.10
    assert spec.output_count_after_culling == 9782
    assert spec.selected_feature_count == 346
    assert spec.l1_ratio_grid == (1.0, 0.95, 0.9)
    assert spec.alpha_fraction_grid == (0.75, 0.5, 0.25, 0.10, 0.05, 0.02, 0.01)
    assert spec.selected_l1_ratio == 1.0
    assert spec.selected_alpha_fraction == 0.10
    assert spec.selected_alpha_absolute == 2.8541
    assert spec.model_selection_criterion == "EBIC"


def test_recovered_screening_comparison_exposes_material_divergences() -> None:
    comparison = recovered_regularized_screening_comparison()
    divergences = comparison.divergence_summary()

    assert divergences["model_family"] == (
        "MultiTaskElasticNetCV",
        "PCA plus de-biased LASSO with final OLS handoff",
    )
    assert divergences["holdout_fraction"] == (0.05, 0.10)
    assert divergences["tuning_subset_file"] == ("tune_vars.csv", None)
    assert divergences["model_selection_criterion"] == (None, "EBIC")


def test_screening_tables_include_both_recovered_variants() -> None:
    specs = screening_spec_table()
    divergences = divergence_table()

    assert specs.shape[0] == 2
    assert set(specs["workflow_name"]) == {
        "archived multitask elastic-net script",
        "notebook PCA plus debiased-lasso screening",
    }
    assert set(divergences["field"]) == {
        "model_family",
        "response_representation",
        "holdout_fraction",
        "tuning_subset_file",
        "model_selection_criterion",
    }
