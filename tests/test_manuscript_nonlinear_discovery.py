"""Tests for manuscript nonlinear-discovery stage."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import bsm_rfm.manuscript_stages as manuscript_stages
from bsm_rfm.manuscript_runtime import (
    build_manuscript_notebook_context,
    load_manuscript_case_study_config,
)
from bsm_rfm.manuscript_stages import (
    NonlinearDiscoverySpec,
    discover_manuscript_nonlinear_transformations,
    nonlinear_discovery_spec_from_case_study_config,
    run_nonlinear_discovery_stage,
    write_nonlinear_discovery_artifacts,
)


def test_nonlinear_discovery_spec_matches_frozen_case_study_contract() -> None:
    config = load_manuscript_case_study_config(Path.cwd())
    spec = nonlinear_discovery_spec_from_case_study_config(config)

    assert spec.method == "gam_plus_restricted_parametric_replacement"
    assert spec.curvature_rule == "edf_gt_1_and_smooth_pvalue_lt_0p01"
    assert spec.replacement_selection_rule == "minimum_training_rmse_against_gam_smooth"
    assert spec.identified_transformations_reference == 112
    assert spec.final_support_transformations_reference == 37
    assert spec.implementation_method == "gam_cubic_smoothing_spline"
    assert spec.implementation_status == "manuscript_aligned"
    assert spec.source_workflow_reference == "private_gam_nonlinear_discovery_workflow"
    assert spec.source_workflow_equivalence_status == (
        "manuscript_aligned_via_scipy_smoothing_spline"
    )


def test_nonlinear_discovery_retains_residual_quadratic_signal_and_writes_artifacts(
    tmp_path: Path,
) -> None:
    sample_ids = list(range(1, 41))
    x1 = [float(value) for value in range(-20, 20)]
    quadratic_signal = [value**2 for value in x1]
    inputs = pd.DataFrame({"sample_id": sample_ids, "x1": x1})
    catalog = pd.DataFrame(
        {
            "feature_name": ["x1"],
            "feature_type": ["first_order"],
            "origin": ["test"],
        }
    )
    holdout = pd.DataFrame({"sample_id": sample_ids, "split": ["train"] * 32 + ["holdout"] * 8})
    pca_scores = pd.DataFrame({"sample_id": sample_ids, "PC1": quadratic_signal})
    retained_terms = pd.DataFrame(
        {
            "feature_name": ["x1"],
            "feature_type": ["first_order"],
        }
    )
    spec = NonlinearDiscoverySpec(
        method="gam_plus_restricted_parametric_replacement",
        curvature_rule="edf_gt_1_and_smooth_pvalue_lt_0p01",
        replacement_selection_rule="minimum_training_rmse_against_gam_smooth",
        identified_transformations_reference=112,
        final_support_transformations_reference=37,
    )

    result = discover_manuscript_nonlinear_transformations(
        inputs,
        catalog,
        holdout,
        pca_scores,
        retained_terms,
        spec,
    )

    assert result.summary.loc[0, "stage"] == "nonlinear_discovery"
    assert result.summary.loc[0, "public_implementation_method"] == ("gam_cubic_smoothing_spline")
    assert result.summary.loc[0, "source_workflow_equivalence_status"] == (
        "manuscript_aligned_via_scipy_smoothing_spline"
    )
    assert result.summary.loc[0, "n_candidate_transformations"] == 1
    assert result.summary.loc[0, "n_retained_transformations"] == 1
    assert result.provenance.loc[0, "public_implementation_status"] == ("manuscript_aligned")
    assert result.transformation_scores.loc[0, "feature_name"] == "x1_squared"
    assert result.transformation_scores.loc[0, "transformation_family"] == "quadratic"
    assert result.transformation_scores.loc[0, "retained"]
    assert result.transformation_scores.loc[0, "empirical_null_retained"]

    paths = write_nonlinear_discovery_artifacts(result, tmp_path)
    assert sorted(paths) == [
        "component_transformation_scores",
        "nonlinear_discovery_provenance",
        "nonlinear_discovery_summary",
        "retained_transformations",
        "transformation_scores",
    ]
    assert paths["retained_transformations"].read_text(encoding="utf-8").startswith("feature_name")
    provenance_text = paths["nonlinear_discovery_provenance"].read_text(encoding="utf-8")
    assert "gam_cubic_smoothing_spline" in provenance_text
    assert "manuscript_aligned" in provenance_text


def test_nonlinear_discovery_generates_supported_transforms_from_retained_terms() -> None:
    sample_ids = list(range(1, 41))
    x1 = [float(value) for value in range(1, 41)]
    pca_signal = [value**2 for value in x1]
    inputs = pd.DataFrame({"sample_id": sample_ids, "x1": x1})
    catalog = pd.DataFrame({"feature_name": ["x1"], "feature_type": ["first_order"]})
    holdout = pd.DataFrame({"sample_id": sample_ids, "split": ["train"] * 32 + ["holdout"] * 8})
    pca_scores = pd.DataFrame({"sample_id": sample_ids, "PC1": pca_signal})
    retained_terms = pd.DataFrame(
        {
            "feature_name": ["x1"],
            "feature_type": ["first_order"],
        }
    )
    spec = NonlinearDiscoverySpec(
        method="gam_plus_restricted_parametric_replacement",
        curvature_rule="edf_gt_1_and_smooth_pvalue_lt_0p01",
        replacement_selection_rule="minimum_training_rmse_against_gam_smooth",
        identified_transformations_reference=112,
        final_support_transformations_reference=37,
    )

    result = discover_manuscript_nonlinear_transformations(
        inputs,
        catalog,
        holdout,
        pca_scores,
        retained_terms,
        spec,
    )

    assert set(result.transformation_scores["feature_name"]) == {
        "inverse_x1",
        "log1p_x1",
        "sqrt_x1",
        "x1_squared",
    }
    assert result.summary.loc[0, "n_candidate_transformations"] == 4


def test_run_nonlinear_discovery_stage_executes_demo_context() -> None:
    context = build_manuscript_notebook_context(
        Path.cwd(),
        "05_nonlinear_discovery.ipynb",
    )

    result = run_nonlinear_discovery_stage(context)

    assert result.nonlinear.summary.loc[0, "stage"] == "nonlinear_discovery"
    assert result.nonlinear.summary.loc[0, "n_candidate_transformations"] == 6
    assert result.artifact_paths["transformation_scores"].exists()
    assert result.artifact_paths["nonlinear_discovery_provenance"].exists()


def test_nonlinear_discovery_reuses_checkpointed_feature_scores(
    monkeypatch,
    tmp_path: Path,
) -> None:
    sample_ids = list(range(1, 41))
    x1 = [float(value) for value in range(1, 41)]
    pca_signal = [value**2 for value in x1]
    inputs = pd.DataFrame({"sample_id": sample_ids, "x1": x1})
    catalog = pd.DataFrame({"feature_name": ["x1"], "feature_type": ["first_order"]})
    holdout = pd.DataFrame({"sample_id": sample_ids, "split": ["train"] * 32 + ["holdout"] * 8})
    pca_scores = pd.DataFrame({"sample_id": sample_ids, "PC1": pca_signal})
    retained_terms = pd.DataFrame(
        {
            "feature_name": ["x1"],
            "feature_type": ["first_order"],
        }
    )
    spec = NonlinearDiscoverySpec(
        method="gam_plus_restricted_parametric_replacement",
        curvature_rule="edf_gt_1_and_smooth_pvalue_lt_0p01",
        replacement_selection_rule="minimum_training_rmse_against_gam_smooth",
        identified_transformations_reference=112,
        final_support_transformations_reference=37,
        n_jobs=1,
    )
    checkpoint_root = tmp_path / "nonlinear_discovery"

    score_counter = {"count": 0}

    def _fake_score_one_nonlinear_feature(
        base_feat: str,
        base_candidates: list[tuple[str, str, str]],
        x_vals,  # noqa: ANN001
        y_scaled,  # noqa: ANN001
        component_names: list[str],
        active_comp_indices: list[int],
        edf_threshold: float,
        gam_p_threshold: float,
    ) -> tuple[str, dict, list[tuple[tuple[str, str], tuple[float, float]]]]:
        _ = (x_vals, y_scaled, edf_threshold, gam_p_threshold)
        score_counter["count"] += 1
        best_component = component_names[active_comp_indices[0]]
        feature_result = {
            "nonlinear": True,
            "best_transform_name": base_candidates[0][0],
            "best_edf": 3.0,
            "best_p": 1.0e-4,
            "best_comp_name": best_component,
            "best_rmse": 0.01,
        }
        cache_entries = [((base_feat, best_component), (3.0, 1.0e-4))]
        return base_feat, feature_result, cache_entries

    monkeypatch.setattr(
        manuscript_stages,
        "_score_one_nonlinear_feature",
        _fake_score_one_nonlinear_feature,
    )
    first = discover_manuscript_nonlinear_transformations(
        inputs,
        catalog,
        holdout,
        pca_scores,
        retained_terms,
        spec,
        checkpoint_dir=checkpoint_root,
    )
    assert score_counter["count"] > 0

    def _should_not_recompute(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("checkpointed nonlinear base-feature scores should be reused")

    monkeypatch.setattr(
        manuscript_stages,
        "_score_one_nonlinear_feature",
        _should_not_recompute,
    )
    second = discover_manuscript_nonlinear_transformations(
        inputs,
        catalog,
        holdout,
        pca_scores,
        retained_terms,
        spec,
        checkpoint_dir=checkpoint_root,
    )
    assert first.summary.equals(second.summary)
