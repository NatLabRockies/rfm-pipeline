"""Tests for manuscript nonlinear-discovery stage."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

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
    assert spec.implementation_method == "residualized_parametric_transform_surrogate"
    assert spec.implementation_status == "source_backed_public_surrogate"
    assert spec.source_workflow_reference == "private_gam_nonlinear_discovery_workflow"
    assert spec.source_workflow_equivalence_status == "not_yet_validated"


def test_nonlinear_discovery_retains_residual_quadratic_signal_and_writes_artifacts(
    tmp_path: Path,
) -> None:
    sample_ids = list(range(1, 41))
    x1 = [float(value) for value in range(-20, 20)]
    quadratic_signal = [value**2 for value in x1]
    inputs = pd.DataFrame({"sample_id": sample_ids, "x1": x1})
    catalog = pd.DataFrame(
        {
            "feature_name": ["x1", "x1_squared"],
            "feature_type": ["first_order", "transformation"],
            "origin": ["test", "test"],
        }
    )
    holdout = pd.DataFrame({"sample_id": sample_ids, "split": ["train"] * 32 + ["holdout"] * 8})
    pca_scores = pd.DataFrame({"sample_id": sample_ids, "PC1": quadratic_signal})
    retained_terms = pd.DataFrame({"feature_name": ["x1_squared"]})
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
    assert result.summary.loc[0, "public_implementation_method"] == (
        "residualized_parametric_transform_surrogate"
    )
    assert result.summary.loc[0, "source_workflow_equivalence_status"] == "not_yet_validated"
    assert result.summary.loc[0, "n_candidate_transformations"] == 1
    assert result.summary.loc[0, "n_retained_transformations"] == 1
    assert result.provenance.loc[0, "public_implementation_status"] == (
        "source_backed_public_surrogate"
    )
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
    assert "residualized_parametric_transform_surrogate" in provenance_text
    assert "not_yet_validated" in provenance_text


def test_run_nonlinear_discovery_stage_executes_demo_context() -> None:
    context = build_manuscript_notebook_context(
        Path.cwd(),
        "05_nonlinear_discovery.ipynb",
    )

    result = run_nonlinear_discovery_stage(context)

    assert result.nonlinear.summary.loc[0, "stage"] == "nonlinear_discovery"
    assert result.nonlinear.summary.loc[0, "n_candidate_transformations"] == 2
    assert result.artifact_paths["transformation_scores"].exists()
    assert result.artifact_paths["nonlinear_discovery_provenance"].exists()
