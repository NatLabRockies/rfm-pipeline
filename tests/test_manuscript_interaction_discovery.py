"""Tests for manuscript interaction-discovery stage."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bsm_rfm.manuscript_runtime import (
    build_manuscript_notebook_context,
    load_manuscript_case_study_config,
)
from bsm_rfm.manuscript_stages import (
    InteractionDiscoverySpec,
    discover_manuscript_interactions,
    interaction_discovery_spec_from_case_study_config,
    run_interaction_discovery_stage,
    write_interaction_discovery_artifacts,
)


def test_interaction_discovery_spec_matches_frozen_case_study_contract() -> None:
    config = load_manuscript_case_study_config(Path.cwd())
    spec = interaction_discovery_spec_from_case_study_config(config)

    assert spec.method == "tree_shap_interaction_values"
    assert spec.aggregation_rule == "max_over_components_of_mean_absolute_shap_interaction"
    assert spec.null_threshold_quantile == 0.995
    assert spec.retained_pairs_reference == 367
    assert spec.permutation_count_B == 200
    assert spec.random_seed == 123
    assert spec.implementation_method == "tree_shap_gradient_boosting"
    assert spec.implementation_status == "manuscript_aligned"
    assert spec.source_workflow_reference == "private_tree_shap_interaction_workflow"
    assert spec.source_workflow_equivalence_status == (
        "manuscript_aligned_via_shap_gradient_boosting"
    )


def test_interaction_discovery_spec_accepts_optional_runtime_overrides() -> None:
    config = {
        "case_study": {
            "interface": {"holdout_random_seed": 456},
            "empirical_null_screen": {"permutation_count_B": 11},
            "interaction_discovery": {
                "method": "tree_shap_interaction_values",
                "aggregation_rule": "max_over_components_of_mean_absolute_shap_interaction",
                "null_threshold_quantile": 0.9,
                "retained_pairs": 12,
                "permutation_count_B": 13,
                "n_tree_estimators": 17,
                "max_tree_depth": 2,
                "max_shap_samples": 41,
            },
        }
    }

    spec = interaction_discovery_spec_from_case_study_config(config)

    assert spec.permutation_count_B == 13
    assert spec.random_seed == 456
    assert spec.n_tree_estimators == 17
    assert spec.max_tree_depth == 2
    assert spec.max_shap_samples == 41


def test_interaction_discovery_retains_residual_pair_signal_and_writes_artifacts(
    tmp_path: Path,
) -> None:
    sample_ids = list(range(1, 41))
    x1 = [-1.0, -1.0, 1.0, 1.0] * 10
    x2 = [-1.0, 1.0, -1.0, 1.0] * 10
    pca_signal = [left * right for left, right in zip(x1, x2, strict=True)]
    inputs = pd.DataFrame({"sample_id": sample_ids, "x1": x1, "x2": x2})
    catalog = pd.DataFrame(
        {
            "feature_name": ["x1", "x2"],
            "feature_type": ["first_order", "first_order"],
            "origin": ["test"] * 2,
        }
    )
    holdout = pd.DataFrame({"sample_id": sample_ids, "split": ["train"] * 32 + ["holdout"] * 8})
    pca_scores = pd.DataFrame({"sample_id": sample_ids, "PC1": pca_signal})
    retained_terms = pd.DataFrame(
        {
            "feature_name": ["x1", "x2"],
            "feature_type": ["first_order", "first_order"],
        }
    )
    spec = InteractionDiscoverySpec(
        method="tree_shap_interaction_values",
        aggregation_rule="max_over_components_of_mean_absolute_shap_interaction",
        null_threshold_quantile=0.95,
        retained_pairs_reference=367,
        permutation_count_B=49,
        random_seed=123,
    )

    result = discover_manuscript_interactions(
        inputs,
        catalog,
        holdout,
        pca_scores,
        retained_terms,
        spec,
    )

    assert result.summary.loc[0, "stage"] == "interaction_discovery"
    assert result.summary.loc[0, "n_candidate_pairs"] == 1
    assert result.summary.loc[0, "n_retained_pairs"] == 1
    assert result.pair_scores.loc[0, "pair_name"] == "x1:x2"
    assert result.pair_scores.loc[0, "retained"]
    assert result.pair_scores.loc[0, "empirical_null_retained"]
    assert result.summary.loc[0, "public_implementation_method"] == "tree_shap_gradient_boosting"
    assert result.summary.loc[0, "source_workflow_equivalence_status"] == (
        "manuscript_aligned_via_shap_gradient_boosting"
    )
    assert result.provenance.loc[0, "manuscript_method"] == "tree_shap_interaction_values"
    assert result.provenance.loc[0, "public_implementation_method"] == "tree_shap_gradient_boosting"

    paths = write_interaction_discovery_artifacts(result, tmp_path)
    assert sorted(paths) == [
        "component_interaction_scores",
        "interaction_discovery_provenance",
        "interaction_discovery_summary",
        "interaction_null_summary",
        "interaction_pair_scores",
        "retained_interaction_pairs",
    ]
    assert paths["retained_interaction_pairs"].read_text(encoding="utf-8").startswith("pair_name")
    provenance_text = paths["interaction_discovery_provenance"].read_text(encoding="utf-8")
    assert "tree_shap_gradient_boosting" in provenance_text
    assert "manuscript_aligned_via_shap_gradient_boosting" in provenance_text


def test_interaction_discovery_generates_all_pairs_from_retained_first_order_terms() -> None:
    sample_ids = list(range(1, 41))
    x1 = [-1.0, -1.0, 1.0, 1.0] * 10
    x2 = [-1.0, 1.0, -1.0, 1.0] * 10
    x3 = [1.0 if value % 2 == 0 else -1.0 for value in sample_ids]
    pca_signal = [left * right for left, right in zip(x1, x2, strict=True)]
    inputs = pd.DataFrame({"sample_id": sample_ids, "x1": x1, "x2": x2, "x3": x3})
    catalog = pd.DataFrame(
        {
            "feature_name": ["x1", "x2", "x3"],
            "feature_type": ["first_order", "first_order", "first_order"],
        }
    )
    holdout = pd.DataFrame({"sample_id": sample_ids, "split": ["train"] * 32 + ["holdout"] * 8})
    pca_scores = pd.DataFrame({"sample_id": sample_ids, "PC1": pca_signal})
    retained_terms = pd.DataFrame(
        {
            "feature_name": ["x1", "x2", "x3"],
            "feature_type": ["first_order", "first_order", "first_order"],
        }
    )
    spec = InteractionDiscoverySpec(
        method="tree_shap_interaction_values",
        aggregation_rule="max_over_components_of_mean_absolute_shap_interaction",
        null_threshold_quantile=0.95,
        retained_pairs_reference=367,
        permutation_count_B=19,
        random_seed=123,
    )

    result = discover_manuscript_interactions(
        inputs,
        catalog,
        holdout,
        pca_scores,
        retained_terms,
        spec,
    )

    assert set(result.pair_scores["pair_name"]) == {"x1:x2", "x1:x3", "x2:x3"}
    assert result.summary.loc[0, "n_candidate_pairs"] == 3


def test_run_interaction_discovery_stage_executes_demo_context() -> None:
    context = build_manuscript_notebook_context(
        Path.cwd(),
        "04_interaction_discovery.ipynb",
    )

    result = run_interaction_discovery_stage(context)

    assert result.interactions.summary.loc[0, "stage"] == "interaction_discovery"
    assert result.interactions.summary.loc[0, "n_candidate_pairs"] == 1
    assert result.artifact_paths["interaction_pair_scores"].exists()
