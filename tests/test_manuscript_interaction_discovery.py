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
    assert spec.implementation_method == "residualized_product_permutation_surrogate"
    assert spec.implementation_status == "source_backed_public_surrogate"
    assert spec.source_workflow_reference == "private_tree_shap_interaction_workflow"
    assert spec.source_workflow_equivalence_status == "not_yet_validated"


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
            "feature_name": ["x1", "x2", "x1:x2"],
            "feature_type": ["first_order", "first_order", "interaction"],
            "origin": ["test"] * 3,
        }
    )
    holdout = pd.DataFrame({"sample_id": sample_ids, "split": ["train"] * 32 + ["holdout"] * 8})
    pca_scores = pd.DataFrame({"sample_id": sample_ids, "PC1": pca_signal})
    retained_terms = pd.DataFrame({"feature_name": ["x1:x2"]})
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
    assert (
        result.summary.loc[0, "public_implementation_method"]
        == "residualized_product_permutation_surrogate"
    )
    assert result.summary.loc[0, "source_workflow_equivalence_status"] == "not_yet_validated"
    assert result.provenance.loc[0, "manuscript_method"] == "tree_shap_interaction_values"
    assert (
        result.provenance.loc[0, "public_implementation_method"]
        == "residualized_product_permutation_surrogate"
    )

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
    assert "source_backed_public_surrogate" in provenance_text
    assert "not_yet_validated" in provenance_text


def test_run_interaction_discovery_stage_executes_demo_context() -> None:
    context = build_manuscript_notebook_context(
        Path.cwd(),
        "04_interaction_discovery.ipynb",
    )

    result = run_interaction_discovery_stage(context)

    assert result.interactions.summary.loc[0, "stage"] == "interaction_discovery"
    assert result.interactions.summary.loc[0, "n_candidate_pairs"] == 1
    assert result.artifact_paths["interaction_pair_scores"].exists()
