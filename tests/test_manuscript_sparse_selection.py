"""Tests for manuscript sparse-selection and stability stage."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bsm_rfm.manuscript_runtime import (
    build_manuscript_notebook_context,
    load_manuscript_case_study_config,
)
from bsm_rfm.manuscript_stages import (
    SparseSelectionStabilitySpec,
    run_sparse_selection_stability_stage,
    select_manuscript_sparse_support,
    sparse_selection_stability_spec_from_case_study_config,
    write_sparse_selection_stability_artifacts,
)


def test_sparse_selection_spec_matches_frozen_case_study_contract() -> None:
    config = load_manuscript_case_study_config(Path.cwd())
    spec = sparse_selection_stability_spec_from_case_study_config(config)

    assert spec.model_class == "l1_penalized_linear_model_per_retained_component"
    assert spec.ebic_gamma == 0.5
    assert spec.support_aggregation_rule == "union_nonzero_support_across_retained_components"
    assert (
        spec.resampling_scheme == "100_subsamples_of_80_percent_rows_without_replacement_seed_123"
    )
    assert spec.subsample_count == 100
    assert spec.subsample_fraction == 0.80
    assert spec.jaccard_threshold == 0.75
    assert spec.spearman_threshold == 0.90
    assert spec.implementation_method == "ebic_l1_component_union_with_subsample_stability"
    assert spec.implementation_status == "source_backed_public_surrogate"
    assert spec.source_workflow_reference == "notebook_pca_debiased_lasso"
    assert spec.source_artifact == "LASSO_to_OLS_v9.ipynb"
    assert spec.source_workflow_equivalence_status == "not_yet_validated"
    assert spec.source_selected_feature_count_reference == 346
    assert spec.random_seed == 123


def test_sparse_selection_retains_stable_signal_feature_and_writes_artifacts(
    tmp_path: Path,
) -> None:
    sample_ids = list(range(1, 61))
    x1 = [float(index) for index in range(60)]
    x2 = [1.0 if index % 2 == 0 else -1.0 for index in range(60)]
    x3 = [float((index % 5) - 2) for index in range(60)]
    inputs = pd.DataFrame({"sample_id": sample_ids, "x1": x1, "x2": x2, "x3": x3})
    catalog = pd.DataFrame(
        {
            "feature_name": ["x1", "x2", "x3"],
            "feature_type": ["first_order", "first_order", "first_order"],
            "origin": ["test", "test", "test"],
        }
    )
    holdout = pd.DataFrame({"sample_id": sample_ids, "split": ["train"] * 48 + ["holdout"] * 12})
    pca_scores = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "PC1": [3.0 * value for value in x1],
            "PC2": [-1.5 * value for value in x1],
        }
    )
    retained_terms = pd.DataFrame({"feature_name": ["x1", "x2", "x3"]})
    retained_pairs = pd.DataFrame({"pair_name": []})
    retained_transformations = pd.DataFrame({"feature_name": []})
    spec = SparseSelectionStabilitySpec(
        model_class="l1_penalized_linear_model_per_retained_component",
        ebic_gamma=0.5,
        support_aggregation_rule="union_nonzero_support_across_retained_components",
        resampling_scheme="8_subsamples_of_80_percent_rows_without_replacement_seed_123",
        subsample_count=8,
        subsample_fraction=0.80,
        jaccard_threshold=0.50,
        spearman_threshold=0.50,
        random_seed=123,
    )

    result = select_manuscript_sparse_support(
        inputs,
        catalog,
        holdout,
        pca_scores,
        retained_terms,
        retained_pairs,
        retained_transformations,
        spec,
    )

    assert result.summary.loc[0, "stage"] == "sparse_selection_and_stability"
    assert result.summary.loc[0, "n_candidate_terms"] == 3
    assert "x1" in result.final_stable_support["feature_name"].tolist()
    assert result.component_model_selection["selected_support_size"].max() >= 1
    x1_row = result.stability_feature_summary.loc[
        result.stability_feature_summary["feature_name"] == "x1"
    ].iloc[0]
    assert bool(x1_row["full_support_selected"])
    assert x1_row["stability_selection_frequency"] >= 0.5

    paths = write_sparse_selection_stability_artifacts(result, tmp_path)
    assert sorted(paths) == [
        "component_coefficients",
        "component_model_selection",
        "final_stable_support",
        "sparse_selection_provenance",
        "sparse_selection_summary",
        "stability_feature_summary",
        "stability_resample_summary",
        "support_candidates",
    ]
    assert paths["final_stable_support"].read_text(encoding="utf-8").startswith("feature_name")
    provenance = pd.read_csv(paths["sparse_selection_provenance"])
    assert provenance.loc[0, "public_implementation_status"] == ("source_backed_public_surrogate")
    assert provenance.loc[0, "source_workflow_reference"] == "notebook_pca_debiased_lasso"
    assert provenance.loc[0, "source_workflow_equivalence_status"] == "not_yet_validated"


def test_run_sparse_selection_stability_stage_executes_demo_context() -> None:
    context = build_manuscript_notebook_context(
        Path.cwd(),
        "06_sparse_selection_and_stability.ipynb",
    )

    result = run_sparse_selection_stability_stage(context)

    assert result.sparse_selection.summary.loc[0, "stage"] == "sparse_selection_and_stability"
    assert result.sparse_selection.summary.loc[0, "n_candidate_terms"] >= 1
    assert result.sparse_selection.summary.loc[0, "n_final_stable_support_terms"] >= 1
    assert not result.sparse_selection.final_stable_support.empty
    assert result.artifact_paths["support_candidates"].exists()
