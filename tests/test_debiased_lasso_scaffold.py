"""Scaffold tests for deterministic de-biased-LASSO implementation.

These tests assert the artifact/schema-level contracts expected from the sparse-selection
and EBIC-path construction stages. They are intentionally conservative and document the
desired artifact schema for a future de-biased-LASSO implementation.
"""

from __future__ import annotations

import pandas as pd

from bsm_rfm.manuscript_stages import (
    SparseSelectionStabilitySpec,
    select_manuscript_sparse_support,
)


def _toy_sparse_inputs():
    sample_ids = list(range(1, 61))
    x1 = [float(i) for i in range(60)]
    x2 = [1.0 if i % 2 == 0 else -1.0 for i in range(60)]
    x3 = [float((i % 5) - 2) for i in range(60)]
    inputs = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "x1": x1,
            "x2": x2,
            "x3": x3,
        }
    )
    catalog = pd.DataFrame(
        {
            "feature_name": ["x1", "x2", "x3"],
            "feature_type": ["first_order"] * 3,
            "origin": ["test"] * 3,
        }
    )
    holdout = pd.DataFrame({"sample_id": sample_ids, "split": ["train"] * 48 + ["holdout"] * 12})
    pca_scores = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "PC1": [3.0 * v for v in x1],
            "PC2": [-1.5 * v for v in x1],
        }
    )
    retained_terms = pd.DataFrame({"feature_name": ["x1", "x2", "x3"]})
    retained_pairs = pd.DataFrame({"pair_name": []})
    retained_transformations = pd.DataFrame({"feature_name": []})

    return (
        inputs,
        catalog,
        holdout,
        pca_scores,
        retained_terms,
        retained_pairs,
        retained_transformations,
    )


def _default_spec():
    return SparseSelectionStabilitySpec(
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


def test_component_model_selection_columns():
    (
        inputs,
        catalog,
        holdout,
        pca_scores,
        retained_terms,
        retained_pairs,
        retained_transformations,
    ) = _toy_sparse_inputs()
    spec = _default_spec()

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

    cols = set(result.component_model_selection.columns)
    expected = {
        "component",
        "selected_alpha",
        "selected_ebic",
        "selected_rss",
        "selected_support_size",
        "n_rows",
        "n_candidate_terms",
        "ebic_gamma",
    }
    assert expected.issubset(cols)


def test_provenance_includes_expected_fields():
    (
        inputs,
        catalog,
        holdout,
        pca_scores,
        retained_terms,
        retained_pairs,
        retained_transformations,
    ) = _toy_sparse_inputs()
    spec = _default_spec()

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

    prov = result.provenance
    for col in (
        "public_implementation_status",
        "source_workflow_reference",
        "source_workflow_equivalence_status",
        "diagnostics",
    ):
        assert col in prov.columns


def test_component_model_selection_numeric_alpha():
    (
        inputs,
        catalog,
        holdout,
        pca_scores,
        retained_terms,
        retained_pairs,
        retained_transformations,
    ) = _toy_sparse_inputs()
    spec = _default_spec()

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
    vals = result.component_model_selection["selected_alpha"].to_numpy(dtype=float)
    assert (vals >= 0.0).all()
