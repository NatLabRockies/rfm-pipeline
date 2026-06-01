from __future__ import annotations

import pandas as pd

from rfm_pipeline.manuscript_stages import (
    SparseSelectionStabilitySpec,
    select_manuscript_sparse_support,
)


def test_sparse_selection_fallback_nonempty_when_global_stability_fails() -> None:
    # Build a small deterministic toy dataset similar to existing tests
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

    # Use impossible thresholds (>1.0) to force global stability to fail and trigger fallback
    spec = SparseSelectionStabilitySpec(
        model_class="l1_penalized_linear_model_per_retained_component",
        ebic_gamma=0.5,
        support_aggregation_rule="union_nonzero_support_across_retained_components",
        resampling_scheme="8_subsamples_of_80_percent_rows_without_replacement_seed_123",
        subsample_count=8,
        subsample_fraction=0.80,
        jaccard_threshold=0.99,
        spearman_threshold=0.99,
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

    # Guard: even when global stability fails, the stage must yield a non-empty final support
    assert not result.final_stable_support.empty, (
        "final_stable_support is empty when it must be non-empty"
    )
