"""R3-S05: ElasticNet interaction path must not fabricate empirical provenance.

Tests:
- `p_value` column in pair_scores is NaN (not 0.0 or 1.0) for all pairs.
- `null_threshold` column in pair_scores is NaN (not 0.0).
- `null_mean` and `null_std` in interaction_null_summary are NaN (not 0.0).
- The function still reports its selection (retained pairs / selection flags present).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from rfm_pipeline.manuscript_stages import (
    InteractionDiscoverySpec,
    discover_manuscript_interactions,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def elasticnet_interaction_result():
    """Run _discover_elasticnet_interactions via the public dispatcher.

    Synthetic data: 3 features, 2 PCA components, 20 training rows.
    Two feature pairs are candidates; planted signal drives one selected.
    """
    rng = np.random.default_rng(42)
    n = 24

    sample_ids = [f"s{i:03d}" for i in range(n)]

    # Input matrix with 3 numeric features + sample_id
    feat_a = rng.standard_normal(n)
    feat_b = rng.standard_normal(n)
    feat_c = rng.standard_normal(n)
    input_matrix = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "feat_a": feat_a,
            "feat_b": feat_b,
            "feat_c": feat_c,
        }
    )

    # All train
    holdout_assignments = pd.DataFrame({"sample_id": sample_ids, "split": ["train"] * n})

    # Two PCA components; comp1 has planted interaction signal
    comp1 = feat_a * feat_b + rng.standard_normal(n) * 0.1
    comp2 = rng.standard_normal(n)
    pca_scores = pd.DataFrame({"sample_id": sample_ids, "comp1": comp1, "comp2": comp2})

    # Retained terms — all three features pass screening
    retained_terms = pd.DataFrame({"feature_name": ["feat_a", "feat_b", "feat_c"]})

    spec = InteractionDiscoverySpec(
        method="elasticnet_interactions",
        aggregation_rule="max_abs",
        null_threshold_quantile=0.95,
        retained_pairs_reference=1,
        permutation_count_B=10,
        random_seed=0,
        elasticnet_l1_ratio=0.5,
        elasticnet_cv_folds=3,
        n_jobs=1,
    )

    return discover_manuscript_interactions(
        input_matrix=input_matrix,
        feature_catalog=pd.DataFrame({"feature_name": ["feat_a", "feat_b", "feat_c"]}),
        holdout_assignments=holdout_assignments,
        pca_scores=pca_scores,
        retained_terms=retained_terms,
        spec=spec,
    )


# ---------------------------------------------------------------------------
# R3-S05 tests
# ---------------------------------------------------------------------------


class TestR3S05ElasticNetProvenance:
    def test_R3_S05_p_value_is_nan_not_zero_or_one(self, elasticnet_interaction_result):
        """p_value column must be NaN for all pairs — not fabricated 0.0 or 1.0."""
        pair_scores = elasticnet_interaction_result.pair_scores
        assert "p_value" in pair_scores.columns, "pair_scores must have a p_value column"
        for val in pair_scores["p_value"]:
            assert val is None or math.isnan(float(val)), (
                f"ElasticNet p_value must be NaN, got {val!r}"
            )

    def test_R3_S05_null_threshold_is_nan_not_zero(self, elasticnet_interaction_result):
        """null_threshold column must be NaN for all pairs — not fabricated 0.0."""
        pair_scores = elasticnet_interaction_result.pair_scores
        assert "null_threshold" in pair_scores.columns
        for val in pair_scores["null_threshold"]:
            assert val is None or math.isnan(float(val)), (
                f"ElasticNet null_threshold must be NaN, got {val!r}"
            )

    def test_R3_S05_null_summary_mean_is_nan(self, elasticnet_interaction_result):
        """null_mean in interaction_null_summary must be NaN — not fabricated 0.0."""
        null_summary = elasticnet_interaction_result.interaction_null_summary
        assert "null_mean" in null_summary.columns
        for val in null_summary["null_mean"]:
            assert val is None or math.isnan(float(val)), (
                f"ElasticNet null_mean must be NaN, got {val!r}"
            )

    def test_R3_S05_null_summary_std_is_nan(self, elasticnet_interaction_result):
        """null_std in interaction_null_summary must be NaN — not fabricated 0.0."""
        null_summary = elasticnet_interaction_result.interaction_null_summary
        assert "null_std" in null_summary.columns
        for val in null_summary["null_std"]:
            assert val is None or math.isnan(float(val)), (
                f"ElasticNet null_std must be NaN, got {val!r}"
            )

    def test_R3_S05_selection_is_still_reported(self, elasticnet_interaction_result):
        """Even with NaN provenance, the result must still carry selection columns."""
        pair_scores = elasticnet_interaction_result.pair_scores
        assert "retained" in pair_scores.columns, "pair_scores must have a retained column"
        assert "empirical_null_retained" in pair_scores.columns
        assert len(pair_scores) > 0, "At least one candidate pair must be present"

    def test_R3_S05_retained_pairs_present(self, elasticnet_interaction_result):
        """retained_pairs table must exist (may be empty but must not be absent)."""
        retained = elasticnet_interaction_result.retained_pairs
        assert isinstance(retained, pd.DataFrame)
