"""R2-S01: Permutation-adequacy guard reconciliation tests.

Acceptance criteria:
- enforce_permutation_adequacy=True (default) raises PermutationAdequacyError for inadequate B.
- enforce_permutation_adequacy=False does not raise for the same inadequate B.
- Default value of the new field is True.
"""

import numpy as np
import pandas as pd
import pytest

from rfm_pipeline import manuscript_stages
from rfm_pipeline.manuscript_stages import (
    InteractionDiscoverySpec,
    PermutationAdequacyError,
    discover_manuscript_interactions,
)


def _make_spec(**kwargs) -> InteractionDiscoverySpec:
    defaults = dict(
        method="tree_shap_interaction_values",
        aggregation_rule="max_over_components_of_mean_absolute_shap_interaction",
        null_threshold_quantile=0.95,
        retained_pairs_reference=1,
        permutation_count_B=3,
        random_seed=42,
    )
    defaults.update(kwargs)
    return InteractionDiscoverySpec(**defaults)


def _make_two_feature_data():
    """Minimal two-feature dataset with non-constant PCA signal."""
    sample_ids = list(range(1, 41))
    x1 = [-1.0, -1.0, 1.0, 1.0] * 10
    x2 = [-1.0, 1.0, -1.0, 1.0] * 10
    pca_signal = [a * b for a, b in zip(x1, x2, strict=True)]
    inputs = pd.DataFrame({"sample_id": sample_ids, "x1": x1, "x2": x2})
    catalog = pd.DataFrame(
        {"feature_name": ["x1", "x2"], "feature_type": ["first_order", "first_order"]}
    )
    holdout = pd.DataFrame({"sample_id": sample_ids, "split": ["train"] * 32 + ["holdout"] * 8})
    pca_scores = pd.DataFrame({"sample_id": sample_ids, "PC1": pca_signal})
    retained_terms = pd.DataFrame(
        {"feature_name": ["x1", "x2"], "feature_type": ["first_order", "first_order"]}
    )
    return inputs, catalog, holdout, pca_scores, retained_terms


def test_enforce_permutation_adequacy_default_is_true() -> None:
    spec = _make_spec()
    assert spec.enforce_permutation_adequacy is True


def test_inadequate_B_raises_with_enforce_true() -> None:
    """With enforce=True (default), an inadequate B raises PermutationAdequacyError."""
    inputs, catalog, holdout, pca_scores, retained_terms = _make_two_feature_data()
    # B=3, quantile=0.95, family_size=1 → min_B=19; 3 < 19 → guard raises
    spec = _make_spec(
        null_threshold_quantile=0.95,
        permutation_count_B=3,
        enforce_permutation_adequacy=True,
    )
    with pytest.raises(PermutationAdequacyError):
        discover_manuscript_interactions(inputs, catalog, holdout, pca_scores, retained_terms, spec)


def test_inadequate_B_does_not_raise_with_enforce_false(monkeypatch) -> None:
    """With enforce=False, the same inadequate B does not raise from the guard."""
    inputs, catalog, holdout, pca_scores, retained_terms = _make_two_feature_data()

    def _fake_score(
        y_base: np.ndarray,
        permute_response: bool,
        *,
        n_pairs: int,
        n_comp: int,
        **_: object,
    ) -> tuple[np.ndarray, np.ndarray]:
        _ = (y_base, permute_response)
        return np.ones(n_pairs, dtype=float), np.ones((n_pairs, n_comp), dtype=float)

    monkeypatch.setattr(manuscript_stages, "_score_interaction_permutation", _fake_score)

    spec = _make_spec(
        null_threshold_quantile=0.95,
        permutation_count_B=3,
        enforce_permutation_adequacy=False,
    )
    result = discover_manuscript_interactions(
        inputs, catalog, holdout, pca_scores, retained_terms, spec
    )
    assert len(result.pair_scores) == 1
