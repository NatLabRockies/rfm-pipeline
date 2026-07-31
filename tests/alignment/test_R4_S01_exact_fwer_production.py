"""R4-S01: Accept fwer_max_stat_exact in the production interaction stage.

Acceptance criteria:
- discover_manuscript_interactions runs without raising when
  family_error_method="fwer_max_stat_exact".
- Planted-interaction pair is retained under the exact method.
- Pure-null fixture retains 0 pairs at alpha=0.05 (fixed seed).
- Unknown method still raises ValueError.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import rfm_pipeline.manuscript_stages as manuscript_stages
from rfm_pipeline.manuscript_stages import (
    InteractionDiscoverySpec,
    discover_manuscript_interactions,
)


def _base_inputs(
    n: int = 40,
    *,
    plant_interaction: bool = False,
    seed: int = 42,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Small synthetic multi-output dataset.

    With ``plant_interaction=True`` the PCA score is exactly x1*x2 (pure
    interaction signal, no main effects).  With ``plant_interaction=False``
    the PCA score is pure noise from the RNG — no true interactions.
    """
    rng = np.random.default_rng(seed)
    sample_ids = list(range(1, n + 1))

    # Two binary features with balanced design + one noise feature
    x1 = np.tile([-1.0, -1.0, 1.0, 1.0], n // 4)
    x2 = np.tile([-1.0, 1.0, -1.0, 1.0], n // 4)
    x3 = rng.choice([-1.0, 1.0], size=n)

    if plant_interaction:
        pca_signal = x1 * x2
    else:
        pca_signal = rng.standard_normal(n)

    inputs = pd.DataFrame({"sample_id": sample_ids, "x1": x1, "x2": x2, "x3": x3})
    catalog = pd.DataFrame(
        {
            "feature_name": ["x1", "x2", "x3"],
            "feature_type": ["first_order"] * 3,
        }
    )
    holdout = pd.DataFrame(
        {"sample_id": sample_ids, "split": ["train"] * (n - n // 5) + ["holdout"] * (n // 5)}
    )
    pca_scores = pd.DataFrame({"sample_id": sample_ids, "PC1": pca_signal})
    retained_terms = pd.DataFrame(
        {
            "feature_name": ["x1", "x2", "x3"],
            "feature_type": ["first_order"] * 3,
        }
    )
    return inputs, catalog, holdout, pca_scores, retained_terms


def _exact_spec(*, permutation_count_B: int = 199, seed: int = 42) -> InteractionDiscoverySpec:
    return InteractionDiscoverySpec(
        method="tree_shap_interaction_values",
        aggregation_rule="max_over_components_of_mean_absolute_shap_interaction",
        null_threshold_quantile=0.95,
        retained_pairs_reference=1,
        permutation_count_B=permutation_count_B,
        random_seed=seed,
        n_jobs=1,
        family_error_method="fwer_max_stat_exact",
        family_error_alpha=0.05,
    )


def test_exact_fwer_production_does_not_raise(monkeypatch) -> None:
    """discover_manuscript_interactions accepts fwer_max_stat_exact without raising."""
    inputs, catalog, holdout, pca_scores, retained_terms = _base_inputs(plant_interaction=True)
    spec = _exact_spec(permutation_count_B=19)

    def _fake_scorer(
        y_base: np.ndarray,
        permute_response: bool,
        *,
        n_pairs: int,
        n_comp: int,
        **_: object,
    ) -> tuple[np.ndarray, np.ndarray]:
        _ = (y_base, permute_response)
        return np.ones(n_pairs, dtype=float), np.ones((n_pairs, n_comp), dtype=float)

    monkeypatch.setattr(manuscript_stages, "_score_interaction_permutation", _fake_scorer)

    result = discover_manuscript_interactions(
        inputs, catalog, holdout, pca_scores, retained_terms, spec
    )
    assert result is not None
    assert "pair_name" in result.pair_scores.columns


def test_exact_fwer_retains_planted_pair() -> None:
    """Planted x1:x2 interaction is retained by the exact maxT method."""
    inputs, catalog, holdout, pca_scores, retained_terms = _base_inputs(plant_interaction=True)
    spec = _exact_spec(permutation_count_B=499, seed=42)

    result = discover_manuscript_interactions(
        inputs, catalog, holdout, pca_scores, retained_terms, spec
    )
    retained = set(result.pair_scores.loc[result.pair_scores["retained"], "pair_name"])
    assert "x1:x2" in retained, f"Planted pair x1:x2 was not retained; retained={retained}"


def test_exact_fwer_null_retains_zero_pairs() -> None:
    """Under pure-null data, exact maxT retains 0 pairs at alpha=0.05."""
    inputs, catalog, holdout, pca_scores, retained_terms = _base_inputs(
        plant_interaction=False, seed=99
    )
    spec = _exact_spec(permutation_count_B=499, seed=99)

    result = discover_manuscript_interactions(
        inputs, catalog, holdout, pca_scores, retained_terms, spec
    )
    n_retained = int(result.summary.loc[0, "n_retained_pairs"])
    assert n_retained == 0, (
        f"Expected 0 retained pairs under null; got {n_retained}. "
        f"pair_scores:\n{result.pair_scores[['pair_name', 'retained']]}"
    )


def test_unknown_method_still_raises() -> None:
    """Unknown family_error_method still raises ValueError."""
    inputs, catalog, holdout, pca_scores, retained_terms = _base_inputs()
    spec = InteractionDiscoverySpec(
        method="tree_shap_interaction_values",
        aggregation_rule="max_over_components_of_mean_absolute_shap_interaction",
        null_threshold_quantile=0.95,
        retained_pairs_reference=1,
        permutation_count_B=19,
        random_seed=42,
        n_jobs=1,
        family_error_method="not_a_real_method",
        family_error_alpha=0.05,
        enforce_permutation_adequacy=False,
    )
    with pytest.raises(ValueError, match="Unknown family_error_method"):
        discover_manuscript_interactions(inputs, catalog, holdout, pca_scores, retained_terms, spec)
