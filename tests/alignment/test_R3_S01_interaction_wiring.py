"""R3-S01: Wire multiplicity-controlled interaction selection into the production path.

The F5 defect: ``discover_manuscript_interactions`` used to retain pairs via an
uncorrected per-pair 0.5% cut (``observed > np.quantile(null, 0.995, axis=0)``).
This slice routes production retention through
``multiplicity_controlled_interaction_selection`` and makes the permutation-
adequacy guard consistent with the corrected family-wise / FDR procedure.

These tests exercise the PRODUCTION entry point
:func:`rfm_pipeline.manuscript_stages.discover_manuscript_interactions` on a
controlled synthetic fixture (pure null + planted-signal variants, fixed seed)
and assert:

- family-wise error control: on the same statistics, the corrected FWER rule
  yields strictly fewer selections than the uncorrected per-pair 0.5% rule
  (surfaced via the ``empirical_null_retained`` diagnostic column).
- planted-signal recovery: a strongly-planted interaction is retained.
- guard consistency: the permutation-adequacy budget is derived from
  ``family_error_alpha`` / ``family_error_method``, not from the removed
  per-pair quantile rule.

Grep guard: ``multiplicity_controlled_interaction_selection`` must be
referenced in the production module, not only in tests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import rfm_pipeline.manuscript_stages as manuscript_stages
from rfm_pipeline.manuscript_stages import (
    InteractionDiscoverySpec,
    PermutationAdequacyError,
    discover_manuscript_interactions,
)

# ---------------------------------------------------------------------------
# Fixture builders (fixed seed; synthetic observed + null pair statistics)
# ---------------------------------------------------------------------------

_N_FEATURES = 8  # -> C(8, 2) = 28 candidate pairs
_N_SAMPLES = 40
_N_TRAIN = 32
_B_PERMS = 400  # >> min_B for both fwer(family=1) and bh(family=28)
_SEED = 20260718


def _build_case_study_tables() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """Return (inputs, catalog, holdout, pca_scores, retained_terms) for 8 features."""
    sample_ids = list(range(1, _N_SAMPLES + 1))
    rng = np.random.default_rng(_SEED)
    feature_names = [f"x{i}" for i in range(_N_FEATURES)]
    x_data = rng.standard_normal((_N_SAMPLES, _N_FEATURES))
    inputs = pd.DataFrame({"sample_id": sample_ids})
    for i, name in enumerate(feature_names):
        inputs[name] = x_data[:, i]
    catalog = pd.DataFrame(
        {
            "feature_name": feature_names,
            "feature_type": ["first_order"] * _N_FEATURES,
            "origin": ["test"] * _N_FEATURES,
        }
    )
    holdout = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "split": ["train"] * _N_TRAIN + ["holdout"] * (_N_SAMPLES - _N_TRAIN),
        }
    )
    pca_scores = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "PC1": rng.standard_normal(_N_SAMPLES),
        }
    )
    retained_terms = pd.DataFrame(
        {
            "feature_name": feature_names,
            "feature_type": ["first_order"] * _N_FEATURES,
        }
    )
    return inputs, catalog, holdout, pca_scores, retained_terms


def _install_synthetic_scorer(
    monkeypatch: pytest.MonkeyPatch,
    *,
    observed_scores: np.ndarray,
    null_statistics: np.ndarray,
) -> None:
    """Monkeypatch _score_interaction_permutation to serve pre-baked scores.

    The controller calls the scorer B+1 times: first with ``permute_response=False``
    (observed), then B times with ``permute_response=True`` (null draws).
    """
    call_counter = {"null_index": 0}

    def _fake_scorer(
        y_base: np.ndarray,
        permute_response: bool,
        *,
        n_pairs: int,
        n_comp: int,
        **_: object,
    ) -> tuple[np.ndarray, np.ndarray]:
        _ = y_base
        assert n_pairs == observed_scores.shape[0]
        if not permute_response:
            scores = observed_scores.astype(float, copy=True)
        else:
            idx = call_counter["null_index"]
            assert idx < null_statistics.shape[0]
            scores = null_statistics[idx].astype(float, copy=True)
            call_counter["null_index"] = idx + 1
        component_scores = np.tile(scores[:, None], (1, n_comp))
        return scores, component_scores

    monkeypatch.setattr(
        manuscript_stages,
        "_score_interaction_permutation",
        _fake_scorer,
    )


def _make_null_and_signal_stats(
    *,
    n_pairs: int,
    B: int,
    signal_pair_index: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (observed_scores, null_statistics) with |N(0, 1)| baseline.

    If ``signal_pair_index`` is not None, the observed score for that pair is set
    well above the null max so that both FWER max-stat and BH-FDR retain it.
    """
    rng = np.random.default_rng(seed)
    null_statistics = np.abs(rng.standard_normal((B, n_pairs)))
    observed_scores = np.abs(rng.standard_normal(n_pairs))
    if signal_pair_index is not None:
        observed_scores[signal_pair_index] = float(null_statistics.max()) + 5.0
    return observed_scores, null_statistics


# ---------------------------------------------------------------------------
# R3-S01 grep guard: the production module must reference the corrected
# selection helper (not only tests).
# ---------------------------------------------------------------------------


def test_R3_S01_production_module_references_multiplicity_helper() -> None:
    module_path = Path(manuscript_stages.__file__)
    source = module_path.read_text(encoding="utf-8")
    # Definition + at least one call site inside the production discovery function.
    assert source.count("multiplicity_controlled_interaction_selection") >= 2, (
        "discover_manuscript_interactions must invoke "
        "multiplicity_controlled_interaction_selection; found only the definition."
    )


# ---------------------------------------------------------------------------
# R3-S01 core assertions: pure-null family control and planted-signal recovery
# through the PRODUCTION discover_manuscript_interactions entry point.
# ---------------------------------------------------------------------------


def _run_production_discovery(
    monkeypatch: pytest.MonkeyPatch,
    *,
    observed_scores: np.ndarray,
    null_statistics: np.ndarray,
    family_error_method: str,
    family_error_alpha: float,
    null_threshold_quantile: float = 0.995,
    permutation_count_B: int = _B_PERMS,
    enforce_permutation_adequacy: bool = True,
) -> pd.DataFrame:
    inputs, catalog, holdout, pca_scores, retained_terms = _build_case_study_tables()
    _install_synthetic_scorer(
        monkeypatch,
        observed_scores=observed_scores,
        null_statistics=null_statistics,
    )
    spec = InteractionDiscoverySpec(
        method="tree_shap_interaction_values",
        aggregation_rule="max_over_components_of_mean_absolute_shap_interaction",
        null_threshold_quantile=null_threshold_quantile,
        retained_pairs_reference=0,
        permutation_count_B=permutation_count_B,
        random_seed=_SEED,
        n_jobs=1,
        parallel_backend="threading",
        enforce_permutation_adequacy=enforce_permutation_adequacy,
        family_error_method=family_error_method,
        family_error_alpha=family_error_alpha,
    )
    result = discover_manuscript_interactions(
        inputs, catalog, holdout, pca_scores, retained_terms, spec
    )
    return result.pair_scores


def test_R3_S01_pure_null_fwer_controls_family_error_vs_per_pair() -> None:
    """On a pure null, corrected FWER retains strictly fewer pairs than the
    uncorrected per-pair 0.5% cut applied to the same statistics."""
    n_pairs = _N_FEATURES * (_N_FEATURES - 1) // 2  # 28
    # Craft a null where per-pair 0.995 rule flags multiple pairs deterministically:
    # set 6 observed scores above the per-pair 99.5-percentile of |N(0,1)| null.
    obs, null = _make_null_and_signal_stats(
        n_pairs=n_pairs, B=_B_PERMS, signal_pair_index=None, seed=_SEED
    )
    per_pair_quantiles = np.quantile(null, 0.995, axis=0)
    flag_columns = [3, 7, 11, 15, 19, 23]  # 6 pairs
    for col in flag_columns:
        obs[col] = per_pair_quantiles[col] + 0.05  # just above per-pair cut

    with pytest.MonkeyPatch.context() as mp:
        scores = _run_production_discovery(
            mp,
            observed_scores=obs,
            null_statistics=null,
            family_error_method="fwer_max_stat",
            family_error_alpha=0.05,
        )

    n_corrected = int(scores["retained"].sum())
    n_per_pair = int(scores["empirical_null_retained"].sum())
    # Per-pair rule retains all crafted flags (>= 6).
    assert n_per_pair >= len(flag_columns), (
        f"Per-pair diagnostic should flag crafted pairs; got {n_per_pair}."
    )
    # Corrected FWER at alpha=0.05 should retain far fewer under this pure-null
    # (max-stat threshold >> any single per-pair 99.5-percentile).
    assert n_corrected < n_per_pair, (
        f"FWER-corrected retention ({n_corrected}) must be strictly less than "
        f"uncorrected per-pair retention ({n_per_pair}) on the same statistics."
    )
    # No observed score exceeds the max-null quantile at alpha=0.05 by construction,
    # so under the pure null the corrected rule retains zero pairs.
    assert n_corrected == 0, (
        f"Under a pure null the corrected FWER rule should retain 0 pairs; got {n_corrected}."
    )


def test_R3_S01_planted_signal_is_retained_under_corrected_rule() -> None:
    """A strongly-planted interaction pair is retained by the corrected rule."""
    n_pairs = _N_FEATURES * (_N_FEATURES - 1) // 2
    planted_col = 5
    obs, null = _make_null_and_signal_stats(
        n_pairs=n_pairs,
        B=_B_PERMS,
        signal_pair_index=planted_col,
        seed=_SEED + 1,
    )

    with pytest.MonkeyPatch.context() as mp:
        scores = _run_production_discovery(
            mp,
            observed_scores=obs,
            null_statistics=null,
            family_error_method="fwer_max_stat",
            family_error_alpha=0.05,
        )

    # Identify the pair by its ordinal position in the candidate list. The
    # production code sorts candidates via itertools.combinations over the input
    # feature order (x0, x1, ..., x7). Cross-check by verifying the planted pair
    # is the unique retained pair.
    retained_rows = scores.loc[scores["retained"]]
    assert len(retained_rows) == 1, (
        f"Expected exactly the planted pair to be retained; got {len(retained_rows)}."
    )
    # The planted pair has the highest interaction_score by construction.
    top_row = scores.sort_values("interaction_score", ascending=False).iloc[0]
    assert bool(top_row["retained"]), "Highest-scoring pair (planted) must be retained."


def test_R3_S01_planted_signal_is_retained_under_bh_fdr_rule() -> None:
    """Guard-ON integration test: BH-FDR retention path with production wiring."""
    n_pairs = _N_FEATURES * (_N_FEATURES - 1) // 2
    planted_col = 2
    obs, null = _make_null_and_signal_stats(
        n_pairs=n_pairs,
        B=600,
        signal_pair_index=planted_col,
        seed=_SEED + 2,
    )

    with pytest.MonkeyPatch.context() as mp:
        scores = _run_production_discovery(
            mp,
            observed_scores=obs,
            null_statistics=null,
            family_error_method="bh_fdr",
            family_error_alpha=0.05,
            permutation_count_B=600,
        )

    top_row = scores.sort_values("interaction_score", ascending=False).iloc[0]
    assert bool(top_row["retained"]), (
        "Highest-scoring pair (planted) must be retained under BH-FDR."
    )


# ---------------------------------------------------------------------------
# R3-S01 guard consistency: adequacy budget matches the corrected inference.
# ---------------------------------------------------------------------------


def test_R3_S01_adequacy_guard_uses_family_error_alpha_for_fwer(monkeypatch) -> None:
    """FWER max-stat: guard requires B >= ceil(1/alpha) - 1 (family_size=1)."""
    n_pairs = _N_FEATURES * (_N_FEATURES - 1) // 2
    obs, null = _make_null_and_signal_stats(
        n_pairs=n_pairs, B=200, signal_pair_index=None, seed=_SEED
    )
    # alpha=0.05 -> min_B = ceil(1/0.05) - 1 = 19. B=15 must fail.
    with pytest.raises(PermutationAdequacyError):
        _run_production_discovery(
            monkeypatch,
            observed_scores=obs,
            null_statistics=null[:15],
            family_error_method="fwer_max_stat",
            family_error_alpha=0.05,
            permutation_count_B=15,
            enforce_permutation_adequacy=True,
        )


def test_R3_S01_adequacy_guard_uses_n_pairs_for_bh_fdr(monkeypatch) -> None:
    """BH-FDR: guard requires B >= ceil(n_pairs/alpha) - 1 (family_size=n_pairs)."""
    n_pairs = _N_FEATURES * (_N_FEATURES - 1) // 2  # 28
    # alpha=0.05, family_size=28 -> min_B = ceil(28/0.05) - 1 = 559. B=200 must fail.
    obs, null = _make_null_and_signal_stats(
        n_pairs=n_pairs, B=200, signal_pair_index=None, seed=_SEED
    )
    with pytest.raises(PermutationAdequacyError):
        _run_production_discovery(
            monkeypatch,
            observed_scores=obs,
            null_statistics=null,
            family_error_method="bh_fdr",
            family_error_alpha=0.05,
            permutation_count_B=200,
            enforce_permutation_adequacy=True,
        )


def test_R3_S01_adequacy_guard_passes_at_corrected_minimum(monkeypatch) -> None:
    """FWER max-stat: guard passes exactly at the corrected minimum B."""
    n_pairs = _N_FEATURES * (_N_FEATURES - 1) // 2
    obs, null = _make_null_and_signal_stats(
        n_pairs=n_pairs, B=200, signal_pair_index=None, seed=_SEED
    )
    # min_B = 19 for alpha=0.05, family_size=1.
    scores = _run_production_discovery(
        monkeypatch,
        observed_scores=obs,
        null_statistics=null[:19],
        family_error_method="fwer_max_stat",
        family_error_alpha=0.05,
        permutation_count_B=19,
        enforce_permutation_adequacy=True,
    )
    assert len(scores) == n_pairs
