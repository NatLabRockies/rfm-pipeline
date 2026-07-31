"""PA-B: Enforce finite-permutation adequacy for exact interaction FWER.

Tests that InteractionDiscoverySpec and interaction_discovery_spec_from_case_study_config
reject configurations where family_error_method='fwer_max_stat_exact' is paired with
insufficient permutation draws, while leaving other methods unaffected.
"""

import pytest

from rfm_pipeline.manuscript_stages import (
    InteractionDiscoverySpec,
    interaction_discovery_spec_from_case_study_config,
)

_BASE_FIELDS = dict(
    method="tree_shap_interaction_values",
    aggregation_rule="max_over_components_of_mean_absolute_shap_interaction",
    null_threshold_quantile=0.995,
    retained_pairs_reference=0,
)


def _make_spec(**overrides) -> InteractionDiscoverySpec:
    fields = {**_BASE_FIELDS, **overrides}
    return InteractionDiscoverySpec(**fields)


def _make_config(
    permutation_count_B: int = 999,
    family_error_method: str = "fwer_max_stat_exact",
    **extra_interaction,
) -> dict:
    interaction = {
        "method": "tree_shap_interaction_values",
        "aggregation_rule": "max_over_components_of_mean_absolute_shap_interaction",
        "null_threshold_quantile": 0.995,
        "retained_pairs": 0,
        "permutation_count_B": permutation_count_B,
        "family_error_method": family_error_method,
        **extra_interaction,
    }
    return {"case_study": {"interaction_discovery": interaction}}


# ---------------------------------------------------------------------------
# Direct spec construction
# ---------------------------------------------------------------------------


def test_exact_fwer_too_few_draws_raises():
    """fwer_max_stat_exact + permutation_count_B < 999 must raise ValueError."""
    with pytest.raises(ValueError, match="fwer_max_stat_exact") as exc_info:
        _make_spec(
            permutation_count_B=200,
            family_error_method="fwer_max_stat_exact",
        )
    assert "200" in str(exc_info.value)


def test_exact_fwer_one_below_minimum_raises():
    with pytest.raises(ValueError, match="fwer_max_stat_exact"):
        _make_spec(
            permutation_count_B=998,
            family_error_method="fwer_max_stat_exact",
        )


def test_exact_fwer_minimum_draws_succeeds():
    """fwer_max_stat_exact + permutation_count_B == 999 must succeed."""
    spec = _make_spec(
        permutation_count_B=999,
        family_error_method="fwer_max_stat_exact",
    )
    assert spec.permutation_count_B == 999


def test_exact_fwer_above_minimum_draws_succeeds():
    spec = _make_spec(
        permutation_count_B=4999,
        family_error_method="fwer_max_stat_exact",
    )
    assert spec.permutation_count_B == 4999


def test_interpolated_fwer_small_draws_succeeds():
    """fwer_max_stat (interpolated) with small permutation_count_B must not raise."""
    spec = _make_spec(
        permutation_count_B=51,
        family_error_method="fwer_max_stat",
    )
    assert spec.permutation_count_B == 51


def test_bh_fdr_small_draws_succeeds():
    """bh_fdr with small permutation_count_B must not raise."""
    spec = _make_spec(
        permutation_count_B=51,
        family_error_method="bh_fdr",
    )
    assert spec.permutation_count_B == 51


def test_min_exact_permutation_draws_below_floor_raises():
    """min_exact_permutation_draws < 999 must raise regardless of method."""
    with pytest.raises(ValueError, match="min_exact_permutation_draws"):
        _make_spec(
            permutation_count_B=999,
            family_error_method="fwer_max_stat_exact",
            min_exact_permutation_draws=100,
        )


def test_min_exact_permutation_draws_default_is_999():
    spec = _make_spec(
        permutation_count_B=999,
        family_error_method="fwer_max_stat_exact",
    )
    assert spec.min_exact_permutation_draws == 999


def test_min_exact_permutation_draws_at_199_floor_succeeds():
    """A case study may lower its floor to 199 (min p_adj = 1/200 = 0.005, well below α)."""
    spec = _make_spec(
        permutation_count_B=199,
        family_error_method="fwer_max_stat_exact",
        min_exact_permutation_draws=199,
    )
    assert spec.min_exact_permutation_draws == 199
    assert spec.permutation_count_B == 199


def test_min_exact_permutation_draws_below_199_floor_raises():
    with pytest.raises(ValueError, match="min_exact_permutation_draws"):
        _make_spec(
            permutation_count_B=198,
            family_error_method="fwer_max_stat_exact",
            min_exact_permutation_draws=198,
        )


def test_reduced_draw_exact_fwer_499_succeeds():
    """A case study may opt into B=499 exact FWER by lowering its floor to 499."""
    spec = _make_spec(
        permutation_count_B=499,
        family_error_method="fwer_max_stat_exact",
        min_exact_permutation_draws=499,
    )
    assert spec.permutation_count_B == 499
    assert spec.min_exact_permutation_draws == 499


def test_min_exact_permutation_draws_can_be_raised():
    """A case study may raise the floor above 999."""
    spec = _make_spec(
        permutation_count_B=4999,
        family_error_method="fwer_max_stat_exact",
        min_exact_permutation_draws=4999,
    )
    assert spec.min_exact_permutation_draws == 4999


def test_min_exact_permutation_draws_raised_floor_enforced():
    """Raising min_exact_permutation_draws also raises the draw requirement."""
    with pytest.raises(ValueError, match="fwer_max_stat_exact"):
        _make_spec(
            permutation_count_B=999,
            family_error_method="fwer_max_stat_exact",
            min_exact_permutation_draws=2000,
        )


# ---------------------------------------------------------------------------
# Config-derived spec
# ---------------------------------------------------------------------------


def test_config_exact_fwer_too_few_raises():
    cfg = _make_config(permutation_count_B=200, family_error_method="fwer_max_stat_exact")
    with pytest.raises(ValueError, match="fwer_max_stat_exact") as exc_info:
        interaction_discovery_spec_from_case_study_config(cfg)
    assert "200" in str(exc_info.value)


def test_config_exact_fwer_sufficient_draws_succeeds():
    cfg = _make_config(permutation_count_B=999, family_error_method="fwer_max_stat_exact")
    spec = interaction_discovery_spec_from_case_study_config(cfg)
    assert spec.permutation_count_B == 999
    assert spec.family_error_method == "fwer_max_stat_exact"


def test_config_interpolated_fwer_small_draws_succeeds():
    cfg = _make_config(permutation_count_B=51, family_error_method="fwer_max_stat")
    spec = interaction_discovery_spec_from_case_study_config(cfg)
    assert spec.permutation_count_B == 51


def test_config_min_exact_permutation_draws_below_floor_raises():
    cfg = _make_config(
        permutation_count_B=999,
        family_error_method="fwer_max_stat_exact",
        min_exact_permutation_draws=100,
    )
    with pytest.raises(ValueError, match="min_exact_permutation_draws"):
        interaction_discovery_spec_from_case_study_config(cfg)


def test_config_reduced_draw_exact_fwer_499_succeeds():
    """Config path: opting into B=499 exact FWER by lowering the floor to 499."""
    cfg = _make_config(
        permutation_count_B=499,
        family_error_method="fwer_max_stat_exact",
        min_exact_permutation_draws=499,
    )
    spec = interaction_discovery_spec_from_case_study_config(cfg)
    assert spec.permutation_count_B == 499
    assert spec.min_exact_permutation_draws == 499
