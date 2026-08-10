"""Gate canonical finite-permutation maxT draw adequacy."""

from __future__ import annotations

import pytest

from rfm_pipeline.interaction_contract import canonical_execution_contract_from_specs
from rfm_pipeline.manuscript_stages import (
    FinalManuscriptArtifactsSpec,
    InteractionDiscoverySpec,
    interaction_discovery_spec_from_case_study_config,
)


def _final_spec() -> FinalManuscriptArtifactsSpec:
    return FinalManuscriptArtifactsSpec(
        final_predictor_count_reference=0,
        final_first_order_input_count_reference=0,
        intermediate_penalized_holdout_nrmse_reference=0.0,
        final_ols_holdout_nrmse_reference=0.0,
        nrmse_denominator_definition="macro",
        nrmse_min_range=1.0e-6,
        nrmse_reference_matrix="Y_train",
    )


def _spec(**overrides: object) -> InteractionDiscoverySpec:
    values: dict[str, object] = {
        "method": "tree_shap_interaction_values",
        "aggregation_rule": "max_over_components_of_mean_absolute_shap_interaction",
        "null_threshold_quantile": 0.995,
        "retained_pairs_reference": 0,
        "permutation_count_B": 199,
        "selection_method": "max_t",
        "selection_alpha": 0.05,
        "minimum_selection_draws": 199,
    }
    values.update(overrides)
    return InteractionDiscoverySpec(**values)


def test_canonical_contract_rejects_draws_below_declared_minimum() -> None:
    with pytest.raises(ValueError, match="draw adequacy"):
        canonical_execution_contract_from_specs(
            _spec(permutation_count_B=198),
            _final_spec(),
        )


def test_canonical_contract_accepts_199_draws() -> None:
    contract = canonical_execution_contract_from_specs(_spec(), _final_spec())
    assert contract.interaction_controls["permutation_count_B"] == 199
    assert contract.interaction_controls["minimum_selection_draws"] == 199


def test_canonical_spec_rejects_lowered_policy_floor() -> None:
    with pytest.raises(ValueError, match="minimum_selection_draws"):
        _spec(minimum_selection_draws=198)


def test_canonical_spec_rejects_legacy_method_alias() -> None:
    with pytest.raises(ValueError, match="selection_method"):
        _spec(selection_method="fwer_max_stat_exact")


def test_case_study_parser_uses_neutral_selection_fields() -> None:
    config = {
        "case_study": {
            "interaction_discovery": {
                "method": "tree_shap_interaction_values",
                "aggregation_rule": "max_over_components_of_mean_absolute_shap_interaction",
                "null_threshold_quantile": 0.995,
                "retained_pairs": 0,
                "permutation_count_B": 199,
                "selection_method": "max_t",
                "selection_alpha": 0.05,
                "minimum_selection_draws": 199,
            }
        }
    }
    spec = interaction_discovery_spec_from_case_study_config(config)
    assert spec.selection_method == "max_t"
    assert spec.selection_alpha == pytest.approx(0.05)


def test_case_study_parser_rejects_legacy_selector_fields() -> None:
    config = {
        "case_study": {
            "interaction_discovery": {
                "family_error_method": "fwer_max_stat_exact",
            }
        }
    }
    with pytest.raises(ValueError, match="Legacy interaction selection fields"):
        interaction_discovery_spec_from_case_study_config(config)
