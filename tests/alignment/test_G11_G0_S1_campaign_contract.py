"""G11-G0-S1: freeze one hashed campaign contract."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from rfm_pipeline.campaign_contract import (
    G11_CONTRACT,
    build_campaign_inventory,
    compute_contract_hash,
    compute_operating_characteristic_table,
    fixed_family_pair_order,
    largest_passing_wilson_count,
    select_resolution_draw_count,
    validate_campaign_inventory,
    validate_contract_hash,
    validate_predictor_design,
)

_EXPECTED_OC = {
    0.05: 0.999742,
    0.075: 0.530682,
    0.09: 0.051731,
    0.10: 0.003760,
}


def test_g11_contract_total_inventory_is_6400():
    inventory = build_campaign_inventory(G11_CONTRACT, compute_contract_hash(G11_CONTRACT))
    assert len(inventory) == 6400


def test_g11_contract_method_name_is_neutral():
    assert G11_CONTRACT.method_name == "max_stat_adjusted_p_mc"
    assert "fwer" not in G11_CONTRACT.method_name
    assert "exact" not in G11_CONTRACT.method_name


def test_g11_contract_B_values():
    assert G11_CONTRACT.B_screen == 3199
    assert G11_CONTRACT.B_interaction == 999


def test_g11_contract_hash_is_stable():
    contract_hash = compute_contract_hash(G11_CONTRACT)
    assert compute_contract_hash(G11_CONTRACT) == contract_hash
    assert compute_contract_hash(replace(G11_CONTRACT, B_screen=3200)) != contract_hash


def test_g11_contract_no_duplicate_seeds():
    inventory = build_campaign_inventory(G11_CONTRACT, compute_contract_hash(G11_CONTRACT))
    assert len({row.seed for row in inventory}) == len(inventory)


def test_g11_contract_no_duplicate_rows():
    inventory = build_campaign_inventory(G11_CONTRACT, compute_contract_hash(G11_CONTRACT))
    assert len({(row.scenario_id, row.replicate_index) for row in inventory}) == len(inventory)


def test_g11_contract_seed_is_scenario_order_independent():
    contract_hash = compute_contract_hash(G11_CONTRACT)
    inventory = build_campaign_inventory(G11_CONTRACT, contract_hash)
    reversed_contract = replace(G11_CONTRACT, scenarios=tuple(reversed(G11_CONTRACT.scenarios)))
    reversed_inventory = build_campaign_inventory(reversed_contract, contract_hash)
    inventory_by_key = {(row.scenario_id, row.replicate_index): row.seed for row in inventory}
    reversed_by_key = {
        (row.scenario_id, row.replicate_index): row.seed for row in reversed_inventory
    }
    assert reversed_by_key == inventory_by_key


def test_g11_contract_rejects_wrong_predictor_count():
    X = np.zeros((8, 158), dtype=np.int8)
    with pytest.raises(ValueError, match="160"):
        validate_predictor_design(X, G11_CONTRACT)


def test_g11_contract_rejects_wrong_binary_types():
    X = np.zeros((8, 160), dtype=float)
    X[:, -2] = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=float)
    X[:, -1] = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=float)
    with pytest.raises(ValueError, match="int8 or bool"):
        validate_predictor_design(X, G11_CONTRACT)


def test_g11_contract_rejects_binary_column_missing_values():
    X = np.zeros((8, 160), dtype=np.int8)
    X[:, -2] = 0
    X[:, -1] = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int8)
    with pytest.raises(ValueError, match=r"\{0, 1\}"):
        validate_predictor_design(X, G11_CONTRACT)


def test_g11_contract_rejects_stale_hash():
    with pytest.raises(ValueError, match="stale"):
        validate_contract_hash(G11_CONTRACT, "deadbeef")


def test_g11_contract_rejects_incomplete_inventory():
    contract_hash = compute_contract_hash(G11_CONTRACT)
    inventory = build_campaign_inventory(G11_CONTRACT, contract_hash)
    incomplete = [row for row in inventory if row.scenario_id != "weak_signal"]
    with pytest.raises(ValueError, match="missing scenario"):
        validate_campaign_inventory(incomplete, G11_CONTRACT, contract_hash)


def test_g11_contract_has_five_null_regimes():
    assert sum(s.kind == "null" for s in G11_CONTRACT.scenarios) == 5


def test_g11_contract_has_three_strong_regimes():
    assert sum(s.kind == "strong" for s in G11_CONTRACT.scenarios) == 3


def test_g11_contract_null_regimes_each_1000_reps():
    assert all(s.n_replicates == 1000 for s in G11_CONTRACT.scenarios if s.kind == "null")


def test_g11_contract_strong_regimes_each_200_reps():
    assert all(s.n_replicates == 200 for s in G11_CONTRACT.scenarios if s.kind == "strong")


def test_g11_contract_uses_prespecified_calibration_and_recovery_scales():
    """Null calibration and recovery must not silently share a development scale."""
    null_scenarios = [scenario for scenario in G11_CONTRACT.scenarios if scenario.kind == "null"]
    recovery_scenarios = [
        scenario for scenario in G11_CONTRACT.scenarios if scenario.kind in {"strong", "stress"}
    ]
    development = [scenario for scenario in G11_CONTRACT.scenarios if scenario.kind == "dev"]

    assert {(s.n_train, s.n_eval, s.n_response) for s in null_scenarios} == {(160, 80, 4)}
    assert {(s.n_train, s.n_eval, s.n_response) for s in recovery_scenarios} == {(2000, 500, 40)}
    assert {(s.n_train, s.n_eval, s.n_response) for s in development} == {(160, 80, 4)}


def test_g11_contract_freezes_no_rerun_campaign_controls():
    assert G11_CONTRACT.resolution_base_draws == 999
    assert G11_CONTRACT.resolution_max_multiplier == 4
    assert G11_CONTRACT.resolution_family_size == 100
    assert G11_CONTRACT.resolution_boundary_interval == pytest.approx((0.04, 0.06))
    assert G11_CONTRACT.fixed_family_replicates == 1000
    assert G11_CONTRACT.bootstrap_draws == 2000
    assert G11_CONTRACT.n_tree_estimators == 250
    assert G11_CONTRACT.n_stability_subsamples == 50
    assert G11_CONTRACT.stability_jaccard_threshold == pytest.approx(0.75)
    assert G11_CONTRACT.stability_spearman_threshold == pytest.approx(0.90)
    assert G11_CONTRACT.power_gate_lower_bound == pytest.approx(0.80)
    assert G11_CONTRACT.pilot_repetitions == 3
    assert G11_CONTRACT.retryable_scheduler_states == (
        "BOOT_FAIL",
        "NODE_FAIL",
        "PREEMPTED",
    )
    assert G11_CONTRACT.recovery_comparators == (
        "proposed_terminal_workflow",
        "oracle_ols",
        "elastic_net_algebraic_library",
        "raw_input_boosted_tree",
    )
    assert G11_CONTRACT.comparator_cv_folds == 3
    assert G11_CONTRACT.elastic_net_alpha_grid == (0.0001, 0.001, 0.01, 0.1, 1.0)
    assert G11_CONTRACT.elastic_net_l1_ratio_grid == (0.1, 0.5, 0.9)
    assert G11_CONTRACT.boosted_tree_candidate_grid == (
        (100, 2, 0.05),
        (250, 3, 0.05),
    )
    assert G11_CONTRACT.comparator_failure_action == "terminal_failure_no_retry"


def _resolution_records(*, unstable_first_comparison: bool) -> list[dict[str, object]]:
    records = []
    base = G11_CONTRACT.resolution_base_draws
    for fixture in ("nondegenerate_null", "strong_planted"):
        for schedule_index in range(G11_CONTRACT.resolution_schedules):
            p_base = np.linspace(0.001, 0.999, G11_CONTRACT.resolution_family_size)
            p_2b = p_base[::-1] if unstable_first_comparison else p_base.copy()
            records.append(
                {
                    "fixture_kind": fixture,
                    "schedule_index": schedule_index,
                    "adjusted_p_values": {
                        str(base): p_base.tolist(),
                        str(2 * base): p_2b.tolist(),
                        str(4 * base): p_2b.tolist(),
                    },
                }
            )
    return records


def test_resolution_selector_uses_smallest_passing_nested_prefix() -> None:
    accepted = select_resolution_draw_count(
        _resolution_records(unstable_first_comparison=False),
        G11_CONTRACT,
    )
    assert accepted["selected_B_interaction"] == 999

    fallback = select_resolution_draw_count(
        _resolution_records(unstable_first_comparison=True),
        G11_CONTRACT,
    )
    assert fallback["selected_B_interaction"] == 1998


def test_resolution_selector_requires_every_schedule_outside_boundary_band() -> None:
    records = _resolution_records(unstable_first_comparison=False)
    first = records[0]
    base = G11_CONTRACT.resolution_base_draws
    p_base = np.asarray(first["adjusted_p_values"][str(base)], dtype=float)
    p_2b = np.asarray(first["adjusted_p_values"][str(2 * base)], dtype=float)
    p_base[0] = 0.10
    p_2b[0] = 0.01
    first["adjusted_p_values"][str(base)] = p_base.tolist()
    first["adjusted_p_values"][str(2 * base)] = p_2b.tolist()

    selected = select_resolution_draw_count(records, G11_CONTRACT)

    assert selected["selected_B_interaction"] == 1998


def test_fixed_family_order_is_hashed_unique_nested_and_type_nonvacuous() -> None:
    order = fixed_family_pair_order(G11_CONTRACT)

    assert len(order) == 1000
    assert len(set(order)) == 1000
    for family_size in G11_CONTRACT.fixed_family_sizes:
        prefix = order[:family_size]
        assert any("binary_0:binary_1" == pair for pair in prefix)
        assert any(pair.startswith("x") and ":binary_" in pair for pair in prefix)
        assert any(pair.startswith("x") and ":x" in pair for pair in prefix)


def test_g11_operating_characteristic_table():
    rates = tuple(_EXPECTED_OC)
    k_max = largest_passing_wilson_count(
        n_replicates=1000,
        gate_value=0.09,
        confidence=0.95,
    )
    assert k_max == 75
    table = compute_operating_characteristic_table(
        n_replicates=1000,
        gate_value=0.09,
        confidence=0.95,
        true_rates=rates,
    )
    for rate, expected in _EXPECTED_OC.items():
        assert table[rate] == pytest.approx(expected, abs=1e-4)
