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
    largest_passing_wilson_count,
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
