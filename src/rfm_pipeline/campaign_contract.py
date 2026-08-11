"""G11 campaign contract — one hashed source of truth for all scientific settings.

This module is the sole authoritative source for B_screen, B_interaction, method
name, scenario grid, DGPs, replicate counts, seed derivation, retry policy,
artifact schema, and campaign inventory for the G11 analysis.

Any change to contract fields changes the contract hash, invalidating all caches.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import tomllib
import yaml


@dataclass(frozen=True, slots=True)
class ScenarioSpec:
    """Frozen scenario contract for one calibration or development regime."""

    id: str
    kind: str
    n_replicates: int
    n_predictor_continuous: int
    n_predictor_binary: int
    n_response: int
    n_train: int
    n_eval: int
    description: str


@dataclass(frozen=True, slots=True)
class CampaignContract:
    """Frozen top-level G11 contract."""

    schema_version: str
    generation: int
    method_name: str
    B_screen: int
    B_interaction: int
    alpha: float
    q_screen: float
    calibration_tolerance: float
    calibration_confidence: float
    gate_value: float
    scenarios: tuple[ScenarioSpec, ...]
    retry_limit: int
    resolution_schedules: int
    fixed_family_sizes: tuple[int, ...]
    artifact_schema_version: int


@dataclass(frozen=True, slots=True)
class ArtifactFieldSpec:
    """Typed terminal-ledger field contract."""

    name: str
    dtype: str


@dataclass(frozen=True, slots=True)
class CampaignInventoryRow:
    """One planned non-development replicate in the G11 inventory."""

    scenario_id: str
    replicate_index: int
    seed: int
    contract_hash: str
    kind: str


TERMINAL_ROW_SCHEMA: tuple[ArtifactFieldSpec, ...] = (
    ArtifactFieldSpec("scenario", "str"),
    ArtifactFieldSpec("replicate_index", "int"),
    ArtifactFieldSpec("seed", "int"),
    ArtifactFieldSpec("contract_hash", "str"),
    ArtifactFieldSpec("source_hash", "str"),
    ArtifactFieldSpec("schedule_hash", "str"),
    ArtifactFieldSpec("screened_count", "int"),
    ArtifactFieldSpec("pair_family_count", "int"),
    ArtifactFieldSpec("truth_pairs", "int"),
    ArtifactFieldSpec("retained_pairs", "int"),
    ArtifactFieldSpec("false_pair_count", "int"),
    ArtifactFieldSpec("terminal_stage", "str"),
    ArtifactFieldSpec("status", "str"),
    ArtifactFieldSpec("exception", "str"),
    ArtifactFieldSpec("runtime_s", "float"),
    ArtifactFieldSpec("peak_memory_mb", "float"),
)

_G11_SCENARIOS: tuple[ScenarioSpec, ...] = (
    ScenarioSpec(
        id="global_null",
        kind="null",
        n_replicates=1000,
        n_predictor_continuous=158,
        n_predictor_binary=2,
        n_response=4,
        n_train=320,
        n_eval=80,
        description="No signal of any kind.",
    ),
    ScenarioSpec(
        id="interaction_null_continuous",
        kind="null",
        n_replicates=1000,
        n_predictor_continuous=158,
        n_predictor_binary=2,
        n_response=4,
        n_train=320,
        n_eval=80,
        description="Continuous and nonlinear main effects without interactions.",
    ),
    ScenarioSpec(
        id="interaction_null_correlated",
        kind="null",
        n_replicates=1000,
        n_predictor_continuous=158,
        n_predictor_binary=2,
        n_response=4,
        n_train=320,
        n_eval=80,
        description="Correlated-input structure without interactions.",
    ),
    ScenarioSpec(
        id="interaction_null_heteroscedastic",
        kind="null",
        n_replicates=1000,
        n_predictor_continuous=158,
        n_predictor_binary=2,
        n_response=4,
        n_train=320,
        n_eval=80,
        description="Frozen heteroscedastic variance function without interactions.",
    ),
    ScenarioSpec(
        id="interaction_null_binary_main",
        kind="null",
        n_replicates=1000,
        n_predictor_continuous=158,
        n_predictor_binary=2,
        n_response=4,
        n_train=320,
        n_eval=80,
        description="Binary main effects only; designed for >=90% nondegenerate families.",
    ),
    ScenarioSpec(
        id="strong_cc",
        kind="strong",
        n_replicates=200,
        n_predictor_continuous=158,
        n_predictor_binary=2,
        n_response=4,
        n_train=320,
        n_eval=80,
        description="Strong continuous-continuous planted interaction.",
    ),
    ScenarioSpec(
        id="strong_bc",
        kind="strong",
        n_replicates=200,
        n_predictor_continuous=158,
        n_predictor_binary=2,
        n_response=4,
        n_train=320,
        n_eval=80,
        description="Strong binary-continuous planted interaction.",
    ),
    ScenarioSpec(
        id="strong_bb",
        kind="strong",
        n_replicates=200,
        n_predictor_continuous=158,
        n_predictor_binary=2,
        n_response=4,
        n_train=320,
        n_eval=80,
        description="Strong binary-binary planted interaction.",
    ),
    ScenarioSpec(
        id="weak_signal",
        kind="stress",
        n_replicates=200,
        n_predictor_continuous=158,
        n_predictor_binary=2,
        n_response=4,
        n_train=320,
        n_eval=80,
        description="Same hierarchical support as strong_cc with low SNR.",
    ),
    ScenarioSpec(
        id="correlated_redundant",
        kind="stress",
        n_replicates=200,
        n_predictor_continuous=158,
        n_predictor_binary=2,
        n_response=4,
        n_train=320,
        n_eval=80,
        description="Dense active set with redundant correlated predictors.",
    ),
    ScenarioSpec(
        id="pure_interaction",
        kind="stress",
        n_replicates=200,
        n_predictor_continuous=158,
        n_predictor_binary=2,
        n_response=4,
        n_train=320,
        n_eval=80,
        description="Pure continuous-continuous interaction with negligible mains.",
    ),
    ScenarioSpec(
        id="nonlinear_misspecified",
        kind="stress",
        n_replicates=200,
        n_predictor_continuous=158,
        n_predictor_binary=2,
        n_response=4,
        n_train=320,
        n_eval=80,
        description="Nonlinear transform plus one out-of-library misspecification.",
    ),
    ScenarioSpec(
        id="dev_resolution",
        kind="dev",
        n_replicates=20,
        n_predictor_continuous=158,
        n_predictor_binary=2,
        n_response=4,
        n_train=320,
        n_eval=80,
        description="Development-only B versus 2B resolution fixture.",
    ),
)

G11_CONTRACT = CampaignContract(
    schema_version="g11_campaign_contract_v1",
    generation=11,
    method_name="max_stat_adjusted_p_mc",
    B_screen=3199,
    B_interaction=999,
    alpha=0.05,
    q_screen=0.05,
    calibration_tolerance=0.04,
    calibration_confidence=0.95,
    gate_value=0.09,
    scenarios=_G11_SCENARIOS,
    retry_limit=0,
    resolution_schedules=10,
    fixed_family_sizes=(10, 100, 1000),
    artifact_schema_version=1,
)


def _contract_payload(contract: CampaignContract) -> dict[str, Any]:
    payload = asdict(contract)
    payload["scenarios"] = sorted(payload["scenarios"], key=lambda scenario: scenario["id"])
    payload["terminal_row_schema"] = [asdict(field) for field in TERMINAL_ROW_SCHEMA]
    return payload


def compute_contract_hash(contract: CampaignContract) -> str:
    """Return the SHA-256 hash of the canonical scientific contract payload."""
    payload = _contract_payload(contract)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def derive_seed(contract_hash: str, scenario_id: str, replicate_index: int) -> int:
    """Derive a scenario-order-independent seed from hash, scenario, and replicate."""
    seed_bytes = hashlib.sha256(
        f"{contract_hash}:{scenario_id}:{replicate_index}".encode()
    ).digest()[:8]
    return int.from_bytes(seed_bytes, byteorder="little", signed=False) % (2**31)


def _non_dev_scenarios(contract: CampaignContract) -> tuple[ScenarioSpec, ...]:
    return tuple(scenario for scenario in contract.scenarios if scenario.kind != "dev")


def _expected_inventory_total(contract: CampaignContract) -> int:
    return sum(scenario.n_replicates for scenario in _non_dev_scenarios(contract))


def validate_campaign_inventory(
    inventory: list[CampaignInventoryRow],
    contract: CampaignContract,
    contract_hash: str,
) -> None:
    """Fail closed on duplicate IDs, duplicate seeds, or incomplete inventories."""
    validate_contract_hash(contract, contract_hash)
    expected_scenarios = {scenario.id: scenario for scenario in _non_dev_scenarios(contract)}
    if _expected_inventory_total(contract) != 6400:
        raise ValueError("G11 contract must define exactly 6400 non-development replicates")

    pair_counts = Counter((row.scenario_id, row.replicate_index) for row in inventory)
    duplicates = [pair for pair, count in pair_counts.items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate (scenario_id, replicate_index) pair(s): {duplicates[:5]}")

    seed_counts = Counter(row.seed for row in inventory)
    duplicate_seeds = [seed for seed, count in seed_counts.items() if count > 1]
    if duplicate_seeds:
        raise ValueError(f"duplicate seeds in inventory: {duplicate_seeds[:5]}")

    seen_scenarios = {row.scenario_id for row in inventory}
    missing_scenarios = sorted(set(expected_scenarios) - seen_scenarios)
    if missing_scenarios:
        raise ValueError(f"missing scenario(s) in inventory: {missing_scenarios}")

    unexpected_scenarios = sorted(seen_scenarios - set(expected_scenarios))
    if unexpected_scenarios:
        raise ValueError(f"unexpected scenario(s) in inventory: {unexpected_scenarios}")

    if len(inventory) != _expected_inventory_total(contract):
        expected_total = _expected_inventory_total(contract)
        raise ValueError(
            f"incomplete inventory: expected {expected_total} rows, got {len(inventory)}"
        )

    for scenario_id, scenario in expected_scenarios.items():
        indices = sorted(row.replicate_index for row in inventory if row.scenario_id == scenario_id)
        expected_indices = list(range(scenario.n_replicates))
        if indices != expected_indices:
            raise ValueError(
                f"incomplete inventory for {scenario_id}: expected replicate indices "
                f"0..{scenario.n_replicates - 1}"
            )
        wrong_hash_rows = [
            row
            for row in inventory
            if row.scenario_id == scenario_id and row.contract_hash != contract_hash
        ]
        if wrong_hash_rows:
            raise ValueError(f"inventory row(s) for {scenario_id} carry a stale contract hash")
        wrong_kind_rows = [
            row for row in inventory if row.scenario_id == scenario_id and row.kind != scenario.kind
        ]
        if wrong_kind_rows:
            raise ValueError(f"inventory row(s) for {scenario_id} carry the wrong scenario kind")


def build_campaign_inventory(
    contract: CampaignContract, contract_hash: str
) -> list[CampaignInventoryRow]:
    """Build and validate the full 6,400-row non-development G11 inventory."""
    inventory = [
        CampaignInventoryRow(
            scenario_id=scenario.id,
            replicate_index=replicate_index,
            seed=derive_seed(contract_hash, scenario.id, replicate_index),
            contract_hash=contract_hash,
            kind=scenario.kind,
        )
        for scenario in _non_dev_scenarios(contract)
        for replicate_index in range(scenario.n_replicates)
    ]
    validate_campaign_inventory(inventory, contract, contract_hash)
    return inventory


def _predictor_dimensions(contract: CampaignContract) -> tuple[int, int]:
    dims = {
        (scenario.n_predictor_continuous, scenario.n_predictor_binary)
        for scenario in contract.scenarios
    }
    if len(dims) != 1:
        raise ValueError("contract scenarios disagree on predictor dimensions")
    return next(iter(dims))


def _validate_binary_column(column: np.ndarray, column_name: str) -> None:
    if column.dtype != object:
        if not (column.dtype == np.int8 or np.issubdtype(column.dtype, np.bool_)):
            raise ValueError(f"{column_name} must use dtype int8 or bool")
        values = {int(value) for value in np.unique(column)}
    else:
        values = set()
        for value in column:
            if not isinstance(value, (bool, np.bool_, int, np.integer)):
                raise ValueError(f"{column_name} must use dtype int8 or bool")
            values.add(int(value))
    if values != {0, 1}:
        raise ValueError(f"{column_name} must contain exactly 2 distinct values {{0, 1}}")


def validate_predictor_design(X: np.ndarray, contract: CampaignContract) -> None:
    """Validate the 158-continuous/2-binary predictor contract."""
    if not isinstance(X, np.ndarray):
        raise ValueError("predictor design must be a numpy.ndarray")
    if X.ndim != 2:
        raise ValueError("predictor design must be a 2D array")

    n_continuous, n_binary = _predictor_dimensions(contract)
    expected_columns = n_continuous + n_binary
    if expected_columns != 160:
        raise ValueError("G11 contract must define exactly 160 predictors")
    if X.shape[1] != expected_columns:
        raise ValueError(
            f"predictor design must have exactly {expected_columns} columns; got {X.shape[1]}"
        )

    for offset in range(n_binary):
        column_index = n_continuous + offset
        _validate_binary_column(X[:, column_index], f"binary column {offset}")


def validate_contract_hash(contract: CampaignContract, stored_hash: str) -> None:
    """Raise if the stored hash no longer matches the scientific contract payload."""
    computed_hash = compute_contract_hash(contract)
    if computed_hash != stored_hash:
        raise ValueError(
            "stale contract hash: stored hash does not match recomputed scientific contract hash"
        )


def _parse_terminal_row_schema(raw_fields: Any) -> tuple[ArtifactFieldSpec, ...]:
    if not isinstance(raw_fields, list):
        raise ValueError("terminal_row_schema must be a list of typed field records")
    parsed = tuple(
        ArtifactFieldSpec(name=field["name"], dtype=field["dtype"]) for field in raw_fields
    )
    if parsed != TERMINAL_ROW_SCHEMA:
        raise ValueError("terminal_row_schema does not match the committed G11 schema")
    return parsed


def _contract_from_mapping(raw: dict[str, Any]) -> CampaignContract:
    _parse_terminal_row_schema(raw.get("terminal_row_schema"))
    scenarios = tuple(ScenarioSpec(**scenario) for scenario in raw["scenarios"])
    return CampaignContract(
        schema_version=raw["schema_version"],
        generation=int(raw["generation"]),
        method_name=raw["method_name"],
        B_screen=int(raw["B_screen"]),
        B_interaction=int(raw["B_interaction"]),
        alpha=float(raw["alpha"]),
        q_screen=float(raw["q_screen"]),
        calibration_tolerance=float(raw["calibration_tolerance"]),
        calibration_confidence=float(raw["calibration_confidence"]),
        gate_value=float(raw["gate_value"]),
        scenarios=scenarios,
        retry_limit=int(raw["retry_limit"]),
        resolution_schedules=int(raw["resolution_schedules"]),
        fixed_family_sizes=tuple(int(value) for value in raw["fixed_family_sizes"]),
        artifact_schema_version=int(raw["artifact_schema_version"]),
    )


def load_contract(path: Path) -> tuple[CampaignContract, str]:
    """Load a TOML or YAML contract file and fail closed on hash drift."""
    suffix = path.suffix.lower()
    if suffix == ".toml":
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    elif suffix in {".yaml", ".yml"}:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        raise ValueError(f"unsupported contract format: {path.suffix}")
    if not isinstance(raw, dict):
        raise ValueError("contract file must deserialize to a mapping")
    stored_hash = raw.get("contract_hash")
    if not isinstance(stored_hash, str) or not stored_hash:
        raise ValueError("contract file must contain a non-empty contract_hash")
    contract = _contract_from_mapping(raw)
    validate_contract_hash(contract, stored_hash)
    return contract, stored_hash


def wilson_upper_bound(k: int, n: int, confidence: float) -> float:
    """Return the one-sided Wilson upper confidence bound for a binomial rate."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= k <= n:
        raise ValueError("k must satisfy 0 <= k <= n")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")

    z = NormalDist().inv_cdf(confidence)
    p_hat = k / n
    z2_over_n = (z * z) / n
    centre = p_hat + z2_over_n / 2.0
    margin = z * math.sqrt((p_hat * (1.0 - p_hat) + z2_over_n / 4.0) / n)
    return (centre + margin) / (1.0 + z2_over_n)


def largest_passing_wilson_count(
    n_replicates: int,
    gate_value: float,
    confidence: float,
) -> int:
    """Derive the largest count whose Wilson upper bound still passes the gate."""
    if not 0.0 <= gate_value <= 1.0:
        raise ValueError("gate_value must lie in [0, 1]")
    passing = -1
    for k in range(n_replicates + 1):
        if wilson_upper_bound(k, n_replicates, confidence) <= gate_value:
            passing = k
        else:
            break
    if passing < 0:
        raise ValueError("no passing count exists for the requested Wilson gate")
    return passing


def _binomial_cdf(k_max: int, n: int, p: float) -> float:
    if not 0.0 <= p <= 1.0:
        raise ValueError("true rates must lie in [0, 1]")
    if k_max < 0:
        return 0.0
    if k_max >= n:
        return 1.0
    if p == 0.0:
        return 1.0
    if p == 1.0:
        return 1.0 if k_max >= n else 0.0

    q = 1.0 - p
    pmf = q**n
    cdf = pmf
    for k in range(1, k_max + 1):
        pmf *= ((n - k + 1) / k) * (p / q)
        cdf += pmf
    return min(1.0, max(0.0, cdf))


def compute_operating_characteristic_table(
    n_replicates: int,
    gate_value: float,
    confidence: float,
    true_rates: tuple[float, ...] | list[float],
) -> dict[float, float]:
    """Derive the exact-binomial pass probabilities for the frozen Wilson gate."""
    k_max = largest_passing_wilson_count(
        n_replicates=n_replicates,
        gate_value=gate_value,
        confidence=confidence,
    )
    return {float(rate): _binomial_cdf(k_max, n_replicates, float(rate)) for rate in true_rates}


def render_contract_toml(
    contract: CampaignContract, contract_hash: str, *, gate: str, status: str
) -> str:
    """Render the committed contract as a TOML document."""
    validate_contract_hash(contract, contract_hash)
    lines = [
        f'schema_version = "{contract.schema_version}"',
        f"generation = {contract.generation}",
        f'gate = "{gate}"',
        f'status = "{status}"',
        f'contract_hash = "{contract_hash}"',
        f'method_name = "{contract.method_name}"',
        f"B_screen = {contract.B_screen}",
        f"B_interaction = {contract.B_interaction}",
        f"alpha = {contract.alpha}",
        f"q_screen = {contract.q_screen}",
        f"calibration_tolerance = {contract.calibration_tolerance}",
        f"calibration_confidence = {contract.calibration_confidence}",
        f"gate_value = {contract.gate_value}",
        f"retry_limit = {contract.retry_limit}",
        f"resolution_schedules = {contract.resolution_schedules}",
        "fixed_family_sizes = [{}]".format(
            ", ".join(str(value) for value in contract.fixed_family_sizes)
        ),
        f"artifact_schema_version = {contract.artifact_schema_version}",
        "",
    ]
    for field in TERMINAL_ROW_SCHEMA:
        lines.extend(
            [
                "[[terminal_row_schema]]",
                f'name = "{field.name}"',
                f'dtype = "{field.dtype}"',
                "",
            ]
        )
    for scenario in contract.scenarios:
        lines.extend(
            [
                "[[scenarios]]",
                f'id = "{scenario.id}"',
                f'kind = "{scenario.kind}"',
                f"n_replicates = {scenario.n_replicates}",
                f"n_predictor_continuous = {scenario.n_predictor_continuous}",
                f"n_predictor_binary = {scenario.n_predictor_binary}",
                f"n_response = {scenario.n_response}",
                f"n_train = {scenario.n_train}",
                f"n_eval = {scenario.n_eval}",
                f'description = "{scenario.description}"',
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "ArtifactFieldSpec",
    "CampaignContract",
    "CampaignInventoryRow",
    "G11_CONTRACT",
    "ScenarioSpec",
    "TERMINAL_ROW_SCHEMA",
    "build_campaign_inventory",
    "compute_contract_hash",
    "compute_operating_characteristic_table",
    "derive_seed",
    "largest_passing_wilson_count",
    "load_contract",
    "render_contract_toml",
    "validate_campaign_inventory",
    "validate_contract_hash",
    "validate_predictor_design",
    "wilson_upper_bound",
]
