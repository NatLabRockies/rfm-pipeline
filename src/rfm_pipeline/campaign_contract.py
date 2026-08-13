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
from itertools import combinations
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
    resolution_base_draws: int
    resolution_max_multiplier: int
    resolution_family_size: int
    resolution_boundary_interval: tuple[float, float]
    resolution_decision_sha256: str
    fixed_family_sizes: tuple[int, ...]
    fixed_family_replicates: int
    fixed_family_pair_order_sha256: str
    bootstrap_draws: int
    n_tree_estimators: int
    n_stability_subsamples: int
    stability_jaccard_threshold: float
    stability_spearman_threshold: float
    power_gate_lower_bound: float
    pilot_repetitions: int
    retryable_scheduler_states: tuple[str, ...]
    recovery_comparators: tuple[str, ...]
    comparator_cv_folds: int
    elastic_net_alpha_grid: tuple[float, ...]
    elastic_net_l1_ratio_grid: tuple[float, ...]
    boosted_tree_candidate_grid: tuple[tuple[int, int, float], ...]
    boosted_tree_response_components: int
    comparator_tuning_metric: str
    comparator_failure_action: str
    artifact_schema_version: int

    def __post_init__(self) -> None:
        """Reject invalid adaptive-resolution identity fields."""
        low, high = self.resolution_boundary_interval
        if not 0.0 <= low < self.alpha < high <= 1.0:
            raise ValueError("resolution boundary interval must strictly bracket alpha")
        if self.resolution_decision_sha256 != "PENDING" and (
            len(self.resolution_decision_sha256) != 64
            or any(value not in "0123456789abcdef" for value in self.resolution_decision_sha256)
        ):
            raise ValueError("resolution decision must be PENDING or a SHA-256 digest")


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


def _canonical_fixed_family_pair_order() -> tuple[str, ...]:
    feature_names = tuple(f"x{index:03d}" for index in range(158)) + (
        "binary_0",
        "binary_1",
    )
    all_pairs = tuple(f"{left}:{right}" for left, right in combinations(feature_names, 2))
    required_prefix = (
        "x000:binary_0",
        "binary_0:binary_1",
        "x000:x001",
    )
    remaining = tuple(pair for pair in all_pairs if pair not in required_prefix)
    return (required_prefix + remaining)[:1000]


_FIXED_FAMILY_PAIR_ORDER = _canonical_fixed_family_pair_order()
_FIXED_FAMILY_PAIR_ORDER_SHA256 = hashlib.sha256(
    json.dumps(
        list(_FIXED_FAMILY_PAIR_ORDER),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
).hexdigest()


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
        n_train=160,
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
        n_train=160,
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
        n_train=160,
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
        n_train=160,
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
        n_train=160,
        n_eval=80,
        description="Binary main effects only; designed for >=90% nondegenerate families.",
    ),
    ScenarioSpec(
        id="strong_cc",
        kind="strong",
        n_replicates=200,
        n_predictor_continuous=158,
        n_predictor_binary=2,
        n_response=40,
        n_train=2000,
        n_eval=500,
        description="Strong continuous-continuous planted interaction.",
    ),
    ScenarioSpec(
        id="strong_bc",
        kind="strong",
        n_replicates=200,
        n_predictor_continuous=158,
        n_predictor_binary=2,
        n_response=40,
        n_train=2000,
        n_eval=500,
        description="Strong binary-continuous planted interaction.",
    ),
    ScenarioSpec(
        id="strong_bb",
        kind="strong",
        n_replicates=200,
        n_predictor_continuous=158,
        n_predictor_binary=2,
        n_response=40,
        n_train=2000,
        n_eval=500,
        description="Strong binary-binary planted interaction.",
    ),
    ScenarioSpec(
        id="weak_signal",
        kind="stress",
        n_replicates=200,
        n_predictor_continuous=158,
        n_predictor_binary=2,
        n_response=40,
        n_train=2000,
        n_eval=500,
        description="Same hierarchical support as strong_cc with low SNR.",
    ),
    ScenarioSpec(
        id="correlated_redundant",
        kind="stress",
        n_replicates=200,
        n_predictor_continuous=158,
        n_predictor_binary=2,
        n_response=40,
        n_train=2000,
        n_eval=500,
        description="Dense active set with redundant correlated predictors.",
    ),
    ScenarioSpec(
        id="pure_interaction",
        kind="stress",
        n_replicates=200,
        n_predictor_continuous=158,
        n_predictor_binary=2,
        n_response=40,
        n_train=2000,
        n_eval=500,
        description="Pure continuous-continuous interaction with negligible mains.",
    ),
    ScenarioSpec(
        id="nonlinear_misspecified",
        kind="stress",
        n_replicates=200,
        n_predictor_continuous=158,
        n_predictor_binary=2,
        n_response=40,
        n_train=2000,
        n_eval=500,
        description="Nonlinear transform plus one out-of-library misspecification.",
    ),
    ScenarioSpec(
        id="dev_resolution",
        kind="dev",
        n_replicates=20,
        n_predictor_continuous=158,
        n_predictor_binary=2,
        n_response=4,
        n_train=160,
        n_eval=80,
        description="Development-only B versus 2B resolution fixture.",
    ),
)

G11_CONTRACT = CampaignContract(
    schema_version="g11_campaign_contract_v9",
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
    retry_limit=1,
    resolution_schedules=10,
    resolution_base_draws=999,
    resolution_max_multiplier=4,
    resolution_family_size=100,
    resolution_boundary_interval=(0.04, 0.06),
    resolution_decision_sha256="PENDING",
    fixed_family_sizes=(10, 100, 1000),
    fixed_family_replicates=1000,
    fixed_family_pair_order_sha256=_FIXED_FAMILY_PAIR_ORDER_SHA256,
    bootstrap_draws=2000,
    n_tree_estimators=250,
    n_stability_subsamples=50,
    stability_jaccard_threshold=0.75,
    stability_spearman_threshold=0.90,
    power_gate_lower_bound=0.80,
    pilot_repetitions=3,
    retryable_scheduler_states=("BOOT_FAIL", "NODE_FAIL", "PREEMPTED"),
    recovery_comparators=(
        "proposed_terminal_workflow",
        "oracle_ols",
        "elastic_net_algebraic_library",
        "raw_input_boosted_tree",
    ),
    comparator_cv_folds=3,
    elastic_net_alpha_grid=(0.0001, 0.001, 0.01, 0.1, 1.0),
    elastic_net_l1_ratio_grid=(0.1, 0.5, 0.9),
    boosted_tree_candidate_grid=((100, 2, 0.05), (250, 3, 0.05)),
    boosted_tree_response_components=4,
    comparator_tuning_metric="training_cv_macro_nrmse",
    comparator_failure_action="terminal_failure_no_retry",
    artifact_schema_version=2,
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


def fixed_family_pair_order(contract: CampaignContract) -> tuple[str, ...]:
    """Return and verify the frozen nested 1,000-pair fixed-family order."""
    if max(contract.fixed_family_sizes) > len(_FIXED_FAMILY_PAIR_ORDER):
        raise ValueError("fixed-family size exceeds the frozen pair order")
    if contract.fixed_family_pair_order_sha256 != _FIXED_FAMILY_PAIR_ORDER_SHA256:
        raise ValueError("fixed-family pair-order hash differs from the canonical order")
    return _FIXED_FAMILY_PAIR_ORDER


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


def select_resolution_draw_count(
    records: list[dict[str, Any]],
    contract: CampaignContract,
) -> dict[str, Any]:
    """Choose the smallest stable nested interaction schedule or fail closed."""
    from scipy.stats import spearmanr

    fixtures = ("nondegenerate_null", "strong_planted")
    expected = {
        (fixture, schedule_index)
        for fixture in fixtures
        for schedule_index in range(contract.resolution_schedules)
    }
    keyed: dict[tuple[str, int], dict[str, Any]] = {}
    for record in records:
        key = (str(record.get("fixture_kind")), int(record.get("schedule_index", -1)))
        if key in keyed:
            raise ValueError(f"duplicate resolution record {key}")
        keyed[key] = record
    if set(keyed) != expected:
        raise ValueError("resolution records do not exactly cover both frozen fixtures")

    base = contract.resolution_base_draws
    candidates = (base, 2 * base, contract.resolution_max_multiplier * base)
    comparisons: dict[str, Any] = {}
    for lower, upper in zip(candidates, candidates[1:], strict=True):
        fixture_diagnostics: dict[str, Any] = {}
        comparison_passes = True
        for fixture in fixtures:
            schedule_diagnostics = []
            for schedule_index in range(contract.resolution_schedules):
                record = keyed[(fixture, schedule_index)]
                values = record.get("adjusted_p_values")
                if not isinstance(values, dict):
                    raise ValueError("resolution record lacks adjusted_p_values")
                lower_p = np.asarray(values.get(str(lower)), dtype=float)
                upper_p = np.asarray(values.get(str(upper)), dtype=float)
                expected_shape = (contract.resolution_family_size,)
                if (
                    lower_p.shape != expected_shape
                    or upper_p.shape != expected_shape
                    or not np.isfinite(lower_p).all()
                    or not np.isfinite(upper_p).all()
                ):
                    raise ValueError("resolution adjusted-p vectors have invalid shape or values")
                lower_selected = lower_p <= contract.alpha
                upper_selected = upper_p <= contract.alpha
                boundary_low, boundary_high = contract.resolution_boundary_interval
                decisive = (upper_p < boundary_low) | (upper_p > boundary_high)
                agreement = lower_selected == upper_selected
                lower_set = set(np.flatnonzero(lower_selected).tolist())
                upper_set = set(np.flatnonzero(upper_selected).tolist())
                union = lower_set | upper_set
                jaccard = 1.0 if not union else len(lower_set & upper_set) / len(union)
                correlation = float(spearmanr(lower_p, upper_p).statistic)
                if not math.isfinite(correlation):
                    correlation = 1.0 if np.array_equal(lower_p, upper_p) else -1.0
                passed = bool(np.all(agreement[decisive]))
                schedule_diagnostics.append(
                    {
                        "schedule_index": schedule_index,
                        "jaccard": jaccard,
                        "spearman": correlation,
                        "decisive_pair_count": int(np.sum(decisive)),
                        "near_boundary_pair_count": int(np.sum(~decisive)),
                        "decisive_disagreement_count": int(np.sum(decisive & ~agreement)),
                        "passed": passed,
                    }
                )
            passing_fraction = sum(row["passed"] for row in schedule_diagnostics) / len(
                schedule_diagnostics
            )
            fixture_passed = all(row["passed"] for row in schedule_diagnostics)
            fixture_diagnostics[fixture] = {
                "passing_fraction": passing_fraction,
                "passed": fixture_passed,
                "schedules": schedule_diagnostics,
            }
            comparison_passes &= fixture_passed
        key = f"{lower}_vs_{upper}"
        comparisons[key] = {
            "lower": lower,
            "upper": upper,
            "passed": comparison_passes,
            "fixtures": fixture_diagnostics,
        }
        if comparison_passes:
            return {
                "status": "ACCEPTED",
                "selected_B_interaction": lower,
                "comparisons": comparisons,
            }
    raise ValueError("no prespecified nested interaction draw comparison passed")


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
        resolution_base_draws=int(raw["resolution_base_draws"]),
        resolution_max_multiplier=int(raw["resolution_max_multiplier"]),
        resolution_family_size=int(raw["resolution_family_size"]),
        resolution_boundary_interval=tuple(
            float(value) for value in raw["resolution_boundary_interval"]
        ),
        resolution_decision_sha256=str(raw["resolution_decision_sha256"]),
        fixed_family_sizes=tuple(int(value) for value in raw["fixed_family_sizes"]),
        fixed_family_replicates=int(raw["fixed_family_replicates"]),
        fixed_family_pair_order_sha256=str(raw["fixed_family_pair_order_sha256"]),
        bootstrap_draws=int(raw["bootstrap_draws"]),
        n_tree_estimators=int(raw["n_tree_estimators"]),
        n_stability_subsamples=int(raw["n_stability_subsamples"]),
        stability_jaccard_threshold=float(raw["stability_jaccard_threshold"]),
        stability_spearman_threshold=float(raw["stability_spearman_threshold"]),
        power_gate_lower_bound=float(raw["power_gate_lower_bound"]),
        pilot_repetitions=int(raw["pilot_repetitions"]),
        retryable_scheduler_states=tuple(str(value) for value in raw["retryable_scheduler_states"]),
        recovery_comparators=tuple(str(value) for value in raw["recovery_comparators"]),
        comparator_cv_folds=int(raw["comparator_cv_folds"]),
        elastic_net_alpha_grid=tuple(float(value) for value in raw["elastic_net_alpha_grid"]),
        elastic_net_l1_ratio_grid=tuple(float(value) for value in raw["elastic_net_l1_ratio_grid"]),
        boosted_tree_candidate_grid=tuple(
            (int(value[0]), int(value[1]), float(value[2]))
            for value in raw["boosted_tree_candidate_grid"]
        ),
        boosted_tree_response_components=int(raw["boosted_tree_response_components"]),
        comparator_tuning_metric=str(raw["comparator_tuning_metric"]),
        comparator_failure_action=str(raw["comparator_failure_action"]),
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
        f"resolution_base_draws = {contract.resolution_base_draws}",
        f"resolution_max_multiplier = {contract.resolution_max_multiplier}",
        f"resolution_family_size = {contract.resolution_family_size}",
        "resolution_boundary_interval = "
        f"[{contract.resolution_boundary_interval[0]}, {contract.resolution_boundary_interval[1]}]",
        f'resolution_decision_sha256 = "{contract.resolution_decision_sha256}"',
        "fixed_family_sizes = [{}]".format(
            ", ".join(str(value) for value in contract.fixed_family_sizes)
        ),
        f"fixed_family_replicates = {contract.fixed_family_replicates}",
        f'fixed_family_pair_order_sha256 = "{contract.fixed_family_pair_order_sha256}"',
        f"bootstrap_draws = {contract.bootstrap_draws}",
        f"n_tree_estimators = {contract.n_tree_estimators}",
        f"n_stability_subsamples = {contract.n_stability_subsamples}",
        f"stability_jaccard_threshold = {contract.stability_jaccard_threshold}",
        f"stability_spearman_threshold = {contract.stability_spearman_threshold}",
        f"power_gate_lower_bound = {contract.power_gate_lower_bound}",
        f"pilot_repetitions = {contract.pilot_repetitions}",
        "retryable_scheduler_states = [{}]".format(
            ", ".join(f'"{value}"' for value in contract.retryable_scheduler_states)
        ),
        "recovery_comparators = [{}]".format(
            ", ".join(f'"{value}"' for value in contract.recovery_comparators)
        ),
        f"comparator_cv_folds = {contract.comparator_cv_folds}",
        "elastic_net_alpha_grid = [{}]".format(
            ", ".join(str(value) for value in contract.elastic_net_alpha_grid)
        ),
        "elastic_net_l1_ratio_grid = [{}]".format(
            ", ".join(str(value) for value in contract.elastic_net_l1_ratio_grid)
        ),
        "boosted_tree_candidate_grid = [{}]".format(
            ", ".join(
                f"[{estimators}, {depth}, {learning_rate}]"
                for estimators, depth, learning_rate in contract.boosted_tree_candidate_grid
            )
        ),
        f"boosted_tree_response_components = {contract.boosted_tree_response_components}",
        f'comparator_tuning_metric = "{contract.comparator_tuning_metric}"',
        f'comparator_failure_action = "{contract.comparator_failure_action}"',
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
    "fixed_family_pair_order",
    "largest_passing_wilson_count",
    "load_contract",
    "render_contract_toml",
    "select_resolution_draw_count",
    "validate_campaign_inventory",
    "validate_contract_hash",
    "validate_predictor_design",
    "wilson_upper_bound",
]
