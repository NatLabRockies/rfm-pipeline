#!/usr/bin/env python
"""BSM G0/B pre-execution controls and typed semi-synthetic DGP.

This module intentionally does not start calibration, scheduler, HPC,
production, holdout, or result-generation work.  It supplies the deterministic
contract, seed ledger, typed 160-field DGP, terminal-ledger validation, and
truthful pre-execution manifest required before those later gates can be
considered.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable, Mapping

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
_CONTRACT_PATH = (
    ROOT / "configs" / "rejected_history" / "method_contract_g0_development.yaml"
)
_DGP_CONTRACT_PATH = ROOT / "configs" / "bsm_dgp_contract.yaml"
_CONTROL_MANIFEST_PATH = ROOT / "docs" / "execution_control" / "control_manifest.json"

STAGE_NAMES = (
    "dgp",
    "noise_train",
    "noise_eval",
    "screening",
    "interaction",
    "nonlinear",
    "selection",
)

_CAMPAIGN_TO_BSM_SCENARIO = {
    "global_null": "global_null",
    "interaction_null_continuous": "interaction_null",
    "interaction_null_correlated": "correlated_interaction_null",
    "interaction_null_heteroscedastic": "heteroscedastic_interaction_null",
    "interaction_null_binary_main": "binary_main_interaction_null",
    "strong_cc": "sparse_strong_hierarchical",
    "strong_bc": "binary_continuous_strong",
    "strong_bb": "binary_binary_strong",
    "weak_signal": "weak_signal",
    "correlated_redundant": "correlated_redundant",
    "pure_interaction": "pure_interaction",
    "nonlinear_misspecified": "nonlinear_misspecified",
}


class ContractError(ValueError):
    """Raised when a contract or its pinned controls cannot be reconciled."""


class DGPValidationError(ValueError):
    """Raised when a generated replicate violates the frozen DGP."""


class TerminalLedgerError(RuntimeError):
    """Raised when planned terminal records are incomplete or inconsistent."""


class PreexecutionBlockedError(RuntimeError):
    """Raised when code attempts scientific execution while G0/A/B are open."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader which rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: yaml.SafeLoader, node: yaml.nodes.MappingNode, deep: bool = False
) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ContractError(f"duplicate contract field {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _require_mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{context} must be a mapping")
    return value


def _require_keys(
    value: Mapping[str, Any],
    *,
    context: str,
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    optional = optional or set()
    missing = required.difference(value)
    unknown = set(value).difference(required | optional)
    if missing:
        raise ContractError(f"{context} is missing required fields: {sorted(missing)}")
    if unknown:
        raise ContractError(f"{context} has unknown fields: {sorted(unknown)}")


def _require_hex(value: Any, context: str) -> str:
    text = str(value)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text.lower()):
        raise ContractError(f"{context} must be a SHA-256 hexadecimal digest")
    return text.lower()


def _repo_path(value: str, context: str) -> Path:
    path = (ROOT / value).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ContractError(f"{context} must remain inside the repository") from exc
    return path


def _load_yaml_text(text: str) -> Any:
    try:
        return yaml.load(text, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise ContractError(f"invalid contract YAML: {exc}") from exc


@dataclass(frozen=True)
class InputSchema:
    n_inputs: int
    continuous_inputs: int
    binary_inputs: int
    binary_input_names: tuple[str, ...]
    metadata_path: str
    all_binary_cells_required: bool


@dataclass(frozen=True)
class ScreeningControls:
    statistic: str
    bh_q: float
    permutation_count_B: int


@dataclass(frozen=True)
class InteractionControls:
    selection_method: str
    alpha: float
    permutation_count_B: int
    minimum_selection_draws: int
    tie_convention: str
    candidate_pair_order: str
    shared_response_row_schedule: bool


@dataclass(frozen=True)
class CalibrationControls:
    one_sided_confidence: float
    tolerance: float
    gate_upper_bound: float
    null_replicates_confirmatory: int
    operating_characteristic_rates: tuple[float, ...]
    statement: str


@dataclass(frozen=True)
class DevelopmentControls:
    replicates_per_scenario: int
    n_train: int
    n_eval: int
    n_outputs: int
    max_attempts: int
    retryable_statuses: tuple[str, ...]


@dataclass(frozen=True)
class DatasetScale:
    """Explicit row/output scale for one campaign scenario."""

    n_train: int
    n_eval: int
    n_outputs: int

    def __post_init__(self) -> None:
        if self.n_train < 4 or self.n_eval < 4 or self.n_outputs < 1:
            raise ValueError("dataset scale requires positive nontrivial dimensions")


@dataclass(frozen=True)
class DGPControls:
    continuous_distribution: str
    binary_distribution: str
    coded_continuous_range: tuple[float, float]
    binary_coding: tuple[float, float]
    intercept: float
    noise_base_sd: float
    heteroscedastic_driver_input_position: int
    heteroscedastic_minimum_multiplier: float
    heteroscedastic_maximum_multiplier: float
    heteroscedastic_function: str
    correlated_block_size: int
    correlated_latent_weight: float
    correlated_noise_weight: float
    strong_effect_floor: float
    transform_library: tuple[str, ...]


@dataclass(frozen=True)
class NondegeneracyControls:
    minimum_pair_family_size: int
    required_rate: float
    global_null_empty_family_allowed: bool
    fixed_family_sizes: tuple[int, ...]


@dataclass(frozen=True)
class TerminalControls:
    required_fields: tuple[str, ...]
    empty_family_outcome: str
    completed_status: str
    failed_status: str
    failure_action: str


@dataclass(frozen=True)
class ScenarioDefinition:
    name: str
    role: str
    n_continuous_main: int
    n_binary_main: int
    interaction_kind: str
    nonlinear_kind: str
    snr: float
    correlated_inputs: bool
    heteroscedastic: bool

    @property
    def is_null(self) -> bool:
        return self.role == "null"


@dataclass(frozen=True)
class ControlSnapshotRecord:
    path: Path
    snapshot_sha256: str
    gates: dict[str, str]
    phase: str


@dataclass(frozen=True)
class ExecutionContract:
    contract_path: Path
    source_sha256: str
    contract_sha256: str
    gate: str
    status: str
    phase: str
    control_snapshot_path: Path
    control_snapshot_sha256: str
    generic_repository: str
    generic_commit: str
    generic_require_clean_checkout: bool
    generic_lockfile: str
    generic_lockfile_sha256: str
    input_schema: InputSchema
    screening: ScreeningControls
    interaction: InteractionControls
    calibration: CalibrationControls
    development: DevelopmentControls
    dgp: DGPControls
    nondegeneracy: NondegeneracyControls
    terminal: TerminalControls
    scenarios: tuple[ScenarioDefinition, ...]
    seed_ledger_path: Path
    seed_ledger_sha256: str
    seed_derivation: str

    def scenario(self, name: str) -> ScenarioDefinition:
        for scenario in self.scenarios:
            if scenario.name == name:
                return scenario
        raise ContractError(f"unknown scenario {name!r}")

    def production_controls(self) -> dict[str, Any]:
        """Return the explicit controls a future pinned production adapter needs."""
        return {
            "phase": self.phase,
            "contract_sha256": self.contract_sha256,
            "control_snapshot_sha256": self.control_snapshot_sha256,
            "screening_statistic": self.screening.statistic,
            "screening_bh_q": self.screening.bh_q,
            "screening_permutation_count_B": self.screening.permutation_count_B,
            "selection_method": self.interaction.selection_method,
            "selection_alpha": self.interaction.alpha,
            "interaction_permutation_count_B": self.interaction.permutation_count_B,
            "minimum_selection_draws": self.interaction.minimum_selection_draws,
            "tie_convention": self.interaction.tie_convention,
        }


@dataclass(frozen=True)
class BSMDGPContract:
    """BSM-only DGP controls; contains no method, scale, seed, or scheduler settings."""

    path: Path
    source_sha256: str
    input_schema: InputSchema
    dgp: DGPControls
    scenarios: tuple[ScenarioDefinition, ...]
    phase: str = "campaign"

    def scenario(self, name: str) -> ScenarioDefinition:
        for scenario in self.scenarios:
            if scenario.name == name:
                return scenario
        raise ContractError(f"unknown BSM DGP scenario {name!r}")


def _parse_input_schema(payload: Mapping[str, Any]) -> InputSchema:
    _require_keys(
        payload,
        context="input_schema",
        required={
            "n_inputs",
            "continuous_inputs",
            "binary_inputs",
            "binary_input_names",
            "metadata_path",
            "all_binary_cells_required",
        },
    )
    names = tuple(str(name) for name in payload["binary_input_names"])
    result = InputSchema(
        n_inputs=int(payload["n_inputs"]),
        continuous_inputs=int(payload["continuous_inputs"]),
        binary_inputs=int(payload["binary_inputs"]),
        binary_input_names=names,
        metadata_path=str(payload["metadata_path"]),
        all_binary_cells_required=bool(payload["all_binary_cells_required"]),
    )
    if result.n_inputs != result.continuous_inputs + result.binary_inputs:
        raise ContractError("input_schema counts do not sum to n_inputs")
    if (
        result.n_inputs != 160
        or result.continuous_inputs != 158
        or result.binary_inputs != 2
    ):
        raise ContractError("the G0/B contract requires the typed 160-column interface")
    if len(result.binary_input_names) != result.binary_inputs:
        raise ContractError("input_schema binary names do not match binary_inputs")
    return result


def _parse_scenarios(raw_scenarios: Any) -> tuple[ScenarioDefinition, ...]:
    if not isinstance(raw_scenarios, list) or not raw_scenarios:
        raise ContractError("scenarios must be a non-empty list")
    required = {
        "name",
        "role",
        "n_continuous_main",
        "n_binary_main",
        "interaction_kind",
        "nonlinear_kind",
        "snr",
        "correlated_inputs",
        "heteroscedastic",
    }
    scenarios: list[ScenarioDefinition] = []
    for raw in raw_scenarios:
        value = _require_mapping(raw, "scenario")
        _require_keys(
            value,
            context=f"scenario {value.get('name', '<unknown>')!r}",
            required=required,
        )
        scenario = ScenarioDefinition(
            name=str(value["name"]),
            role=str(value["role"]),
            n_continuous_main=int(value["n_continuous_main"]),
            n_binary_main=int(value["n_binary_main"]),
            interaction_kind=str(value["interaction_kind"]),
            nonlinear_kind=str(value["nonlinear_kind"]),
            snr=float(value["snr"]),
            correlated_inputs=bool(value["correlated_inputs"]),
            heteroscedastic=bool(value["heteroscedastic"]),
        )
        if scenario.role not in {"null", "strong_alternative", "stress"}:
            raise ContractError(f"scenario {scenario.name!r} has an invalid role")
        if scenario.interaction_kind not in {
            "none",
            "continuous_continuous",
            "binary_continuous",
            "binary_binary",
        }:
            raise ContractError(
                f"scenario {scenario.name!r} has an invalid interaction kind"
            )
        if scenario.nonlinear_kind not in {"none", "quadratic", "sine"}:
            raise ContractError(
                f"scenario {scenario.name!r} has an invalid nonlinear kind"
            )
        if scenario.snr <= 0.0:
            raise ContractError(f"scenario {scenario.name!r} must have positive SNR")
        scenarios.append(scenario)
    names = [scenario.name for scenario in scenarios]
    if len(names) != len(set(names)):
        raise ContractError("scenario names must be unique")
    required_names = {
        "global_null",
        "interaction_null",
        "correlated_interaction_null",
        "heteroscedastic_interaction_null",
        "binary_main_interaction_null",
        "sparse_strong_hierarchical",
        "binary_continuous_strong",
        "binary_binary_strong",
    }
    if not required_names.issubset(names):
        raise ContractError("scenario grid omits a required G0/B regime")
    return tuple(scenarios)


def _parse_dgp_payload(dgp_raw: Mapping[str, Any]) -> DGPControls:
    _require_keys(
        dgp_raw,
        context="dgp",
        required={
            "continuous_distribution",
            "binary_distribution",
            "coded_continuous_range",
            "binary_coding",
            "intercept",
            "noise_base_sd",
            "row_heteroscedasticity",
            "correlated_block",
            "strong_effect_floor",
            "transform_library",
        },
    )
    hetero = _require_mapping(
        dgp_raw["row_heteroscedasticity"], "dgp.row_heteroscedasticity"
    )
    _require_keys(
        hetero,
        context="dgp.row_heteroscedasticity",
        required={
            "driver_input_position",
            "minimum_multiplier",
            "maximum_multiplier",
            "function",
        },
    )
    correlated = _require_mapping(dgp_raw["correlated_block"], "dgp.correlated_block")
    _require_keys(
        correlated,
        context="dgp.correlated_block",
        required={"size", "latent_weight", "noise_weight"},
    )
    result = DGPControls(
        continuous_distribution=str(dgp_raw["continuous_distribution"]),
        binary_distribution=str(dgp_raw["binary_distribution"]),
        coded_continuous_range=tuple(
            float(v) for v in dgp_raw["coded_continuous_range"]
        ),
        binary_coding=tuple(float(v) for v in dgp_raw["binary_coding"]),
        intercept=float(dgp_raw["intercept"]),
        noise_base_sd=float(dgp_raw["noise_base_sd"]),
        heteroscedastic_driver_input_position=int(hetero["driver_input_position"]),
        heteroscedastic_minimum_multiplier=float(hetero["minimum_multiplier"]),
        heteroscedastic_maximum_multiplier=float(hetero["maximum_multiplier"]),
        heteroscedastic_function=str(hetero["function"]),
        correlated_block_size=int(correlated["size"]),
        correlated_latent_weight=float(correlated["latent_weight"]),
        correlated_noise_weight=float(correlated["noise_weight"]),
        strong_effect_floor=float(dgp_raw["strong_effect_floor"]),
        transform_library=tuple(str(value) for value in dgp_raw["transform_library"]),
    )
    if (
        result.continuous_distribution != "uniform_published_range"
        or result.binary_distribution != "balanced_four_cell"
        or result.coded_continuous_range != (-1.0, 1.0)
        or result.binary_coding != (-1.0, 1.0)
        or result.noise_base_sd <= 0.0
        or result.heteroscedastic_minimum_multiplier <= 0.0
        or result.heteroscedastic_maximum_multiplier
        <= result.heteroscedastic_minimum_multiplier
        or result.strong_effect_floor <= 0.0
    ):
        raise ContractError("DGP controls are not valid for the frozen typed design")
    return result


def load_bsm_dgp_contract(path: Path | str = _DGP_CONTRACT_PATH) -> BSMDGPContract:
    """Load the production BSM DGP-only contract and reject scientific duplicates."""
    resolved = Path(path)
    if not resolved.is_file():
        raise ContractError(f"BSM DGP contract does not exist: {resolved}")
    text = resolved.read_text(encoding="utf-8")
    raw = _require_mapping(_load_yaml_text(text), "BSM DGP contract")
    _require_keys(
        raw,
        context="BSM DGP contract",
        required={"schema_version", "input_schema", "dgp", "scenarios"},
    )
    if int(raw["schema_version"]) != 1:
        raise ContractError("unsupported BSM DGP contract schema_version")
    forbidden = {
        "screening",
        "interaction",
        "calibration",
        "development",
        "seed_ledger",
        "scheduler",
        "replicates",
    }
    if forbidden.intersection(raw):
        raise ContractError("BSM DGP contract contains duplicated campaign controls")
    return BSMDGPContract(
        path=resolved,
        source_sha256=_sha256_bytes(text.encode("utf-8")),
        input_schema=_parse_input_schema(
            _require_mapping(raw["input_schema"], "input_schema")
        ),
        dgp=_parse_dgp_payload(_require_mapping(raw["dgp"], "dgp")),
        scenarios=_parse_scenarios(raw["scenarios"]),
    )


def load_execution_contract_text(
    text: str, *, contract_path: Path = _CONTRACT_PATH
) -> ExecutionContract:
    """Parse one canonical contract without defaults or duplicate keys."""
    root = _require_mapping(_load_yaml_text(text), "contract")
    _require_keys(
        root,
        context="contract",
        required={
            "schema_version",
            "gate",
            "status",
            "phase",
            "control_snapshot",
            "pinned_generic_source",
            "input_schema",
            "screening",
            "interaction",
            "calibration",
            "development",
            "dgp",
            "nondegeneracy",
            "terminal_records",
            "scenarios",
            "seed_ledger",
        },
    )
    if int(root["schema_version"]) != 2:
        raise ContractError("unsupported contract schema_version")
    if (
        root["gate"] != "G0"
        or root["status"] != "OPEN"
        or root["phase"] != "development"
    ):
        raise ContractError("only the G0 OPEN development contract may be loaded")

    control = _require_mapping(root["control_snapshot"], "control_snapshot")
    _require_keys(control, context="control_snapshot", required={"path", "sha256"})
    control_path = _repo_path(str(control["path"]), "control_snapshot.path")
    control_sha256 = _require_hex(control["sha256"], "control_snapshot.sha256")

    generic = _require_mapping(root["pinned_generic_source"], "pinned_generic_source")
    _require_keys(
        generic,
        context="pinned_generic_source",
        required={
            "repository",
            "commit",
            "require_clean_checkout",
            "lockfile",
            "lockfile_sha256",
        },
    )
    generic_commit = str(generic["commit"])
    if len(generic_commit) != 40 or any(
        char not in "0123456789abcdef" for char in generic_commit
    ):
        raise ContractError("pinned_generic_source.commit must be a full Git SHA")
    generic_lockfile = str(generic["lockfile"])
    generic_lockfile_path = _repo_path(
        generic_lockfile, "pinned_generic_source.lockfile"
    )
    generic_lockfile_sha256 = _require_hex(
        generic["lockfile_sha256"],
        "pinned_generic_source.lockfile_sha256",
    )
    if (
        not generic_lockfile_path.is_file()
        or _sha256_path(generic_lockfile_path) != generic_lockfile_sha256
    ):
        raise ContractError(
            "pinned_generic_source lockfile checksum differs from the contract"
        )

    screening_raw = _require_mapping(root["screening"], "screening")
    _require_keys(
        screening_raw,
        context="screening",
        required={"statistic", "bh_q", "permutation_count_B"},
    )
    screening = ScreeningControls(
        statistic=str(screening_raw["statistic"]),
        bh_q=float(screening_raw["bh_q"]),
        permutation_count_B=int(screening_raw["permutation_count_B"]),
    )
    if screening.statistic != "coefficient_row_l2_norm" or screening.bh_q != 0.05:
        raise ContractError(
            "screening controls must use the frozen statistic and q=0.05"
        )
    if screening.permutation_count_B < 3199:
        raise ContractError("screening permutation_count_B must be at least 3199")

    interaction_raw = _require_mapping(root["interaction"], "interaction")
    _require_keys(
        interaction_raw,
        context="interaction",
        required={
            "selection_method",
            "alpha",
            "permutation_count_B",
            "minimum_selection_draws",
            "tie_convention",
            "candidate_pair_order",
            "shared_response_row_schedule",
        },
    )
    interaction = InteractionControls(
        selection_method=str(interaction_raw["selection_method"]),
        alpha=float(interaction_raw["alpha"]),
        permutation_count_B=int(interaction_raw["permutation_count_B"]),
        minimum_selection_draws=int(interaction_raw["minimum_selection_draws"]),
        tie_convention=str(interaction_raw["tie_convention"]),
        candidate_pair_order=str(interaction_raw["candidate_pair_order"]),
        shared_response_row_schedule=bool(
            interaction_raw["shared_response_row_schedule"]
        ),
    )
    if interaction.selection_method != "max_stat_adjusted_p_mc":
        raise ContractError(
            "interaction selection_method must use the neutral canonical label"
        )
    if interaction.alpha != 0.05 or interaction.tie_convention != ">=":
        raise ContractError(
            "interaction alpha or tie convention differs from the G0 contract"
        )
    if interaction.permutation_count_B < interaction.minimum_selection_draws:
        raise ContractError("interaction draw count is below its declared minimum")
    if interaction.permutation_count_B < 999:
        raise ContractError("interaction permutation_count_B must be at least 999")
    if 1.0 / (interaction.permutation_count_B + 1) > interaction.alpha:
        raise ContractError(
            "interaction draw count cannot resolve the configured alpha"
        )
    if not interaction.shared_response_row_schedule:
        raise ContractError(
            "interaction schedule must be shared across the candidate family"
        )

    calibration_raw = _require_mapping(root["calibration"], "calibration")
    _require_keys(
        calibration_raw,
        context="calibration",
        required={
            "one_sided_confidence",
            "tolerance",
            "gate_upper_bound",
            "null_replicates_confirmatory",
            "operating_characteristic_rates",
            "statement",
        },
    )
    calibration = CalibrationControls(
        one_sided_confidence=float(calibration_raw["one_sided_confidence"]),
        tolerance=float(calibration_raw["tolerance"]),
        gate_upper_bound=float(calibration_raw["gate_upper_bound"]),
        null_replicates_confirmatory=int(
            calibration_raw["null_replicates_confirmatory"]
        ),
        operating_characteristic_rates=tuple(
            float(rate) for rate in calibration_raw["operating_characteristic_rates"]
        ),
        statement=str(calibration_raw["statement"]),
    )
    if calibration.one_sided_confidence != 0.95:
        raise ContractError("calibration must use a one-sided 95% Wilson bound")
    if calibration.gate_upper_bound != interaction.alpha + calibration.tolerance:
        raise ContractError(
            "calibration gate_upper_bound must equal alpha plus tolerance"
        )
    if calibration.null_replicates_confirmatory < 1000:
        raise ContractError(
            "confirmatory null ledger must contain at least 1000 replicates"
        )

    development_raw = _require_mapping(root["development"], "development")
    _require_keys(
        development_raw,
        context="development",
        required={
            "replicates_per_scenario",
            "n_train",
            "n_eval",
            "n_outputs",
            "retry_policy",
        },
    )
    retry = _require_mapping(
        development_raw["retry_policy"], "development.retry_policy"
    )
    _require_keys(
        retry,
        context="development.retry_policy",
        required={"max_attempts", "retryable_statuses"},
    )
    development = DevelopmentControls(
        replicates_per_scenario=int(development_raw["replicates_per_scenario"]),
        n_train=int(development_raw["n_train"]),
        n_eval=int(development_raw["n_eval"]),
        n_outputs=int(development_raw["n_outputs"]),
        max_attempts=int(retry["max_attempts"]),
        retryable_statuses=tuple(str(value) for value in retry["retryable_statuses"]),
    )
    if (
        development.replicates_per_scenario < 2
        or development.n_train < 4
        or development.n_eval < 4
        or development.n_outputs < 1
        or development.max_attempts != 1
        or development.retryable_statuses
    ):
        raise ContractError("development controls violate the frozen no-retry control")

    dgp_raw = _require_mapping(root["dgp"], "dgp")
    _require_keys(
        dgp_raw,
        context="dgp",
        required={
            "continuous_distribution",
            "binary_distribution",
            "coded_continuous_range",
            "binary_coding",
            "intercept",
            "noise_base_sd",
            "row_heteroscedasticity",
            "correlated_block",
            "strong_effect_floor",
            "transform_library",
        },
    )
    hetero = _require_mapping(
        dgp_raw["row_heteroscedasticity"], "dgp.row_heteroscedasticity"
    )
    _require_keys(
        hetero,
        context="dgp.row_heteroscedasticity",
        required={
            "driver_input_position",
            "minimum_multiplier",
            "maximum_multiplier",
            "function",
        },
    )
    correlated = _require_mapping(dgp_raw["correlated_block"], "dgp.correlated_block")
    _require_keys(
        correlated,
        context="dgp.correlated_block",
        required={"size", "latent_weight", "noise_weight"},
    )
    dgp = DGPControls(
        continuous_distribution=str(dgp_raw["continuous_distribution"]),
        binary_distribution=str(dgp_raw["binary_distribution"]),
        coded_continuous_range=tuple(
            float(v) for v in dgp_raw["coded_continuous_range"]
        ),
        binary_coding=tuple(float(v) for v in dgp_raw["binary_coding"]),
        intercept=float(dgp_raw["intercept"]),
        noise_base_sd=float(dgp_raw["noise_base_sd"]),
        heteroscedastic_driver_input_position=int(hetero["driver_input_position"]),
        heteroscedastic_minimum_multiplier=float(hetero["minimum_multiplier"]),
        heteroscedastic_maximum_multiplier=float(hetero["maximum_multiplier"]),
        heteroscedastic_function=str(hetero["function"]),
        correlated_block_size=int(correlated["size"]),
        correlated_latent_weight=float(correlated["latent_weight"]),
        correlated_noise_weight=float(correlated["noise_weight"]),
        strong_effect_floor=float(dgp_raw["strong_effect_floor"]),
        transform_library=tuple(str(value) for value in dgp_raw["transform_library"]),
    )
    if (
        dgp.continuous_distribution != "uniform_published_range"
        or dgp.binary_distribution != "balanced_four_cell"
        or dgp.coded_continuous_range != (-1.0, 1.0)
        or dgp.binary_coding != (-1.0, 1.0)
        or dgp.noise_base_sd <= 0.0
        or dgp.heteroscedastic_minimum_multiplier <= 0.0
        or dgp.heteroscedastic_maximum_multiplier
        <= dgp.heteroscedastic_minimum_multiplier
        or dgp.strong_effect_floor <= 0.0
    ):
        raise ContractError("DGP controls are not valid for the frozen typed design")

    nondegeneracy_raw = _require_mapping(root["nondegeneracy"], "nondegeneracy")
    _require_keys(
        nondegeneracy_raw,
        context="nondegeneracy",
        required={
            "minimum_pair_family_size",
            "required_rate",
            "global_null_empty_family_allowed",
            "fixed_family_sizes",
        },
    )
    nondegeneracy = NondegeneracyControls(
        minimum_pair_family_size=int(nondegeneracy_raw["minimum_pair_family_size"]),
        required_rate=float(nondegeneracy_raw["required_rate"]),
        global_null_empty_family_allowed=bool(
            nondegeneracy_raw["global_null_empty_family_allowed"]
        ),
        fixed_family_sizes=tuple(
            int(value) for value in nondegeneracy_raw["fixed_family_sizes"]
        ),
    )
    if (
        nondegeneracy.minimum_pair_family_size != 2
        or nondegeneracy.required_rate != 0.90
        or nondegeneracy.fixed_family_sizes != (10, 100, 1000)
    ):
        raise ContractError("nondegeneracy controls differ from the G0 contract")

    terminal_raw = _require_mapping(root["terminal_records"], "terminal_records")
    _require_keys(
        terminal_raw,
        context="terminal_records",
        required={
            "required_fields",
            "empty_family_outcome",
            "completed_status",
            "failed_status",
            "failure_action",
        },
    )
    terminal = TerminalControls(
        required_fields=tuple(str(value) for value in terminal_raw["required_fields"]),
        empty_family_outcome=str(terminal_raw["empty_family_outcome"]),
        completed_status=str(terminal_raw["completed_status"]),
        failed_status=str(terminal_raw["failed_status"]),
        failure_action=str(terminal_raw["failure_action"]),
    )
    if (
        terminal.empty_family_outcome != "NO_INTERACTION_CANDIDATES"
        or terminal.completed_status != "ANALYSIS_COMPLETE"
        or terminal.failed_status != "FAILED"
        or terminal.failure_action != "exit_nonzero"
    ):
        raise ContractError("terminal-record controls differ from the G0/B contract")

    seed_raw = _require_mapping(root["seed_ledger"], "seed_ledger")
    _require_keys(
        seed_raw, context="seed_ledger", required={"path", "sha256", "derivation"}
    )
    seed_ledger_path = _repo_path(str(seed_raw["path"]), "seed_ledger.path")
    seed_ledger_sha256 = _require_hex(seed_raw["sha256"], "seed_ledger.sha256")
    seed_derivation = str(seed_raw["derivation"])
    if seed_derivation != "sha256_contract_phase_scenario_replicate_stage":
        raise ContractError("seed ledger uses an unrecognized derivation")

    identity_payload = {
        key: value for key, value in root.items() if key != "seed_ledger"
    }
    identity_payload["seed_ledger"] = {
        "path": str(seed_raw["path"]),
        "derivation": str(seed_raw["derivation"]),
    }
    contract_sha256 = _sha256_bytes(_canonical_json(identity_payload))
    return ExecutionContract(
        contract_path=contract_path,
        source_sha256=_sha256_bytes(text.encode("utf-8")),
        contract_sha256=contract_sha256,
        gate=str(root["gate"]),
        status=str(root["status"]),
        phase=str(root["phase"]),
        control_snapshot_path=control_path,
        control_snapshot_sha256=control_sha256,
        generic_repository=str(generic["repository"]),
        generic_commit=generic_commit,
        generic_require_clean_checkout=bool(generic["require_clean_checkout"]),
        generic_lockfile=generic_lockfile,
        generic_lockfile_sha256=generic_lockfile_sha256,
        input_schema=_parse_input_schema(
            _require_mapping(root["input_schema"], "input_schema")
        ),
        screening=screening,
        interaction=interaction,
        calibration=calibration,
        development=development,
        dgp=dgp,
        nondegeneracy=nondegeneracy,
        terminal=terminal,
        scenarios=_parse_scenarios(root["scenarios"]),
        seed_ledger_path=seed_ledger_path,
        seed_ledger_sha256=seed_ledger_sha256,
        seed_derivation=seed_derivation,
    )


def load_execution_contract(path: Path | str = _CONTRACT_PATH) -> ExecutionContract:
    """Load the only BSM scientific-settings source."""
    contract_path = Path(path)
    if not contract_path.is_file():
        raise ContractError(f"contract file does not exist: {contract_path}")
    return load_execution_contract_text(
        contract_path.read_text(encoding="utf-8"), contract_path=contract_path
    )


def verify_pinned_control_snapshot(
    contract: ExecutionContract,
) -> ControlSnapshotRecord:
    """Verify the tracked control snapshot and manifest before any DGP work."""
    if not contract.control_snapshot_path.is_file():
        raise ContractError("pinned control snapshot is absent")
    if _sha256_path(contract.control_snapshot_path) != contract.control_snapshot_sha256:
        raise ContractError(
            "pinned control snapshot checksum differs from the contract"
        )
    if not _CONTROL_MANIFEST_PATH.is_file():
        raise ContractError("control manifest is absent")
    payload = json.loads(_CONTROL_MANIFEST_PATH.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "control_snapshot",
        "control_snapshot_sha256",
        "source_document",
        "source_sections",
        "gates",
        "phase",
        "execution_permitted",
    }
    if set(payload) != required:
        raise ContractError("control manifest fields do not match the pinned schema")
    manifest_path = _CONTROL_MANIFEST_PATH.parent / str(payload["control_snapshot"])
    if manifest_path.resolve() != contract.control_snapshot_path.resolve():
        raise ContractError("control manifest points to a different snapshot")
    manifest_sha = _require_hex(
        payload["control_snapshot_sha256"], "control manifest snapshot"
    )
    if manifest_sha != contract.control_snapshot_sha256:
        raise ContractError("control manifest and contract snapshot hashes differ")
    gates = payload["gates"]
    if gates != {"G0": "OPEN", "A": "OPEN", "B": "OPEN"}:
        raise ContractError("control manifest must keep G0, A, and B open")
    if (
        payload["phase"] != contract.phase
        or payload["execution_permitted"] is not False
    ):
        raise ContractError("control manifest permits an unapproved execution phase")
    return ControlSnapshotRecord(
        path=contract.control_snapshot_path,
        snapshot_sha256=manifest_sha,
        gates=dict(gates),
        phase=str(payload["phase"]),
    )


@dataclass(frozen=True)
class SeedLedgerEntry:
    phase: str
    scenario: str
    replicate_index: int
    seed: int
    stage_seeds: dict[str, int]

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.phase, self.scenario, self.replicate_index)


@dataclass(frozen=True)
class SeedLedger:
    path: Path
    sha256: str
    contract_sha256: str
    phase: str
    entries: tuple[SeedLedgerEntry, ...]


def _seed_from_parts(*parts: object) -> int:
    digest = hashlib.sha256(
        "\x1f".join(str(part) for part in parts).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**31 - 1) + 1


def derive_replicate_seed(
    contract_sha256: str,
    phase: str,
    scenario: str,
    replicate_index: int,
) -> int:
    """Derive a scenario-keyed seed without iteration-order dependence."""
    return _seed_from_parts(
        contract_sha256, phase, scenario, replicate_index, "replicate"
    )


def derive_stage_seed(
    contract_sha256: str,
    phase: str,
    scenario: str,
    replicate_index: int,
    stage: str,
) -> int:
    """Derive a named stage seed from the same immutable identity."""
    if stage not in STAGE_NAMES:
        raise ContractError(f"unknown stage seed name {stage!r}")
    return _seed_from_parts(contract_sha256, phase, scenario, replicate_index, stage)


def load_seed_ledger(contract: ExecutionContract) -> SeedLedger:
    """Load and verify every planned development replicate before it can run."""
    path = contract.seed_ledger_path
    if not path.is_file():
        raise ContractError("seed ledger is absent")
    actual_sha = _sha256_path(path)
    if actual_sha != contract.seed_ledger_sha256:
        raise ContractError("seed ledger checksum differs from the contract")
    raw = _require_mapping(
        _load_yaml_text(path.read_text(encoding="utf-8")), "seed ledger"
    )
    _require_keys(
        raw,
        context="seed ledger",
        required={"schema_version", "phase", "contract_sha256", "entries"},
    )
    if int(raw["schema_version"]) != 1 or raw["phase"] != contract.phase:
        raise ContractError("seed ledger schema or phase differs from the contract")
    if raw["contract_sha256"] != contract.contract_sha256:
        raise ContractError("seed ledger is bound to a different contract")
    if not isinstance(raw["entries"], list):
        raise ContractError("seed ledger entries must be a list")
    entries: list[SeedLedgerEntry] = []
    for raw_entry in raw["entries"]:
        entry = _require_mapping(raw_entry, "seed ledger entry")
        _require_keys(
            entry,
            context="seed ledger entry",
            required={"phase", "scenario", "replicate_index"},
        )
        if entry["phase"] != contract.phase:
            raise ContractError("seed ledger entry phase differs from the contract")
        scenario = str(entry["scenario"])
        replicate_index = int(entry["replicate_index"])
        if replicate_index < 0:
            raise ContractError("seed ledger replicate_index must be non-negative")
        contract.scenario(scenario)
        entries.append(
            SeedLedgerEntry(
                phase=contract.phase,
                scenario=scenario,
                replicate_index=replicate_index,
                seed=derive_replicate_seed(
                    contract.contract_sha256,
                    contract.phase,
                    scenario,
                    replicate_index,
                ),
                stage_seeds={
                    stage: derive_stage_seed(
                        contract.contract_sha256,
                        contract.phase,
                        scenario,
                        replicate_index,
                        stage,
                    )
                    for stage in STAGE_NAMES
                },
            )
        )
    keys = [entry.key for entry in entries]
    if len(keys) != len(set(keys)):
        raise ContractError("seed ledger contains duplicate scenario/replicate keys")
    expected = {
        (contract.phase, scenario.name, replicate_index)
        for scenario in contract.scenarios
        for replicate_index in range(contract.development.replicates_per_scenario)
    }
    actual = set(keys)
    if actual != expected:
        missing = sorted(expected.difference(actual))
        extra = sorted(actual.difference(expected))
        raise ContractError(
            "seed ledger does not match planned replicates; "
            f"missing={missing}; extra={extra}"
        )
    return SeedLedger(
        path=path,
        sha256=actual_sha,
        contract_sha256=contract.contract_sha256,
        phase=contract.phase,
        entries=tuple(entries),
    )


@dataclass(frozen=True)
class BSMInputDesign:
    """One typed BSM input interface used by every recovery regime."""

    input_names: tuple[str, ...]
    continuous_input_names: tuple[str, ...]
    binary_input_names: tuple[str, ...]
    lower: dict[str, float]
    upper: dict[str, float]

    @property
    def n_inputs(self) -> int:
        return len(self.input_names)

    @property
    def schema_sha256(self) -> str:
        return _sha256_bytes(
            _canonical_json(
                {
                    "input_names": self.input_names,
                    "continuous_input_names": self.continuous_input_names,
                    "binary_input_names": self.binary_input_names,
                    "lower": self.lower,
                    "upper": self.upper,
                }
            )
        )


def load_bsm_input_design(
    metadata_path: Path | str | None = None,
    *,
    contract: ExecutionContract | None = None,
) -> BSMInputDesign:
    """Load and validate the single 160-column schema from published metadata."""
    contract = contract or load_execution_contract()
    path = (
        Path(metadata_path)
        if metadata_path is not None
        else _repo_path(
            contract.input_schema.metadata_path, "input_schema.metadata_path"
        )
    )
    if not path.is_file():
        raise ContractError(f"input metadata does not exist: {path}")
    meta = _require_mapping(
        _load_yaml_text(path.read_text(encoding="utf-8")), "input metadata"
    )
    entries = meta.get("inputs")
    if not isinstance(entries, list):
        raise ContractError("input metadata must contain an inputs list")
    names: list[str] = []
    continuous: list[str] = []
    binary: list[str] = []
    lower: dict[str, float] = {}
    upper: dict[str, float] = {}
    for raw in entries:
        item = _require_mapping(raw, "input metadata entry")
        name = str(item["name"])
        if name in names:
            raise ContractError(f"input metadata contains duplicate name {name!r}")
        lo = float(item.get("min_sample_value", 0.0))
        hi = float(item.get("max_sample_value", 1.0))
        if not hi > lo:
            raise ContractError(f"input metadata range is invalid for {name!r}")
        names.append(name)
        lower[name] = lo
        upper[name] = hi
        if bool(item.get("is_binary_scenario", False)):
            binary.append(name)
        else:
            continuous.append(name)
    design = BSMInputDesign(
        input_names=tuple(names),
        continuous_input_names=tuple(continuous),
        binary_input_names=tuple(binary),
        lower=lower,
        upper=upper,
    )
    if (
        design.n_inputs != contract.input_schema.n_inputs
        or len(design.continuous_input_names) != contract.input_schema.continuous_inputs
        or design.binary_input_names != contract.input_schema.binary_input_names
    ):
        raise ContractError(
            "published metadata does not reconcile with the typed contract schema"
        )
    return design


def canonical_pair_family(feature_names: Iterable[str]) -> tuple[str, ...]:
    """Return the sole lexicographic unordered-pair representation."""
    names = tuple(sorted(str(name) for name in feature_names))
    if len(names) != len(set(names)):
        raise ContractError("candidate family contains duplicate feature names")
    return tuple(
        f"{left}:{right}"
        for index, left in enumerate(names)
        for right in names[index + 1 :]
    )


def _balanced_binary_cells(n_rows: int, rng: np.random.Generator) -> np.ndarray:
    if n_rows < 4:
        raise DGPValidationError("all four binary cells require at least four rows")
    cells = np.array(((0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)), dtype=float)
    repeats, remainder = divmod(n_rows, len(cells))
    result = np.vstack([np.tile(cells, (repeats, 1)), cells[:remainder]])
    rng.shuffle(result, axis=0)
    return result


def _sample_inputs(
    design: BSMInputDesign,
    n_rows: int,
    rng: np.random.Generator,
    *,
    correlated: bool,
    contract: ExecutionContract,
) -> tuple[np.ndarray, dict[tuple[int, int], int], dict[str, Any]]:
    """Sample all continuous and binary columns under the frozen DGP."""
    X = np.empty((n_rows, design.n_inputs), dtype=float)
    index = {name: position for position, name in enumerate(design.input_names)}
    binary_cells = _balanced_binary_cells(n_rows, rng)
    for position, name in enumerate(design.binary_input_names):
        X[:, index[name]] = binary_cells[:, position]
    continuous = design.continuous_input_names
    correlated_positions: list[int] = []
    if correlated:
        block_size = min(contract.dgp.correlated_block_size, len(continuous))
        latent = rng.uniform(-1.0, 1.0, size=n_rows)
        for position, name in enumerate(continuous[:block_size]):
            raw = (
                contract.dgp.correlated_latent_weight * latent
                + contract.dgp.correlated_noise_weight
                * rng.uniform(-1.0, 1.0, size=n_rows)
            )
            X[:, index[name]] = design.lower[name] + (raw + 1.0) * 0.5 * (
                design.upper[name] - design.lower[name]
            )
            correlated_positions.append(index[name])
    else:
        block_size = 0
    for name in continuous[block_size:]:
        X[:, index[name]] = rng.uniform(
            design.lower[name], design.upper[name], size=n_rows
        )
    counts = {
        cell: int(np.sum(np.all(binary_cells == np.array(cell, dtype=float), axis=1)))
        for cell in ((0, 0), (0, 1), (1, 0), (1, 1))
    }
    correlation: dict[str, Any] = {}
    if correlated_positions:
        block = X[:, correlated_positions]
        ranks = np.argsort(np.argsort(block, axis=0), axis=0).astype(float)
        matrix = np.corrcoef(ranks, rowvar=False)
        upper = matrix[np.triu_indices_from(matrix, k=1)]
        correlation = {
            "block_size": len(correlated_positions),
            "mean_pairwise_rank_correlation": float(np.mean(upper)),
        }
    return X, counts, correlation


def _coded_features(design: BSMInputDesign, X: np.ndarray) -> np.ndarray:
    """Use fixed metadata coding, never a split-specific sample center."""
    coded = np.empty_like(X, dtype=float)
    binary = set(design.binary_input_names)
    for position, name in enumerate(design.input_names):
        if name in binary:
            coded[:, position] = 2.0 * X[:, position] - 1.0
        else:
            midpoint = 0.5 * (design.lower[name] + design.upper[name])
            half_range = 0.5 * (design.upper[name] - design.lower[name])
            coded[:, position] = (X[:, position] - midpoint) / half_range
    return coded


@dataclass(frozen=True)
class TermId:
    kind: str
    inputs: tuple[str, ...]
    transform: str | None = None

    @property
    def canonical_id(self) -> str:
        suffix = f":{self.transform}" if self.transform else ""
        return f"{self.kind}:{'|'.join(self.inputs)}{suffix}"


@dataclass(frozen=True)
class TruthTerm:
    term_id: TermId
    coefficient: float
    output_loadings: tuple[float, ...]
    in_library: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "term_id": self.term_id.canonical_id,
            "kind": self.term_id.kind,
            "inputs": list(self.term_id.inputs),
            "transform": self.term_id.transform,
            "coefficient": self.coefficient,
            "output_loadings": list(self.output_loadings),
            "in_library": self.in_library,
        }


@dataclass(frozen=True)
class DGPTruth:
    """Typed generative basis shared exactly by train and evaluation rows."""

    input_schema_sha256: str
    intercept: tuple[float, ...]
    terms: tuple[TruthTerm, ...]
    surface_sha256: str

    def _term_column(
        self, design: BSMInputDesign, coded: np.ndarray, term: TruthTerm
    ) -> np.ndarray:
        index = {name: position for position, name in enumerate(design.input_names)}
        values = [coded[:, index[name]] for name in term.term_id.inputs]
        if term.term_id.kind == "main":
            return values[0]
        if term.term_id.kind in {
            "continuous_continuous_interaction",
            "binary_continuous_interaction",
            "binary_binary_interaction",
        }:
            return values[0] * values[1]
        if term.term_id.kind == "transformation":
            if term.term_id.transform == "quadratic":
                return values[0] ** 2
            if term.term_id.transform == "sine":
                return np.sin(3.0 * values[0])
        raise DGPValidationError(
            f"unsupported typed truth term {term.term_id.canonical_id!r}"
        )

    def evaluate(self, design: BSMInputDesign, X: np.ndarray) -> np.ndarray:
        if design.schema_sha256 != self.input_schema_sha256:
            raise DGPValidationError(
                "truth ledger was bound to a different input schema"
            )
        coded = _coded_features(design, X)
        response = np.tile(np.asarray(self.intercept, dtype=float), (X.shape[0], 1))
        for term in self.terms:
            response += (
                term.coefficient
                * self._term_column(design, coded, term)[:, None]
                * np.asarray(term.output_loadings, dtype=float)[None, :]
            )
        return response

    def to_records(self) -> list[dict[str, Any]]:
        return [term.to_dict() for term in self.terms]

    @property
    def interaction_ids(self) -> tuple[str, ...]:
        return tuple(
            term.term_id.canonical_id
            for term in self.terms
            if term.term_id.kind.endswith("_interaction")
        )

    def in_library_design(self, design: BSMInputDesign, X: np.ndarray) -> np.ndarray:
        """Return the exact planted in-library columns for the oracle OLS comparator."""
        coded = _coded_features(design, X)
        columns = [
            self._term_column(design, coded, term)
            for term in self.terms
            if term.in_library
        ]
        if not columns:
            return np.empty((X.shape[0], 0), dtype=float)
        return np.column_stack(columns).astype(float, copy=False)


def _loadings(n_outputs: int, *, sign: float = 1.0) -> tuple[float, ...]:
    return tuple(float(sign * value) for value in np.linspace(1.0, 1.25, n_outputs))


def _typed_term(
    kind: str,
    inputs: Iterable[str],
    *,
    coefficient: float,
    n_outputs: int,
    transform: str | None = None,
    in_library: bool = True,
) -> TruthTerm:
    return TruthTerm(
        term_id=TermId(kind=kind, inputs=tuple(inputs), transform=transform),
        coefficient=float(coefficient),
        output_loadings=_loadings(n_outputs, sign=1.0 if coefficient >= 0.0 else -1.0),
        in_library=in_library,
    )


def _build_truth(
    contract: ExecutionContract,
    design: BSMInputDesign,
    scenario: ScenarioDefinition,
    entry: SeedLedgerEntry,
    *,
    n_outputs: int,
) -> DGPTruth:
    """Build a deterministic typed basis before sampling response noise."""
    del (
        entry
    )  # Identity remains represented by the ledger; coefficients are contract-fixed.
    terms: list[TruthTerm] = []
    continuous = design.continuous_input_names
    binary = design.binary_input_names
    for name in continuous[: scenario.n_continuous_main]:
        terms.append(
            _typed_term(
                "main",
                (name,),
                coefficient=contract.dgp.strong_effect_floor,
                n_outputs=n_outputs,
            )
        )
    for name in binary[: scenario.n_binary_main]:
        terms.append(
            _typed_term(
                "main",
                (name,),
                coefficient=contract.dgp.strong_effect_floor,
                n_outputs=n_outputs,
            )
        )
    if scenario.interaction_kind == "continuous_continuous":
        inputs = (continuous[0], continuous[1])
    elif scenario.interaction_kind == "binary_continuous":
        inputs = (binary[0], continuous[0])
    elif scenario.interaction_kind == "binary_binary":
        inputs = (binary[0], binary[1])
    else:
        inputs = ()
    if inputs:
        terms.append(
            _typed_term(
                f"{scenario.interaction_kind}_interaction",
                inputs,
                coefficient=contract.dgp.strong_effect_floor + 0.25,
                n_outputs=n_outputs,
            )
        )
    if scenario.nonlinear_kind != "none":
        terms.append(
            _typed_term(
                "transformation",
                (continuous[0],),
                coefficient=contract.dgp.strong_effect_floor,
                n_outputs=n_outputs,
                transform=scenario.nonlinear_kind,
                in_library=scenario.nonlinear_kind in contract.dgp.transform_library,
            )
        )
    intercept = tuple(float(contract.dgp.intercept) for _ in range(n_outputs))
    identity = {
        "input_schema_sha256": design.schema_sha256,
        "intercept": intercept,
        "terms": [term.to_dict() for term in terms],
    }
    return DGPTruth(
        input_schema_sha256=design.schema_sha256,
        intercept=intercept,
        terms=tuple(terms),
        surface_sha256=_sha256_bytes(_canonical_json(identity)),
    )


def _row_noise_scale(
    contract: ExecutionContract,
    design: BSMInputDesign,
    X: np.ndarray,
    scenario: ScenarioDefinition,
) -> np.ndarray:
    if not scenario.heteroscedastic:
        return np.ones(X.shape[0], dtype=float)
    driver_position = contract.dgp.heteroscedastic_driver_input_position
    name = design.continuous_input_names[driver_position]
    coded = _coded_features(design, X)[:, design.input_names.index(name)]
    return contract.dgp.heteroscedastic_minimum_multiplier + (
        contract.dgp.heteroscedastic_maximum_multiplier
        - contract.dgp.heteroscedastic_minimum_multiplier
    ) * np.abs(coded)


def _noise_standard_deviation(
    noiseless: np.ndarray,
    intercept: tuple[float, ...],
    snr: float,
    floor: float,
) -> np.ndarray:
    signal = noiseless - np.asarray(intercept, dtype=float)[None, :]
    signal_sd = signal.std(axis=0, ddof=1)
    return np.maximum(float(floor), signal_sd / math.sqrt(snr))


@dataclass(frozen=True)
class GeneratedData:
    X_train: np.ndarray
    Y_train: np.ndarray
    X_eval: np.ndarray
    Y_eval: np.ndarray
    noiseless_train: np.ndarray
    noiseless_eval: np.ndarray
    row_noise_scale_train: np.ndarray
    row_noise_scale_eval: np.ndarray
    noise_standard_deviation_train: np.ndarray
    noise_standard_deviation_eval: np.ndarray
    feature_names: tuple[str, ...]
    truth: DGPTruth
    surface_sha256: str
    binary_cell_counts_train: dict[tuple[int, int], int]
    binary_cell_counts_eval: dict[tuple[int, int], int]
    correlation_record: dict[str, Any]
    realized_ranges_train: dict[str, tuple[float, float]]
    realized_ranges_eval: dict[str, tuple[float, float]]
    realized_heteroscedasticity_ratio_train: float
    realized_heteroscedasticity_ratio_eval: float
    realized_snr_train: tuple[float, ...]
    realized_snr_eval: tuple[float, ...]

    @property
    def truth_ledger(self) -> list[dict[str, Any]]:
        return self.truth.to_records()


def _realized_ranges(
    design: BSMInputDesign, X: np.ndarray
) -> dict[str, tuple[float, float]]:
    return {
        name: (float(X[:, position].min()), float(X[:, position].max()))
        for position, name in enumerate(design.input_names)
    }


def _realized_snr(
    noiseless: np.ndarray, response: np.ndarray, intercept: tuple[float, ...]
) -> tuple[float, ...]:
    signal = noiseless - np.asarray(intercept, dtype=float)[None, :]
    noise = response - noiseless
    signal_var = signal.var(axis=0, ddof=1)
    noise_var = noise.var(axis=0, ddof=1)
    return tuple(
        float(signal_value / noise_value) if noise_value > 0.0 else math.inf
        for signal_value, noise_value in zip(signal_var, noise_var, strict=True)
    )


def _validate_generated_data(
    contract: ExecutionContract,
    design: BSMInputDesign,
    scenario: ScenarioDefinition,
    data: GeneratedData,
) -> None:
    matrices = (
        data.X_train,
        data.Y_train,
        data.X_eval,
        data.Y_eval,
        data.noiseless_train,
        data.noiseless_eval,
    )
    if any(not np.isfinite(matrix).all() for matrix in matrices):
        raise DGPValidationError("generated DGP contains a non-finite matrix")
    if data.feature_names != design.input_names:
        raise DGPValidationError(
            "generated data does not preserve the typed input order"
        )
    required_cells = {(0, 0), (0, 1), (1, 0), (1, 1)}
    if contract.input_schema.all_binary_cells_required and (
        set(data.binary_cell_counts_train) != required_cells
        or set(data.binary_cell_counts_eval) != required_cells
        or min(data.binary_cell_counts_train.values()) < 1
        or min(data.binary_cell_counts_eval.values()) < 1
    ):
        raise DGPValidationError("a train or evaluation binary cell is absent")
    for X in (data.X_train, data.X_eval):
        for position, name in enumerate(design.input_names):
            if name in design.binary_input_names:
                if not set(np.unique(X[:, position])).issubset({0.0, 1.0}):
                    raise DGPValidationError(f"binary input {name!r} is not binary")
            elif (
                X[:, position].min() < design.lower[name]
                or X[:, position].max() > design.upper[name]
            ):
                raise DGPValidationError(
                    f"continuous input {name!r} leaves its published range"
                )
    if scenario.heteroscedastic:
        if (
            data.realized_heteroscedasticity_ratio_train <= 1.10
            or data.realized_heteroscedasticity_ratio_eval <= 1.10
        ):
            raise DGPValidationError("row-level heteroscedasticity was not realized")
    if data.surface_sha256 != data.truth.surface_sha256:
        raise DGPValidationError(
            "generated surface identity differs from the typed truth ledger"
        )
    if not np.allclose(
        data.noiseless_train, data.truth.evaluate(design, data.X_train), atol=1e-12
    ):
        raise DGPValidationError(
            "train response surface does not match the typed truth"
        )
    if not np.allclose(
        data.noiseless_eval, data.truth.evaluate(design, data.X_eval), atol=1e-12
    ):
        raise DGPValidationError(
            "evaluation response surface does not match the typed truth"
        )


def generate_bsm_dataset(
    contract: ExecutionContract,
    design: BSMInputDesign,
    scenario: ScenarioDefinition,
    entry: SeedLedgerEntry,
    *,
    scale: DatasetScale | None = None,
) -> GeneratedData:
    """Generate one validated 160-input replicate from the frozen typed DGP."""
    if entry.phase != contract.phase or entry.scenario != scenario.name:
        raise DGPValidationError(
            "seed ledger entry does not match the requested scenario"
        )
    resolved_scale = scale or DatasetScale(
        n_train=contract.development.n_train,
        n_eval=contract.development.n_eval,
        n_outputs=contract.development.n_outputs,
    )
    rng = np.random.default_rng(entry.stage_seeds["dgp"])
    X_train, train_cells, train_correlation = _sample_inputs(
        design,
        resolved_scale.n_train,
        rng,
        correlated=scenario.correlated_inputs,
        contract=contract,
    )
    X_eval, eval_cells, eval_correlation = _sample_inputs(
        design,
        resolved_scale.n_eval,
        rng,
        correlated=scenario.correlated_inputs,
        contract=contract,
    )
    truth = _build_truth(
        contract,
        design,
        scenario,
        entry,
        n_outputs=resolved_scale.n_outputs,
    )
    noiseless_train = truth.evaluate(design, X_train)
    noiseless_eval = truth.evaluate(design, X_eval)
    row_scale_train = _row_noise_scale(contract, design, X_train, scenario)
    row_scale_eval = _row_noise_scale(contract, design, X_eval, scenario)
    train_sd = _noise_standard_deviation(
        noiseless_train,
        truth.intercept,
        scenario.snr,
        contract.dgp.noise_base_sd,
    )
    # The evaluation distribution uses the training-frozen noise scale.  It is
    # never recalibrated from evaluation outcomes or evaluation signal bytes.
    eval_sd = train_sd.copy()
    train_noise_rng = np.random.default_rng(entry.stage_seeds["noise_train"])
    eval_noise_rng = np.random.default_rng(entry.stage_seeds["noise_eval"])
    Y_train = noiseless_train + train_noise_rng.standard_normal(
        noiseless_train.shape
    ) * (row_scale_train[:, None] * train_sd[None, :])
    Y_eval = noiseless_eval + eval_noise_rng.standard_normal(noiseless_eval.shape) * (
        row_scale_eval[:, None] * eval_sd[None, :]
    )
    data = GeneratedData(
        X_train=X_train,
        Y_train=Y_train,
        X_eval=X_eval,
        Y_eval=Y_eval,
        noiseless_train=noiseless_train,
        noiseless_eval=noiseless_eval,
        row_noise_scale_train=row_scale_train,
        row_noise_scale_eval=row_scale_eval,
        noise_standard_deviation_train=train_sd,
        noise_standard_deviation_eval=eval_sd,
        feature_names=design.input_names,
        truth=truth,
        surface_sha256=truth.surface_sha256,
        binary_cell_counts_train=train_cells,
        binary_cell_counts_eval=eval_cells,
        correlation_record={
            "train": train_correlation,
            "evaluation": eval_correlation,
        },
        realized_ranges_train=_realized_ranges(design, X_train),
        realized_ranges_eval=_realized_ranges(design, X_eval),
        realized_heteroscedasticity_ratio_train=float(
            (row_scale_train.max() / row_scale_train.min()) ** 2
        ),
        realized_heteroscedasticity_ratio_eval=float(
            (row_scale_eval.max() / row_scale_eval.min()) ** 2
        ),
        realized_snr_train=_realized_snr(noiseless_train, Y_train, truth.intercept),
        realized_snr_eval=_realized_snr(noiseless_eval, Y_eval, truth.intercept),
    )
    _validate_generated_data(contract, design, scenario, data)
    return data


@dataclass(frozen=True)
class TerminalRecord:
    """One terminal outcome for one planned scenario/replicate/attempt."""

    phase: str
    scenario: str
    replicate_index: int
    attempt: int
    status: str
    terminal_outcome: str
    terminal_stage: str
    contract_sha256: str
    control_snapshot_sha256: str
    seed: int
    schedule_sha256: str
    screened_count: int
    pair_family_count: int
    truth_interaction_ids: tuple[str, ...]
    retained_interaction_ids: tuple[str, ...]
    false_pair_count: int
    exception: str | None
    runtime_seconds: float
    max_rss_bytes: int

    def __post_init__(self) -> None:
        if self.status not in {"ANALYSIS_COMPLETE", "FAILED"}:
            raise TerminalLedgerError("terminal record has an invalid status")
        if self.attempt != 0:
            raise TerminalLedgerError(
                "the frozen retry policy permits only attempt zero"
            )
        if self.pair_family_count < 0 or self.false_pair_count < 0:
            raise TerminalLedgerError("terminal record counts must be non-negative")
        if self.status == "ANALYSIS_COMPLETE":
            if self.terminal_outcome == "FAILED":
                raise TerminalLedgerError(
                    "analysis-complete record cannot carry FAILED outcome"
                )
            if (
                self.pair_family_count == 0
                and self.terminal_outcome != "NO_INTERACTION_CANDIDATES"
            ):
                raise TerminalLedgerError(
                    "empty family must use NO_INTERACTION_CANDIDATES"
                )
            if (
                self.pair_family_count > 0
                and self.terminal_outcome == "NO_INTERACTION_CANDIDATES"
            ):
                raise TerminalLedgerError(
                    "non-empty family cannot use NO_INTERACTION_CANDIDATES"
                )
            if self.pair_family_count == 0 and self.false_pair_count != 0:
                raise TerminalLedgerError(
                    "empty family cannot have false retained pairs"
                )
        elif self.terminal_outcome != "FAILED" or not self.exception:
            raise TerminalLedgerError("FAILED record must carry an exception")

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.phase, self.scenario, self.replicate_index)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_terminal_ledger(
    contract: ExecutionContract,
    ledger: SeedLedger,
    records: Iterable[TerminalRecord],
) -> tuple[TerminalRecord, ...]:
    """Fail closed unless every planned key has one successful terminal record."""
    rows = tuple(records)
    for row in rows:
        if row.status == contract.terminal.failed_status:
            raise TerminalLedgerError(
                f"failed terminal record for {row.key}: {row.exception}"
            )
    keys = [row.key for row in rows]
    if len(keys) != len(set(keys)):
        raise TerminalLedgerError("duplicate terminal record")
    expected = {entry.key for entry in ledger.entries}
    actual = set(keys)
    if expected != actual:
        missing = sorted(expected.difference(actual))
        extra = sorted(actual.difference(expected))
        raise TerminalLedgerError(
            f"missing terminal record(s)={missing}; unplanned={extra}"
        )
    by_key = {entry.key: entry for entry in ledger.entries}
    for row in rows:
        entry = by_key[row.key]
        if (
            row.contract_sha256 != contract.contract_sha256
            or row.control_snapshot_sha256 != contract.control_snapshot_sha256
            or row.seed != entry.seed
        ):
            raise TerminalLedgerError(
                f"terminal record identity mismatch for {row.key}"
            )
        if row.status != contract.terminal.completed_status:
            raise TerminalLedgerError(
                f"terminal record is not analysis-complete for {row.key}"
            )
        if row.phase != contract.phase or row.scenario not in {
            s.name for s in contract.scenarios
        }:
            raise TerminalLedgerError(
                f"terminal record has an unplanned phase or scenario: {row.key}"
            )
        if (
            not math.isfinite(row.runtime_seconds)
            or row.runtime_seconds < 0.0
            or row.max_rss_bytes < 0
        ):
            raise TerminalLedgerError(
                f"terminal record has invalid resource fields for {row.key}"
            )
    return rows


def one_sided_wilson_upper(events: int, denominator: int, confidence: float) -> float:
    """Return the one-sided Wilson upper bound using z=Phi^-1(confidence)."""
    if denominator <= 0:
        raise TerminalLedgerError("Wilson denominator must be positive")
    if events < 0 or events > denominator:
        raise TerminalLedgerError("Wilson event count is outside its denominator")
    z = NormalDist().inv_cdf(confidence)
    denominator_with_z = denominator + z**2
    center = (events + z**2 / 2.0) / denominator_with_z
    half_width = z * math.sqrt(
        events * (denominator - events) / denominator + z**2 / 4.0
    )
    return min(1.0, center + half_width / denominator_with_z)


def largest_passing_event_count(
    calibration: CalibrationControls, denominator: int
) -> int:
    """Derive, rather than hard-code, the largest event count passing the gate."""
    passing = [
        events
        for events in range(denominator + 1)
        if one_sided_wilson_upper(events, denominator, calibration.one_sided_confidence)
        <= calibration.gate_upper_bound
    ]
    if not passing:
        raise TerminalLedgerError("calibration rule has no passing event count")
    return max(passing)


def binomial_operating_characteristic(
    calibration: CalibrationControls,
    denominator: int,
) -> dict[float, float]:
    """Compute pass probabilities from the frozen Wilson rule and stated rates."""
    largest = largest_passing_event_count(calibration, denominator)
    probabilities: dict[float, float] = {}
    for probability in calibration.operating_characteristic_rates:
        probabilities[probability] = float(
            sum(
                math.comb(denominator, events)
                * probability**events
                * (1.0 - probability) ** (denominator - events)
                for events in range(largest + 1)
            )
        )
    return probabilities


def aggregate_null_terminal_records(
    contract: ExecutionContract,
    records: Iterable[TerminalRecord],
) -> dict[str, Any]:
    """Aggregate a null regime without dropping empty valid candidate families."""
    rows = tuple(records)
    if not rows:
        raise TerminalLedgerError(
            "null aggregation requires at least one terminal record"
        )
    scenarios = {row.scenario for row in rows}
    if len(scenarios) != 1:
        raise TerminalLedgerError("null aggregation must not pool scenarios")
    keys = [row.key for row in rows]
    if len(keys) != len(set(keys)):
        raise TerminalLedgerError("null aggregation contains duplicate terminal record")
    scenario = contract.scenario(next(iter(scenarios)))
    if not scenario.is_null:
        raise TerminalLedgerError(
            "only a contract null regime may enter null aggregation"
        )
    for row in rows:
        if row.status != contract.terminal.completed_status:
            raise TerminalLedgerError(
                "null aggregation encountered a non-complete record"
            )
        if row.contract_sha256 != contract.contract_sha256:
            raise TerminalLedgerError("null aggregation contract mismatch")
    denominator = len(rows)
    false_pair_replicates = sum(row.false_pair_count > 0 for row in rows)
    n_empty = sum(
        row.terminal_outcome == contract.terminal.empty_family_outcome for row in rows
    )
    n_one_pair = sum(row.pair_family_count == 1 for row in rows)
    n_nondegenerate = sum(
        row.pair_family_count >= contract.nondegeneracy.minimum_pair_family_size
        for row in rows
    )
    fwer_proportion = false_pair_replicates / denominator
    upper = one_sided_wilson_upper(
        false_pair_replicates,
        denominator,
        contract.calibration.one_sided_confidence,
    )
    return {
        "scenario": scenario.name,
        "denominator": denominator,
        "n_false_pair_replicates": false_pair_replicates,
        "fwer_proportion": fwer_proportion,
        "wilson_upper": upper,
        "passes_calibration": upper <= contract.calibration.gate_upper_bound,
        "n_empty_family": n_empty,
        "n_one_pair_family": n_one_pair,
        "nondegenerate_rate": n_nondegenerate / denominator,
    }


def build_preexecution_manifest(
    contract: ExecutionContract,
    design: BSMInputDesign,
    ledger: SeedLedger,
) -> dict[str, Any]:
    """Build a status-only manifest that cannot be mistaken for a result."""
    return {
        "schema_version": 1,
        "kind": "bsm_g0b_preexecution_control",
        "execution_status": "NOT_EXECUTED",
        "phase": contract.phase,
        "gate_status": {"G0": "OPEN", "A": "OPEN", "B": "OPEN"},
        "contract_sha256": contract.contract_sha256,
        "control_snapshot_sha256": contract.control_snapshot_sha256,
        "seed_ledger_sha256": ledger.sha256,
        "input_schema": {
            "n_inputs": design.n_inputs,
            "continuous_inputs": len(design.continuous_input_names),
            "binary_inputs": len(design.binary_input_names),
            "binary_input_names": list(design.binary_input_names),
            "schema_sha256": design.schema_sha256,
        },
        "candidate_schema": {
            "n_inputs": design.n_inputs,
            "binary_inputs_included": True,
            "order": "published_metadata_order",
        },
        "planned_replicates": len(ledger.entries),
        "calibration_status": "NOT_RUN",
        "terminal_records_status": "NOT_GENERATED",
        "generic_source": {
            "repository": contract.generic_repository,
            "commit": contract.generic_commit,
            "require_clean_checkout": contract.generic_require_clean_checkout,
            "lockfile": contract.generic_lockfile,
            "lockfile_sha256": contract.generic_lockfile_sha256,
        },
    }


def write_preexecution_manifest(
    output_path: Path,
    contract: ExecutionContract,
    design: BSMInputDesign,
    ledger: SeedLedger,
) -> None:
    """Write one new status-only manifest; never overwrite a prior root."""
    if output_path.exists():
        raise PreexecutionBlockedError(
            f"refusing to overwrite existing output: {output_path}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            build_preexecution_manifest(contract, design, ledger),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _verify_pinned_generic_source(contract: ExecutionContract) -> None:
    """Require the named generic source commit to be clean before score execution."""
    spec = importlib.util.find_spec("rfm_pipeline")
    if spec is None or spec.origin is None:
        raise PreexecutionBlockedError("pinned rfm-pipeline package is unavailable")
    package_path = Path(spec.origin).resolve()
    repository = next(
        (
            candidate
            for candidate in (package_path.parent, *package_path.parents)
            if (candidate / ".git").exists()
            and (candidate / "src" / "rfm_pipeline").resolve() == package_path.parent
            and (candidate / "pixi.lock").is_file()
        ),
        None,
    )
    if repository is None:
        raise PreexecutionBlockedError(
            "rfm-pipeline source is not a pinned Git checkout"
        )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        text=True,
    ).strip()
    if commit != contract.generic_commit:
        raise PreexecutionBlockedError(
            "rfm-pipeline commit differs from the BSM contract pin"
        )
    if contract.generic_require_clean_checkout:
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=repository,
            text=True,
        ).strip()
        if dirty:
            raise PreexecutionBlockedError("rfm-pipeline checkout is dirty")


def campaign_seed_entry(
    *,
    campaign_contract: Any,
    scenario_id: str,
    replicate_index: int,
    bsm_scenario_name: str,
) -> SeedLedgerEntry:
    """Derive one BSM stage ledger from the canonical campaign identity."""
    from rfm_pipeline.campaign_contract import compute_contract_hash, derive_seed

    contract_hash = compute_contract_hash(campaign_contract)
    seed = derive_seed(contract_hash, scenario_id, replicate_index)
    return SeedLedgerEntry(
        phase="campaign",
        scenario=bsm_scenario_name,
        replicate_index=replicate_index,
        seed=seed,
        stage_seeds={
            stage: derive_seed(contract_hash, f"{scenario_id}:{stage}", replicate_index)
            for stage in STAGE_NAMES
        },
    )


def generate_campaign_dataset(
    *,
    campaign_contract: Any,
    bsm_contract: ExecutionContract,
    design: BSMInputDesign,
    scenario_id: str,
    replicate_index: int,
    scale: DatasetScale | None = None,
) -> GeneratedData:
    """Generate one exact campaign row using RFM dimensions and seed identity."""
    scenario_spec = next(
        (
            scenario
            for scenario in campaign_contract.scenarios
            if scenario.id == scenario_id
        ),
        None,
    )
    if scenario_spec is None or scenario_spec.kind == "dev":
        raise ContractError(
            f"unknown non-development campaign scenario {scenario_id!r}"
        )
    if (
        scenario_spec.n_predictor_continuous
        != bsm_contract.input_schema.continuous_inputs
        or scenario_spec.n_predictor_binary != bsm_contract.input_schema.binary_inputs
    ):
        raise ContractError(
            "campaign scenario differs from the typed BSM input contract"
        )
    try:
        bsm_scenario_name = _CAMPAIGN_TO_BSM_SCENARIO[scenario_id]
    except KeyError as exc:
        raise ContractError(
            f"campaign scenario {scenario_id!r} has no BSM DGP mapping"
        ) from exc
    bsm_scenario = bsm_contract.scenario(bsm_scenario_name)
    entry = campaign_seed_entry(
        campaign_contract=campaign_contract,
        scenario_id=scenario_id,
        replicate_index=replicate_index,
        bsm_scenario_name=bsm_scenario_name,
    )
    # The legacy contract phase is irrelevant to campaign identity; generation
    # checks only that the entry and selected BSM DGP scenario agree.
    entry = replace(entry, phase=bsm_contract.phase)
    return generate_bsm_dataset(
        bsm_contract,
        design,
        bsm_scenario,
        entry,
        scale=(
            scale
            if scale is not None
            else DatasetScale(
                n_train=scenario_spec.n_train,
                n_eval=scenario_spec.n_eval,
                n_outputs=scenario_spec.n_response,
            )
        ),
    )


def _verify_execution_authorization(
    path: Path,
    *,
    contract_hash: str,
    phase: str,
    expected_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not path.is_file():
        raise PreexecutionBlockedError("campaign execution authorization is absent")
    payload = _require_mapping(
        json.loads(path.read_text(encoding="utf-8")), "authorization"
    )
    required = {
        "schema_version",
        "status",
        "phase",
        "run_id",
        "contract_hash",
        "source_hash",
        "lock_hash",
        "campaign_inventory_hash",
        "preflight_sha256",
        "submission_plan_sha256",
        "prerequisite_sha256",
        "execution_permitted",
        "resource_freeze_sha256",
        "authorization_sha256",
    }
    if set(payload) != required:
        raise PreexecutionBlockedError("campaign execution authorization fields differ")
    identity = {
        key: value for key, value in payload.items() if key != "authorization_sha256"
    }
    if payload["authorization_sha256"] != _sha256_bytes(_canonical_json(identity)):
        raise PreexecutionBlockedError("campaign execution authorization hash differs")
    if (
        int(payload["schema_version"]) != 2
        or payload["status"] != "ACCEPTED"
        or payload["phase"] != phase
        or payload["contract_hash"] != contract_hash
        or payload["execution_permitted"] is not True
    ):
        raise PreexecutionBlockedError(
            "campaign execution authorization is not valid for this phase"
        )
    if expected_identity is not None:
        for field in ("run_id", "source_hash", "lock_hash"):
            if payload.get(field) != expected_identity.get(field):
                raise PreexecutionBlockedError(
                    f"campaign execution authorization has stale {field}"
                )
    if not str(payload["run_id"]).strip():
        raise PreexecutionBlockedError("campaign execution authorization has no run ID")
    for field in (
        "source_hash",
        "lock_hash",
        "campaign_inventory_hash",
        "preflight_sha256",
        "submission_plan_sha256",
    ):
        _require_hex(payload[field], f"execution authorization {field}")
    prerequisite_sha256 = payload["prerequisite_sha256"]
    expected_prerequisites = {
        "development": set(),
        "gate_b": {"resolution"},
        "fixed_family": {"gate_b"},
        "gate_p": {"gate_b", "fixed_family_supplement"},
        "gate_c": {"gate_b", "fixed_family_supplement", "applied_bootstrap"},
    }[phase]
    if not isinstance(prerequisite_sha256, dict) or set(prerequisite_sha256) != (
        expected_prerequisites
    ):
        raise PreexecutionBlockedError(
            "campaign execution authorization prerequisite identities differ"
        )
    for operation, digest in prerequisite_sha256.items():
        _require_hex(digest, f"execution authorization prerequisite {operation}")
    _require_hex(payload["resource_freeze_sha256"], "resource freeze authorization")
    return payload


def run_pipeline(
    data: GeneratedData,
    contract: BSMDGPContract,
    *,
    campaign_contract_path: Path = ROOT / "configs" / "g11_campaign_contract.toml",
    artifact_dir: Path,
    authorization_manifest: Path,
    phase: str,
    seed: int = 0,
    include_recovery_comparators: bool = True,
) -> Any:
    """Invoke the future pinned production adapter with no local fallback.

    The current G0/A/B state deliberately blocks this path before scoring.  A
    future approved contract must carry the same resolved controls into the
    pinned generic runner; this driver never reconstructs or substitutes them.
    """
    import pandas as pd
    from rfm_pipeline import run_production_recovery_pipeline
    from rfm_pipeline.campaign_contract import load_contract
    from rfm_pipeline.manuscript_stages import build_manuscript_feature_design
    from rfm_pipeline.recovery_study import (
        run_recovery_comparators,
        write_recovery_comparator_result,
    )

    execution_contract, _ = load_contract(campaign_contract_path)
    from rfm_pipeline.campaign_contract import compute_contract_hash

    contract_hash = compute_contract_hash(execution_contract)
    _verify_execution_authorization(
        authorization_manifest,
        contract_hash=contract_hash,
        phase=phase,
    )
    design = load_bsm_input_design(contract=contract)
    if design.input_names != data.feature_names:
        raise DGPValidationError(
            "campaign data feature order differs from the BSM DGP contract"
        )

    result = run_production_recovery_pipeline(
        pd.DataFrame(data.X_train, columns=data.feature_names),
        data.Y_train,
        pd.DataFrame(data.X_eval, columns=data.feature_names),
        data.Y_eval,
        execution_contract=execution_contract,
        artifact_dir=artifact_dir,
        seed=seed,
    )
    if not include_recovery_comparators:
        return result
    if result.algebraic_candidate_names:
        train_inputs = pd.DataFrame(data.X_train, columns=data.feature_names)
        train_inputs.insert(0, "sample_id", np.arange(data.X_train.shape[0]))
        eval_inputs = pd.DataFrame(data.X_eval, columns=data.feature_names)
        eval_inputs.insert(0, "sample_id", np.arange(data.X_eval.shape[0]))
        catalog = pd.DataFrame({"feature_name": list(result.algebraic_candidate_names)})
        algebraic_train = (
            build_manuscript_feature_design(train_inputs, catalog)
            .drop(columns=["sample_id"])
            .to_numpy(dtype=float)
        )
        algebraic_eval = (
            build_manuscript_feature_design(eval_inputs, catalog)
            .drop(columns=["sample_id"])
            .to_numpy(dtype=float)
        )
    else:
        algebraic_train = np.zeros((data.X_train.shape[0], 1), dtype=float)
        algebraic_eval = np.zeros((data.X_eval.shape[0], 1), dtype=float)
    comparators = run_recovery_comparators(
        X_train=data.X_train,
        Y_train=data.Y_train,
        X_eval=data.X_eval,
        oracle_train_design=data.truth.in_library_design(design, data.X_train),
        oracle_eval_design=data.truth.in_library_design(design, data.X_eval),
        algebraic_train_design=algebraic_train,
        algebraic_eval_design=algebraic_eval,
        proposed_predictions=result.eval_predictions,
        execution_contract=execution_contract,
        seed=seed,
    )
    write_recovery_comparator_result(comparators, artifact_dir / "comparators")
    return replace(
        result,
        comparator_predictions=comparators.predictions,
        comparator_schedule_sha256=comparators.schedule_sha256,
        comparator_hyperparameters=comparators.selected_hyperparameters,
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify BSM G0/B pre-execution controls."
    )
    parser.add_argument("--contract", type=Path, default=_CONTRACT_PATH)
    parser.add_argument(
        "--write-preexecution-manifest",
        type=Path,
        metavar="PATH",
        help="write one status-only control manifest at a new path",
    )
    parser.add_argument(
        "--execute-development",
        action="store_true",
        help="rejected while G0, A, and B remain OPEN",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Verify deterministic controls without starting scientific computation."""
    args = _parse_args(argv)
    contract = load_execution_contract(args.contract)
    verify_pinned_control_snapshot(contract)
    ledger = load_seed_ledger(contract)
    design = load_bsm_input_design(contract=contract)
    if args.execute_development:
        raise PreexecutionBlockedError(
            "development execution is not authorized by the OPEN G0/A/B control state"
        )
    if args.write_preexecution_manifest:
        write_preexecution_manifest(
            args.write_preexecution_manifest, contract, design, ledger
        )
    print(
        "[bsm_g0b] verified OPEN development controls "
        f"(contract={contract.contract_sha256}, "
        f"planned_replicates={len(ledger.entries)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
