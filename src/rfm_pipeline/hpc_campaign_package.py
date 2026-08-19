"""Content-addressed NO-SUBMIT Kestrel HPC packaging for G11."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import resource
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from datetime import date
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import yaml

from rfm_pipeline.campaign_contract import (
    G11_CONTRACT,
    CampaignContract,
    build_campaign_inventory,
    compute_contract_hash,
    derive_seed,
    fixed_family_pair_order,
    load_contract,
    render_contract_toml,
)

_HASH_FIELDS = (
    "source_hash",
    "config_hash",
    "lock_hash",
    "input_hash",
    "schedule_hash",
    "parent_hash",
    "output_hash",
)
_PLACEHOLDER_TOKENS = ("{TODO", "TBD", "<PLACEHOLDER>", "{user}", "{run_id}")
_RESOLUTION_FIXTURES = ("nondegenerate_null", "strong_planted")
_PILOT_STAGES = (
    "scheduler_diagnostic",
    "pilot_conditioning",
    "pilot_screening",
    "pilot_interaction_score",
    "pilot_interaction_reduce",
    "pilot_nonlinear",
    "pilot_sparse_full",
    "pilot_sparse_resample",
    "pilot_terminal_fit",
    "pilot_gate_b_null",
    "pilot_recovery",
    "pilot_holdout_predict",
    "pilot_bootstrap",
)
_PILOT_ENRICHED_CANDIDATE_COUNT = 367
_PILOT_APPLIED_MODEL_COUNT = 5
_PILOT_OUTPUT_COUNT_BY_PROFILE = {"p0": 500, "p1": 1000, "p2": 2000}
_PILOT_PROFILE_GRID: dict[str, tuple[tuple[str, str, int, int, int, int, int], ...]] = {
    # profile_id, partition, estimated CPUs, GiB, seconds, task size, block size
    "scheduler_diagnostic": (("p0", "debug", 1, 8, 180, 1, 1),),
    "pilot_conditioning": (
        ("p0", "debug", 8, 32, 2880, 23495, 500),
        ("p1", "debug", 16, 64, 2880, 23495, 1000),
        ("p2", "debug", 32, 128, 2880, 23495, 2000),
    ),
    "pilot_screening": (
        ("p0", "debug", 8, 24, 2880, G11_CONTRACT.B_screen, 100),
        ("p1", "debug", 16, 32, 2880, G11_CONTRACT.B_screen, 200),
        ("p2", "debug", 32, 48, 2880, G11_CONTRACT.B_screen, 400),
    ),
    "pilot_interaction_score": (
        ("p0", "short", 24, 64, 10_800, G11_CONTRACT.B_interaction, 25),
        ("p1", "short", 52, 112, 10_800, G11_CONTRACT.B_interaction, 50),
        ("p2", "short", 80, 160, 10_800, G11_CONTRACT.B_interaction, 100),
    ),
    "pilot_interaction_reduce": (
        ("p0", "debug", 4, 12, 600, G11_CONTRACT.B_interaction, 25),
        ("p1", "debug", 8, 24, 600, G11_CONTRACT.B_interaction, 50),
        ("p2", "debug", 16, 48, 600, G11_CONTRACT.B_interaction, 100),
    ),
    "pilot_nonlinear": (
        ("p0", "debug", 8, 24, 2880, 160, 10),
        ("p1", "debug", 16, 48, 2880, 160, 20),
        ("p2", "debug", 32, 96, 2880, 160, 40),
    ),
    "pilot_sparse_full": (
        ("p0", "debug", 8, 32, 2880, _PILOT_ENRICHED_CANDIDATE_COUNT, 40),
        ("p1", "debug", 16, 64, 2880, _PILOT_ENRICHED_CANDIDATE_COUNT, 80),
        ("p2", "debug", 32, 128, 2880, _PILOT_ENRICHED_CANDIDATE_COUNT, 160),
    ),
    "pilot_sparse_resample": (
        ("p0", "debug", 8, 32, 2880, 50, 1),
        ("p1", "debug", 16, 64, 2880, 50, 5),
        ("p2", "debug", 32, 128, 2880, 50, 10),
    ),
    "pilot_terminal_fit": (
        ("p0", "debug", 8, 48, 2880, 23495, 500),
        ("p1", "debug", 16, 96, 2880, 23495, 1000),
        ("p2", "debug", 32, 160, 2880, 23495, 2000),
    ),
    "pilot_gate_b_null": (
        ("p0", "debug", 24, 64, 2880, 1, 1),
        ("p1", "debug", 48, 128, 2880, 1, 1),
        ("p2", "debug", 80, 192, 2880, 1, 1),
    ),
    "pilot_recovery": (
        ("p0", "debug", 24, 64, 2880, 1, 1),
        ("p1", "debug", 48, 128, 2880, 1, 1),
        ("p2", "debug", 80, 192, 2880, 1, 1),
    ),
    "pilot_holdout_predict": (
        ("p0", "debug", 4, 16, 600, 23495, 500),
        ("p1", "debug", 8, 32, 600, 23495, 1000),
        ("p2", "debug", 16, 64, 600, 23495, 2000),
    ),
    "pilot_bootstrap": (
        ("p0", "debug", 8, 24, 1200, G11_CONTRACT.bootstrap_draws, 50),
        ("p1", "debug", 16, 48, 1200, G11_CONTRACT.bootstrap_draws, 100),
        ("p2", "debug", 32, 96, 1200, G11_CONTRACT.bootstrap_draws, 200),
    ),
}
_PILOT_FULL_TASK_STAGES = {
    "scheduler_diagnostic",
    "pilot_conditioning",
    "pilot_interaction_reduce",
    "pilot_sparse_full",
    "pilot_terminal_fit",
    "pilot_gate_b_null",
    "pilot_recovery",
}


def _pilot_executed_work_units(row: dict[str, Any]) -> int:
    if str(row["stage"]) == "pilot_bootstrap":
        return (
            int(row["block_size"])
            * _PILOT_OUTPUT_COUNT_BY_PROFILE[str(row["profile_id"])]
            * _PILOT_APPLIED_MODEL_COUNT
        )
    if str(row["stage"]) in _PILOT_FULL_TASK_STAGES:
        if row["stage"] in {"pilot_conditioning", "pilot_terminal_fit"}:
            return int(row["block_size"])
        return int(row["task_size"])
    return int(row["block_size"])


@dataclass(frozen=True, slots=True)
class ClusterConfig:
    """Committed Kestrel execution surface."""

    account: str
    max_array_size: int
    max_array_concurrency: int
    cpu_cores_per_node: int
    memory_per_node_gb: int
    partitions: tuple[str, ...]
    project_root: str
    scratch_template: str
    kill_wait: str
    job_requeue: bool
    allocation_unit: str
    cpu_charge_factor: float
    qos_factor: float
    allocation_quota: str
    rfm_repository_root: str
    bsm_repository_root: str
    scientific_adapter_relative_path: str
    scientific_adapter_sha256: str
    bsm_recovery_driver_relative_path: str
    bsm_recovery_driver_sha256: str
    bsm_dgp_contract_relative_path: str
    bsm_dgp_contract_sha256: str
    applied_data_preparer_relative_path: str
    applied_data_preparer_sha256: str
    applied_config_relative_path: str
    applied_config_sha256: str
    applied_data_root: str
    applied_data_manifest_sha256: str
    authorization_root: str


@dataclass(frozen=True, slots=True)
class ResourceEnvelope:
    """Estimated versus requested resources for one job class."""

    estimated_cpu_cores: int
    requested_cpu_cores: int
    estimated_memory_gb: int
    requested_memory_gb: int
    estimated_walltime_seconds: int
    requested_walltime_seconds: int

    @property
    def requested_walltime_hms(self) -> str:
        """Return requested walltime as ``HH:MM:SS``."""
        hours, remainder = divmod(self.requested_walltime_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


@dataclass(frozen=True, slots=True)
class CampaignEnvelope:
    """Whole-campaign allocation forecast with a shared 20% reserve.

    The pre-pilot estimate is diagnostic.  Once pilot telemetry freezes the
    resource profile, this forecast becomes the 25,000-AU admission gate.

    The storage cap is a package admission guard (one GiB per worker
    artifact), not a claim about an available Kestrel filesystem quota.
    Kestrel allocation units apply the documented CPU charge and QoS factors
    to allocated node-hours.
    """

    estimated_node_hours: float
    requested_node_hours: float
    estimated_au: float
    requested_au: int
    stage_allocations: tuple[StageAllocationEstimate, ...]
    estimated_storage_gb: int
    requested_storage_gb: int
    estimated_inode_count: int
    requested_inode_count: int
    estimated_retry_attempts: int
    requested_retry_attempts: int
    allocation_quota_readiness_blocker: bool


@dataclass(frozen=True, slots=True)
class StageAllocationEstimate:
    """Estimated/requested node-hours and Kestrel allocation units for one stage."""

    stage_name: str
    estimated_node_hours: float
    requested_node_hours: float
    estimated_au: float
    requested_au: float


@dataclass(frozen=True, slots=True)
class StagePlan:
    """One campaign stage package."""

    name: str
    partition: str
    job_count: int
    parent_stage_name: str | None
    worker_resources: ResourceEnvelope
    reducer_resources: ResourceEnvelope
    manifest_path: Path
    worker_script_path: Path
    worker_script_paths: tuple[Path, ...]
    audit_script_path: Path
    audit_output_dir: Path
    reducer_script_path: Path
    output_root: Path
    reducer_output_dir: Path
    reducer_output_hash: str
    reducer_expected_range: tuple[int, int]


@dataclass(frozen=True, slots=True)
class CampaignDAG:
    """Generated content-addressed campaign package."""

    run_id: str
    package_mode: str
    repo_root: Path
    output_dir: Path
    cluster: ClusterConfig
    scheduler_submission_permitted: bool
    source_hash: str
    config_hash: str
    lock_hash: str
    contract_config_path: Path
    campaign_inventory_path: Path
    campaign_inventory_hash: str
    readiness_blockers: tuple[str, ...]
    telemetry_schema: tuple[dict[str, str], ...]
    pilot_matrix: tuple[dict[str, Any], ...]
    post_pilot_selection: dict[str, dict[str, Any]]
    campaign_envelope: CampaignEnvelope
    stages: tuple[StagePlan, ...]


def generate_campaign_package(
    *,
    output_dir: str | Path,
    repo_root: str | Path | None = None,
    config_path: str | Path | None = None,
    contract: CampaignContract = G11_CONTRACT,
    resource_freeze: dict[str, Any] | None = None,
    package_mode: str = "full",
    completed_observed_au_for_admission: float | None = None,
    postprocessing_reserved_au_for_admission: float = 0.0,
) -> CampaignDAG:
    """Build manifests, scripts, hashes, telemetry, and readiness artifacts."""
    if package_mode not in {"full", "confirmatory", "downstream"}:
        raise ValueError("package_mode must be full, confirmatory, or downstream")
    if package_mode in {"confirmatory", "downstream"} and (
        resource_freeze is None or contract.resolution_decision_sha256 == "PENDING"
    ):
        raise ValueError(
            "confirmatory and downstream packages require frozen resources and resolution bytes"
        )
    if completed_observed_au_for_admission is not None:
        if package_mode not in {"confirmatory", "downstream"}:
            raise ValueError(
                "observed completed allocation admission is confirmatory/downstream-only"
            )
        if not math.isfinite(completed_observed_au_for_admission) or (
            completed_observed_au_for_admission < 0
        ):
            raise ValueError("completed observed admission AUs must be nonnegative")
        if not math.isfinite(postprocessing_reserved_au_for_admission) or (
            postprocessing_reserved_au_for_admission < 0
        ):
            raise ValueError("postprocessing admission reserve must be nonnegative")
    elif postprocessing_reserved_au_for_admission != 0:
        raise ValueError(
            "postprocessing admission reserve requires observed completed allocation AUs"
        )
    resolved_repo_root = (
        Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parents[2]
    )
    resolved_output_dir = Path(output_dir).resolve()
    if resolved_output_dir.exists() and any(resolved_output_dir.iterdir()):
        raise ValueError("campaign output directory must be empty")
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    resolved_config_path = (
        Path(config_path).resolve()
        if config_path is not None
        else resolved_repo_root / "configs" / "hpc" / "g11_kestrel_campaign.yml"
    )

    raw_config = _load_campaign_config(resolved_config_path)
    scheduler_submission_permitted = bool(raw_config["scheduler_submission_permitted"])
    if scheduler_submission_permitted:
        raise ValueError("NO-SUBMIT package requires scheduler_submission_permitted: false")

    cluster = _cluster_from_mapping(raw_config["cluster"])

    source_hash = _hash_python_tree(resolved_repo_root / "src" / "rfm_pipeline")
    lock_hash = _hash_file(resolved_repo_root / "pixi.lock")
    config_hash = compute_contract_hash(contract)
    if resource_freeze is not None:
        _validate_resource_freeze(resource_freeze)
        if (
            resource_freeze["source_hash"] != source_hash
            or resource_freeze["lock_hash"] != lock_hash
            or (
                resource_freeze["config_hash"] != config_hash
                and contract.resolution_decision_sha256 == "PENDING"
            )
        ):
            raise ValueError("resource freeze source, lock, or base-contract identity differs")

    contract_dir = resolved_output_dir / "contract"
    contract_dir.mkdir(parents=True, exist_ok=True)
    contract_config_path = contract_dir / "g11_campaign_contract.toml"
    contract_config_path.write_text(
        render_contract_toml(contract, config_hash, gate="HPC", status="NO_SUBMIT"),
        encoding="utf-8",
    )
    if resource_freeze is not None:
        (contract_dir / "resource_freeze.json").write_text(
            json.dumps(resource_freeze, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    fixed_family_order_path = contract_dir / "fixed_family_pair_order.json"
    fixed_family_order_path.write_text(
        json.dumps(list(fixed_family_pair_order(contract)), indent=2) + "\n",
        encoding="utf-8",
    )

    telemetry_schema = tuple(
        {
            "name": name,
            "type": dtype,
        }
        for name, dtype in (
            ("stage", "str"),
            ("shard_id", "str"),
            ("status", "str"),
            ("attempt", "int"),
            ("elapsed_seconds", "float"),
            ("total_cpu_seconds", "float"),
            ("cpu_time_seconds", "float"),
            ("max_rss_bytes", "int"),
            ("bytes_read", "int"),
            ("bytes_written", "int"),
            ("task_size", "int"),
            ("block_size", "int"),
            ("executed_work_units", "int"),
            ("profile_id", "str"),
            ("partition", "str"),
            ("requested_cpus", "int"),
            ("requested_memory_gb", "int"),
            ("requested_walltime_seconds", "int"),
            ("hostname", "str"),
            ("slurm_job_id", "str"),
            ("slurm_array_job_id", "str_or_null"),
            ("slurm_array_task_id", "str_or_null"),
            ("scheduler_state", "str"),
            ("scheduler_exit_code", "str"),
            ("scheduler_elapsed_seconds", "int"),
            ("allocated_nodes", "int"),
            ("allocated_cpus", "int"),
            ("allocated_tres", "str"),
            ("scheduler_max_rss_bytes", "int"),
            ("source_hash", "str"),
            ("config_hash", "str"),
            ("lock_hash", "str"),
            ("input_hash", "str"),
            ("schedule_hash", "str"),
            ("parent_hash", "str"),
            ("output_hash", "str"),
        )
    )
    pilot_matrix = _build_pilot_matrix(cluster)
    post_pilot_selection = _select_post_pilot_profiles(pilot_matrix)
    readiness_blockers = [
        "scheduler_submission_permitted is false; package is NO-SUBMIT by construction",
    ]
    quota_unknown = cluster.allocation_quota == "UNKNOWN"
    if quota_unknown:
        readiness_blockers.append(
            "live remaining allocation is unknown; run aus_report and freeze the available AU value"
        )
    if cluster.applied_data_manifest_sha256 == "UNKNOWN":
        readiness_blockers.append(
            "prepared applied-data manifest hash is unknown; run the deterministic "
            "train/holdout splitter and bind its manifest SHA-256"
        )
    if resource_freeze is None:
        readiness_blockers.append(
            "production resources and block sizes are not frozen from complete pilot telemetry"
        )
    if contract.resolution_decision_sha256 == "PENDING":
        readiness_blockers.append(
            "confirmatory seeds are provisional until a successful resolution "
            "decision is content-bound"
        )

    package_hash = _stable_hash(
        {
            "run_id": raw_config["run_id"],
            "source_hash": source_hash,
            "config_hash": config_hash,
            "lock_hash": lock_hash,
        }
    )
    stages = _build_stage_plans(
        output_dir=resolved_output_dir,
        run_id=raw_config["run_id"],
        cluster=cluster,
        source_hash=source_hash,
        config_hash=config_hash,
        lock_hash=lock_hash,
        package_hash=package_hash,
        contract_config_path=contract_config_path,
        contract=contract,
        resource_freeze=resource_freeze,
        package_mode=package_mode,
    )
    campaign_inventory_path = resolved_output_dir / "campaign_inventory.jsonl"
    campaign_rows = _build_campaign_table(stages)
    _write_jsonl(campaign_inventory_path, campaign_rows)
    campaign_inventory_hash = _hash_file(campaign_inventory_path)

    prior_allocations: tuple[StageAllocationEstimate, ...] = ()
    prior_worker_count = 0
    prior_stage_count = 0
    if package_mode == "confirmatory":
        assert resource_freeze is not None
        prior_allocations, prior_worker_count, prior_stage_count = (
            _completed_adaptive_allocation_estimates(
                cluster=cluster,
                contract=contract,
                resource_freeze=resource_freeze,
            )
        )
    campaign_envelope = _build_campaign_envelope(
        stages,
        cluster=cluster,
        retry_limit=contract.retry_limit,
        allocation_quota_is_unknown=quota_unknown,
        prior_allocations=prior_allocations,
        prior_worker_count=prior_worker_count,
        prior_stage_count=prior_stage_count,
    )
    if not quota_unknown:
        try:
            allocation_quota = int(cluster.allocation_quota)
        except ValueError as exc:
            raise ValueError(
                "allocation_quota must be UNKNOWN or a positive integer AU limit."
            ) from exc
        if allocation_quota <= 0:
            raise ValueError("allocation_quota must be a positive integer AU limit.")
        admission_requested_au = float(campaign_envelope.requested_au)
        admission_error = "telemetry-based whole-campaign projection"
        if completed_observed_au_for_admission is not None:
            remaining_estimated_au = sum(
                _stage_allocation_estimate(stage, cluster=cluster).estimated_au for stage in stages
            )
            reserve_fraction = 0.20 if contract.retry_limit else 0.0
            admission_requested_au = (
                completed_observed_au_for_admission
                + math.ceil(remaining_estimated_au * (1.0 + reserve_fraction))
                + postprocessing_reserved_au_for_admission
            )
            admission_error = "observed-prior confirmatory projection"
        if resource_freeze is None and allocation_quota < admission_requested_au:
            readiness_blockers.append(
                "provisional pre-pilot forecast exceeds the campaign AU budget; "
                "run the bounded pilot and require the telemetry-based final package to fit"
            )
        if resource_freeze is not None and allocation_quota < admission_requested_au:
            raise ValueError(
                f"{admission_error} exceeds allocation_quota: "
                f"{admission_requested_au:g} > {allocation_quota}"
            )
    dag = CampaignDAG(
        run_id=raw_config["run_id"],
        package_mode=package_mode,
        repo_root=resolved_repo_root,
        output_dir=resolved_output_dir,
        cluster=cluster,
        scheduler_submission_permitted=scheduler_submission_permitted,
        source_hash=source_hash,
        config_hash=config_hash,
        lock_hash=lock_hash,
        contract_config_path=contract_config_path,
        campaign_inventory_path=campaign_inventory_path,
        campaign_inventory_hash=campaign_inventory_hash,
        readiness_blockers=tuple(readiness_blockers),
        telemetry_schema=telemetry_schema,
        pilot_matrix=pilot_matrix,
        post_pilot_selection=post_pilot_selection,
        campaign_envelope=campaign_envelope,
        stages=stages,
    )
    validate_campaign_table(dag, contract=contract)
    _write_package_metadata(dag)
    return dag


def finalize_post_resolution_package(
    *,
    output_dir: str | Path,
    resource_freeze_path: str | Path,
    resolution_decision_path: str | Path,
    repo_root: str | Path | None = None,
    config_path: str | Path | None = None,
    base_contract: CampaignContract = G11_CONTRACT,
) -> CampaignDAG:
    """Regenerate fresh confirmatory manifests from accepted pilot and resolution evidence."""
    resolved_repo_root = (
        Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parents[2]
    )
    freeze_path = Path(resource_freeze_path).resolve()
    decision_path = Path(resolution_decision_path).resolve()
    resource_freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    _validate_resource_freeze(resource_freeze)
    if (
        resource_freeze["source_hash"]
        != _hash_python_tree(resolved_repo_root / "src" / "rfm_pipeline")
        or resource_freeze["lock_hash"] != _hash_file(resolved_repo_root / "pixi.lock")
        or resource_freeze["config_hash"] != compute_contract_hash(base_contract)
    ):
        raise ValueError(
            "resource freeze does not belong to the current source, lock, and base contract"
        )
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if (
        decision.get("operation") != "resolution"
        or decision.get("status") != "completed"
        or decision.get("decision") != "ACCEPTED"
        or int(decision.get("terminal_record_count", 0)) != 20
        or decision.get("contract_hash") != compute_contract_hash(base_contract)
    ):
        raise ValueError("resolution decision is not valid for the base campaign contract")
    selected_b = int(decision.get("selected_B_interaction", 0))
    permitted = {
        base_contract.resolution_base_draws,
        2 * base_contract.resolution_base_draws,
    }
    if selected_b not in permitted:
        raise ValueError("resolution decision selected a non-prespecified interaction schedule")
    destination = Path(output_dir).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("final campaign output directory must be empty")
    decision_sha256 = _hash_file(decision_path)
    selected_contract = replace(
        base_contract,
        B_interaction=selected_b,
        resolution_decision_sha256=decision_sha256,
    )
    return generate_campaign_package(
        output_dir=destination,
        repo_root=resolved_repo_root,
        config_path=config_path,
        contract=selected_contract,
        resource_freeze=resource_freeze,
        package_mode="confirmatory",
    )


def validate_hpc_preflight(
    dag: CampaignDAG,
    *,
    evidence: dict[str, Any],
    target_phase: str,
    output_path: str | Path | None = None,
    run_command: Any | None = None,
) -> dict[str, Any]:
    """Validate same-day, non-submitting Kestrel evidence for exact package bytes."""
    permitted_phases = {"pilot", "development", "gate_b", "gate_p", "gate_c"}
    if target_phase not in permitted_phases:
        raise ValueError(
            "preflight target_phase must be pilot, development, gate_b, gate_p, or gate_c"
        )
    if target_phase in {"pilot", "development"} and dag.package_mode != "full":
        raise ValueError("pilot/development preflight requires a full package")
    if target_phase in {"gate_b", "gate_p", "gate_c"} and dag.package_mode != "confirmatory":
        raise ValueError("confirmatory preflight requires a post-resolution confirmatory package")
    required = {
        "observed_date",
        "account",
        "project_root",
        "rfm_git_clean",
        "rfm_git_commit",
        "bsm_git_clean",
        "bsm_git_commit",
        "remaining_au",
        "project_available_bytes",
        "project_available_inodes",
        "scratch_available_bytes",
        "scratch_available_inodes",
        "sbatch_test_only",
    }
    if set(evidence) != required:
        raise ValueError("HPC preflight evidence fields differ from the frozen schema")
    if evidence["observed_date"] != date.today().isoformat():
        raise ValueError("HPC preflight evidence is not from the current date")
    if evidence["account"] != dag.cluster.account:
        raise ValueError("HPC preflight account differs from the package")
    if evidence["project_root"] != dag.cluster.project_root:
        raise ValueError("HPC preflight project root differs from the package")
    for prefix in ("rfm", "bsm"):
        if (
            evidence[f"{prefix}_git_clean"] is not True
            or re.fullmatch(r"[0-9a-f]{40}", str(evidence[f"{prefix}_git_commit"])) is None
        ):
            raise ValueError(f"HPC preflight {prefix.upper()} checkout is not clean and pinned")
    runner = subprocess.run if run_command is None else run_command
    for prefix, repository_root in (
        ("rfm", dag.cluster.rfm_repository_root),
        ("bsm", dag.cluster.bsm_repository_root),
    ):
        status = runner(
            ["git", "-C", repository_root, "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        )
        commit = runner(
            ["git", "-C", repository_root, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        if status.stdout.strip():
            raise ValueError(f"Kestrel {prefix.upper()} checkout is dirty")
        if commit.stdout.strip() != evidence[f"{prefix}_git_commit"]:
            raise ValueError(f"Kestrel {prefix.upper()} commit differs from preflight evidence")
    remaining_au = int(evidence["remaining_au"])
    if remaining_au <= 0:
        raise ValueError("same-day remaining AU must be positive")
    selected_stages = _selected_stages_for_phase(dag, target_phase)
    if not selected_stages:
        raise ValueError("HPC preflight target phase has no packaged stages")
    allocations = {item.stage_name: item for item in dag.campaign_envelope.stage_allocations}
    contract, loaded_contract_hash = load_contract(dag.contract_config_path)
    if loaded_contract_hash != dag.config_hash:
        raise ValueError("HPC preflight contract bytes differ from the package")
    required_au = math.ceil(
        1.20 * sum(allocations[stage.name].estimated_au for stage in selected_stages)
    )
    if remaining_au < required_au:
        raise ValueError("same-day remaining AU is below the target-phase envelope")
    worker_count = sum(stage.job_count for stage in selected_stages)
    required_storage_gb = math.ceil(1.20 * (worker_count + len(selected_stages)))
    required_inodes = math.ceil(1.20 * (worker_count * 6 + len(selected_stages) * 6))
    required_scratch_bytes = required_storage_gb * 1024**3
    if int(evidence["scratch_available_bytes"]) < required_scratch_bytes:
        raise ValueError("scratch free space is below the campaign storage envelope")
    if int(evidence["scratch_available_inodes"]) < required_inodes:
        raise ValueError("scratch free inodes are below the campaign inode envelope")
    if int(evidence["project_available_bytes"]) < 2 * 1024**3:
        raise ValueError("project space has less than the frozen 2-GiB metadata/log reserve")
    if int(evidence["project_available_inodes"]) < 10_000:
        raise ValueError("project filesystem has fewer than 10,000 free metadata/log inodes")

    if (
        _hash_python_tree(Path(dag.cluster.rfm_repository_root) / "src" / "rfm_pipeline")
        != dag.source_hash
    ):
        raise ValueError("Kestrel RFM source bytes differ from the package")
    if _hash_file(Path(dag.cluster.rfm_repository_root) / "pixi.lock") != dag.lock_hash:
        raise ValueError("Kestrel RFM lock bytes differ from the package")
    for relative_path, expected_hash in (
        (
            dag.cluster.scientific_adapter_relative_path,
            dag.cluster.scientific_adapter_sha256,
        ),
        (
            dag.cluster.bsm_recovery_driver_relative_path,
            dag.cluster.bsm_recovery_driver_sha256,
        ),
        (
            dag.cluster.bsm_dgp_contract_relative_path,
            dag.cluster.bsm_dgp_contract_sha256,
        ),
        (
            dag.cluster.applied_data_preparer_relative_path,
            dag.cluster.applied_data_preparer_sha256,
        ),
    ):
        if _hash_file(Path(dag.cluster.bsm_repository_root) / relative_path) != expected_hash:
            raise ValueError(f"Kestrel BSM file bytes differ: {relative_path}")
    if target_phase == "gate_p":
        if (
            _hash_file(
                Path(dag.cluster.bsm_repository_root) / dag.cluster.applied_config_relative_path
            )
            != dag.cluster.applied_config_sha256
        ):
            raise ValueError("Kestrel applied config bytes differ from the package")
        if (
            _hash_file(Path(dag.cluster.applied_data_root) / "dataset_manifest.json")
            != dag.cluster.applied_data_manifest_sha256
        ):
            raise ValueError("Kestrel applied-data manifest differs from the package")

    expected_scripts = {
        str(path): _hash_file(path)
        for stage in selected_stages
        for path in (
            *stage.worker_script_paths,
            stage.audit_script_path,
            stage.reducer_script_path,
        )
    }
    if evidence["sbatch_test_only"] != expected_scripts:
        raise ValueError("sbatch --test-only evidence does not exactly cover packaged scripts")
    for stage in selected_stages:
        for path in (stage.output_root, stage.reducer_output_dir, stage.audit_output_dir):
            if path.exists() and any(path.iterdir()):
                raise ValueError(f"HPC output root is not empty: {path}")
    ignored_blocker_prefixes = {
        "pilot": (
            "production resources",
            "confirmatory seeds",
            "prepared applied-data",
            "provisional pre-pilot forecast",
        ),
        "development": ("confirmatory seeds", "prepared applied-data"),
        "gate_b": ("prepared applied-data",),
        "gate_p": (),
        "gate_c": (),
    }[target_phase]
    unresolved = [
        blocker
        for blocker in dag.readiness_blockers
        if not blocker.startswith("scheduler_submission_permitted is false")
        and not blocker.startswith(ignored_blocker_prefixes)
    ]
    if unresolved:
        raise ValueError(f"HPC package retains readiness blockers: {unresolved}")
    payload = {
        "schema_version": 1,
        "status": "HPC_SUBMISSION_READY",
        "target_phase": target_phase,
        "run_id": dag.run_id,
        "source_hash": dag.source_hash,
        "config_hash": dag.config_hash,
        "lock_hash": dag.lock_hash,
        "campaign_inventory_hash": dag.campaign_inventory_hash,
        "readiness_blockers": list(dag.readiness_blockers),
        "resource_freeze_sha256": next(
            str(record["resource_freeze_sha256"])
            for stage in dag.stages
            if stage.name not in _PILOT_STAGES
            for record in load_stage_manifest(stage.manifest_path)[:1]
        ),
        "observed_date": evidence["observed_date"],
        "remaining_au": remaining_au,
        "required_au": required_au,
        "required_storage_gb": required_storage_gb,
        "required_inodes": required_inodes,
        "rfm_git_commit": evidence["rfm_git_commit"],
        "bsm_git_commit": evidence["bsm_git_commit"],
        "evidence_sha256": _stable_hash(evidence),
    }
    payload["preflight_sha256"] = _stable_hash(payload)
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return payload


def _selected_stages_for_phase(dag: CampaignDAG, target_phase: str) -> list[StagePlan]:
    if target_phase == "pilot":
        return [stage for stage in dag.stages if stage.name in _PILOT_STAGES]
    if target_phase == "development":
        return [stage for stage in dag.stages if stage.name == "resolution"]
    if target_phase == "gate_b":
        return [
            stage for stage in dag.stages if stage.name in {"gate_b", "fixed_family_supplement"}
        ]
    if target_phase == "fixed_family":
        return [stage for stage in dag.stages if stage.name == "fixed_family_supplement"]
    if target_phase == "gate_p":
        return [stage for stage in dag.stages if stage.name.startswith("applied_")]
    if target_phase == "gate_c":
        return [stage for stage in dag.stages if stage.name == "recovery"]
    raise ValueError("unknown campaign preflight phase")


def _filesystem_availability(path: str | Path) -> tuple[int, int]:
    """Return filesystem-available bytes and inodes at an existing path."""
    location = Path(path)
    if not location.is_dir():
        raise ValueError(f"HPC capacity probe path is not an existing directory: {location}")
    stats = os.statvfs(location)
    return (int(stats.f_bavail * stats.f_frsize), int(stats.f_favail))


def _nearest_existing_directory(path: str | Path) -> Path:
    location = Path(path)
    while not location.exists() and location != location.parent:
        location = location.parent
    if not location.is_dir():
        raise ValueError(f"HPC capacity probe has no existing directory ancestor: {path}")
    return location


def _parse_lustre_size(raw: str) -> int:
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([KMGTPE]?)", raw.strip(), re.IGNORECASE)
    if match is None:
        raise ValueError(f"unrecognized Lustre quota size: {raw}")
    factors = {
        "": 1,
        "K": 1024,
        "M": 1024**2,
        "G": 1024**3,
        "T": 1024**4,
        "P": 1024**5,
        "E": 1024**6,
    }
    return int(float(match.group(1)) * factors[match.group(2).upper()])


def _project_quota_availability(
    project_root: str,
    *,
    run_command: Any,
) -> tuple[int, int, dict[str, Any]]:
    project = run_command(
        ["lfs", "project", "-d", project_root],
        check=True,
        capture_output=True,
        text=True,
    )
    project_match = re.search(r"(?m)^\s*([0-9]+)\s+([P-])\s+", project.stdout)
    if project_match is None:
        raise ValueError("could not parse the Lustre project ID")
    project_id = project_match.group(1)
    if project_id == "0" or project_match.group(2) != "P":
        available_bytes, available_inodes = _filesystem_availability(project_root)
        return (
            available_bytes,
            available_inodes,
            {
                "project_command": ["lfs", "project", "-d", project_root],
                "project_stdout": project.stdout,
                "project_id": project_id,
                "quota_mode": "filesystem_available_no_project_quota",
            },
        )
    quota = run_command(
        ["lfs", "quota", "-hp", project_id, project_root],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line.strip() for line in quota.stdout.splitlines() if line.strip()]
    root_index = next(
        (index for index, line in enumerate(lines) if line == project_root),
        None,
    )
    if root_index is None or root_index + 1 >= len(lines):
        raise ValueError("could not locate the project quota values")
    fields = lines[root_index + 1].split()
    if len(fields) < 7:
        raise ValueError("project quota output has fewer fields than expected")
    used_bytes = _parse_lustre_size(fields[0])
    quota_bytes = _parse_lustre_size(fields[1])
    if quota_bytes <= 0 or used_bytes > quota_bytes:
        raise ValueError("project byte quota is absent, invalid, or exceeded")
    used_inodes = int(fields[4])
    quota_inodes = int(fields[5])
    if quota_inodes > 0:
        available_inodes = quota_inodes - used_inodes
        if available_inodes < 0:
            raise ValueError("project inode quota is exceeded")
    else:
        _, available_inodes = _filesystem_availability(project_root)
    audit = {
        "project_command": ["lfs", "project", "-d", project_root],
        "project_stdout": project.stdout,
        "quota_command": ["lfs", "quota", "-hp", project_id, project_root],
        "quota_stdout": quota.stdout,
        "project_id": project_id,
        "used_bytes": used_bytes,
        "quota_bytes": quota_bytes,
        "used_inodes": used_inodes,
        "quota_inodes": quota_inodes,
        "quota_mode": "project_quota",
    }
    return quota_bytes - used_bytes, available_inodes, audit


def run_hpc_live_smoke(
    dag: CampaignDAG,
    *,
    target_phase: str,
    remaining_au: int,
    evidence_dir: str | Path,
    run_command: Any | None = None,
) -> dict[str, Any]:
    """Run read-only Kestrel probes and exact ``sbatch --test-only`` checks.

    This path never invokes ordinary ``sbatch`` and never creates a scientific
    output root. It writes the raw AU/quota/test-only evidence before validating
    the same-day preflight contract.
    """
    runner = subprocess.run if run_command is None else run_command
    destination = Path(evidence_dir).resolve()
    if dag.cluster.allocation_quota == "UNKNOWN":
        raise ValueError("live smoke requires a numeric allocation_quota ceiling")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("live-smoke evidence directory must be empty")
    destination.mkdir(parents=True, exist_ok=True)
    selected_stages = _selected_stages_for_phase(dag, target_phase)
    if not selected_stages:
        raise ValueError("live-smoke phase has no packaged stages")
    if int(remaining_au) <= 0:
        raise ValueError("live-smoke remaining AU must be positive")

    aus_report = runner(
        ["aus_report"],
        check=True,
        capture_output=True,
        text=True,
    )
    if not aus_report.stdout.strip():
        raise ValueError("aus_report returned no allocation evidence")
    (destination / "aus_report.txt").write_text(aus_report.stdout, encoding="utf-8")

    association_command = [
        "sacctmgr",
        "-n",
        "-P",
        "show",
        "assoc",
        f"account={dag.cluster.account}",
        f"user={os.environ.get('USER', '')}",
        "format=Cluster,Account,User",
    ]
    association = runner(
        association_command,
        check=True,
        capture_output=True,
        text=True,
    )
    if (
        re.search(
            rf"^kestrel\|{re.escape(dag.cluster.account)}\|[^|\s]+\s*$",
            association.stdout,
            flags=re.MULTILINE,
        )
        is None
    ):
        raise ValueError("sacctmgr does not confirm the configured Kestrel account association")
    normalized_au_report = aus_report.stdout.replace(",", "")
    aus_report_binds_budget = (
        re.search(rf"\b{re.escape(dag.cluster.account)}\b", aus_report.stdout) is not None
        and re.search(rf"\b{int(remaining_au)}\b", normalized_au_report) is not None
    )
    if not aus_report_binds_budget and int(remaining_au) != int(dag.cluster.allocation_quota):
        raise ValueError(
            "an account absent from aus_report must use the configured allocation_quota ceiling"
        )

    repository_evidence: dict[str, Any] = {}
    repository_audit: dict[str, Any] = {}
    for prefix, repository_root in (
        ("rfm", dag.cluster.rfm_repository_root),
        ("bsm", dag.cluster.bsm_repository_root),
    ):
        status = runner(
            ["git", "-C", repository_root, "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        )
        commit = runner(
            ["git", "-C", repository_root, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        repository_evidence[f"{prefix}_git_clean"] = not bool(status.stdout.strip())
        repository_evidence[f"{prefix}_git_commit"] = commit.stdout.strip()
        repository_audit[prefix] = {
            "repository_root": repository_root,
            "status_stdout": status.stdout,
            "commit_stdout": commit.stdout,
        }

    project_bytes, project_inodes, project_audit = _project_quota_availability(
        dag.cluster.project_root,
        run_command=runner,
    )
    scratch_run_root = Path(_resolved_scratch_template(dag.cluster.scratch_template, dag.run_id))
    scratch_capacity_root = _nearest_existing_directory(scratch_run_root.parent)
    scratch_bytes, scratch_inodes = _filesystem_availability(scratch_capacity_root)

    sbatch_hashes: dict[str, str] = {}
    sbatch_audit: list[dict[str, Any]] = []
    for stage in selected_stages:
        for script_path in (
            *stage.worker_script_paths,
            stage.audit_script_path,
            stage.reducer_script_path,
        ):
            command = ["sbatch", "--test-only", str(script_path)]
            result = runner(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
            script_hash = _hash_file(script_path)
            sbatch_hashes[str(script_path)] = script_hash
            sbatch_audit.append(
                {
                    "command": command,
                    "script_sha256": script_hash,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            )

    evidence = {
        "observed_date": date.today().isoformat(),
        "account": dag.cluster.account,
        "project_root": dag.cluster.project_root,
        **repository_evidence,
        "remaining_au": int(remaining_au),
        "project_available_bytes": project_bytes,
        "project_available_inodes": project_inodes,
        "scratch_available_bytes": scratch_bytes,
        "scratch_available_inodes": scratch_inodes,
        "sbatch_test_only": sbatch_hashes,
    }
    audit = {
        "schema_version": 1,
        "status": "READ_ONLY_PROBES_COMPLETED",
        "observed_date": evidence["observed_date"],
        "aus_report_command": ["aus_report"],
        "aus_report_sha256": _hash_file(destination / "aus_report.txt"),
        "account_association": {
            "command": association_command,
            "stdout": association.stdout,
            "stderr": association.stderr,
            "budget_binding": (
                "aus_report" if aus_report_binds_budget else "configured_allocation_quota"
            ),
        },
        "repositories": repository_audit,
        "project_quota": project_audit,
        "scratch_capacity_root": str(scratch_capacity_root),
        "sbatch_test_only": sbatch_audit,
    }
    audit["audit_sha256"] = _stable_hash(audit)
    evidence_path = destination / "evidence.json"
    audit_path = destination / "live_smoke_audit.json"
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    preflight = validate_hpc_preflight(
        dag,
        evidence=evidence,
        target_phase=target_phase,
        output_path=destination / "preflight.json",
        run_command=runner,
    )
    return {
        "evidence": evidence,
        "audit": audit,
        "preflight": preflight,
    }


def write_phase_authorization(
    dag: CampaignDAG,
    *,
    phase: str,
    resource_freeze_path: str | Path,
    preflight_path: str | Path,
    prerequisite_paths: tuple[str | Path, ...] = (),
    output_path: str | Path,
) -> dict[str, Any]:
    """Write the exact BSM authorization schema only after phase gates pass."""
    if phase not in {"development", "gate_b", "fixed_family", "gate_p", "gate_c"}:
        raise ValueError("unknown campaign authorization phase")
    freeze = json.loads(Path(resource_freeze_path).read_text(encoding="utf-8"))
    _validate_resource_freeze(freeze)
    resource_hash = str(freeze["resource_freeze_sha256"])
    freeze_contract_matches = freeze.get("config_hash") == dag.config_hash
    if dag.package_mode in {"confirmatory", "downstream"}:
        freeze_contract_matches = freeze.get("config_hash") == compute_contract_hash(G11_CONTRACT)
    if (
        freeze.get("source_hash") != dag.source_hash
        or freeze.get("lock_hash") != dag.lock_hash
        or not freeze_contract_matches
    ):
        raise ValueError("phase authorization resource freeze identity differs")
    preflight = json.loads(Path(preflight_path).read_text(encoding="utf-8"))
    preflight_identity = {
        key: value for key, value in preflight.items() if key != "preflight_sha256"
    }
    if (
        preflight.get("status") != "HPC_SUBMISSION_READY"
        or preflight.get("config_hash") != dag.config_hash
        or preflight.get("source_hash") != dag.source_hash
        or preflight.get("lock_hash") != dag.lock_hash
        or preflight.get("campaign_inventory_hash") != dag.campaign_inventory_hash
        or preflight.get("observed_date") != date.today().isoformat()
        or preflight.get("resource_freeze_sha256") != resource_hash
        or preflight.get("preflight_sha256") != _stable_hash(preflight_identity)
    ):
        raise ValueError("phase authorization preflight identity differs")
    if preflight.get("target_phase") != phase:
        raise ValueError("phase authorization uses the wrong preflight target")

    prerequisites = [
        json.loads(Path(path).read_text(encoding="utf-8")) for path in prerequisite_paths
    ]
    expected_operations = {
        "development": set(),
        "gate_b": {"resolution"},
        "fixed_family": {"gate_b"},
        "gate_p": {"gate_b", "fixed_family_supplement"},
        "gate_c": {"gate_b", "fixed_family_supplement", "applied_bootstrap"},
    }[phase]
    by_operation = {str(item.get("operation", "")): item for item in prerequisites}
    if set(by_operation) != expected_operations:
        raise ValueError("phase authorization prerequisite set differs")
    for operation, item in by_operation.items():
        if item.get("status") != "completed":
            raise ValueError(f"phase prerequisite {operation} is incomplete or stale")
        if operation != "resolution" and item.get("contract_hash") != dag.config_hash:
            raise ValueError(f"phase prerequisite {operation} has a stale contract")
        if operation in {"resolution", "gate_b", "fixed_family_supplement"}:
            expected_decision = "ACCEPTED" if operation == "resolution" else "PASS"
            decision_field = "decision"
            if item.get(decision_field) != expected_decision:
                raise ValueError(f"phase prerequisite {operation} did not pass")
    if phase != "development":
        expected_stage = {
            "gate_b": "resolution",
            "fixed_family": {"gate_b": "gate_b"},
            "gate_p": {
                "gate_b": "gate_b",
                "fixed_family_supplement": "fixed_family_supplement",
            },
            "gate_c": {
                "gate_b": "gate_b",
                "fixed_family_supplement": "fixed_family_supplement",
                "applied_bootstrap": "applied_bootstrap",
            },
        }[phase]
        for path, item in zip(prerequisite_paths, prerequisites, strict=True):
            operation = str(item["operation"])
            if operation == "resolution":
                continue
            if phase == "fixed_family" and operation == "gate_b":
                adoption_identity = {
                    key: value for key, value in item.items() if key != "adoption_sha256"
                }
                if (
                    item.get("adoption_sha256") != _stable_hash(adoption_identity)
                    or item.get("adopted_from_contract_hash") == dag.config_hash
                    or not re.fullmatch(
                        r"[0-9a-f]{64}",
                        str(item.get("adopted_gate_b_decision_sha256", "")),
                    )
                    or not re.fullmatch(
                        r"[0-9a-f]{64}",
                        str(item.get("contract_amendment_sha256", "")),
                    )
                ):
                    raise ValueError("fixed-family Gate-B adoption evidence differs")
                continue
            stage_name = str(expected_stage[operation])
            stage = _stage_by_name(dag, stage_name)
            expected_path = (
                stage.reducer_output_dir
                / {
                    "resolution": "resolution_decision.json",
                    "gate_b": "gate_b_decision.json",
                    "fixed_family_supplement": "fixed_family_decision.json",
                    "applied_bootstrap": "scientific_stage_summary.json",
                }[operation]
            )
            if Path(path).resolve() != expected_path.resolve():
                raise ValueError(
                    f"phase prerequisite {operation} is not the packaged reducer output"
                )
            reducer_result = stage.reducer_output_dir / "reduced_result.json"
            success_marker = stage.reducer_output_dir / "_SUCCESS.json"
            if not reducer_result.is_file() or not success_marker.is_file():
                raise ValueError(
                    f"phase prerequisite {operation} lacks packaged reducer output evidence"
                )
            marker = json.loads(success_marker.read_text(encoding="utf-8"))
            result = json.loads(reducer_result.read_text(encoding="utf-8"))
            scientific = result.get("scientific_reduction", {})
            _validate_scientific_artifacts(scientific, allowed_root=stage.reducer_output_dir)
            prerequisite_hash = _hash_file(Path(path))
            artifact_is_bound = any(
                Path(str(artifact.get("path", ""))).resolve() == Path(path).resolve()
                and artifact.get("sha256") == prerequisite_hash
                for artifact in scientific.get("scientific_artifacts", [])
            )
            if (
                marker.get("stage") != stage_name
                or marker.get("status") != "completed"
                or marker.get("output_hash") != stage.reducer_output_hash
                or marker.get("artifact_sha256") != _hash_file(reducer_result)
                or not artifact_is_bound
            ):
                raise ValueError(f"phase prerequisite {operation} reducer output identity differs")
    if phase == "gate_b":
        decision_path = next(
            Path(path)
            for path, item in zip(prerequisite_paths, prerequisites, strict=True)
            if item.get("operation") == "resolution"
        )
        contract, _ = load_contract(dag.contract_config_path)
        if _hash_file(decision_path) != contract.resolution_decision_sha256:
            raise ValueError("Gate-B authorization resolution bytes differ from the contract")
        if int(by_operation["resolution"].get("selected_B_interaction", 0)) != (
            contract.B_interaction
        ):
            raise ValueError("Gate-B authorization resolution selection differs from the contract")

    phase_plan = build_campaign_phase_plan(dag, phase=phase)
    identity = {
        "schema_version": 2,
        "status": "ACCEPTED",
        "phase": phase,
        "run_id": dag.run_id,
        "contract_hash": dag.config_hash,
        "source_hash": dag.source_hash,
        "lock_hash": dag.lock_hash,
        "campaign_inventory_hash": dag.campaign_inventory_hash,
        "submission_plan_sha256": str(phase_plan["submission_plan_sha256"]),
        "preflight_sha256": str(preflight["preflight_sha256"]),
        "prerequisite_sha256": {
            str(item["operation"]): _hash_file(Path(path))
            for path, item in zip(prerequisite_paths, prerequisites, strict=True)
        },
        "execution_permitted": True,
        "resource_freeze_sha256": resource_hash,
    }
    authorization = {**identity, "authorization_sha256": _stable_hash(identity)}
    destination = Path(output_path)
    if destination.exists():
        raise ValueError("phase authorization path already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(authorization, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return authorization


def load_stage_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Load one stage JSONL manifest and fail closed on malformed rows."""
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise ValueError(f"manifest does not exist: {manifest_path}")

    records: list[dict[str, Any]] = []
    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:  # pragma: no cover - exercised by ValueError boundary
            raise ValueError(
                f"manifest {manifest_path} line {line_number} is not valid JSON"
            ) from exc
        if not isinstance(record, dict):
            raise ValueError(f"manifest {manifest_path} line {line_number} must be a JSON object")
        _validate_manifest_record(record, manifest_path=manifest_path, line_number=line_number)
        records.append(record)
    if not records:
        raise ValueError(f"manifest {manifest_path} contains no records")
    return records


def validate_resume_artifacts(stage: StagePlan) -> dict[str, int]:
    """Validate atomic per-shard success markers for one stage."""
    records = load_stage_manifest(stage.manifest_path)
    summary = {"completed": 0, "failed": 0, "pending": 0}
    for record in records:
        shard_dir = Path(record["output_dir"])
        success_marker = shard_dir / "_SUCCESS.json"
        failed_marker = shard_dir / "_FAILED.json"
        if success_marker.exists():
            payload = json.loads(success_marker.read_text(encoding="utf-8"))
            if payload.get("stage") != record["stage"]:
                raise ValueError(f"resume artifact stage mismatch for {record['shard_id']}")
            if payload.get("shard_id") != record["shard_id"]:
                raise ValueError(f"resume artifact shard_id mismatch for {record['shard_id']}")
            if payload.get("output_hash") != record["output_hash"]:
                raise ValueError(f"resume artifact output_hash mismatch for {record['shard_id']}")
            artifact_path = shard_dir / "result.json"
            if not artifact_path.is_file():
                raise ValueError(f"resume artifact is missing for {record['shard_id']}")
            artifact_sha256 = _hash_file(artifact_path)
            if payload.get("artifact_sha256") != artifact_sha256:
                raise ValueError(f"resume artifact byte hash mismatch for {record['shard_id']}")
            result = json.loads(artifact_path.read_text(encoding="utf-8"))
            operation = str(record.get("operation", ""))
            if operation != "scheduler_diagnostic" and not operation.startswith("pilot_"):
                _validate_scientific_artifacts(result, allowed_root=shard_dir)
            if payload.get("parent_hash") != record["parent_hash"]:
                raise ValueError(f"resume artifact parent_hash mismatch for {record['shard_id']}")
            if payload.get("status") != "completed":
                raise ValueError(f"resume artifact status mismatch for {record['shard_id']}")
            if int(payload.get("attempt", 0)) <= int(record["attempt"]):
                raise ValueError(
                    f"resume artifact attempt did not advance for {record['shard_id']}"
                )
            summary["completed"] += 1
        elif failed_marker.exists():
            summary["failed"] += 1
        else:
            summary["pending"] += 1
    return summary


def validate_stage_dependencies(dag: CampaignDAG, stage_name: str) -> None:
    """Fail closed when an upstream reducer is missing or failed."""
    stage = _stage_by_name(dag, stage_name)
    if stage.parent_stage_name is None:
        return

    parent = _stage_by_name(dag, stage.parent_stage_name)
    failed_marker = parent.reducer_output_dir / "_FAILED.json"
    success_marker = parent.reducer_output_dir / "_SUCCESS.json"

    if failed_marker.exists():
        raise ValueError(f"upstream dependency {parent.name} failed")
    if not success_marker.exists():
        raise ValueError(f"upstream dependency {parent.name} is not complete")

    payload = json.loads(success_marker.read_text(encoding="utf-8"))
    if payload.get("stage") != parent.name:
        raise ValueError(f"upstream dependency {parent.name} success marker has wrong stage")
    if payload.get("output_hash") != parent.reducer_output_hash:
        raise ValueError(f"upstream dependency {parent.name} success marker hash mismatch")
    if payload.get("status") != "completed":
        raise ValueError(f"upstream dependency {parent.name} did not complete successfully")
    artifact_path = parent.reducer_output_dir / "reduced_result.json"
    if not artifact_path.is_file() or payload.get("artifact_sha256") != _hash_file(artifact_path):
        raise ValueError(f"upstream dependency {parent.name} reducer artifact is invalid")
    reduced = json.loads(artifact_path.read_text(encoding="utf-8"))
    scientific = reduced.get("scientific_reduction")
    if scientific is not None:
        _validate_scientific_artifacts(scientific, allowed_root=parent.reducer_output_dir)


def build_submission_plan(dag: CampaignDAG) -> dict[str, Any]:
    """Build an exact, ordered NO-SUBMIT plan with fail-closed dependencies."""
    pilot_steps: list[dict[str, Any]] = []
    production_steps: list[dict[str, Any]] = []
    reducer_step_by_stage: dict[str, str] = {}
    for stage in dag.stages:
        is_pilot = stage.name in _PILOT_STAGES
        destination = pilot_steps if is_pilot else production_steps
        locked = not is_pilot
        unlock_requirement = (
            ""
            if is_pilot
            else (
                "accepted_resource_freeze_and_independent_gate_b_pass"
                if stage.name.startswith("applied_")
                else "accepted_resource_freeze_and_phase_specific_upstream_gate"
            )
        )
        parent_steps = (
            []
            if stage.parent_stage_name is None
            else [reducer_step_by_stage[stage.parent_stage_name]]
        )
        worker_step_ids: list[str] = []
        for index, script_path in enumerate(stage.worker_script_paths):
            suffix = f":{index:04d}" if len(stage.worker_script_paths) > 1 else ""
            step_id = f"{stage.name}:worker{suffix}"
            worker_step_ids.append(step_id)
            destination.append(
                {
                    "step_id": step_id,
                    "stage": stage.name,
                    "action": "worker",
                    "command": ["sbatch", "--parsable", str(script_path)],
                    "dependency_type": "afterok" if parent_steps else None,
                    "depends_on": parent_steps,
                    "locked": locked,
                    "unlock_requirement": unlock_requirement,
                }
            )
        audit_step_id = f"{stage.name}:audit"
        destination.append(
            {
                "step_id": audit_step_id,
                "stage": stage.name,
                "action": "audit",
                "command": ["sbatch", "--parsable", str(stage.audit_script_path)],
                "dependency_type": "afterany",
                "depends_on": worker_step_ids,
                "locked": locked,
                "unlock_requirement": unlock_requirement,
            }
        )
        reducer_step_id = f"{stage.name}:reduce"
        destination.append(
            {
                "step_id": reducer_step_id,
                "stage": stage.name,
                "action": "reduce",
                "command": ["sbatch", "--parsable", str(stage.reducer_script_path)],
                "dependency_type": "afterok",
                "depends_on": [*worker_step_ids, audit_step_id],
                "locked": locked,
                "unlock_requirement": unlock_requirement,
            }
        )
        reducer_step_by_stage[stage.name] = reducer_step_id
    plan = {
        "schema_version": 1,
        "status": "NO_SUBMIT",
        "scheduler_submission_permitted": False,
        "run_id": dag.run_id,
        "source_hash": dag.source_hash,
        "config_hash": dag.config_hash,
        "lock_hash": dag.lock_hash,
        "campaign_inventory_hash": dag.campaign_inventory_hash,
        "pilot_steps": pilot_steps,
        "production_steps": production_steps,
    }
    plan["submission_plan_sha256"] = _stable_hash(plan)
    return plan


def execute_submission_plan(
    plan: dict[str, Any],
    *,
    run_command: Any,
    authorize: bool,
    preflight: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Submit only the reviewed pilot plan; any scheduler ambiguity is terminal."""
    if not authorize:
        raise PermissionError("explicit pilot submission authorization is required")
    if preflight is None:
        raise PermissionError("validated pilot preflight evidence is required")
    preflight_identity = {
        key: value for key, value in preflight.items() if key != "preflight_sha256"
    }
    if (
        preflight.get("status") != "HPC_SUBMISSION_READY"
        or preflight.get("target_phase") != "pilot"
        or preflight.get("source_hash") != plan.get("source_hash")
        or preflight.get("config_hash") != plan.get("config_hash")
        or preflight.get("lock_hash") != plan.get("lock_hash")
        or preflight.get("campaign_inventory_hash") != plan.get("campaign_inventory_hash")
        or preflight.get("observed_date") != date.today().isoformat()
        or preflight.get("preflight_sha256") != _stable_hash(preflight_identity)
    ):
        raise PermissionError("pilot preflight is stale or differs from the submission plan")
    if plan.get("status") != "NO_SUBMIT" or plan.get("scheduler_submission_permitted") is not False:
        raise ValueError("submission client requires an unchanged reviewed NO_SUBMIT plan")
    job_ids: dict[str, str] = {}
    for step in plan["pilot_steps"]:
        if step.get("locked"):
            raise PermissionError(f"pilot step {step['step_id']} is locked")
        dependencies = [job_ids.get(step_id) for step_id in step["depends_on"]]
        if any(job_id is None for job_id in dependencies):
            raise RuntimeError(f"submission dependency is missing for {step['step_id']}")
        command = [str(value) for value in step["command"]]
        if dependencies:
            dependency_type = str(step["dependency_type"])
            command.insert(
                2,
                f"--dependency={dependency_type}:{':'.join(str(value) for value in dependencies)}",
            )
        completed = run_command(command)
        if int(completed.returncode) != 0:
            detail = str(completed.stderr).strip() or "scheduler command failed"
            raise RuntimeError(f"scheduler submission failed for {step['step_id']}: {detail}")
        raw_job_id = str(completed.stdout).strip()
        job_id = raw_job_id.split(";", maxsplit=1)[0]
        if re.fullmatch(r"[0-9]+", job_id) is None:
            raise RuntimeError(f"scheduler returned no valid job ID for {step['step_id']}")
        job_ids[str(step["step_id"])] = job_id
    return job_ids


def build_campaign_phase_plan(dag: CampaignDAG, *, phase: str) -> dict[str, Any]:
    """Build one independently gated development or confirmatory tranche."""
    if phase == "development" and dag.package_mode != "full":
        raise ValueError("development phase plan requires a full package")
    if phase != "development" and dag.package_mode not in {"confirmatory", "downstream"}:
        raise ValueError("confirmatory phase plan requires a confirmatory/downstream package")
    stage_predicate = {
        "development": lambda name: name == "resolution",
        "gate_b": lambda name: name in {"gate_b", "fixed_family_supplement"},
        "fixed_family": lambda name: name == "fixed_family_supplement",
        "gate_p": lambda name: name.startswith("applied_"),
        "gate_c": lambda name: name == "recovery",
    }.get(phase)
    if stage_predicate is None:
        raise ValueError("phase must be development, gate_b, fixed_family, gate_p, or gate_c")
    full = build_submission_plan(dag)
    selected = [
        dict(step) for step in full["production_steps"] if stage_predicate(str(step["stage"]))
    ]
    selected_ids = {str(step["step_id"]) for step in selected}
    for step in selected:
        retained_dependencies = [
            dependency for dependency in step["depends_on"] if dependency in selected_ids
        ]
        step["depends_on"] = retained_dependencies
        step["dependency_type"] = step["dependency_type"] if retained_dependencies else None
        step["locked"] = False
        step["unlock_requirement"] = f"accepted_{phase}_authorization"
    plan = {
        "schema_version": 1,
        "status": "NO_SUBMIT",
        "scheduler_submission_permitted": False,
        "phase": phase,
        "run_id": dag.run_id,
        "source_hash": dag.source_hash,
        "config_hash": dag.config_hash,
        "lock_hash": dag.lock_hash,
        "campaign_inventory_hash": dag.campaign_inventory_hash,
        "steps": selected,
    }
    plan["submission_plan_sha256"] = _stable_hash(plan)
    return plan


def execute_campaign_phase_plan(
    plan: dict[str, Any],
    *,
    run_command: Any,
    authorize: bool,
    preflight: dict[str, Any],
    phase_authorization: dict[str, Any],
) -> dict[str, str]:
    """Submit one preflighted, phase-authorized tranche and nothing else."""
    if not authorize:
        raise PermissionError("explicit campaign phase submission authorization is required")
    plan_identity = {key: value for key, value in plan.items() if key != "submission_plan_sha256"}
    if (
        plan.get("status") != "NO_SUBMIT"
        or plan.get("scheduler_submission_permitted") is not False
        or plan.get("submission_plan_sha256") != _stable_hash(plan_identity)
    ):
        raise ValueError("campaign submission plan is not an unchanged NO-SUBMIT plan")
    preflight_identity = {
        key: value for key, value in preflight.items() if key != "preflight_sha256"
    }
    if (
        preflight.get("status") != "HPC_SUBMISSION_READY"
        or preflight.get("target_phase") != plan.get("phase")
        or preflight.get("source_hash") != plan.get("source_hash")
        or preflight.get("config_hash") != plan.get("config_hash")
        or preflight.get("lock_hash") != plan.get("lock_hash")
        or preflight.get("campaign_inventory_hash") != plan.get("campaign_inventory_hash")
        or preflight.get("observed_date") != date.today().isoformat()
        or preflight.get("preflight_sha256") != _stable_hash(preflight_identity)
    ):
        raise PermissionError("campaign preflight is stale or differs from the plan")
    authorization_identity = {
        key: value for key, value in phase_authorization.items() if key != "authorization_sha256"
    }
    if (
        phase_authorization.get("schema_version") != 2
        or phase_authorization.get("status") != "ACCEPTED"
        or phase_authorization.get("phase") != plan.get("phase")
        or phase_authorization.get("run_id") != plan.get("run_id")
        or phase_authorization.get("contract_hash") != plan.get("config_hash")
        or phase_authorization.get("source_hash") != plan.get("source_hash")
        or phase_authorization.get("lock_hash") != plan.get("lock_hash")
        or phase_authorization.get("campaign_inventory_hash") != plan.get("campaign_inventory_hash")
        or phase_authorization.get("submission_plan_sha256") != plan.get("submission_plan_sha256")
        or phase_authorization.get("preflight_sha256") != preflight.get("preflight_sha256")
        or phase_authorization.get("execution_permitted") is not True
        or phase_authorization.get("resource_freeze_sha256")
        != preflight.get("resource_freeze_sha256")
        or phase_authorization.get("authorization_sha256") != _stable_hash(authorization_identity)
    ):
        raise PermissionError("campaign phase authorization is stale or differs")

    job_ids: dict[str, str] = {}
    for step in plan["steps"]:
        dependencies = [job_ids.get(step_id) for step_id in step["depends_on"]]
        if any(job_id is None for job_id in dependencies):
            raise RuntimeError(f"submission dependency is missing for {step['step_id']}")
        command = [str(value) for value in step["command"]]
        if dependencies:
            command.insert(
                2,
                f"--dependency={step['dependency_type']}:"
                f"{':'.join(str(value) for value in dependencies)}",
            )
        completed = run_command(command)
        if int(completed.returncode) != 0:
            detail = str(completed.stderr).strip() or "scheduler command failed"
            raise RuntimeError(f"scheduler submission failed for {step['step_id']}: {detail}")
        job_id = str(completed.stdout).strip().split(";", maxsplit=1)[0]
        if re.fullmatch(r"[0-9]+", job_id) is None:
            raise RuntimeError(f"scheduler returned no valid job ID for {step['step_id']}")
        job_ids[str(step["step_id"])] = job_id
    return job_ids


def plan_retry_attempts(
    records: list[dict[str, Any]],
    scheduler_states: dict[str, str],
) -> list[dict[str, Any]]:
    """Plan the sole permitted retry without changing scientific identity.

    Only scheduler infrastructure states frozen in the campaign contract may
    be retried. Scientific failures and exhausted attempts remain terminal.
    The returned record is byte-for-byte identical in all scientific identity
    fields; only ``attempt`` and ``status`` change.
    """
    by_shard: dict[str, dict[str, Any]] = {}
    for record in records:
        shard_id = str(record.get("shard_id", ""))
        if not shard_id or shard_id in by_shard:
            raise ValueError("retry planning requires unique non-empty shard IDs")
        by_shard[shard_id] = record
    unknown = sorted(set(scheduler_states).difference(by_shard))
    if unknown:
        raise ValueError(f"retry state contains unknown shard IDs: {unknown}")

    retries: list[dict[str, Any]] = []
    for shard_id, scheduler_state in scheduler_states.items():
        record = by_shard[shard_id]
        retryable = tuple(str(value) for value in record["retryable_scheduler_states"])
        attempt = int(record["attempt"])
        max_retries = int(record["max_retries"])
        if scheduler_state not in retryable or attempt >= max_retries:
            continue
        retry = dict(record)
        retry["attempt"] = attempt + 1
        retry["status"] = "pending_retry"
        retries.append(retry)
    return retries


def prepare_stage_retry_package(
    dag: CampaignDAG,
    *,
    stage_name: str,
    scheduler_states: dict[str, str],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Render a complete NO-SUBMIT retry, audit, and reduction dependency chain."""
    stage = _stage_by_name(dag, stage_name)
    records = load_stage_manifest(stage.manifest_path)
    retries = plan_retry_attempts(records, scheduler_states)
    if not retries:
        raise ValueError("no scheduler state is eligible for the sole frozen retry")
    retry_by_shard = {str(record["shard_id"]): record for record in retries}
    merged = [retry_by_shard.get(str(record["shard_id"]), record) for record in records]
    destination = Path(output_dir).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("retry package output directory must be empty")
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "retry_manifest.jsonl"
    _write_jsonl(manifest_path, merged)
    script_paths: list[Path] = []
    for task_id, record in enumerate(merged):
        if record["shard_id"] not in retry_by_shard:
            continue
        resources = ResourceEnvelope(**record["worker_resources"])
        script_path = destination / f"retry-{task_id:04d}.slurm"
        script_path.write_text(
            _render_worker_script(
                stage_name=stage.name,
                partition=str(record.get("partition", stage.partition)),
                cluster=dag.cluster,
                resources=resources,
                manifest_path=manifest_path,
                output_root=stage.output_root,
                contract_config_path=dag.contract_config_path,
                contract_hash=dag.config_hash,
                job_count=len(merged),
                run_id=dag.run_id,
                task_id=task_id,
                task_range_start=int(record.get("dispatch_range_start", 0)),
                task_range_end=int(record.get("dispatch_range_end", len(merged))),
            ),
            encoding="utf-8",
        )
        script_paths.append(script_path)
    retry_audit_output_dir = destination / "audit"
    audit_script_path = destination / "retry-audit.slurm"
    reducer_script_path = destination / "retry-reduce.slurm"
    parent_hashes = {str(record["parent_hash"]) for record in merged}
    if len(parent_hashes) != 1:
        raise ValueError("retry stage manifest has more than one parent hash")
    audit_script_path.write_text(
        _render_audit_script(
            stage_name=stage.name,
            partition=stage.partition,
            cluster=dag.cluster,
            resources=stage.reducer_resources,
            manifest_path=manifest_path,
            audit_output_dir=retry_audit_output_dir,
            contract_hash=dag.config_hash,
            run_id=dag.run_id,
        ),
        encoding="utf-8",
    )
    reducer_script_path.write_text(
        _render_reducer_script(
            stage_name=stage.name,
            partition=stage.partition,
            cluster=dag.cluster,
            resources=stage.reducer_resources,
            manifest_path=manifest_path,
            reducer_output_dir=stage.reducer_output_dir,
            audit_success_path=retry_audit_output_dir / "_SUCCESS.json",
            contract_config_path=dag.contract_config_path,
            contract_hash=dag.config_hash,
            job_count=len(merged),
            parent_hash=parent_hashes.pop(),
            run_id=dag.run_id,
        ),
        encoding="utf-8",
    )
    worker_steps = [
        {
            "step_id": f"{stage.name}:retry-worker:{index:04d}",
            "stage": stage.name,
            "action": "worker",
            "command": ["sbatch", "--parsable", str(script_path)],
            "dependency_type": None,
            "depends_on": [],
        }
        for index, script_path in enumerate(script_paths)
    ]
    worker_step_ids = [str(step["step_id"]) for step in worker_steps]
    audit_step_id = f"{stage.name}:retry-audit"
    submission_steps = [
        *worker_steps,
        {
            "step_id": audit_step_id,
            "stage": stage.name,
            "action": "audit",
            "command": ["sbatch", "--parsable", str(audit_script_path)],
            "dependency_type": "afterany",
            "depends_on": worker_step_ids,
        },
        {
            "step_id": f"{stage.name}:retry-reduce",
            "stage": stage.name,
            "action": "reduce",
            "command": ["sbatch", "--parsable", str(reducer_script_path)],
            "dependency_type": "afterok",
            "depends_on": [*worker_step_ids, audit_step_id],
        },
    ]
    submission_scripts = [*script_paths, audit_script_path, reducer_script_path]
    phase = "pilot" if stage.name in _PILOT_STAGES else str(merged[0].get("phase", ""))
    if phase not in {"pilot", "development", "gate_b", "gate_p", "gate_c"}:
        raise ValueError("retry stage has no valid campaign phase")
    payload = {
        "schema_version": 2,
        "status": "NO_SUBMIT",
        "scheduler_submission_permitted": False,
        "phase": phase,
        "run_id": dag.run_id,
        "source_hash": dag.source_hash,
        "config_hash": dag.config_hash,
        "lock_hash": dag.lock_hash,
        "campaign_inventory_hash": dag.campaign_inventory_hash,
        "stage": stage.name,
        "original_manifest_sha256": _hash_file(stage.manifest_path),
        "retry_manifest": str(manifest_path),
        "retry_manifest_sha256": _hash_file(manifest_path),
        "retry_shard_ids": sorted(retry_by_shard),
        "retry_scripts": [str(path) for path in script_paths],
        "retry_script_sha256": {str(path): _hash_file(path) for path in script_paths},
        "audit_script": str(audit_script_path),
        "reducer_script": str(reducer_script_path),
        "submission_steps": submission_steps,
        "submission_script_sha256": {str(path): _hash_file(path) for path in submission_scripts},
    }
    payload["retry_package_sha256"] = _stable_hash(payload)
    (destination / "retry_package.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def execute_retry_submission_plan(
    retry_package: dict[str, Any],
    *,
    run_command: Any,
    authorize: bool,
    preflight: dict[str, Any],
    phase_authorization: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Test and submit one rebuilt retry chain, failing closed on any ambiguity."""
    if not authorize:
        raise PermissionError("explicit retry submission authorization is required")
    package_identity = {
        key: value for key, value in retry_package.items() if key != "retry_package_sha256"
    }
    if (
        retry_package.get("schema_version") != 2
        or retry_package.get("status") != "NO_SUBMIT"
        or retry_package.get("scheduler_submission_permitted") is not False
        or retry_package.get("retry_package_sha256") != _stable_hash(package_identity)
    ):
        raise ValueError("retry package is not an unchanged NO-SUBMIT package")
    preflight_identity = {
        key: value for key, value in preflight.items() if key != "preflight_sha256"
    }
    identity_fields = ("source_hash", "config_hash", "lock_hash", "campaign_inventory_hash")
    if (
        preflight.get("status") != "HPC_SUBMISSION_READY"
        or preflight.get("target_phase") != retry_package.get("phase")
        or preflight.get("observed_date") != date.today().isoformat()
        or any(preflight.get(field) != retry_package.get(field) for field in identity_fields)
        or preflight.get("preflight_sha256") != _stable_hash(preflight_identity)
    ):
        raise PermissionError("retry preflight is stale or differs from the retry package")
    if retry_package.get("phase") != "pilot":
        if phase_authorization is None:
            raise PermissionError("confirmatory retry requires phase authorization")
        authorization_identity = {
            key: value
            for key, value in phase_authorization.items()
            if key != "authorization_sha256"
        }
        if (
            phase_authorization.get("schema_version") != 2
            or phase_authorization.get("status") != "ACCEPTED"
            or phase_authorization.get("phase") != retry_package.get("phase")
            or phase_authorization.get("run_id") != retry_package.get("run_id")
            or phase_authorization.get("contract_hash") != retry_package.get("config_hash")
            or any(
                phase_authorization.get(field) != retry_package.get(field)
                for field in ("source_hash", "lock_hash", "campaign_inventory_hash")
            )
            or phase_authorization.get("preflight_sha256") != preflight.get("preflight_sha256")
            or phase_authorization.get("execution_permitted") is not True
            or phase_authorization.get("resource_freeze_sha256")
            != preflight.get("resource_freeze_sha256")
            or phase_authorization.get("authorization_sha256")
            != _stable_hash(authorization_identity)
        ):
            raise PermissionError("retry phase authorization is stale or differs")

    expected_scripts = retry_package.get("submission_script_sha256")
    actual_scripts = {
        str(path): _hash_file(Path(str(path))) for path in dict(expected_scripts or {})
    }
    if not actual_scripts or actual_scripts != expected_scripts:
        raise ValueError("retry submission script bytes differ from the reviewed package")
    for path in actual_scripts:
        tested = run_command(["sbatch", "--test-only", path])
        if int(tested.returncode) != 0:
            detail = str(tested.stderr).strip() or "scheduler test-only check failed"
            raise RuntimeError(f"retry sbatch --test-only failed for {path}: {detail}")

    job_ids: dict[str, str] = {}
    for step in retry_package["submission_steps"]:
        dependencies = [job_ids.get(step_id) for step_id in step["depends_on"]]
        if any(job_id is None for job_id in dependencies):
            raise RuntimeError(f"retry submission dependency is missing for {step['step_id']}")
        command = [str(value) for value in step["command"]]
        if dependencies:
            command.insert(
                2,
                f"--dependency={step['dependency_type']}:"
                f"{':'.join(str(value) for value in dependencies)}",
            )
        completed = run_command(command)
        if int(completed.returncode) != 0:
            detail = str(completed.stderr).strip() or "scheduler command failed"
            raise RuntimeError(f"retry submission failed for {step['step_id']}: {detail}")
        job_id = str(completed.stdout).strip().split(";", maxsplit=1)[0]
        if re.fullmatch(r"[0-9]+", job_id) is None:
            raise RuntimeError(f"scheduler returned no valid job ID for {step['step_id']}")
        job_ids[str(step["step_id"])] = job_id
    return job_ids


def _load_campaign_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("campaign config must deserialize to a mapping")
    if "hpc_campaign" in raw:
        raw = raw["hpc_campaign"]
    required = {"run_id", "scheduler_submission_permitted", "cluster"}
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(f"campaign config missing required field(s): {missing}")
    if not isinstance(raw["cluster"], dict):
        raise ValueError("campaign config cluster must be a mapping")
    return raw


def _cluster_from_mapping(raw: dict[str, Any]) -> ClusterConfig:
    cluster = ClusterConfig(
        account=str(raw["account"]),
        max_array_size=int(raw["max_array_size"]),
        max_array_concurrency=int(raw["max_array_concurrency"]),
        cpu_cores_per_node=int(raw["cpu_cores_per_node"]),
        memory_per_node_gb=int(raw["memory_per_node_gb"]),
        partitions=tuple(str(value) for value in raw["partitions"]),
        project_root=str(raw["project_root"]),
        scratch_template=str(raw["scratch_template"]),
        kill_wait=str(raw["kill_wait"]),
        job_requeue=bool(raw["job_requeue"]),
        allocation_unit=str(raw["allocation_unit"]),
        cpu_charge_factor=float(raw["cpu_charge_factor"]),
        qos_factor=float(raw["qos_factor"]),
        allocation_quota=str(raw["allocation_quota"]),
        rfm_repository_root=str(raw["rfm_repository_root"]),
        bsm_repository_root=str(raw["bsm_repository_root"]),
        scientific_adapter_relative_path=str(raw["scientific_adapter_relative_path"]),
        scientific_adapter_sha256=str(raw["scientific_adapter_sha256"]),
        bsm_recovery_driver_relative_path=str(raw["bsm_recovery_driver_relative_path"]),
        bsm_recovery_driver_sha256=str(raw["bsm_recovery_driver_sha256"]),
        bsm_dgp_contract_relative_path=str(raw["bsm_dgp_contract_relative_path"]),
        bsm_dgp_contract_sha256=str(raw["bsm_dgp_contract_sha256"]),
        applied_data_preparer_relative_path=str(raw["applied_data_preparer_relative_path"]),
        applied_data_preparer_sha256=str(raw["applied_data_preparer_sha256"]),
        applied_config_relative_path=str(raw["applied_config_relative_path"]),
        applied_config_sha256=str(raw["applied_config_sha256"]),
        applied_data_root=str(raw["applied_data_root"]),
        applied_data_manifest_sha256=str(raw["applied_data_manifest_sha256"]),
        authorization_root=str(raw["authorization_root"]),
    )
    if cluster.account != "nationalpfa":
        raise ValueError("Kestrel package must target account=nationalpfa")
    if cluster.max_array_size <= 0 or cluster.max_array_concurrency <= 0:
        raise ValueError("Kestrel array limits must be positive")
    if cluster.max_array_concurrency > cluster.max_array_size:
        raise ValueError("max_array_concurrency cannot exceed max_array_size")
    if cluster.cpu_cores_per_node != 104 or cluster.memory_per_node_gb != 240:
        raise ValueError("Kestrel package must use the committed 104-core / 240-GB node shape")
    if cluster.allocation_unit != "AU":
        raise ValueError("Kestrel package allocation_unit must be AU")
    if cluster.cpu_charge_factor != 10.0 or cluster.qos_factor <= 0.0:
        raise ValueError("Kestrel CPU charge and QoS factors must be explicit and positive")
    if not cluster.partitions:
        raise ValueError("Kestrel package must list at least one partition")
    for field_name in (
        "scientific_adapter_sha256",
        "bsm_recovery_driver_sha256",
        "bsm_dgp_contract_sha256",
        "applied_data_preparer_sha256",
        "applied_config_sha256",
    ):
        if re.fullmatch(r"[0-9a-f]{64}", getattr(cluster, field_name)) is None:
            raise ValueError(f"cluster {field_name} must be a SHA-256 digest")
    if (
        cluster.applied_data_manifest_sha256 != "UNKNOWN"
        and re.fullmatch(r"[0-9a-f]{64}", cluster.applied_data_manifest_sha256) is None
    ):
        raise ValueError("cluster applied_data_manifest_sha256 must be UNKNOWN or SHA-256")
    for field_name in (
        "rfm_repository_root",
        "bsm_repository_root",
        "applied_data_root",
        "authorization_root",
    ):
        if not getattr(cluster, field_name).startswith("/"):
            raise ValueError(f"cluster {field_name} must be an absolute path")
    return cluster


def _build_pilot_matrix(cluster: ClusterConfig) -> tuple[dict[str, Any], ...]:
    matrix = tuple(row for stage in _PILOT_STAGES for row in _pilot_profile_rows(stage))
    for row in matrix:
        if row["partition"] not in cluster.partitions:
            raise ValueError(f"pilot matrix uses uncommitted partition {row['partition']}")
    return matrix


def _select_post_pilot_profiles(
    pilot_matrix: tuple[dict[str, Any], ...],
) -> dict[str, dict[str, Any]]:
    return {
        stage: {
            "status": "PENDING_TELEMETRY",
            "candidate_schedule_hashes": [
                row["schedule_hash"] for row in pilot_matrix if row["stage"] == stage
            ],
            "selector": "minimum_projected_au_subject_to_resource_bounds",
        }
        for stage in dict.fromkeys(str(row["stage"]) for row in pilot_matrix)
    }


def collect_pilot_accounting(
    dag: CampaignDAG,
    *,
    submission_job_ids: dict[str, str],
    output_dir: str | Path,
    run_command: Any | None = None,
    stage_names: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Join verified worker results to complete top-level Slurm accounting."""
    selected_names = set(_PILOT_STAGES if stage_names is None else stage_names)
    if not selected_names or not selected_names <= set(_PILOT_STAGES):
        raise ValueError("pilot accounting stage selection is empty or invalid")
    plan_steps = [
        step
        for step in build_submission_plan(dag)["pilot_steps"]
        if str(step["stage"]) in selected_names
    ]
    expected_step_ids = {str(step["step_id"]) for step in plan_steps}
    if set(submission_job_ids) != expected_step_ids:
        raise ValueError("pilot accounting job IDs do not exactly cover the selected plan")
    if any(re.fullmatch(r"[0-9]+", str(value)) is None for value in submission_job_ids.values()):
        raise ValueError("pilot accounting contains an invalid scheduler job ID")

    destination = Path(output_dir).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("pilot accounting output directory must be empty")
    destination.mkdir(parents=True, exist_ok=True)
    runner = subprocess.run if run_command is None else run_command
    field_names = (
        "job_id",
        "job_name",
        "partition",
        "scheduler_state",
        "scheduler_exit_code",
        "scheduler_elapsed_seconds",
        "allocated_nodes",
        "allocated_cpus",
        "requested_cpus",
        "requested_memory",
        "allocated_tres",
        "cpu_time_raw",
        "total_cpu",
        "max_rss",
        "max_disk_read",
        "max_disk_write",
    )
    job_ids = sorted(set(submission_job_ids.values()), key=int)
    completed = runner(
        [
            "sacct",
            "-j",
            ",".join(job_ids),
            "-nP",
            "--format="
            + ",".join(
                (
                    "JobIDRaw",
                    "JobName",
                    "Partition",
                    "State",
                    "ExitCode",
                    "ElapsedRaw",
                    "AllocNodes",
                    "AllocCPUS",
                    "ReqCPUS",
                    "ReqMem",
                    "AllocTRES%120",
                    "CPUTimeRAW",
                    "TotalCPU",
                    "MaxRSS",
                    "MaxDiskRead",
                    "MaxDiskWrite",
                )
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    raw_text = str(completed.stdout)
    raw_path = destination / "sacct_raw.psv"
    raw_path.write_text(raw_text, encoding="utf-8")
    all_scheduler_rows: dict[str, dict[str, Any]] = {}
    for line in raw_text.splitlines():
        values = line.split("|")
        if len(values) != len(field_names):
            raise ValueError("pilot sacct output does not match the frozen field schema")
        raw = dict(zip(field_names, values, strict=True))
        job_id = str(raw["job_id"])
        if job_id in all_scheduler_rows:
            raise ValueError(f"pilot sacct output duplicates job row {job_id}")
        try:
            is_top_level = job_id in job_ids
            all_scheduler_rows[job_id] = {
                **raw,
                "scheduler_elapsed_seconds": int(raw["scheduler_elapsed_seconds"]),
                "allocated_nodes": int(raw["allocated_nodes"]),
                "allocated_cpus": int(raw["allocated_cpus"]),
                "requested_cpus": int(raw["requested_cpus"]) if is_top_level else 0,
                "cpu_time_raw": int(raw["cpu_time_raw"]),
            }
        except ValueError as exc:
            raise ValueError(f"pilot sacct row {job_id} has invalid numeric fields") from exc
    scheduler_rows = {
        job_id: all_scheduler_rows[job_id] for job_id in job_ids if job_id in all_scheduler_rows
    }
    if set(scheduler_rows) != set(job_ids):
        raise ValueError("pilot sacct output lacks complete top-level job coverage")
    for job_id, row in scheduler_rows.items():
        if row["scheduler_state"] != "COMPLETED" or row["scheduler_exit_code"] != "0:0":
            raise ValueError(f"pilot scheduler job {job_id} did not complete successfully")
        if (
            row["scheduler_elapsed_seconds"] <= 0
            or row["allocated_nodes"] <= 0
            or row["allocated_cpus"] <= 0
            or not str(row["allocated_tres"]).strip()
        ):
            raise ValueError(f"pilot scheduler job {job_id} has incomplete allocation evidence")

    def _slurm_memory_bytes(raw: str) -> int:
        match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([KMGTP]?)", raw.strip(), flags=re.IGNORECASE)
        if match is None:
            if not raw.strip():
                return 0
            raise ValueError(f"pilot sacct MaxRSS value is invalid: {raw}")
        scale = 1024 ** ("KMGTP".find(match.group(2).upper()) + 1) if match.group(2) else 1
        return math.ceil(float(match.group(1)) * scale)

    telemetry: list[dict[str, Any]] = []
    worker_job_ids: set[str] = set()
    for stage in dag.stages:
        if stage.name not in selected_names:
            continue
        validate_resume_artifacts(stage)
        records = load_stage_manifest(stage.manifest_path)
        worker_steps = [
            step
            for step in plan_steps
            if step["stage"] == stage.name and step["action"] == "worker"
        ]
        if len(worker_steps) != len(records):
            raise ValueError(f"pilot worker plan differs from manifest for {stage.name}")
        for record, step in zip(records, worker_steps, strict=True):
            job_id = str(submission_job_ids[str(step["step_id"])])
            result_path = Path(record["output_dir"]) / "result.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if str(result.get("slurm_job_id", "")) != job_id:
                raise ValueError(f"pilot worker result differs from submitted job ID {job_id}")
            scheduler = scheduler_rows[job_id]
            child_rss = max(
                (
                    _slurm_memory_bytes(str(row["max_rss"]))
                    for child_id, row in all_scheduler_rows.items()
                    if child_id.startswith(f"{job_id}.")
                ),
                default=0,
            )
            scheduler_max_rss_bytes = max(
                _slurm_memory_bytes(str(scheduler["max_rss"])),
                child_rss,
            )
            telemetry.append(
                {
                    **result,
                    "scheduler_state": scheduler["scheduler_state"],
                    "scheduler_exit_code": scheduler["scheduler_exit_code"],
                    "scheduler_elapsed_seconds": scheduler["scheduler_elapsed_seconds"],
                    "allocated_nodes": scheduler["allocated_nodes"],
                    "allocated_cpus": scheduler["allocated_cpus"],
                    "allocated_tres": scheduler["allocated_tres"],
                    "scheduler_max_rss_bytes": scheduler_max_rss_bytes,
                    "scheduler_cpu_time_raw": scheduler["cpu_time_raw"],
                    "scheduler_total_cpu": scheduler["total_cpu"],
                    "scheduler_max_rss": scheduler["max_rss"],
                    "scheduler_max_disk_read": scheduler["max_disk_read"],
                    "scheduler_max_disk_write": scheduler["max_disk_write"],
                }
            )
            worker_job_ids.add(job_id)

    def _observed_au(ids: set[str]) -> float:
        return sum(
            int(scheduler_rows[job_id]["scheduler_elapsed_seconds"])
            / 3600.0
            * int(scheduler_rows[job_id]["allocated_nodes"])
            * dag.cluster.cpu_charge_factor
            * dag.cluster.qos_factor
            for job_id in ids
        )

    canonical_telemetry = sorted(
        telemetry, key=lambda row: (str(row["stage"]), str(row["profile_id"]))
    )
    payload = {
        "schema_version": 1,
        "status": "COMPLETE",
        "run_id": dag.run_id,
        "source_hash": dag.source_hash,
        "config_hash": dag.config_hash,
        "lock_hash": dag.lock_hash,
        "campaign_inventory_hash": dag.campaign_inventory_hash,
        "submission_job_ids": dict(sorted(submission_job_ids.items())),
        "sacct_raw_sha256": _hash_file(raw_path),
        "pilot_telemetry": canonical_telemetry,
        "pilot_telemetry_sha256": _stable_hash(canonical_telemetry),
        "observed_worker_au": _observed_au(worker_job_ids),
        "observed_total_au": _observed_au(set(job_ids)),
    }
    payload["pilot_accounting_sha256"] = _stable_hash(payload)
    (destination / "pilot_accounting.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def select_pilot_resources(
    *,
    pilot_matrix: tuple[dict[str, Any], ...],
    telemetry: list[dict[str, Any]],
    cluster: ClusterConfig,
    source_hash: str,
    config_hash: str,
    lock_hash: str,
) -> dict[str, Any]:
    """Validate complete pilot evidence and freeze the minimum-projected-AU profiles."""
    expected = {(str(row["stage"]), str(row["profile_id"])): row for row in pilot_matrix}
    if len(expected) != len(pilot_matrix):
        raise ValueError("pilot matrix contains duplicate stage/profile identities")
    observed: dict[tuple[str, str], dict[str, Any]] = {}
    for row in telemetry:
        identity = (str(row.get("stage", "")), str(row.get("profile_id", "")))
        if identity not in expected:
            raise ValueError(f"unexpected pilot telemetry identity {identity}")
        if identity in observed:
            raise ValueError(f"duplicate pilot telemetry identity {identity}")
        profile = expected[identity]
        if row.get("status") != "completed":
            raise ValueError(f"pilot {identity} did not complete")
        if (
            row.get("source_hash") != source_hash
            or row.get("config_hash") != config_hash
            or row.get("lock_hash") != lock_hash
        ):
            raise ValueError(f"pilot {identity} source, contract, or lock identity differs")
        if row.get("schedule_hash") != profile["schedule_hash"]:
            raise ValueError(f"pilot {identity} schedule hash differs from its profile")
        for field, telemetry_field in (
            ("block_size", "block_size"),
            ("executed_work_units", "executed_work_units"),
            ("task_size", "task_size"),
            ("cpus", "requested_cpus"),
            ("memory_gb", "requested_memory_gb"),
            ("walltime_seconds", "requested_walltime_seconds"),
            ("partition", "partition"),
        ):
            if row.get(telemetry_field) != profile[field]:
                label = field.replace("_", " ")
                raise ValueError(f"pilot {identity} {label} differs from its frozen profile")
        elapsed = float(row.get("elapsed_seconds", 0.0))
        max_rss = max(
            int(row.get("max_rss_bytes", -1)),
            int(row.get("scheduler_max_rss_bytes", -1)),
        )
        if not 0.0 < elapsed <= int(profile["walltime_seconds"]):
            raise ValueError(f"pilot {identity} elapsed time exceeds its walltime")
        if not 0 <= max_rss <= int(profile["memory_gb"]) * 1024**3:
            raise ValueError(f"pilot {identity} memory use exceeds its request")
        for field in (
            "total_cpu_seconds",
            "cpu_time_seconds",
            "bytes_read",
            "bytes_written",
        ):
            if float(row.get(field, -1.0)) < 0.0:
                raise ValueError(f"pilot {identity} has invalid {field}")
        if row.get("scheduler_state") != "COMPLETED" or row.get("scheduler_exit_code") != "0:0":
            raise ValueError(f"pilot {identity} scheduler job did not complete successfully")
        scheduler_elapsed = int(row.get("scheduler_elapsed_seconds", 0))
        allocated_nodes = int(row.get("allocated_nodes", 0))
        allocated_cpus = int(row.get("allocated_cpus", 0))
        if not 0 < scheduler_elapsed <= int(profile["walltime_seconds"]):
            raise ValueError(f"pilot {identity} scheduler elapsed time exceeds its walltime")
        if allocated_nodes < 1 or allocated_cpus < int(profile["cpus"]):
            raise ValueError(f"pilot {identity} scheduler allocation is incomplete")
        if re.fullmatch(r"[0-9]+", str(row.get("slurm_job_id", ""))) is None:
            raise ValueError(f"pilot {identity} scheduler job ID is invalid")
        if (
            not str(row.get("hostname", "")).strip()
            or not str(row.get("allocated_tres", "")).strip()
        ):
            raise ValueError(f"pilot {identity} scheduler identity is incomplete")
        observed[identity] = row

    missing = sorted(set(expected) - set(observed))
    if missing:
        raise ValueError(f"missing pilot telemetry for {missing}")

    selections: dict[str, dict[str, Any]] = {}
    for stage in dict.fromkeys(str(row["stage"]) for row in pilot_matrix):
        profiles = [row for row in pilot_matrix if row["stage"] == stage]
        evidence = [observed[(stage, str(profile["profile_id"]))] for profile in profiles]
        if stage == "scheduler_diagnostic":
            selections[stage] = {
                "selected_profile_id": "p0",
                "status": "ACCEPTED",
            }
            continue

        candidates: list[tuple[float, str, dict[str, Any], dict[str, Any]]] = []
        for profile, result in zip(profiles, evidence, strict=True):
            projected_au_per_work_unit = (
                float(result["scheduler_elapsed_seconds"])
                / 3600.0
                * int(result["allocated_nodes"])
                * cluster.cpu_charge_factor
                * cluster.qos_factor
                / int(profile["executed_work_units"])
            )
            candidates.append(
                (
                    projected_au_per_work_unit,
                    str(profile["profile_id"]),
                    profile,
                    result,
                )
            )
        projected_au, selected_id, selected_profile, selected_result = min(
            candidates, key=lambda value: (value[0], value[1])
        )
        selected_block_size = int(selected_profile["block_size"])
        if stage == "pilot_bootstrap":
            target_work_units = selected_block_size * 23495 * _PILOT_APPLIED_MODEL_COUNT
        else:
            target_work_units = (
                int(selected_profile["task_size"])
                if stage in _PILOT_FULL_TASK_STAGES
                else selected_block_size
            )
        if stage == "pilot_bootstrap":
            largest_scaled_rss_gb = (
                max(
                    int(selected_result["max_rss_bytes"]),
                    int(selected_result["scheduler_max_rss_bytes"]),
                )
                / 1024**3
                * 23495
                / _PILOT_OUTPUT_COUNT_BY_PROFILE[selected_id]
            )
        else:
            largest_scaled_rss_gb = (
                max(
                    int(selected_result["max_rss_bytes"]),
                    int(selected_result["scheduler_max_rss_bytes"]),
                )
                / 1024**3
                * target_work_units
                / int(selected_profile["executed_work_units"])
            )
        requested_memory_gb = math.ceil(1.5 * largest_scaled_rss_gb + 2.0)
        if requested_memory_gb > cluster.memory_per_node_gb:
            raise ValueError(f"pilot {stage} memory bound crosses the standard node class")
        selected_scaled_seconds = (
            float(selected_result["scheduler_elapsed_seconds"])
            / int(selected_profile["executed_work_units"])
            * target_work_units
        )
        requested_walltime_seconds = math.ceil(1.5 * selected_scaled_seconds + 300.0)
        if requested_walltime_seconds <= 4 * 3600:
            partition = "short"
        elif requested_walltime_seconds <= 2 * 86400:
            partition = "standard"
        else:
            raise ValueError(f"pilot {stage} projects beyond the supported standard partition")
        selections[stage] = {
            "status": "ACCEPTED",
            "selected_profile_id": selected_id,
            "selected_schedule_hash": selected_profile["schedule_hash"],
            "requested_cpus": int(selected_profile["cpus"]),
            "requested_memory_gb": requested_memory_gb,
            "requested_walltime_seconds": requested_walltime_seconds,
            "partition": partition,
            "task_size": int(selected_profile["task_size"]),
            "block_size": selected_block_size,
            "production_target_work_units": target_work_units,
            "projected_au_per_work_unit": projected_au,
            "observed_scheduler_elapsed_seconds": int(
                observed[(stage, selected_id)]["scheduler_elapsed_seconds"]
            ),
            "observed_allocated_nodes": int(observed[(stage, selected_id)]["allocated_nodes"]),
        }

    canonical_telemetry = sorted(
        telemetry, key=lambda row: (str(row["stage"]), str(row["profile_id"]))
    )
    freeze = {
        "schema_version": 1,
        "status": "ACCEPTED",
        "selector": "minimum_projected_au_subject_to_resource_bounds",
        "source_hash": source_hash,
        "config_hash": config_hash,
        "lock_hash": lock_hash,
        "telemetry_sha256": _stable_hash(canonical_telemetry),
        "selections": selections,
    }
    freeze["resource_freeze_sha256"] = _stable_hash(freeze)
    return freeze


def _validate_resource_freeze(freeze: dict[str, Any]) -> None:
    """Validate a complete immutable post-pilot resource decision."""
    if freeze.get("status") != "ACCEPTED" or freeze.get("schema_version") != 1:
        raise ValueError("resource freeze is not an accepted schema-v1 decision")
    recorded_hash = str(freeze.get("resource_freeze_sha256", ""))
    identity = {key: value for key, value in freeze.items() if key != "resource_freeze_sha256"}
    if (
        re.fullmatch(r"[0-9a-f]{64}", recorded_hash) is None
        or _stable_hash(identity) != recorded_hash
    ):
        raise ValueError("resource freeze self-hash differs")
    selections = freeze.get("selections")
    if not isinstance(selections, dict) or set(selections) != set(_PILOT_STAGES):
        raise ValueError("resource freeze does not cover every frozen pilot kernel")
    for stage in _PILOT_STAGES:
        selection = selections[stage]
        if not isinstance(selection, dict) or selection.get("status") != "ACCEPTED":
            raise ValueError(f"resource freeze does not accept {stage}")
        if stage == "scheduler_diagnostic":
            continue
        for field in (
            "requested_cpus",
            "requested_memory_gb",
            "requested_walltime_seconds",
            "block_size",
        ):
            if int(selection.get(field, 0)) <= 0:
                raise ValueError(f"resource freeze {stage} has invalid {field}")
        if selection.get("partition") not in {"short", "standard"}:
            raise ValueError(f"resource freeze {stage} has invalid production partition")
    for field in ("source_hash", "config_hash", "lock_hash", "telemetry_sha256"):
        if re.fullmatch(r"[0-9a-f]{64}", str(freeze.get(field, ""))) is None:
            raise ValueError(f"resource freeze has invalid {field}")


def _frozen_resource_envelope(
    selection: dict[str, Any], *, walltime_scale: float = 1.0
) -> tuple[str, ResourceEnvelope]:
    requested_cpu = int(selection["requested_cpus"])
    requested_memory = int(selection["requested_memory_gb"])
    requested_walltime = math.ceil(int(selection["requested_walltime_seconds"]) * walltime_scale)
    if requested_walltime <= 4 * 3600:
        partition = "short"
    elif requested_walltime <= 2 * 86400:
        partition = "standard"
    else:
        raise ValueError("resource-frozen workload exceeds the supported two-day envelope")
    return partition, ResourceEnvelope(
        estimated_cpu_cores=max(1, math.floor(requested_cpu / 1.25)),
        requested_cpu_cores=requested_cpu,
        estimated_memory_gb=max(1, math.floor(requested_memory / 1.25)),
        requested_memory_gb=requested_memory,
        estimated_walltime_seconds=max(1, math.floor(requested_walltime / 1.25)),
        requested_walltime_seconds=requested_walltime,
    )


def _apply_resource_freeze_to_stage_spec(
    spec: dict[str, Any],
    freeze: dict[str, Any],
    *,
    contract: CampaignContract,
) -> dict[str, Any]:
    """Apply measured resources and native block sizes to a production stage."""
    selections = freeze["selections"]
    stage = str(spec["name"])
    worker_kernel = {
        "resolution": "pilot_interaction_score",
        "gate_b": "pilot_recovery",
        "fixed_family_supplement": "pilot_interaction_score",
        "applied_conditioning": "pilot_conditioning",
        "applied_screening": "pilot_screening",
        "applied_interaction": "pilot_interaction_score",
        "applied_nonlinear": "pilot_nonlinear",
        "applied_sparse_full": "pilot_sparse_full",
        "applied_sparse_resample": "pilot_sparse_resample",
        "applied_terminal_fit": "pilot_terminal_fit",
        "applied_ablation_fit": "pilot_terminal_fit",
        "applied_holdout_predict": "pilot_holdout_predict",
        "applied_bootstrap": "pilot_bootstrap",
        "recovery": "pilot_recovery",
    }.get(stage)
    if worker_kernel is None:
        return spec

    selection = selections[worker_kernel]
    walltime_scale = 1.0
    if stage in {"gate_b", "recovery"}:
        walltime_scale = max(
            1.0,
            contract.B_interaction / G11_CONTRACT.B_interaction,
        )
    if stage in {"resolution", "fixed_family_supplement"}:
        pilot_pair_draws = int(selection["block_size"]) * 12720
        target_pair_draws = (
            contract.resolution_base_draws
            * contract.resolution_max_multiplier
            * contract.resolution_family_size
            if stage == "resolution"
            else contract.B_interaction * max(contract.fixed_family_sizes)
        )
        walltime_scale = max(1.0, target_pair_draws / pilot_pair_draws)
    partition, worker_resources = _frozen_resource_envelope(
        selection, walltime_scale=walltime_scale
    )
    updated = {**spec, "partition": partition, "worker_resources": worker_resources}
    if stage == "gate_b":
        null_partition, null_resources = _frozen_resource_envelope(
            selections["pilot_gate_b_null"],
            walltime_scale=walltime_scale,
        )
        updated["worker_resources_by_scenario_kind"] = {
            "null": {
                "partition": null_partition,
                "worker_resources": null_resources,
                "dispatch_range_start": 0,
                "dispatch_range_end": sum(
                    scenario.n_replicates
                    for scenario in contract.scenarios
                    if scenario.kind == "null"
                ),
            },
            "strong": {
                "partition": partition,
                "worker_resources": worker_resources,
                "dispatch_range_start": sum(
                    scenario.n_replicates
                    for scenario in contract.scenarios
                    if scenario.kind == "null"
                ),
                "dispatch_range_end": sum(
                    scenario.n_replicates
                    for scenario in contract.scenarios
                    if scenario.kind in {"null", "strong"}
                ),
            },
        }

    block_kernel_by_stage = {
        "applied_screening": "pilot_screening",
        "applied_interaction": "pilot_interaction_score",
        "applied_nonlinear": "pilot_nonlinear",
        "applied_sparse_resample": "pilot_sparse_resample",
        "applied_holdout_predict": "pilot_holdout_predict",
        "applied_bootstrap": "pilot_bootstrap",
    }
    totals = {
        "applied_screening": contract.B_screen,
        "applied_interaction": contract.B_interaction,
        "applied_nonlinear": 160,
        "applied_sparse_resample": contract.n_stability_subsamples,
        "applied_holdout_predict": 23495,
        "applied_bootstrap": contract.bootstrap_draws,
    }
    if stage in block_kernel_by_stage:
        block_size = int(selections[block_kernel_by_stage[stage]]["block_size"])
        updated["block_size_override"] = block_size
        updated["job_count"] = math.ceil(totals[stage] / block_size)

    if stage in {"resolution", "fixed_family_supplement", "applied_interaction"}:
        _, reducer_resources = _frozen_resource_envelope(selections["pilot_interaction_reduce"])
        updated["reducer_resources"] = reducer_resources
    return updated


def _pilot_profile_rows(stage_name: str) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for profile in _PILOT_PROFILE_GRID[stage_name]:
        profile_id, partition, cpu, memory_gb, walltime_seconds, task_size, block_size = profile
        resources = _resource_envelope(cpu, memory_gb, walltime_seconds)
        row = {
            "stage": stage_name,
            "profile_id": profile_id,
            "partition": partition,
            "cpus": resources.requested_cpu_cores,
            "memory_gb": resources.requested_memory_gb,
            "walltime_seconds": resources.requested_walltime_seconds,
            "task_size": task_size,
            "block_size": block_size,
            "timing_repetitions": 1,
            "worker_resources": asdict(resources),
        }
        row["executed_work_units"] = _pilot_executed_work_units(row)
        row["schedule_hash"] = _stable_hash(row)
        rows.append(row)
    return tuple(rows)


def _build_stage_plans(
    *,
    output_dir: Path,
    run_id: str,
    cluster: ClusterConfig,
    source_hash: str,
    config_hash: str,
    lock_hash: str,
    package_hash: str,
    contract_config_path: Path,
    contract: CampaignContract,
    resource_freeze: dict[str, Any] | None,
    package_mode: str,
) -> tuple[StagePlan, ...]:
    null_jobs = sum(
        scenario.n_replicates for scenario in contract.scenarios if scenario.kind == "null"
    )
    strong_jobs = sum(
        scenario.n_replicates for scenario in contract.scenarios if scenario.kind == "strong"
    )
    recovery_jobs = sum(
        scenario.n_replicates for scenario in contract.scenarios if scenario.kind == "stress"
    )

    def _spec(
        name: str,
        parent: str | None,
        jobs: int,
        *,
        partition: str,
        cpu: int,
        memory_gb: int,
        walltime_seconds: int,
    ) -> dict[str, Any]:
        return {
            "name": name,
            "partition": partition,
            "job_count": jobs,
            "parent_stage_name": parent,
            "worker_resources": _resource_envelope(cpu, memory_gb, walltime_seconds),
            "reducer_resources": _resource_envelope(4, 8, 300),
        }

    stage_specs = (
        _pilot_stage_spec("scheduler_diagnostic", None),
        _pilot_stage_spec("pilot_conditioning", "scheduler_diagnostic"),
        _pilot_stage_spec("pilot_screening", "pilot_conditioning"),
        _pilot_stage_spec("pilot_interaction_score", "pilot_screening"),
        _pilot_stage_spec("pilot_interaction_reduce", "pilot_interaction_score"),
        _pilot_stage_spec("pilot_nonlinear", "pilot_interaction_reduce"),
        _pilot_stage_spec("pilot_sparse_full", "pilot_nonlinear"),
        _pilot_stage_spec("pilot_sparse_resample", "pilot_sparse_full"),
        _pilot_stage_spec("pilot_terminal_fit", "pilot_sparse_resample"),
        _pilot_stage_spec("pilot_gate_b_null", "pilot_terminal_fit"),
        _pilot_stage_spec("pilot_recovery", "pilot_gate_b_null"),
        _pilot_stage_spec("pilot_holdout_predict", "pilot_recovery"),
        _pilot_stage_spec("pilot_bootstrap", "pilot_holdout_predict"),
        _spec(
            "resolution",
            "pilot_bootstrap",
            20,
            partition="short",
            cpu=24,
            memory_gb=48,
            walltime_seconds=1800,
        ),
        _spec(
            "gate_b",
            "resolution",
            null_jobs + strong_jobs,
            partition="short",
            cpu=64,
            memory_gb=128,
            walltime_seconds=7200,
        ),
        _spec(
            "fixed_family_supplement",
            "gate_b",
            contract.fixed_family_replicates,
            partition="shared",
            cpu=16,
            memory_gb=32,
            walltime_seconds=1800,
        ),
        _spec(
            "applied_conditioning",
            "fixed_family_supplement",
            1,
            partition="short",
            cpu=24,
            memory_gb=64,
            walltime_seconds=1800,
        ),
        _spec(
            "applied_screening",
            "applied_conditioning",
            math.ceil(contract.B_screen / 200),
            partition="short",
            cpu=64,
            memory_gb=128,
            walltime_seconds=7200,
        ),
        _spec(
            "applied_interaction",
            "applied_screening",
            math.ceil(contract.B_interaction / 50),
            partition="short",
            cpu=80,
            memory_gb=160,
            walltime_seconds=10800,
        ),
        _spec(
            "applied_nonlinear",
            "applied_interaction",
            math.ceil(160 / 20),
            partition="short",
            cpu=48,
            memory_gb=96,
            walltime_seconds=5400,
        ),
        _spec(
            "applied_sparse_full",
            "applied_nonlinear",
            1,
            partition="short",
            cpu=64,
            memory_gb=128,
            walltime_seconds=7200,
        ),
        _spec(
            "applied_sparse_resample",
            "applied_sparse_full",
            contract.n_stability_subsamples,
            partition="short",
            cpu=64,
            memory_gb=128,
            walltime_seconds=7200,
        ),
        _spec(
            "applied_terminal_fit",
            "applied_sparse_resample",
            1,
            partition="short",
            cpu=64,
            memory_gb=160,
            walltime_seconds=7200,
        ),
        _spec(
            "applied_ablation_fit",
            "applied_terminal_fit",
            5,
            partition="short",
            cpu=64,
            memory_gb=160,
            walltime_seconds=7200,
        ),
        _spec(
            "applied_holdout_authorize",
            "applied_ablation_fit",
            1,
            partition="shared",
            cpu=4,
            memory_gb=8,
            walltime_seconds=300,
        ),
        _spec(
            "applied_holdout_predict",
            "applied_holdout_authorize",
            math.ceil(23495 / 2000),
            partition="short",
            cpu=24,
            memory_gb=96,
            walltime_seconds=1800,
        ),
        _spec(
            "applied_eligibility",
            "applied_holdout_predict",
            1,
            partition="shared",
            cpu=8,
            memory_gb=32,
            walltime_seconds=900,
        ),
        _spec(
            "applied_bootstrap",
            "applied_eligibility",
            math.ceil(contract.bootstrap_draws / 100),
            partition="short",
            cpu=48,
            memory_gb=96,
            walltime_seconds=3600,
        ),
        _spec(
            "recovery",
            "applied_bootstrap",
            recovery_jobs,
            partition="short",
            cpu=80,
            memory_gb=160,
            walltime_seconds=10800,
        ),
    )
    if resource_freeze is not None:
        stage_specs = tuple(
            _apply_resource_freeze_to_stage_spec(spec, resource_freeze, contract=contract)
            if str(spec["name"]) not in _PILOT_STAGES
            else spec
            for spec in stage_specs
        )
    if package_mode == "confirmatory":
        stage_specs = tuple(
            {
                **spec,
                "parent_stage_name": (
                    None if spec["name"] == "gate_b" else spec["parent_stage_name"]
                ),
            }
            for spec in stage_specs
            if spec["name"] not in {*_PILOT_STAGES, "resolution"}
        )
    elif package_mode == "downstream":
        stage_specs = tuple(
            {
                **spec,
                "parent_stage_name": (
                    None if spec["name"] == "fixed_family_supplement" else spec["parent_stage_name"]
                ),
            }
            for spec in stage_specs
            if spec["name"] not in {*_PILOT_STAGES, "resolution", "gate_b"}
        )

    plans: list[StagePlan] = []
    reducer_hash_by_stage: dict[str, str] = {}
    for spec in stage_specs:
        if spec["partition"] not in cluster.partitions:
            raise ValueError(
                f"stage {spec['name']} uses partition {spec['partition']} not in config"
            )
        if int(spec["job_count"]) > cluster.max_array_size:
            raise ValueError(f"stage {spec['name']} exceeds Kestrel MaxArraySize")

        stage_dir = output_dir / "stages" / str(spec["name"])
        manifest_path = stage_dir / "manifest.jsonl"
        worker_script_path = stage_dir / "worker.slurm"
        reducer_script_path = stage_dir / "reduce.slurm"
        audit_script_path = stage_dir / "audit.slurm"
        scratch_stage_dir = (
            Path(_resolved_scratch_template(cluster.scratch_template, run_id))
            / "stages"
            / str(spec["name"])
        )
        output_root = scratch_stage_dir / "results"
        audit_output_dir = stage_dir / "audit"
        reducer_output_dir = scratch_stage_dir / "reducer"
        parent_hash = (
            package_hash
            if spec["parent_stage_name"] is None
            else reducer_hash_by_stage[str(spec["parent_stage_name"])]
        )
        reducer_output_hash = _stable_hash(
            {
                "stage": spec["name"],
                "job_count": spec["job_count"],
                "parent_hash": parent_hash,
                "source_hash": source_hash,
                "config_hash": config_hash,
                "lock_hash": lock_hash,
            }
        )
        records = _build_manifest_records(
            run_id=run_id,
            stage_name=str(spec["name"]),
            partition=str(spec["partition"]),
            job_count=int(spec["job_count"]),
            output_root=output_root,
            parent_hash=parent_hash,
            source_hash=source_hash,
            config_hash=config_hash,
            lock_hash=lock_hash,
            worker_resources=spec["worker_resources"],
            contract_config_path=contract_config_path,
            cluster=cluster,
            contract=contract,
            block_size_override=spec.get("block_size_override"),
            resource_freeze_sha256=(
                "PENDING"
                if resource_freeze is None
                else str(resource_freeze["resource_freeze_sha256"])
            ),
            stage_manifest_root=output_dir / "stages",
            campaign_inventory_path=output_dir / "campaign_inventory.jsonl",
            worker_resources_by_scenario_kind=spec.get("worker_resources_by_scenario_kind"),
            package_mode=package_mode,
        )
        _write_jsonl(manifest_path, records)
        if str(spec["name"]) in _PILOT_STAGES:
            worker_script_paths = []
            for task_id, record in enumerate(records):
                task_resources = ResourceEnvelope(**record["worker_resources"])
                task_script_path = stage_dir / f"worker-{task_id:04d}.slurm"
                task_script_path.write_text(
                    _render_worker_script(
                        stage_name=str(spec["name"]),
                        partition=str(record["partition"]),
                        cluster=cluster,
                        resources=task_resources,
                        manifest_path=manifest_path,
                        output_root=output_root,
                        contract_config_path=contract_config_path,
                        contract_hash=config_hash,
                        job_count=int(spec["job_count"]),
                        run_id=run_id,
                        task_id=task_id,
                    ),
                    encoding="utf-8",
                )
                worker_script_paths.append(task_script_path)
            worker_script_path = worker_script_paths[0]
        elif str(spec["name"]) == "gate_b" and spec.get("worker_resources_by_scenario_kind"):
            worker_script_paths = []
            ranges = (("null", 0, null_jobs), ("strong", null_jobs, null_jobs + strong_jobs))
            for scenario_kind, range_start, range_end in ranges:
                resource_class = spec["worker_resources_by_scenario_kind"][scenario_kind]
                task_script_path = stage_dir / f"worker-{scenario_kind}.slurm"
                task_script_path.write_text(
                    _render_worker_script(
                        stage_name=str(spec["name"]),
                        partition=str(resource_class["partition"]),
                        cluster=cluster,
                        resources=resource_class["worker_resources"],
                        manifest_path=manifest_path,
                        output_root=output_root,
                        contract_config_path=contract_config_path,
                        contract_hash=config_hash,
                        job_count=int(spec["job_count"]),
                        run_id=run_id,
                        task_id=None,
                        task_range_start=range_start,
                        task_range_end=range_end,
                    ),
                    encoding="utf-8",
                )
                worker_script_paths.append(task_script_path)
            worker_script_path = worker_script_paths[0]
        else:
            worker_script_path.write_text(
                _render_worker_script(
                    stage_name=str(spec["name"]),
                    partition=str(spec["partition"]),
                    cluster=cluster,
                    resources=spec["worker_resources"],
                    manifest_path=manifest_path,
                    output_root=output_root,
                    contract_config_path=contract_config_path,
                    contract_hash=config_hash,
                    job_count=int(spec["job_count"]),
                    run_id=run_id,
                    task_id=None,
                ),
                encoding="utf-8",
            )
            worker_script_paths = [worker_script_path]
        audit_script_path.write_text(
            _render_audit_script(
                stage_name=str(spec["name"]),
                partition=str(spec["partition"]),
                cluster=cluster,
                resources=spec["reducer_resources"],
                manifest_path=manifest_path,
                audit_output_dir=audit_output_dir,
                contract_hash=config_hash,
                run_id=run_id,
            ),
            encoding="utf-8",
        )
        reducer_script_path.write_text(
            _render_reducer_script(
                stage_name=str(spec["name"]),
                partition=str(spec["partition"]),
                cluster=cluster,
                resources=spec["reducer_resources"],
                manifest_path=manifest_path,
                reducer_output_dir=reducer_output_dir,
                audit_success_path=audit_output_dir / "_SUCCESS.json",
                contract_config_path=contract_config_path,
                contract_hash=config_hash,
                job_count=int(spec["job_count"]),
                parent_hash=parent_hash,
                run_id=run_id,
            ),
            encoding="utf-8",
        )
        plan = StagePlan(
            name=str(spec["name"]),
            partition=str(spec["partition"]),
            job_count=int(spec["job_count"]),
            parent_stage_name=(
                None if spec["parent_stage_name"] is None else str(spec["parent_stage_name"])
            ),
            worker_resources=spec["worker_resources"],
            reducer_resources=spec["reducer_resources"],
            manifest_path=manifest_path,
            worker_script_path=worker_script_path,
            worker_script_paths=tuple(worker_script_paths),
            audit_script_path=audit_script_path,
            audit_output_dir=audit_output_dir,
            reducer_script_path=reducer_script_path,
            output_root=output_root,
            reducer_output_dir=reducer_output_dir,
            reducer_output_hash=reducer_output_hash,
            reducer_expected_range=(0, int(spec["job_count"])),
        )
        plans.append(plan)
        reducer_hash_by_stage[plan.name] = reducer_output_hash
    return tuple(plans)


def _pilot_stage_spec(name: str, parent: str | None) -> dict[str, Any]:
    rows = _pilot_profile_rows(name)
    partitions = {str(row["partition"]) for row in rows}
    if len(partitions) != 1:
        raise ValueError(f"pilot stage {name} profiles must share one partition")
    envelopes = [ResourceEnvelope(**row["worker_resources"]) for row in rows]
    return {
        "name": name,
        "partition": partitions.pop(),
        "job_count": len(rows),
        "parent_stage_name": parent,
        "worker_resources": ResourceEnvelope(
            estimated_cpu_cores=max(value.estimated_cpu_cores for value in envelopes),
            requested_cpu_cores=max(value.requested_cpu_cores for value in envelopes),
            estimated_memory_gb=max(value.estimated_memory_gb for value in envelopes),
            requested_memory_gb=max(value.requested_memory_gb for value in envelopes),
            estimated_walltime_seconds=max(value.estimated_walltime_seconds for value in envelopes),
            requested_walltime_seconds=max(value.requested_walltime_seconds for value in envelopes),
        ),
        "reducer_resources": _resource_envelope(4, 8, 300),
    }


def _resource_envelope(
    estimated_cpu_cores: int,
    estimated_memory_gb: int,
    estimated_walltime_seconds: int,
) -> ResourceEnvelope:
    return ResourceEnvelope(
        estimated_cpu_cores=estimated_cpu_cores,
        requested_cpu_cores=math.ceil(estimated_cpu_cores * 1.25),
        estimated_memory_gb=estimated_memory_gb,
        requested_memory_gb=math.ceil(estimated_memory_gb * 1.25),
        estimated_walltime_seconds=estimated_walltime_seconds,
        requested_walltime_seconds=math.ceil(estimated_walltime_seconds * 1.25),
    )


def _build_campaign_envelope(
    stages: tuple[StagePlan, ...],
    *,
    cluster: ClusterConfig,
    retry_limit: int,
    allocation_quota_is_unknown: bool,
    prior_allocations: tuple[StageAllocationEstimate, ...] = (),
    prior_worker_count: int = 0,
    prior_stage_count: int = 0,
) -> CampaignEnvelope:
    """Calculate a whole-package planning forecast without asserting a live quota.

    Each planned worker is capped at one GiB of durable artifact output and
    six durable files. A 20% campaign reserve covers ordinary runtime overrun
    and the small fraction of workers that may require the one allowed retry.
    It does not assume that every successful worker consumes its walltime and
    then runs a second time.
    """
    stage_allocations = (
        *prior_allocations,
        *(_stage_allocation_estimate(stage, cluster=cluster) for stage in stages),
    )
    estimated_node_hours = sum(estimate.estimated_node_hours for estimate in stage_allocations)
    estimated_au = sum(estimate.estimated_au for estimate in stage_allocations)
    total_workers = prior_worker_count + sum(stage.job_count for stage in stages)
    total_stages = prior_stage_count + len(stages)
    estimated_storage_gb = total_workers + total_stages
    estimated_inode_count = total_workers * 6 + total_stages * 6
    reserve_fraction = 0.20 if retry_limit else 0.0
    requested_retry_attempts = math.ceil(total_workers * reserve_fraction)
    return CampaignEnvelope(
        estimated_node_hours=estimated_node_hours,
        requested_node_hours=estimated_node_hours * (1.0 + reserve_fraction),
        estimated_au=estimated_au,
        requested_au=math.ceil(estimated_au * (1.0 + reserve_fraction)),
        stage_allocations=stage_allocations,
        estimated_storage_gb=estimated_storage_gb,
        requested_storage_gb=math.ceil(estimated_storage_gb * (1.0 + reserve_fraction)),
        estimated_inode_count=estimated_inode_count,
        requested_inode_count=math.ceil(estimated_inode_count * (1.0 + reserve_fraction)),
        estimated_retry_attempts=0,
        requested_retry_attempts=requested_retry_attempts,
        allocation_quota_readiness_blocker=allocation_quota_is_unknown,
    )


def _completed_adaptive_allocation_estimates(
    *,
    cluster: ClusterConfig,
    contract: CampaignContract,
    resource_freeze: dict[str, Any],
) -> tuple[tuple[StageAllocationEstimate, ...], int, int]:
    """Reconstruct pilot and resolution use omitted from a confirmatory package.

    A confirmatory package is created only after these adaptive phases finish,
    but their allocation still counts against the manuscript's 25,000-AU cap.
    """
    estimates: list[StageAllocationEstimate] = []
    worker_count = 0
    reducer_resources = _resource_envelope(4, 8, 300)
    for stage_name in _PILOT_STAGES:
        rows = _pilot_profile_rows(stage_name)
        estimates.append(
            _allocation_estimate_from_resources(
                stage_name=stage_name,
                workers=tuple(
                    (
                        ResourceEnvelope(**row["worker_resources"]),
                        str(row["partition"]),
                        1,
                    )
                    for row in rows
                ),
                reducer_resources=reducer_resources,
                reducer_partition=str(rows[0]["partition"]),
                cluster=cluster,
            )
        )
        worker_count += len(rows)

    resolution_spec = _apply_resource_freeze_to_stage_spec(
        {
            "name": "resolution",
            "partition": "short",
            "job_count": 20,
            "worker_resources": _resource_envelope(24, 48, 1800),
            "reducer_resources": reducer_resources,
        },
        resource_freeze,
        contract=contract,
    )
    estimates.append(
        _allocation_estimate_from_resources(
            stage_name="resolution",
            workers=(
                (
                    resolution_spec["worker_resources"],
                    str(resolution_spec["partition"]),
                    int(resolution_spec["job_count"]),
                ),
            ),
            reducer_resources=resolution_spec["reducer_resources"],
            reducer_partition=str(resolution_spec["partition"]),
            cluster=cluster,
        )
    )
    worker_count += int(resolution_spec["job_count"])
    return tuple(estimates), worker_count, len(estimates)


def _allocation_estimate_from_resources(
    *,
    stage_name: str,
    workers: tuple[tuple[ResourceEnvelope, str, int], ...],
    reducer_resources: ResourceEnvelope,
    reducer_partition: str,
    cluster: ClusterConfig,
) -> StageAllocationEstimate:
    estimated_node_hours = sum(
        count
        * _allocated_node_count(
            resources,
            cluster=cluster,
            allow_fraction=partition == "shared",
        )
        * resources.estimated_walltime_seconds
        / 3600
        for resources, partition, count in workers
    )
    requested_node_hours = sum(
        count
        * _allocated_node_count(
            resources,
            cluster=cluster,
            requested=True,
            allow_fraction=partition == "shared",
        )
        * resources.requested_walltime_seconds
        / 3600
        for resources, partition, count in workers
    )
    estimated_node_hours += (
        _allocated_node_count(
            reducer_resources,
            cluster=cluster,
            allow_fraction=reducer_partition == "shared",
        )
        * reducer_resources.estimated_walltime_seconds
        / 3600
        * 2  # one exact-coverage audit plus one reducer
    )
    requested_node_hours += (
        _allocated_node_count(
            reducer_resources,
            cluster=cluster,
            requested=True,
            allow_fraction=reducer_partition == "shared",
        )
        * reducer_resources.requested_walltime_seconds
        / 3600
        * 2  # one exact-coverage audit plus one reducer
    )
    charge = cluster.cpu_charge_factor * cluster.qos_factor
    return StageAllocationEstimate(
        stage_name=stage_name,
        estimated_node_hours=estimated_node_hours,
        requested_node_hours=requested_node_hours,
        estimated_au=estimated_node_hours * charge,
        requested_au=requested_node_hours * charge,
    )


def _stage_allocation_estimate(
    stage: StagePlan,
    *,
    cluster: ClusterConfig,
) -> StageAllocationEstimate:
    """Calculate Kestrel allocation usage from allocated node count and walltime."""
    worker_records = load_stage_manifest(stage.manifest_path)
    worker_estimated_node_hours = 0.0
    worker_requested_node_hours = 0.0
    for record in worker_records:
        resources = ResourceEnvelope(**record["worker_resources"])
        partition = str(record.get("partition", stage.partition))
        worker_estimated_node_hours += (
            _allocated_node_count(
                resources,
                cluster=cluster,
                allow_fraction=partition == "shared",
            )
            * resources.estimated_walltime_seconds
            / 3600
        )
        worker_requested_node_hours += (
            _allocated_node_count(
                resources,
                cluster=cluster,
                requested=True,
                allow_fraction=partition == "shared",
            )
            * resources.requested_walltime_seconds
            / 3600
        )
    reducer_estimated_nodes = _allocated_node_count(
        stage.reducer_resources,
        cluster=cluster,
        allow_fraction=stage.partition == "shared",
    )
    reducer_requested_nodes = _allocated_node_count(
        stage.reducer_resources,
        cluster=cluster,
        requested=True,
        allow_fraction=stage.partition == "shared",
    )
    estimated_node_hours = (
        worker_estimated_node_hours
        + 2 * reducer_estimated_nodes * stage.reducer_resources.estimated_walltime_seconds / 3600
    )
    requested_node_hours = (
        worker_requested_node_hours
        + 2 * reducer_requested_nodes * stage.reducer_resources.requested_walltime_seconds / 3600
    )
    charge = cluster.cpu_charge_factor * cluster.qos_factor
    return StageAllocationEstimate(
        stage_name=stage.name,
        estimated_node_hours=estimated_node_hours,
        requested_node_hours=requested_node_hours,
        estimated_au=estimated_node_hours * charge,
        requested_au=requested_node_hours * charge,
    )


def _stage_retry_requested_node_hours(stage: StagePlan, *, cluster: ClusterConfig) -> float:
    """Return the worker-only requested node-hours for one full retry pass."""
    total = 0.0
    for record in load_stage_manifest(stage.manifest_path):
        resources = ResourceEnvelope(**record["worker_resources"])
        partition = str(record.get("partition", stage.partition))
        total += (
            _allocated_node_count(
                resources,
                cluster=cluster,
                requested=True,
                allow_fraction=partition == "shared",
            )
            * resources.requested_walltime_seconds
            / 3600
        )
    return total


def _stage_retry_requested_allocation(stage: StagePlan, *, cluster: ClusterConfig) -> float:
    return (
        _stage_retry_requested_node_hours(stage, cluster=cluster)
        * cluster.cpu_charge_factor
        * cluster.qos_factor
    )


def _allocated_node_count(
    resources: ResourceEnvelope,
    *,
    cluster: ClusterConfig,
    requested: bool = False,
    allow_fraction: bool = False,
) -> float:
    """Return charged Kestrel node equivalents for one resource request."""
    cpu_cores = resources.requested_cpu_cores if requested else resources.estimated_cpu_cores
    memory_gb = resources.requested_memory_gb if requested else resources.estimated_memory_gb
    fraction = max(
        cpu_cores / cluster.cpu_cores_per_node,
        memory_gb / cluster.memory_per_node_gb,
    )
    return fraction if allow_fraction else float(math.ceil(fraction))


def _build_manifest_records(
    *,
    run_id: str,
    stage_name: str,
    partition: str,
    job_count: int,
    output_root: Path,
    parent_hash: str,
    source_hash: str,
    config_hash: str,
    lock_hash: str,
    worker_resources: ResourceEnvelope,
    contract_config_path: Path,
    cluster: ClusterConfig,
    contract: CampaignContract,
    block_size_override: int | None,
    resource_freeze_sha256: str,
    stage_manifest_root: Path,
    campaign_inventory_path: Path,
    worker_resources_by_scenario_kind: dict[str, dict[str, Any]] | None,
    package_mode: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if stage_name in _PILOT_STAGES:
        descriptors = []
        for index, profile in enumerate(_pilot_profile_rows(stage_name)):
            descriptor = {
                "operation": stage_name,
                "pilot_repetition": index,
                "profile_id": profile["profile_id"],
                "partition": profile["partition"],
                "task_size": profile["task_size"],
                "block_size": profile["block_size"],
                "timing_repetitions": profile["timing_repetitions"],
                "executed_work_units": profile["executed_work_units"],
                "worker_resources": profile["worker_resources"],
                "expected_range_start": index,
                "expected_range_end": index + 1,
            }
            if stage_name in {
                "pilot_sparse_resample",
                "pilot_holdout_predict",
                "pilot_bootstrap",
            }:
                descriptor["upstream_shard_id"] = f"task-{index:04d}"
            descriptors.append(descriptor)
    elif stage_name == "resolution":
        descriptors = [
            {
                "operation": "resolution",
                "fixture_kind": fixture,
                "schedule_index": schedule_index,
                "seed": derive_seed(config_hash, f"resolution:{fixture}", schedule_index),
                "base_draws": contract.resolution_base_draws,
                "nested_schedule_draws": (
                    contract.resolution_base_draws * contract.resolution_max_multiplier
                ),
                "family_size": contract.resolution_family_size,
                "family_order_path": str(
                    contract_config_path.parent / "fixed_family_pair_order.json"
                ),
                "family_order_file_sha256": _hash_file(
                    contract_config_path.parent / "fixed_family_pair_order.json"
                ),
                "expected_range_start": index,
                "expected_range_end": index + 1,
            }
            for index, (fixture, schedule_index) in enumerate(
                (fixture, schedule_index)
                for fixture in _RESOLUTION_FIXTURES
                for schedule_index in range(contract.resolution_schedules)
            )
        ]
    elif stage_name == "gate_b":
        descriptors = _replicate_descriptors(
            {"null", "strong"}, operation="gate_b", contract=contract
        )
    elif stage_name == "recovery":
        descriptors = _replicate_descriptors({"stress"}, operation="recovery", contract=contract)
        gate_b_manifest_path = stage_manifest_root / "gate_b" / "manifest.jsonl"
        for descriptor in descriptors:
            descriptor["gate_b_manifest_path"] = str(gate_b_manifest_path)
    elif stage_name == "fixed_family_supplement":
        family_order_path = contract_config_path.parent / "fixed_family_pair_order.json"
        descriptors = [
            {
                "operation": "fixed_family_supplement",
                "global_null_replicate_index": replicate_index,
                "seed": derive_seed(config_hash, "fixed_family_supplement", replicate_index),
                "family_sizes": list(contract.fixed_family_sizes),
                "family_order_sha256": contract.fixed_family_pair_order_sha256,
                "family_order_path": str(family_order_path),
                "family_order_file_sha256": _hash_file(family_order_path),
                "expected_range_start": index,
                "expected_range_end": index + 1,
            }
            for index, replicate_index in enumerate(range(contract.fixed_family_replicates))
        ]
    elif stage_name == "applied_conditioning":
        descriptors = [
            {
                "operation": stage_name,
                "applied_rows": 30000,
                "applied_inputs": 160,
                "applied_outputs": 23495,
                "expected_range_start": 0,
                "expected_range_end": 1,
            }
        ]
    elif stage_name == "applied_screening":
        descriptors = _block_descriptors(
            operation=stage_name,
            total=contract.B_screen,
            block_size=block_size_override or 200,
            unit="permutation_draw",
        )
    elif stage_name == "applied_interaction":
        descriptors = _block_descriptors(
            operation=stage_name,
            total=contract.B_interaction,
            block_size=block_size_override or 50,
            unit="complete_family_permutation_draw",
        )
    elif stage_name == "applied_nonlinear":
        descriptors = _block_descriptors(
            operation=stage_name,
            total=160,
            block_size=block_size_override or 20,
            unit="typed_base_feature",
        )
    elif stage_name == "applied_sparse_full":
        descriptors = [
            {
                "operation": stage_name,
                "sparse_work_unit": "frozen_full_fit",
                "expected_range_start": 0,
                "expected_range_end": 1,
            }
        ]
    elif stage_name == "applied_sparse_resample":
        descriptors = _block_descriptors(
            operation=stage_name,
            total=contract.n_stability_subsamples,
            block_size=block_size_override or 1,
            unit="stability_subsample",
        )
    elif stage_name == "applied_terminal_fit":
        descriptors = [
            {
                "operation": stage_name,
                "terminal_steps": ["hc3", "pruning", "final_ols", "model_freeze"],
                "expected_range_start": 0,
                "expected_range_end": 1,
            }
        ]
    elif stage_name == "applied_ablation_fit":
        ablations = (
            "null_mean",
            "main_effects_ols",
            "screened_ols",
            "penalized_ols",
            "final_ols",
        )
        descriptors = [
            {
                "operation": stage_name,
                "ablation_model": model,
                "expected_range_start": index,
                "expected_range_end": index + 1,
            }
            for index, model in enumerate(ablations)
        ]
    elif stage_name == "applied_holdout_authorize":
        descriptors = [
            {
                "operation": stage_name,
                "authorization_action": "verify_model_freezes_then_unseal_holdout_response",
                "expected_range_start": 0,
                "expected_range_end": 1,
            }
        ]
    elif stage_name == "applied_holdout_predict":
        descriptors = _block_descriptors(
            operation=stage_name,
            total=23495,
            block_size=block_size_override or 2000,
            unit="frozen_output_column",
        )
    elif stage_name == "applied_eligibility":
        descriptors = [
            {
                "operation": stage_name,
                "eligibility_rule": "variance_and_relative_range_from_training_only",
                "expected_range_start": 0,
                "expected_range_end": 1,
            }
        ]
    elif stage_name == "applied_bootstrap":
        descriptors = _block_descriptors(
            operation=stage_name,
            total=contract.bootstrap_draws,
            block_size=block_size_override or 100,
            unit="four_scenario_stratified_row_bootstrap_draw",
        )
    else:  # pragma: no cover - guarded by stage plan construction
        raise ValueError(f"unsupported stage {stage_name}")

    if len(descriptors) != job_count:
        raise ValueError(
            f"stage {stage_name} expected {job_count} jobs but built {len(descriptors)}"
        )

    for index, descriptor in enumerate(descriptors):
        resource_class = (
            worker_resources_by_scenario_kind.get(str(descriptor.get("scenario_kind")), {})
            if worker_resources_by_scenario_kind is not None
            else {}
        )
        record_resources = (
            resource_class["worker_resources"] if resource_class else worker_resources
        )
        record_partition = (
            str(resource_class["partition"])
            if resource_class
            else str(descriptor.get("partition", partition))
        )
        dispatch_range_start = int(resource_class["dispatch_range_start"]) if resource_class else 0
        dispatch_range_end = (
            int(resource_class["dispatch_range_end"]) if resource_class else job_count
        )
        if stage_name not in _PILOT_STAGES:
            phase_by_stage = {
                "resolution": "development",
                "gate_b": "gate_b",
                "fixed_family_supplement": (
                    "fixed_family" if package_mode == "downstream" else "gate_b"
                ),
                "recovery": "gate_c",
            }
            descriptor.setdefault("phase", phase_by_stage.get(stage_name, "gate_p"))
            if stage_name.startswith("applied_"):
                descriptor.setdefault(
                    "seed",
                    derive_seed(config_hash, f"applied:{stage_name}", 0),
                )
            descriptor.update(
                {
                    "resource_freeze_sha256": resource_freeze_sha256,
                    "rfm_repository_root": cluster.rfm_repository_root,
                    "bsm_repository_root": cluster.bsm_repository_root,
                    "scientific_adapter_path": str(
                        Path(cluster.bsm_repository_root) / cluster.scientific_adapter_relative_path
                    ),
                    "scientific_adapter_sha256": cluster.scientific_adapter_sha256,
                    "bsm_recovery_driver_path": str(
                        Path(cluster.bsm_repository_root)
                        / cluster.bsm_recovery_driver_relative_path
                    ),
                    "bsm_recovery_driver_sha256": cluster.bsm_recovery_driver_sha256,
                    "bsm_dgp_contract_path": str(
                        Path(cluster.bsm_repository_root) / cluster.bsm_dgp_contract_relative_path
                    ),
                    "bsm_dgp_contract_sha256": cluster.bsm_dgp_contract_sha256,
                    "applied_data_preparer_path": str(
                        Path(cluster.bsm_repository_root)
                        / cluster.applied_data_preparer_relative_path
                    ),
                    "applied_data_preparer_sha256": (cluster.applied_data_preparer_sha256),
                    "applied_config_path": str(
                        Path(cluster.bsm_repository_root) / cluster.applied_config_relative_path
                    ),
                    "applied_config_sha256": cluster.applied_config_sha256,
                    "applied_data_root": cluster.applied_data_root,
                    "applied_data_manifest_sha256": cluster.applied_data_manifest_sha256,
                    "execution_authorization_path": str(
                        Path(cluster.authorization_root) / run_id / f"{descriptor['phase']}.json"
                    ),
                }
            )
        shard_id = f"task-{index:04d}"
        schedule_hash = _stable_hash({"stage": stage_name, **descriptor})
        input_hash = _stable_hash(
            {
                "stage": stage_name,
                "descriptor": descriptor,
                "parent_hash": parent_hash,
                "source_hash": source_hash,
                "config_hash": config_hash,
                "lock_hash": lock_hash,
                "contract_config_path": str(contract_config_path),
            }
        )
        output_dir = output_root / shard_id
        output_hash = _stable_hash(
            {
                "stage": stage_name,
                "shard_id": shard_id,
                "output_dir": str(output_dir),
                "input_hash": input_hash,
            }
        )
        rows.append(
            {
                "schema_version": 1,
                "run_id": run_id,
                "campaign_inventory_path": str(campaign_inventory_path),
                "stage": stage_name,
                "shard_id": shard_id,
                "interaction_sharding": "draw-block",
                "status": "pending",
                "attempt": 0,
                "max_retries": contract.retry_limit,
                "retryable_scheduler_states": list(contract.retryable_scheduler_states),
                "source_hash": source_hash,
                "config_hash": config_hash,
                "lock_hash": lock_hash,
                "input_hash": input_hash,
                "schedule_hash": schedule_hash,
                "parent_hash": parent_hash,
                "output_hash": output_hash,
                "output_dir": str(output_dir),
                "success_marker": str(output_dir / "_SUCCESS.json"),
                "contract_config_path": str(contract_config_path),
                "partition": record_partition,
                "worker_resources": asdict(record_resources),
                "dispatch_range_start": dispatch_range_start,
                "dispatch_range_end": dispatch_range_end,
                **descriptor,
            }
        )
    return rows


def _block_descriptors(
    *, operation: str, total: int, block_size: int, unit: str
) -> list[dict[str, Any]]:
    """Enumerate complete non-overlapping native work units."""
    return [
        {
            "operation": operation,
            "work_unit": unit,
            "block_start": start,
            "block_end": min(total, start + block_size),
            "block_size": min(total, start + block_size) - start,
            "expected_range_start": index,
            "expected_range_end": index + 1,
        }
        for index, start in enumerate(range(0, total, block_size))
    ]


def _replicate_descriptors(
    kinds: set[str], *, operation: str, contract: CampaignContract
) -> list[dict[str, Any]]:
    if operation not in {"gate_b", "recovery"}:
        raise ValueError("replicate operation must be gate_b or recovery")
    descriptors: list[dict[str, Any]] = []
    index = 0
    contract_hash = compute_contract_hash(contract)
    for scenario in contract.scenarios:
        if scenario.kind not in kinds:
            continue
        for replicate_index in range(scenario.n_replicates):
            descriptors.append(
                {
                    "operation": operation,
                    "scenario_id": scenario.id,
                    "scenario_kind": scenario.kind,
                    "replicate_index": replicate_index,
                    "seed": derive_seed(contract_hash, scenario.id, replicate_index),
                    "n_train": scenario.n_train,
                    "n_eval": scenario.n_eval,
                    "n_response": scenario.n_response,
                    "n_predictor_continuous": scenario.n_predictor_continuous,
                    "n_predictor_binary": scenario.n_predictor_binary,
                    "phase": "gate_b" if operation == "gate_b" else "gate_c",
                    "recovery_comparators": (
                        list(contract.recovery_comparators)
                        if scenario.kind in {"strong", "stress"}
                        else []
                    ),
                    "comparator_contract": {
                        "cv_folds": contract.comparator_cv_folds,
                        "elastic_net_alpha_grid": list(contract.elastic_net_alpha_grid),
                        "elastic_net_l1_ratio_grid": list(contract.elastic_net_l1_ratio_grid),
                        "boosted_tree_candidate_grid": [
                            list(value) for value in contract.boosted_tree_candidate_grid
                        ],
                        "tuning_metric": contract.comparator_tuning_metric,
                        "failure_action": contract.comparator_failure_action,
                    },
                    "acceptance_rule": (
                        "one_sided_wilson_upper_le_0p09"
                        if scenario.kind == "null"
                        else (
                            "discovery_power_lower_bound_ge_0p80"
                            if scenario.kind == "strong"
                            else "prespecified_terminal_recovery_estimands_complete"
                        )
                    ),
                    "expected_artifacts": [
                        "terminal_record.json",
                        "truth_ledger.json",
                        "interaction_score_blocks",
                    ]
                    + (
                        ["comparator_predictions.npz"]
                        if scenario.kind in {"strong", "stress"}
                        else []
                    ),
                    "artifact_schema_version": contract.artifact_schema_version,
                    "expected_range_start": index,
                    "expected_range_end": index + 1,
                }
            )
            index += 1
    return descriptors


def _build_campaign_table(stages: tuple[StagePlan, ...]) -> list[dict[str, Any]]:
    """Flatten every worker and reducer into one exact campaign inventory."""
    rows: list[dict[str, Any]] = []
    for stage in stages:
        records = load_stage_manifest(stage.manifest_path)
        rows.extend(
            {
                **record,
                "record_type": "worker",
                "analysis_unit_id": f"worker:{stage.name}:{record['shard_id']}",
            }
            for record in records
        )
        rows.append(
            {
                "schema_version": 1,
                "record_type": "reducer",
                "analysis_unit_id": f"reducer:{stage.name}",
                "stage": stage.name,
                "parent_stage_name": stage.parent_stage_name,
                "expected_worker_count": stage.job_count,
                "expected_range_start": 0,
                "expected_range_end": stage.job_count,
                "manifest_sha256": _hash_file(stage.manifest_path),
                "reducer_output_hash": stage.reducer_output_hash,
                "expected_artifact": str(stage.reducer_output_dir / "reduced_result.json"),
                "dependency_mode": "afterok_complete_validator",
            }
        )
    return rows


def validate_campaign_table(
    dag: CampaignDAG,
    *,
    rows: list[dict[str, Any]] | None = None,
    contract: CampaignContract | None = None,
) -> None:
    """Prove one-to-one campaign coverage and reject malformed native blocks."""
    if contract is None:
        from rfm_pipeline.campaign_contract import load_contract

        contract, loaded_hash = load_contract(dag.contract_config_path)
        if loaded_hash != dag.config_hash:
            raise ValueError("campaign table contract differs from the DAG hash")
    inventory = rows
    if inventory is None:
        inventory = [
            json.loads(line)
            for line in dag.campaign_inventory_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if _hash_file(dag.campaign_inventory_path) != dag.campaign_inventory_hash:
            raise ValueError("campaign inventory byte hash differs from package metadata")

    unit_ids = [str(row.get("analysis_unit_id", "")) for row in inventory]
    if len(unit_ids) != len(set(unit_ids)):
        raise ValueError("duplicate analysis unit in campaign inventory")
    expected_total = sum(stage.job_count + 1 for stage in dag.stages)
    if len(inventory) != expected_total:
        raise ValueError(
            f"campaign inventory row count is {len(inventory)}; expected {expected_total}"
        )

    workers = [row for row in inventory if row.get("record_type") == "worker"]
    reducers = [row for row in inventory if row.get("record_type") == "reducer"]
    if len(reducers) != len(dag.stages):
        raise ValueError("campaign inventory must contain exactly one reducer per stage")
    for stage in dag.stages:
        stage_workers = [row for row in workers if row.get("stage") == stage.name]
        if len(stage_workers) != stage.job_count:
            raise ValueError(f"campaign inventory has incomplete worker coverage for {stage.name}")
        manifest_records = load_stage_manifest(stage.manifest_path)
        expected_schedules = [record["schedule_hash"] for record in manifest_records]
        observed_schedules = [row.get("schedule_hash") for row in stage_workers]
        if observed_schedules != expected_schedules:
            raise ValueError(f"campaign inventory differs from manifest order for {stage.name}")
        stage_reducers = [row for row in reducers if row.get("stage") == stage.name]
        if len(stage_reducers) != 1:
            raise ValueError(f"campaign inventory reducer coverage differs for {stage.name}")
        if int(stage_reducers[0]["expected_worker_count"]) != stage.job_count:
            raise ValueError(f"campaign inventory reducer range differs for {stage.name}")

    block_totals = {
        "applied_screening": contract.B_screen,
        "applied_interaction": contract.B_interaction,
        "applied_nonlinear": 160,
        "applied_sparse_resample": contract.n_stability_subsamples,
        "applied_holdout_predict": 23495,
        "applied_bootstrap": contract.bootstrap_draws,
    }
    for stage_name, expected_end in block_totals.items():
        stage_blocks = sorted(
            (row for row in workers if row.get("stage") == stage_name),
            key=lambda row: int(row["block_start"]),
        )
        cursor = 0
        for row in stage_blocks:
            start = int(row["block_start"])
            end = int(row["block_end"])
            if end <= start:
                raise ValueError(f"campaign inventory contains zero-width block in {stage_name}")
            if start != cursor:
                raise ValueError(f"campaign inventory block coverage has a gap in {stage_name}")
            cursor = end
        if cursor != expected_end:
            raise ValueError(
                f"campaign inventory block coverage for {stage_name} ends at {cursor}, "
                f"expected {expected_end}"
            )

    scientific = [row for row in workers if row.get("stage") in {"gate_b", "recovery"}]
    contract_hash = compute_contract_hash(contract)
    expected_scientific = build_campaign_inventory(contract, contract_hash)
    if dag.package_mode == "downstream":
        expected_scientific = [row for row in expected_scientific if row.kind == "stress"]
    expected_identity = {
        (row.scenario_id, row.replicate_index, row.seed) for row in expected_scientific
    }
    observed_identity = {
        (str(row["scenario_id"]), int(row["replicate_index"]), int(row["seed"]))
        for row in scientific
    }
    if len(scientific) != len(expected_scientific) or observed_identity != expected_identity:
        raise ValueError("campaign inventory does not exactly cover its scientific records")
    if any(
        tuple(row.get("recovery_comparators", ()))
        != (contract.recovery_comparators if row.get("scenario_kind") != "null" else ())
        for row in scientific
    ):
        raise ValueError("campaign inventory recovery comparator contract differs")


def _render_worker_script(
    *,
    stage_name: str,
    partition: str,
    cluster: ClusterConfig,
    resources: ResourceEnvelope,
    manifest_path: Path,
    output_root: Path,
    contract_config_path: Path,
    contract_hash: str,
    job_count: int,
    run_id: str,
    task_id: int | None,
    task_range_start: int = 0,
    task_range_end: int | None = None,
) -> str:
    range_end = job_count if task_range_end is None else task_range_end
    if not 0 <= task_range_start < range_end <= job_count:
        raise ValueError("worker task range is empty or outside the manifest")
    array_size = range_end - task_range_start
    scratch_root = _resolved_scratch_template(cluster.scratch_template, run_id)
    array_directive = (
        f"#SBATCH --array=0-{array_size - 1}%{min(array_size, cluster.max_array_concurrency)}"
        if task_id is None
        else ""
    )
    output_pattern = (
        f"{stage_name}-%A_%a.out" if task_id is None else f"{stage_name}-task-{task_id:04d}-%j.out"
    )
    if task_id is not None:
        task_argument = str(task_id)
    elif task_range_start:
        task_argument = f"$((SLURM_ARRAY_TASK_ID + {task_range_start}))"
    else:
        task_argument = "${SLURM_ARRAY_TASK_ID:-0}"
    runtime_python = (
        Path(cluster.rfm_repository_root) / ".pixi" / "envs" / "default" / "bin" / "python"
    )
    return f"""#!/bin/bash
#SBATCH --job-name=g11-{stage_name}-worker
#SBATCH --account={cluster.account}
#SBATCH --partition={partition}
#SBATCH --time={resources.requested_walltime_hms}
#SBATCH --cpus-per-task={resources.requested_cpu_cores}
#SBATCH --mem={resources.requested_memory_gb}G
{array_directive}
#SBATCH --output={cluster.project_root}/{run_id}/logs/{output_pattern}
#SBATCH --signal=USR1@{_kill_wait_seconds(cluster.kill_wait)}
#SBATCH --no-requeue
set -euo pipefail

# NO-SUBMIT package: script generation only; manual review required before any scheduler action.
RUN_ID="{run_id}"
PROJECT_ROOT="{cluster.project_root}"
SCRATCH_ROOT="{scratch_root}"
MANIFEST_PATH="{manifest_path}"
OUTPUT_ROOT="{output_root}"
RUNTIME_PYTHON="{runtime_python}"

cd "{cluster.rfm_repository_root}"

"$RUNTIME_PYTHON" -m rfm_pipeline.hpc_campaign_package worker \\
  --manifest "$MANIFEST_PATH" \\
  --stage "{stage_name}" \\
  --task-id "{task_argument}" \\
  --output-root "$OUTPUT_ROOT" \\
  --scratch-root "$SCRATCH_ROOT" \\
  --contract-config "{contract_config_path}" \\
  --contract-hash "{contract_hash}" \\
  --expected-range-start "{task_range_start}" \\
  --expected-range-end "{range_end}"
"""


def _render_reducer_script(
    *,
    stage_name: str,
    partition: str,
    cluster: ClusterConfig,
    resources: ResourceEnvelope,
    manifest_path: Path,
    reducer_output_dir: Path,
    audit_success_path: Path,
    contract_config_path: Path,
    contract_hash: str,
    job_count: int,
    parent_hash: str,
    run_id: str,
) -> str:
    runtime_python = (
        Path(cluster.rfm_repository_root) / ".pixi" / "envs" / "default" / "bin" / "python"
    )
    return f"""#!/bin/bash
#SBATCH --job-name=g11-{stage_name}-reduce
#SBATCH --account={cluster.account}
#SBATCH --partition={partition}
#SBATCH --time={resources.requested_walltime_hms}
#SBATCH --cpus-per-task={resources.requested_cpu_cores}
#SBATCH --mem={resources.requested_memory_gb}G
#SBATCH --output={cluster.project_root}/{run_id}/logs/{stage_name}-reduce-%j.out
#SBATCH --signal=USR1@{_kill_wait_seconds(cluster.kill_wait)}
#SBATCH --no-requeue
set -euo pipefail

# NO-SUBMIT package: reducer script generated for review only.
MANIFEST_PATH="{manifest_path}"
REDUCER_OUTPUT_DIR="{reducer_output_dir}"
RUNTIME_PYTHON="{runtime_python}"

cd "{cluster.rfm_repository_root}"

"$RUNTIME_PYTHON" -m rfm_pipeline.hpc_campaign_package reduce \\
  --manifest "$MANIFEST_PATH" \\
  --stage "{stage_name}" \\
  --output-root "$REDUCER_OUTPUT_DIR" \\
  --audit-success "{audit_success_path}" \\
  --contract-config "{contract_config_path}" \\
  --contract-hash "{contract_hash}" \\
  --expected-range-start 0 \\
  --expected-range-end "{job_count}" \\
  --expected-parent-hash "{parent_hash}"
"""


def _render_audit_script(
    *,
    stage_name: str,
    partition: str,
    cluster: ClusterConfig,
    resources: ResourceEnvelope,
    manifest_path: Path,
    audit_output_dir: Path,
    contract_hash: str,
    run_id: str,
) -> str:
    runtime_python = (
        Path(cluster.rfm_repository_root) / ".pixi" / "envs" / "default" / "bin" / "python"
    )
    return f"""#!/bin/bash
#SBATCH --job-name=g11-{stage_name}-audit
#SBATCH --account={cluster.account}
#SBATCH --partition={partition}
#SBATCH --time={resources.requested_walltime_hms}
#SBATCH --cpus-per-task={resources.requested_cpu_cores}
#SBATCH --mem={resources.requested_memory_gb}G
#SBATCH --output={cluster.project_root}/{run_id}/logs/{stage_name}-audit-%j.out
#SBATCH --signal=USR1@{_kill_wait_seconds(cluster.kill_wait)}
#SBATCH --no-requeue
set -euo pipefail

# afterany failure audit and exact-coverage validator; it never performs science.
RUNTIME_PYTHON="{runtime_python}"
cd "{cluster.rfm_repository_root}"

"$RUNTIME_PYTHON" -m rfm_pipeline.hpc_campaign_package audit \
  --manifest "{manifest_path}" \
  --stage "{stage_name}" \
  --output-root "{audit_output_dir}" \
  --contract-hash "{contract_hash}"
"""


def _write_package_metadata(dag: CampaignDAG) -> None:
    submission_plan = build_submission_plan(dag)
    package_summary = {
        "run_id": dag.run_id,
        "package_mode": dag.package_mode,
        "scheduler_submission_permitted": dag.scheduler_submission_permitted,
        "source_hash": dag.source_hash,
        "config_hash": dag.config_hash,
        "lock_hash": dag.lock_hash,
        "campaign_inventory_path": str(dag.campaign_inventory_path),
        "campaign_inventory_hash": dag.campaign_inventory_hash,
        "readiness_blockers": list(dag.readiness_blockers),
        "campaign_envelope": asdict(dag.campaign_envelope),
        "stages": [
            {
                "name": stage.name,
                "partition": stage.partition,
                "job_count": stage.job_count,
                "manifest_path": str(stage.manifest_path),
                "worker_script_path": str(stage.worker_script_path),
                "worker_script_paths": [str(path) for path in stage.worker_script_paths],
                "audit_script_path": str(stage.audit_script_path),
                "reducer_script_path": str(stage.reducer_script_path),
                "reducer_output_hash": stage.reducer_output_hash,
            }
            for stage in dag.stages
        ],
    }
    (dag.output_dir / "package_summary.json").write_text(
        json.dumps(package_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (dag.output_dir / "telemetry_schema.json").write_text(
        json.dumps(list(dag.telemetry_schema), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (dag.output_dir / "pilot_matrix.json").write_text(
        json.dumps(list(dag.pilot_matrix), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (dag.output_dir / "post_pilot_selection.json").write_text(
        json.dumps(dag.post_pilot_selection, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (dag.output_dir / "readiness_blockers.json").write_text(
        json.dumps({"blockers": list(dag.readiness_blockers)}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (dag.output_dir / "submission_plan.json").write_text(
        json.dumps(submission_plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _stage_by_name(dag: CampaignDAG, stage_name: str) -> StagePlan:
    for stage in dag.stages:
        if stage.name == stage_name:
            return stage
    raise ValueError(f"unknown stage {stage_name}")


def _resolved_scratch_template(template: str, run_id: str) -> str:
    user = os.environ.get("USER", "").strip()
    if not user or "/" in user:
        raise ValueError("USER must be a non-empty path-safe account name")
    resolved = template.replace("{user}", user).replace("{run_id}", run_id)
    if "{" in resolved or "}" in resolved or not resolved.startswith("/"):
        raise ValueError("scratch template did not resolve to an absolute placeholder-free path")
    return resolved


def _kill_wait_seconds(raw: str) -> int:
    stripped = raw.strip().lower()
    if not stripped.endswith("s"):
        raise ValueError("kill_wait must use second units")
    return int(stripped[:-1])


def _validate_manifest_record(
    record: dict[str, Any],
    *,
    manifest_path: Path,
    line_number: int,
) -> None:
    required = {
        "schema_version",
        "stage",
        "shard_id",
        "interaction_sharding",
        "status",
        "attempt",
        "max_retries",
        "output_dir",
        "success_marker",
        "contract_config_path",
        "worker_resources",
        "expected_range_start",
        "expected_range_end",
        *_HASH_FIELDS,
    }
    missing = sorted(required - set(record))
    if missing:
        raise ValueError(
            f"manifest {manifest_path} line {line_number} missing required field(s): {missing}"
        )
    if record["interaction_sharding"] != "draw-block":
        raise ValueError(
            f"manifest {manifest_path} line {line_number} must use draw-block sharding"
        )
    for field in _HASH_FIELDS:
        value = record[field]
        if not isinstance(value, str) or not value:
            raise ValueError(f"manifest {manifest_path} line {line_number} has empty {field}")
        if any(token in value for token in _PLACEHOLDER_TOKENS):
            raise ValueError(
                f"manifest {manifest_path} line {line_number} contains placeholder text in {field}"
            )
        if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
            raise ValueError(f"manifest {manifest_path} line {line_number} has malformed {field}")
    if any(
        isinstance(value, str) and any(token in value for token in _PLACEHOLDER_TOKENS)
        for value in record.values()
    ):
        raise ValueError(f"manifest {manifest_path} line {line_number} contains placeholder text")
    if int(record["attempt"]) < 0 or int(record["max_retries"]) < 0:
        raise ValueError(f"manifest {manifest_path} line {line_number} has invalid attempt counts")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _hash_file(path: Path) -> str:
    if not path.exists():
        raise ValueError(f"required hash input does not exist: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_scientific_artifacts(result: dict[str, Any], *, allowed_root: Path) -> None:
    """Verify every nested scientific artifact, not only the JSON wrapper."""
    artifacts = result.get("scientific_artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("scientific result has no nested artifact inventory")
    root = allowed_root.resolve(strict=True)
    seen_paths: set[Path] = set()
    seen_relative_paths: set[str] = set()
    for index, item in enumerate(artifacts):
        if not isinstance(item, dict) or set(item) != {
            "path",
            "relative_path",
            "sha256",
            "bytes",
        }:
            raise ValueError(f"scientific artifact {index} has a malformed inventory row")
        path = Path(str(item["path"])).resolve(strict=True)
        try:
            relative = path.relative_to(root)
        except ValueError as exc:
            raise ValueError("scientific artifact escapes its allowed output root") from exc
        relative_text = relative.as_posix()
        if relative_text in {"", "."} or relative_text != str(item["relative_path"]):
            raise ValueError("scientific artifact relative path is inconsistent")
        if path in seen_paths or relative_text in seen_relative_paths:
            raise ValueError("scientific artifact inventory contains a duplicate")
        seen_paths.add(path)
        seen_relative_paths.add(relative_text)
        if not path.is_file() or path.is_symlink():
            raise ValueError("scientific artifact inventory entry is not a regular file")
        if int(item["bytes"]) != path.stat().st_size:
            raise ValueError("scientific artifact byte count differs from the inventory")
        expected_hash = str(item["sha256"])
        if re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None:
            raise ValueError("scientific artifact hash is malformed")
        if _hash_file(path) != expected_hash:
            raise ValueError("scientific artifact byte hash differs from the inventory")


def _hash_python_tree(path: Path) -> str:
    if not path.exists():
        raise ValueError(f"python source tree does not exist: {path}")
    digest = hashlib.sha256()
    for file_path in sorted(path.rglob("*.py")):
        digest.update(str(file_path.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _validate_runtime_identity(records: list[dict[str, Any]]) -> None:
    """Bind every worker, audit, and reducer to the packaged checkout bytes."""
    if not records:
        raise ValueError("runtime identity requires at least one manifest record")
    expected_source = str(records[0]["source_hash"])
    expected_lock = str(records[0]["lock_hash"])
    if any(
        record["source_hash"] != expected_source or record["lock_hash"] != expected_lock
        for record in records
    ):
        raise ValueError("manifest mixes source or lock identities")
    checkout = Path.cwd().resolve()
    if _hash_python_tree(checkout / "src" / "rfm_pipeline") != expected_source:
        raise ValueError("runtime RFM source tree differs from the manifest")
    if _hash_file(checkout / "pixi.lock") != expected_lock:
        raise ValueError("runtime RFM lock file differs from the manifest")


def _stable_hash(payload: Any) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _stable_seed(scenario_id: str, replicate_index: int) -> int:
    return derive_seed(compute_contract_hash(G11_CONTRACT), scenario_id, replicate_index)


def _max_rss_bytes(raw_max_rss: int | float) -> int:
    """Normalize ``ru_maxrss`` to bytes on Linux HPC and macOS development hosts."""
    multiplier = 1 if platform.system() == "Darwin" else 1024
    return int(raw_max_rss * multiplier)


def _prepare_shard_for_attempt(record: dict[str, Any], shard_dir: Path) -> None:
    """Preserve an interrupted prior attempt before the sole authorized retry."""
    shard_dir.mkdir(parents=True, exist_ok=True)
    if not any(shard_dir.iterdir()):
        return
    attempt = int(record["attempt"])
    if attempt <= 0:
        raise RuntimeError("partial shard requires a validated retry manifest")
    archive = shard_dir.parent / f".{shard_dir.name}.partial-attempt-{attempt - 1}"
    if archive.exists():
        raise RuntimeError("partial-attempt archive already exists")
    shard_dir.replace(archive)
    shard_dir.mkdir(parents=True)


def _cli_worker(args: argparse.Namespace) -> int:
    records = load_stage_manifest(args.manifest)
    _validate_runtime_identity(records)
    task_id = int(args.task_id)
    if task_id < 0 or task_id >= len(records):
        raise ValueError(f"task_id {task_id} out of range for manifest with {len(records)} records")
    record = records[task_id]
    if record["stage"] != args.stage:
        raise ValueError("worker stage differs from its manifest record")
    if record["config_hash"] != args.contract_hash:
        raise ValueError("worker contract hash differs from its manifest record")
    if Path(args.contract_config).resolve() != Path(record["contract_config_path"]).resolve():
        raise ValueError("worker contract path differs from its manifest record")
    _, loaded_contract_hash = load_contract(Path(args.contract_config))
    if loaded_contract_hash != args.contract_hash:
        raise ValueError("worker contract bytes differ from the requested contract hash")
    expected_start = int(record.get("dispatch_range_start", 0))
    expected_end = int(record.get("dispatch_range_end", len(records)))
    if (
        int(args.expected_range_start) != expected_start
        or int(args.expected_range_end) != expected_end
        or not expected_start <= task_id < expected_end
    ):
        raise ValueError("worker expected range differs from its manifest resource class")

    shard_dir = Path(record["output_dir"])
    result_path = shard_dir / "result.json"
    success_path = shard_dir / "_SUCCESS.json"
    if success_path.exists():
        stage = StagePlan(
            name=str(record["stage"]),
            partition="resume-validation",
            job_count=len(records),
            parent_stage_name=None,
            worker_resources=ResourceEnvelope(1, 1, 1, 1, 1, 1),
            reducer_resources=ResourceEnvelope(1, 1, 1, 1, 1, 1),
            manifest_path=Path(args.manifest),
            worker_script_path=Path(),
            worker_script_paths=(),
            audit_script_path=Path(),
            audit_output_dir=Path(),
            reducer_script_path=Path(),
            output_root=Path(args.output_root),
            reducer_output_dir=Path(),
            reducer_output_hash="",
            reducer_expected_range=(0, len(records)),
        )
        validate_resume_artifacts(stage)
        return 0
    _prepare_shard_for_attempt(record, shard_dir)

    started = time.perf_counter()
    cpu_started = time.process_time()
    usage_started = resource.getrusage(resource.RUSAGE_SELF)
    result = _execute_worker_operation(record, shard_dir=shard_dir)
    usage_finished = resource.getrusage(resource.RUSAGE_SELF)
    result.update(
        {
            "schema_version": 2,
            "stage": record["stage"],
            "shard_id": record["shard_id"],
            "input_hash": record["input_hash"],
            "schedule_hash": record["schedule_hash"],
            "elapsed_seconds": time.perf_counter() - started,
            "total_cpu_seconds": (
                float(usage_finished.ru_utime + usage_finished.ru_stime)
                - float(usage_started.ru_utime + usage_started.ru_stime)
            ),
            "cpu_time_seconds": time.process_time() - cpu_started,
            "max_rss_bytes": _max_rss_bytes(usage_finished.ru_maxrss),
            "bytes_read": max(0, int(usage_finished.ru_inblock - usage_started.ru_inblock)) * 512,
            "bytes_written": max(0, int(usage_finished.ru_oublock - usage_started.ru_oublock))
            * 512,
            "task_size": int(record.get("task_size", 1)),
            "block_size": int(record.get("block_size", 1)),
            "executed_work_units": int(record.get("executed_work_units", 1)),
            "profile_id": str(record.get("profile_id", "production")),
            "partition": str(record.get("partition", "")),
            "requested_cpus": int(record["worker_resources"]["requested_cpu_cores"]),
            "requested_memory_gb": int(record["worker_resources"]["requested_memory_gb"]),
            "requested_walltime_seconds": int(
                record["worker_resources"]["requested_walltime_seconds"]
            ),
            "status": "completed",
            "attempt": int(record["attempt"]) + 1,
            "hostname": platform.node(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "source_hash": record["source_hash"],
            "config_hash": record["config_hash"],
            "lock_hash": record["lock_hash"],
            "parent_hash": record["parent_hash"],
            "output_hash": record["output_hash"],
        }
    )
    pending_result = shard_dir / "result.pending.json"
    pending_result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pending_result.replace(result_path)
    marker = {
        "schema_version": 2,
        "stage": record["stage"],
        "shard_id": record["shard_id"],
        "output_hash": record["output_hash"],
        "artifact_sha256": _hash_file(result_path),
        "parent_hash": record["parent_hash"],
        "attempt": int(record["attempt"]) + 1,
        "status": "completed",
    }
    pending_marker = shard_dir / "_SUCCESS.pending.json"
    pending_marker.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pending_marker.replace(success_path)
    print(json.dumps(marker, sort_keys=True))
    return 0


def _load_content_hashed_adapter(record: dict[str, Any]) -> ModuleType:
    """Load the exact scientific adapter bound into a manifest record."""
    adapter_path = Path(str(record.get("scientific_adapter_path", "")))
    expected_hash = str(record.get("scientific_adapter_sha256", ""))
    if not adapter_path.is_file():
        raise ValueError("production operation has no scientific adapter")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        raise ValueError("scientific adapter hash is absent or malformed")
    if _hash_file(adapter_path) != expected_hash:
        raise ValueError("scientific adapter hash differs from the manifest")
    module_name = f"_g11_scientific_adapter_{expected_hash}"
    module_spec = importlib.util.spec_from_file_location(module_name, adapter_path)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError("could not load the content-addressed scientific adapter")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def _execute_scientific_reduce(
    records: list[dict[str, Any]], *, output_dir: Path
) -> dict[str, Any]:
    """Dispatch a production reducer through the same immutable adapter."""
    if not records:
        raise ValueError("scientific reducer requires at least one manifest record")
    operation = str(records[0].get("operation", ""))
    adapter_identity = (
        records[0].get("scientific_adapter_path"),
        records[0].get("scientific_adapter_sha256"),
    )
    if any(str(record.get("operation", "")) != operation for record in records):
        raise ValueError("scientific reducer received mixed operations")
    if any(
        (
            record.get("scientific_adapter_path"),
            record.get("scientific_adapter_sha256"),
        )
        != adapter_identity
        for record in records
    ):
        raise ValueError("scientific reducer received mixed adapter identities")
    module = _load_content_hashed_adapter(records[0])
    reduce_stage = getattr(module, "reduce_scientific_stage", None)
    if not callable(reduce_stage):
        raise ValueError("scientific adapter lacks reduce_scientific_stage")
    result = reduce_stage(records, output_dir)
    if not isinstance(result, dict) or result.get("status") != "completed":
        raise RuntimeError("scientific adapter did not return a completed reduction")
    _validate_scientific_artifacts(result, allowed_root=output_dir)
    return result


def _execute_worker_operation(record: dict[str, Any], *, shard_dir: Path) -> dict[str, Any]:
    """Execute one real stage kernel without ever submitting scheduler work."""
    operation = str(record.get("operation", record["stage"]))
    if operation == "scheduler_diagnostic":
        return {
            "operation": operation,
            "status": "completed",
            "hostname": platform.node(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        }
    pilot_dispatch = {
        "pilot_conditioning": _run_pilot_conditioning,
        "pilot_screening": _run_pilot_screening,
        "pilot_interaction_score": _run_pilot_interaction_score,
        "pilot_interaction_reduce": _run_pilot_interaction_reduce,
        "pilot_nonlinear": _run_pilot_nonlinear,
        "pilot_sparse_full": _run_pilot_sparse_full,
        "pilot_sparse_resample": _run_pilot_sparse_resample,
        "pilot_terminal_fit": _run_pilot_terminal_fit,
        "pilot_gate_b_null": _run_pilot_gate_b_null,
        "pilot_recovery": _run_pilot_recovery,
        "pilot_holdout_predict": _run_pilot_holdout_predict,
        "pilot_bootstrap": _run_pilot_bootstrap,
    }
    if operation in pilot_dispatch:
        return pilot_dispatch[operation](record, shard_dir)
    resource_freeze_sha256 = str(record.get("resource_freeze_sha256", ""))
    if re.fullmatch(r"[0-9a-f]{64}", resource_freeze_sha256) is None:
        raise ValueError("production operation lacks an accepted resource freeze")
    module = _load_content_hashed_adapter(record)
    execute = getattr(module, "execute_scientific_work_unit", None)
    if not callable(execute):
        raise ValueError("scientific adapter lacks execute_scientific_work_unit")
    result = execute(record, shard_dir)
    if not isinstance(result, dict) or result.get("status") != "completed":
        raise RuntimeError("scientific adapter did not return a completed result")
    _validate_scientific_artifacts(result, allowed_root=shard_dir)
    return result


def _pilot_seed(record: dict[str, Any]) -> int:
    return int(str(record["schedule_hash"])[:16], 16) % (2**32)


def _pilot_training_tables(
    record: dict[str, Any],
    *,
    n_rows: int = 28_500,
    n_components: int = 17,
    seed: int | None = None,
) -> tuple[Any, Any, Any, Any]:
    """Return a deterministic typed 158-continuous/2-binary pilot fixture."""
    import pandas as pd

    rng = np.random.default_rng(_pilot_seed(record) if seed is None else seed)
    continuous = rng.standard_normal((n_rows, 158))
    binary = rng.integers(0, 2, size=(n_rows, 2), dtype=np.int8)
    values = np.column_stack([continuous, binary])
    feature_names = [f"x{index:03d}" for index in range(158)] + ["binary_0", "binary_1"]
    sample_ids = [f"pilot-{index:04d}" for index in range(n_rows)]
    inputs = pd.DataFrame(values, columns=feature_names)
    inputs.insert(0, "sample_id", sample_ids)
    catalog = pd.DataFrame({"feature_name": feature_names, "feature_type": ["first_order"] * 160})
    assignments = pd.DataFrame({"sample_id": sample_ids, "split": ["train"] * n_rows})
    components = np.column_stack(
        [
            2.0 * continuous[:, index % 6]
            + 1.5 * continuous[:, (index + 1) % 6] * continuous[:, (index + 2) % 6]
            + rng.normal(scale=0.5, size=n_rows)
            for index in range(n_components)
        ]
    )
    pca_scores = pd.DataFrame(
        components,
        columns=[f"PC{index + 1}" for index in range(n_components)],
    )
    pca_scores.insert(0, "sample_id", sample_ids)
    return inputs, catalog, assignments, pca_scores


def _pilot_worker_count() -> int:
    """Use the scheduler-granted CPU count exactly as production specs do."""
    return max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))


def _run_pilot_screening(record: dict[str, Any], shard_dir: Path) -> dict[str, Any]:
    from rfm_pipeline.manuscript_stages import (
        EmpiricalNullScreeningSpec,
        score_screening_draw_block,
    )

    inputs, catalog, assignments, pca_scores = _pilot_training_tables(record)
    spec = EmpiricalNullScreeningSpec(
        statistic="coefficient_row_l2_norm",
        permutation_count_B=G11_CONTRACT.B_screen,
        bh_q_screen=G11_CONTRACT.q_screen,
        retained_terms_reference=0,
        random_seed=_pilot_seed(record),
        n_jobs=_pilot_worker_count(),
    )
    draw_end = min(G11_CONTRACT.B_screen, max(1, int(record["block_size"])))
    block = score_screening_draw_block(
        inputs,
        catalog,
        assignments,
        pca_scores,
        spec,
        draw_start=0,
        draw_end=draw_end,
    )
    block.write(shard_dir / "screening_block")
    return {
        "operation": "pilot_screening",
        "status": "completed",
        "kernel": "score_screening_draw_block",
        "artifact_hash": block.artifact_hash,
        "draw_start": 0,
        "draw_end": draw_end,
        "n_training_rows": block.n_training_rows,
        "n_features": len(block.feature_names),
    }


def _run_pilot_conditioning(record: dict[str, Any], shard_dir: Path) -> dict[str, Any]:
    import pandas as pd

    from rfm_pipeline.manuscript_stages import OutputConditioningSpec, condition_manuscript_outputs

    rng = np.random.default_rng(_pilot_seed(record))
    n_rows = 28_500
    n_outputs = max(2, int(record["block_size"]))
    sample_ids = [f"pilot-{index:04d}" for index in range(n_rows)]
    latent = rng.standard_normal((n_rows, 17))
    loadings = rng.standard_normal((17, n_outputs))
    response = latent @ loadings + rng.normal(scale=0.1, size=(n_rows, n_outputs))
    outputs = pd.DataFrame(response, columns=[f"output_{index:05d}" for index in range(n_outputs)])
    outputs.insert(0, "sample_id", sample_ids)
    assignments = pd.DataFrame({"sample_id": sample_ids, "split": "train"})
    result = condition_manuscript_outputs(
        outputs,
        assignments,
        OutputConditioningSpec(
            epsilon_var=1.0e-12,
            epsilon_snr=1.0e-2,
            snr_delta=1.0e-12,
            method="pca",
            retained_components=17,
            retained_variance_fraction=0.90,
        ),
    )
    path = shard_dir / "conditioning_summary.csv"
    result.summary.to_csv(path, index=False)
    return {
        "operation": "pilot_conditioning",
        "status": "completed",
        "kernel": "condition_manuscript_outputs",
        "artifact_sha256": _hash_file(path),
        "n_rows": n_rows,
        "n_outputs": n_outputs,
        "n_retained_components": len(result.pca_explained_variance),
    }


def _pilot_interaction_spec(record: dict[str, Any]) -> Any:
    from rfm_pipeline.manuscript_stages import InteractionDiscoverySpec

    return InteractionDiscoverySpec(
        method=G11_CONTRACT.interaction_detector_method,
        aggregation_rule="max_over_components_by_detector",
        null_threshold_quantile=0.95,
        retained_pairs_reference=0,
        permutation_count_B=G11_CONTRACT.B_interaction,
        random_seed=_pilot_seed(record),
        n_tree_estimators=G11_CONTRACT.n_tree_estimators,
        max_tree_depth=3,
        max_shap_samples=500,
        n_jobs=_pilot_worker_count(),
        selection_method="max_stat_adjusted_p_mc",
        selection_alpha=G11_CONTRACT.alpha,
        tree_family_alpha=G11_CONTRACT.tree_family_alpha,
        binary_binary_family_alpha=G11_CONTRACT.binary_binary_family_alpha,
        family_partition_method=G11_CONTRACT.family_partition_method,
        binary_binary_method=G11_CONTRACT.binary_binary_method,
        binary_binary_minimum_cell_count=G11_CONTRACT.binary_binary_minimum_cell_count,
        minimum_selection_draws=G11_CONTRACT.B_interaction,
    )


def _run_pilot_interaction_score(record: dict[str, Any], shard_dir: Path) -> dict[str, Any]:
    import pandas as pd

    from rfm_pipeline.interaction_contract import canonical_execution_contract_from_specs
    from rfm_pipeline.manuscript_stages import score_interaction_draw_block

    inputs, catalog, assignments, pca_scores = _pilot_training_tables(record)
    retained_names = [f"x{index:03d}" for index in range(158)] + [
        "binary_0",
        "binary_1",
    ]
    retained = pd.DataFrame({"feature_name": retained_names, "feature_type": ["first_order"] * 160})
    spec = _pilot_interaction_spec(record)
    contract = canonical_execution_contract_from_specs(spec)
    draw_end = min(G11_CONTRACT.B_interaction, max(1, int(record["block_size"])))
    artifact = score_interaction_draw_block(
        inputs,
        catalog,
        assignments,
        pca_scores,
        retained,
        spec,
        draw_start=0,
        draw_end=draw_end,
        contract=contract,
    )
    artifact.write_to(shard_dir / "interaction_block")
    return {
        "operation": "pilot_interaction_score",
        "status": "completed",
        "kernel": "score_interaction_draw_block",
        "artifact_checksum": artifact.payload_sha256,
        "draw_start": 0,
        "draw_end": draw_end,
        "n_candidate_pairs": len(artifact.pair_names),
        "n_tree_estimators": spec.n_tree_estimators,
    }


def _run_pilot_interaction_reduce(record: dict[str, Any], shard_dir: Path) -> dict[str, Any]:
    from itertools import combinations

    from rfm_pipeline.interaction_contract import (
        ScoreOnlyInteractionArtifact,
        build_control_snapshot,
        canonical_execution_contract_from_specs,
    )
    from rfm_pipeline.manuscript_stages import reduce_score_only_interaction_artifacts

    rng = np.random.default_rng(_pilot_seed(record))
    names = tuple([f"x{index:03d}" for index in range(158)] + ["binary_0", "binary_1"])
    pairs = tuple(f"{left}:{right}" for left, right in combinations(names, 2))
    spec = _pilot_interaction_spec(record)
    contract = canonical_execution_contract_from_specs(spec)
    feature_matrix = rng.standard_normal((160, len(names)))
    feature_matrix[:, -2:] = rng.integers(0, 2, size=(160, 2))
    pair_detectors = tuple(
        "studentized_binary_factorial"
        if left.startswith("binary_") and right.startswith("binary_")
        else "tree_shap"
        for left, right in combinations(names, 2)
    )
    snapshot = build_control_snapshot(
        contract,
        candidate_pair_names=pairs,
        candidate_pair_detectors=pair_detectors,
        training_sample_ids=np.arange(160, dtype=np.int64),
        feature_matrix=feature_matrix,
        response_matrix=rng.standard_normal((160, 4)),
        component_names=("PC1", "PC2", "PC3", "PC4"),
    )
    observed = np.abs(rng.standard_normal(len(pairs)))
    null = np.abs(rng.standard_normal((G11_CONTRACT.B_interaction, len(pairs))))
    artifacts = []
    block_size = max(1, int(record["block_size"]))
    for start in range(0, G11_CONTRACT.B_interaction, block_size):
        end = min(G11_CONTRACT.B_interaction, start + block_size)
        artifacts.append(
            ScoreOnlyInteractionArtifact(
                status="score_only_completed",
                draw_range_start=start,
                draw_range_end=end,
                pair_names=pairs,
                observed_scores=observed if start == 0 else np.empty(0),
                null_scores=null[start:end],
                draw_ids=np.arange(start, end, dtype=np.int64),
                control_snapshot=snapshot,
            )
        )
    result = reduce_score_only_interaction_artifacts(
        artifacts,
        spec=spec,
        contract=contract,
        expected_pair_names=pairs,
    )
    output_path = shard_dir / "interaction_pair_scores.csv"
    result.pair_scores.to_csv(output_path, index=False)
    return {
        "operation": "pilot_interaction_reduce",
        "status": "completed",
        "kernel": "reduce_score_only_interaction_artifacts",
        "artifact_sha256": _hash_file(output_path),
        "n_draw_blocks": len(artifacts),
        "n_candidate_pairs": len(result.pair_scores),
        "n_retained_pairs": len(result.retained_pairs),
    }


def _run_pilot_nonlinear(record: dict[str, Any], shard_dir: Path) -> dict[str, Any]:
    import pandas as pd

    from rfm_pipeline.manuscript_stages import (
        NonlinearDiscoverySpec,
        discover_manuscript_nonlinear_transformations,
    )

    inputs, catalog, assignments, pca_scores = _pilot_training_tables(record)
    retained = pd.DataFrame({"feature_name": catalog["feature_name"].astype(str)})
    block_size = min(158, max(1, int(record["block_size"])))
    result = discover_manuscript_nonlinear_transformations(
        inputs,
        catalog,
        assignments,
        pca_scores,
        retained,
        NonlinearDiscoverySpec(
            method="gam_plus_restricted_parametric_replacement",
            curvature_rule="edf_gt_1_and_smooth_pvalue_lt_0p01",
            replacement_selection_rule="minimum_training_rmse_against_gam_smooth",
            identified_transformations_reference=0,
            final_support_transformations_reference=0,
            n_jobs=_pilot_worker_count(),
        ),
        active_feature_indices=set(range(block_size)),
    )
    path = shard_dir / "nonlinear_summary.csv"
    result.summary.to_csv(path, index=False)
    return {
        "operation": "pilot_nonlinear",
        "status": "completed",
        "kernel": "discover_manuscript_nonlinear_transformations",
        "artifact_sha256": _hash_file(path),
        "active_base_features": block_size,
    }


def _pilot_sparse_spec() -> Any:
    from rfm_pipeline.manuscript_stages import SparseSelectionStabilitySpec

    return SparseSelectionStabilitySpec(
        model_class="l1_penalized_linear_model_per_retained_component",
        subsample_count=G11_CONTRACT.n_stability_subsamples,
        subsample_fraction=0.80,
        jaccard_threshold=G11_CONTRACT.stability_jaccard_threshold,
        spearman_threshold=G11_CONTRACT.stability_spearman_threshold,
        random_seed=123,
        n_jobs=_pilot_worker_count(),
    )


def _pilot_sparse_inputs(
    record: dict[str, Any],
    *,
    n_rows: int = 28_500,
    n_components: int = 17,
) -> tuple[Any, ...]:
    from itertools import combinations

    import pandas as pd

    fixture_seed = derive_seed(
        str(record["config_hash"]),
        "pilot_sparse_fixture",
        int(record["pilot_repetition"]),
    )
    inputs, catalog, assignments, pca_scores = _pilot_training_tables(
        record,
        n_rows=n_rows,
        n_components=n_components,
        seed=fixture_seed,
    )
    retained = pd.DataFrame({"feature_name": catalog["feature_name"].astype(str)})
    feature_names = catalog["feature_name"].astype(str).tolist()
    pair_names = [f"{left}:{right}" for left, right in combinations(feature_names, 2)][
        : _PILOT_ENRICHED_CANDIDATE_COUNT - len(feature_names)
    ]
    retained_pairs = pd.DataFrame({"pair_name": pair_names})
    retained_transformations = pd.DataFrame({"feature_name": []})
    return (
        inputs,
        catalog,
        assignments,
        pca_scores,
        retained,
        retained_pairs,
        retained_transformations,
    )


def _run_pilot_sparse_full(record: dict[str, Any], shard_dir: Path) -> dict[str, Any]:
    from rfm_pipeline.manuscript_stages import fit_sparse_full_selection_artifact

    full_fit = fit_sparse_full_selection_artifact(
        *_pilot_sparse_inputs(record),
        _pilot_sparse_spec(),
    )
    artifact_dir = shard_dir / "sparse_full_fit"
    full_fit.write(artifact_dir)
    return {
        "operation": "pilot_sparse_full",
        "status": "completed",
        "kernel": "fit_sparse_full_selection_artifact",
        "artifact_hash": full_fit.artifact_hash,
        "n_candidate_terms": len(full_fit.candidate_names),
    }


def _run_pilot_sparse_resample(record: dict[str, Any], shard_dir: Path) -> dict[str, Any]:
    from rfm_pipeline.manuscript_stages import (
        SparseFullFitArtifact,
        score_sparse_stability_resample,
    )

    full_path = (
        _pilot_stage_root(
            shard_dir,
            "pilot_sparse_full",
            upstream_shard_id=str(record["upstream_shard_id"]),
        )
        / "sparse_full_fit"
    )
    full_fit = SparseFullFitArtifact.read(full_path)
    block_size = min(G11_CONTRACT.n_stability_subsamples, int(record["block_size"]))
    hashes = []
    for resample_id in range(1, block_size + 1):
        block = score_sparse_stability_resample(
            *_pilot_sparse_inputs(record),
            _pilot_sparse_spec(),
            full_fit=full_fit,
            resample_id=resample_id,
        )
        block.write(shard_dir / f"resample-{resample_id:04d}")
        hashes.append(block.artifact_hash)
    return {
        "operation": "pilot_sparse_resample",
        "status": "completed",
        "kernel": "score_sparse_stability_resample",
        "full_fit_hash": full_fit.artifact_hash,
        "resample_count": block_size,
        "block_hashes": hashes,
    }


def _pilot_final_spec() -> Any:
    from rfm_pipeline.manuscript_stages import FinalManuscriptArtifactsSpec

    return FinalManuscriptArtifactsSpec(
        final_predictor_count_reference=0,
        final_first_order_input_count_reference=0,
        intermediate_penalized_holdout_nrmse_reference=0.0,
        final_ols_holdout_nrmse_reference=0.0,
        nrmse_denominator_definition="training_response_range",
        nrmse_min_range=1.0e-12,
        nrmse_reference_matrix="Y_train",
        bootstrap_count=2,
        inferential_filter_alpha=0.05,
        inferential_filter_interval_method=(
            "hc3_wald_95_percent_drop_if_zero_compatible_for_all_outputs"
        ),
        n_jobs=1,
    )


def _run_pilot_terminal_fit(record: dict[str, Any], shard_dir: Path) -> dict[str, Any]:
    import pandas as pd

    from rfm_pipeline.manuscript_stages import (
        terminal_train_fit_and_freeze,
        write_terminal_train_fit_and_freeze,
    )

    rng = np.random.default_rng(_pilot_seed(record))
    x = rng.standard_normal((28_500, _PILOT_ENRICHED_CANDIDATE_COUNT))
    n_outputs = max(2, int(record["block_size"]))
    coefficients = rng.normal(size=(10, n_outputs))
    y = x[:, :10] @ coefficients + rng.normal(scale=0.5, size=(28_500, n_outputs))
    x_frame = pd.DataFrame(
        x,
        columns=[f"candidate_{index:03d}" for index in range(_PILOT_ENRICHED_CANDIDATE_COUNT)],
    )
    y_frame = pd.DataFrame(
        y,
        columns=[f"output_{index:05d}" for index in range(n_outputs)],
    )
    terminal = terminal_train_fit_and_freeze(
        x_frame,
        y_frame,
        spec=_pilot_final_spec(),
        contract_hash=str(record["config_hash"]),
    )
    output_dir = shard_dir / "terminal"
    write_terminal_train_fit_and_freeze(terminal=terminal, output_dir=output_dir)
    return {
        "operation": "pilot_terminal_fit",
        "status": "completed",
        "kernel": "terminal_train_fit_and_freeze",
        "freeze_hash": terminal.freeze_result.freeze_manifest.freeze_hash,
        "n_prefilter_features": len(terminal.prefilter_feature_names),
        "n_hc3_features": len(terminal.hc3_feature_names),
        "n_final_features": len(terminal.final_feature_names),
        "n_outputs": n_outputs,
    }


def _run_pilot_recovery(record: dict[str, Any], shard_dir: Path) -> dict[str, Any]:
    """Exercise the production recovery API on one full-scale strong fixture."""
    import pandas as pd

    from rfm_pipeline.manuscript_stages import build_manuscript_feature_design
    from rfm_pipeline.recovery_study import (
        run_production_recovery_pipeline,
        run_recovery_comparators,
        write_recovery_comparator_result,
    )

    rng = np.random.default_rng(_pilot_seed(record))
    x_train = rng.standard_normal((2000, 160))
    x_eval = rng.standard_normal((500, 160))
    x_train[:, 158:] = rng.integers(0, 2, size=(2000, 2))
    x_eval[:, 158:] = rng.integers(0, 2, size=(500, 2))
    coefficients = rng.normal(size=(6, 40))
    y_train = x_train[:, :6] @ coefficients
    y_eval = x_eval[:, :6] @ coefficients
    interaction_coef = rng.normal(size=40)
    y_train += (x_train[:, 0] * x_train[:, 1])[:, None] * interaction_coef
    y_eval += (x_eval[:, 0] * x_eval[:, 1])[:, None] * interaction_coef
    y_train += rng.normal(scale=0.5, size=y_train.shape)
    y_eval += rng.normal(scale=0.5, size=y_eval.shape)
    feature_names = [f"x{index:03d}" for index in range(158)] + [
        "binary_0",
        "binary_1",
    ]
    train_feature_frame = pd.DataFrame(x_train, columns=feature_names)
    eval_feature_frame = pd.DataFrame(x_eval, columns=feature_names)
    result = run_production_recovery_pipeline(
        train_feature_frame,
        y_train,
        eval_feature_frame,
        y_eval,
        execution_contract=G11_CONTRACT,
        artifact_dir=shard_dir / "recovery_pipeline",
        seed=_pilot_seed(record),
    )
    train_inputs = train_feature_frame.copy()
    train_inputs.insert(0, "sample_id", np.arange(len(x_train)))
    eval_inputs = eval_feature_frame.copy()
    eval_inputs.insert(0, "sample_id", np.arange(len(x_eval)))
    algebraic_catalog = pd.DataFrame({"feature_name": list(result.algebraic_candidate_names)})
    algebraic_train = build_manuscript_feature_design(
        train_inputs,
        algebraic_catalog,
    ).drop(columns=["sample_id"])
    algebraic_eval = build_manuscript_feature_design(
        eval_inputs,
        algebraic_catalog,
    ).drop(columns=["sample_id"])
    oracle_train = np.column_stack([x_train[:, :6], x_train[:, 0] * x_train[:, 1]])
    oracle_eval = np.column_stack([x_eval[:, :6], x_eval[:, 0] * x_eval[:, 1]])
    comparators = run_recovery_comparators(
        X_train=x_train,
        Y_train=y_train,
        X_eval=x_eval,
        oracle_train_design=oracle_train,
        oracle_eval_design=oracle_eval,
        algebraic_train_design=algebraic_train.to_numpy(dtype=float),
        algebraic_eval_design=algebraic_eval.to_numpy(dtype=float),
        proposed_predictions=result.eval_predictions,
        execution_contract=G11_CONTRACT,
        seed=_pilot_seed(record),
    )
    write_recovery_comparator_result(comparators, shard_dir / "comparators")
    predictions_path = shard_dir / "recovery_eval_predictions.npy"
    np.save(predictions_path, result.eval_predictions, allow_pickle=False)
    return {
        "operation": "pilot_recovery",
        "status": "completed",
        "kernel": "run_production_recovery_pipeline",
        "predictions_sha256": _hash_file(predictions_path),
        "screening_retained": len(result.screening_retained_set),
        "interaction_retained": len(result.interaction_retained_set),
        "terminal_support": len(result.final_selected_support),
        "comparator_schedule_sha256": comparators.schedule_sha256,
        "comparator_names": sorted(comparators.predictions),
    }


def _run_pilot_gate_b_null(record: dict[str, Any], shard_dir: Path) -> dict[str, Any]:
    """Exercise the submitted method on one exact-scale Gate-B null fixture."""
    from rfm_pipeline.recovery_study import run_production_recovery_pipeline

    rng = np.random.default_rng(_pilot_seed(record))
    x_train = rng.standard_normal((160, 160))
    x_eval = rng.standard_normal((80, 160))
    x_train[:, 158:] = rng.integers(0, 2, size=(160, 2))
    x_eval[:, 158:] = rng.integers(0, 2, size=(80, 2))
    y_train = rng.standard_normal((160, 4))
    y_eval = rng.standard_normal((80, 4))
    result = run_production_recovery_pipeline(
        x_train,
        y_train,
        x_eval,
        y_eval,
        execution_contract=G11_CONTRACT,
        artifact_dir=shard_dir / "gate_b_null_pipeline",
        seed=_pilot_seed(record),
    )
    predictions_path = shard_dir / "gate_b_null_eval_predictions.npy"
    np.save(predictions_path, result.eval_predictions, allow_pickle=False)
    return {
        "operation": "pilot_gate_b_null",
        "status": "completed",
        "kernel": "run_production_recovery_pipeline",
        "predictions_sha256": _hash_file(predictions_path),
        "screening_retained": len(result.screening_retained_set),
        "interaction_retained": len(result.interaction_retained_set),
        "terminal_support": len(result.final_selected_support),
        "comparators_executed": False,
    }


def _pilot_stage_root(shard_dir: Path, stage_name: str, *, upstream_shard_id: str) -> Path:
    return shard_dir.parents[2] / stage_name / "results" / upstream_shard_id


def _run_pilot_holdout_predict(record: dict[str, Any], shard_dir: Path) -> dict[str, Any]:
    from rfm_pipeline.manuscript_stages import (
        _load_train_fit_and_freeze,
        holdout_predict_from_files,
        write_frozen_prediction_matrices,
    )

    freeze_dir = (
        _pilot_stage_root(
            shard_dir,
            "pilot_terminal_fit",
            upstream_shard_id=str(record["upstream_shard_id"]),
        )
        / "terminal"
        / "model"
    )
    freeze = _load_train_fit_and_freeze(freeze_dir)
    rng = np.random.default_rng(_pilot_seed(record))
    n_rows = 1_500
    n_features = len(freeze.freeze_manifest.feature_names)
    n_outputs = len(freeze.freeze_manifest.output_names)
    x_path = shard_dir / "x_holdout.npy"
    y_path = shard_dir / "y_holdout.npy"
    np.save(x_path, rng.standard_normal((n_rows, n_features)), allow_pickle=False)
    np.save(y_path, rng.standard_normal((n_rows, n_outputs)), allow_pickle=False)
    frozen_by_model = []
    for _model_index in range(_PILOT_APPLIED_MODEL_COUNT):
        frozen_by_model.append(
            holdout_predict_from_files(
                freeze_dir=freeze_dir,
                x_holdout_path=x_path,
                y_holdout_path=y_path,
                holdout_ids=tuple(f"pilot-holdout-{index:04d}" for index in range(n_rows)),
                strata=np.asarray([f"scenario-{index % 4}" for index in range(n_rows)]),
            )
        )
    output_dir = shard_dir / "frozen_predictions"
    for model_index, frozen in enumerate(frozen_by_model):
        write_frozen_prediction_matrices(
            frozen=frozen,
            output_dir=output_dir / f"model-{model_index}",
        )
    return {
        "operation": "pilot_holdout_predict",
        "status": "completed",
        "kernel": "holdout_predict_from_files",
        "freeze_hash": frozen_by_model[0].freeze_hash,
        "n_holdout_rows": n_rows,
        "n_outputs": n_outputs,
        "model_count": len(frozen_by_model),
        "artifact_sha256": _hash_file(output_dir / "model-0" / "frozen_predictions.npz"),
    }


def _run_pilot_bootstrap(record: dict[str, Any], shard_dir: Path) -> dict[str, Any]:
    from rfm_pipeline.manuscript_stages import (
        bootstrap_metric_block,
        read_frozen_prediction_matrices,
    )

    source_dir = (
        _pilot_stage_root(
            shard_dir,
            "pilot_holdout_predict",
            upstream_shard_id=str(record["upstream_shard_id"]),
        )
        / "frozen_predictions"
    )
    draw_end = min(G11_CONTRACT.bootstrap_draws, max(1, int(record["block_size"])))
    blocks = []
    for model_index in range(_PILOT_APPLIED_MODEL_COUNT):
        frozen = read_frozen_prediction_matrices(source_dir / f"model-{model_index}")
        block = bootstrap_metric_block(
            frozen,
            draw_start=0,
            draw_end=draw_end,
            random_seed=_pilot_seed(record),
        )
        block_path = shard_dir / f"model-{model_index}-bootstrap_block.npz"
        np.savez_compressed(
            block_path,
            per_draw_macro_nrmse=block.per_draw_macro_nrmse,
            draw_ids=block.draw_ids,
        )
        blocks.append(block)
    return {
        "operation": "pilot_bootstrap",
        "status": "completed",
        "kernel": "bootstrap_metric_block",
        "freeze_hash": blocks[0].freeze_hash,
        "bootstrap_schedule_hash": blocks[0].schedule_hash,
        "draw_start": blocks[0].draw_start,
        "draw_end": blocks[0].draw_end,
        "eligible_count": blocks[0].eligible_count,
        "model_count": len(blocks),
        "artifact_sha256": _hash_file(shard_dir / "model-0-bootstrap_block.npz"),
    }


def _cli_reduce(args: argparse.Namespace) -> int:
    records = load_stage_manifest(args.manifest)
    _validate_runtime_identity(records)
    if any(
        Path(record["contract_config_path"]).resolve() != Path(args.contract_config).resolve()
        for record in records
    ):
        raise ValueError("reducer contract path differs from its manifest")
    _, loaded_contract_hash = load_contract(Path(args.contract_config))
    if loaded_contract_hash != args.contract_hash:
        raise ValueError("reducer contract bytes differ from the requested contract hash")
    audit_success_path = Path(args.audit_success)
    if not audit_success_path.is_file():
        raise ValueError("reducer requires a completed exact-coverage audit")
    audit_marker = json.loads(audit_success_path.read_text(encoding="utf-8"))
    audit_result_path = audit_success_path.parent / "audit_result.json"
    if (
        audit_marker.get("stage") != args.stage
        or audit_marker.get("status") != "completed"
        or not audit_result_path.is_file()
        or audit_marker.get("artifact_sha256") != _hash_file(audit_result_path)
    ):
        raise ValueError("reducer rejected invalid exact-coverage audit evidence")
    if int(args.expected_range_start) != 0 or int(args.expected_range_end) != len(records):
        raise ValueError("reducer expected range differs from the complete stage manifest")
    artifact_hashes: list[str] = []
    for record in records:
        shard_dir = Path(record["output_dir"])
        marker_path = shard_dir / "_SUCCESS.json"
        artifact_path = shard_dir / "result.json"
        if not marker_path.is_file() or not artifact_path.is_file():
            raise ValueError(f"reducer found incomplete shard {record['shard_id']}")
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        artifact_hash = _hash_file(artifact_path)
        if (
            marker.get("stage") != args.stage
            or marker.get("shard_id") != record["shard_id"]
            or marker.get("artifact_sha256") != artifact_hash
            or marker.get("output_hash") != record["output_hash"]
            or marker.get("parent_hash") != record["parent_hash"]
            or marker.get("status") != "completed"
        ):
            raise ValueError(f"reducer rejected shard identity {record['shard_id']}")
        operation = str(record.get("operation", ""))
        if operation != "scheduler_diagnostic" and not operation.startswith("pilot_"):
            _validate_scientific_artifacts(
                json.loads(artifact_path.read_text(encoding="utf-8")),
                allowed_root=shard_dir,
            )
        artifact_hashes.append(artifact_hash)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    result_path = output_root / "reduced_result.json"
    summary = {
        "schema_version": 2,
        "stage": args.stage,
        "records": len(records),
        "contract_hash": args.contract_hash,
        "parent_hash": args.expected_parent_hash,
        "artifact_hashes": artifact_hashes,
        "status": "completed",
    }
    if not args.stage.startswith("pilot_") and args.stage != "scheduler_diagnostic":
        summary["scientific_reduction"] = _execute_scientific_reduce(
            records,
            output_dir=output_root,
        )
    result_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    reducer_output_hash = _stable_hash(
        {
            "stage": args.stage,
            "job_count": len(records),
            "parent_hash": args.expected_parent_hash,
            "source_hash": records[0]["source_hash"],
            "config_hash": records[0]["config_hash"],
            "lock_hash": records[0]["lock_hash"],
        }
    )
    marker = {
        "schema_version": 2,
        "stage": args.stage,
        "status": "completed",
        "output_hash": reducer_output_hash,
        "artifact_sha256": _hash_file(result_path),
    }
    (output_root / "_SUCCESS.json").write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(marker, sort_keys=True))
    return 0


def _cli_audit(args: argparse.Namespace) -> int:
    """Validate exact worker coverage before any scientific reducer may run."""
    records = load_stage_manifest(args.manifest)
    _validate_runtime_identity(records)
    if any(record["stage"] != args.stage for record in records):
        raise ValueError("audit stage differs from its manifest")
    if any(record["config_hash"] != args.contract_hash for record in records):
        raise ValueError("audit contract hash differs from its manifest")
    stage = StagePlan(
        name=args.stage,
        partition="audit-validation",
        job_count=len(records),
        parent_stage_name=None,
        worker_resources=ResourceEnvelope(1, 1, 1, 1, 1, 1),
        reducer_resources=ResourceEnvelope(1, 1, 1, 1, 1, 1),
        manifest_path=Path(args.manifest),
        worker_script_path=Path(),
        worker_script_paths=(),
        audit_script_path=Path(),
        audit_output_dir=Path(args.output_root),
        reducer_script_path=Path(),
        output_root=Path(records[0]["output_dir"]).parent,
        reducer_output_dir=Path(),
        reducer_output_hash="",
        reducer_expected_range=(0, len(records)),
    )
    summary = validate_resume_artifacts(stage)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    ledger_rows = []
    for record in records:
        shard_dir = Path(record["output_dir"])
        state = (
            "COMPLETED"
            if (shard_dir / "_SUCCESS.json").is_file()
            else "FAILED"
            if (shard_dir / "_FAILED.json").is_file()
            else "MISSING"
        )
        ledger_rows.append(
            {
                "stage": args.stage,
                "shard_id": record["shard_id"],
                "attempt": int(record["attempt"]) + 1,
                "state": state,
                "input_hash": record["input_hash"],
                "schedule_hash": record["schedule_hash"],
                "retry_permitted": False,
                "retry_reason": "requires separately validated scheduler infrastructure state",
            }
        )
    _write_jsonl(output_root / "attempt_ledger.jsonl", ledger_rows)
    if summary != {"completed": len(records), "failed": 0, "pending": 0}:
        failure = {
            "stage": args.stage,
            "status": "failed",
            "coverage": summary,
            "contract_hash": args.contract_hash,
        }
        (output_root / "_FAILED.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raise RuntimeError(f"audit rejected incomplete stage {args.stage}: {summary}")
    result_path = output_root / "audit_result.json"
    result = {
        "stage": args.stage,
        "status": "completed",
        "coverage": summary,
        "contract_hash": args.contract_hash,
        "manifest_sha256": _hash_file(Path(args.manifest)),
        "attempt_ledger_sha256": _hash_file(output_root / "attempt_ledger.jsonl"),
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    marker = {
        "stage": args.stage,
        "status": "completed",
        "artifact_sha256": _hash_file(result_path),
    }
    (output_root / "_SUCCESS.json").write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(marker, sort_keys=True))
    return 0


def _cli_live_smoke(args: argparse.Namespace) -> int:
    dag = generate_campaign_package(
        output_dir=args.package_root,
        repo_root=args.repo_root,
        config_path=args.config,
        package_mode="full",
    )
    result = run_hpc_live_smoke(
        dag,
        target_phase="pilot",
        remaining_au=args.remaining_au,
        evidence_dir=args.evidence_root,
    )
    preflight = result["preflight"]
    print(
        json.dumps(
            {
                "status": preflight["status"],
                "target_phase": preflight["target_phase"],
                "required_au": preflight["required_au"],
                "preflight_sha256": preflight["preflight_sha256"],
                "package_root": str(Path(args.package_root).resolve()),
                "evidence_root": str(Path(args.evidence_root).resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI surface used by generated scripts for validation-only dry execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    worker = subparsers.add_parser("worker")
    worker.add_argument("--manifest", required=True)
    worker.add_argument("--stage", required=True)
    worker.add_argument("--task-id", required=True, type=int)
    worker.add_argument("--output-root", required=True)
    worker.add_argument("--scratch-root", required=True)
    worker.add_argument("--contract-config", required=True)
    worker.add_argument("--contract-hash", required=True)
    worker.add_argument("--expected-range-start", required=True, type=int)
    worker.add_argument("--expected-range-end", required=True, type=int)
    worker.set_defaults(func=_cli_worker)

    reducer = subparsers.add_parser("reduce")
    reducer.add_argument("--manifest", required=True)
    reducer.add_argument("--stage", required=True)
    reducer.add_argument("--output-root", required=True)
    reducer.add_argument("--audit-success", required=True)
    reducer.add_argument("--contract-config", required=True)
    reducer.add_argument("--contract-hash", required=True)
    reducer.add_argument("--expected-range-start", required=True, type=int)
    reducer.add_argument("--expected-range-end", required=True, type=int)
    reducer.add_argument("--expected-parent-hash", required=True)
    reducer.set_defaults(func=_cli_reduce)

    audit = subparsers.add_parser("audit")
    audit.add_argument("--manifest", required=True)
    audit.add_argument("--stage", required=True)
    audit.add_argument("--output-root", required=True)
    audit.add_argument("--contract-hash", required=True)
    audit.set_defaults(func=_cli_audit)

    live_smoke = subparsers.add_parser(
        "live-smoke",
        help="run read-only Kestrel pilot probes and sbatch --test-only checks",
    )
    live_smoke.add_argument("--package-root", required=True)
    live_smoke.add_argument("--evidence-root", required=True)
    live_smoke.add_argument("--repo-root", required=True)
    live_smoke.add_argument("--config", required=True)
    live_smoke.add_argument("--remaining-au", required=True, type=int)
    live_smoke.set_defaults(func=_cli_live_smoke)

    args = parser.parse_args(argv)
    return int(args.func(args))


__all__ = [
    "CampaignDAG",
    "CampaignEnvelope",
    "ClusterConfig",
    "ResourceEnvelope",
    "StagePlan",
    "build_campaign_phase_plan",
    "build_submission_plan",
    "collect_pilot_accounting",
    "execute_campaign_phase_plan",
    "execute_retry_submission_plan",
    "execute_submission_plan",
    "finalize_post_resolution_package",
    "generate_campaign_package",
    "load_stage_manifest",
    "main",
    "plan_retry_attempts",
    "prepare_stage_retry_package",
    "run_hpc_live_smoke",
    "select_pilot_resources",
    "validate_hpc_preflight",
    "validate_campaign_table",
    "validate_resume_artifacts",
    "validate_stage_dependencies",
    "write_phase_authorization",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
