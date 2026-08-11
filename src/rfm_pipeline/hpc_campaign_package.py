"""Content-addressed NO-SUBMIT Kestrel HPC packaging for G11."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from rfm_pipeline.campaign_contract import G11_CONTRACT, compute_contract_hash, render_contract_toml

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
_RESOLUTION_FIXTURES = ("B", "2B")
_ROOT_STAGE_NAME = "resolution"


@dataclass(frozen=True, slots=True)
class ClusterConfig:
    """Committed Kestrel execution surface."""

    account: str
    max_array_size: int
    cpu_cores_per_node: int
    memory_per_node_gb: int
    partitions: tuple[str, ...]
    project_root: str
    scratch_template: str
    kill_wait: str
    job_requeue: bool
    allocation_unit: str
    allocation_quota: str


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
    """Conservative pre-pilot package capacity envelope.

    The storage cap is a package admission guard (one GiB per worker
    artifact), not a claim about an available Kestrel filesystem quota.
    Kestrel allocation units are node-hours, not CPU-core-hours.
    """

    estimated_au_node_hours: float
    requested_au_node_hours: int
    stage_au_node_hours: tuple[StageAllocationEstimate, ...]
    estimated_storage_gb: int
    requested_storage_gb: int
    estimated_inode_count: int
    requested_inode_count: int
    estimated_retry_attempts: int
    requested_retry_attempts: int
    allocation_quota_readiness_blocker: bool


@dataclass(frozen=True, slots=True)
class StageAllocationEstimate:
    """Estimated and requested Kestrel allocation node-hours for one stage."""

    stage_name: str
    estimated_au_node_hours: float
    requested_au_node_hours: float


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
    reducer_script_path: Path
    output_root: Path
    reducer_output_dir: Path
    reducer_output_hash: str
    reducer_expected_range: tuple[int, int]


@dataclass(frozen=True, slots=True)
class CampaignDAG:
    """Generated content-addressed campaign package."""

    run_id: str
    repo_root: Path
    output_dir: Path
    cluster: ClusterConfig
    scheduler_submission_permitted: bool
    source_hash: str
    config_hash: str
    lock_hash: str
    contract_config_path: Path
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
) -> CampaignDAG:
    """Build manifests, scripts, hashes, telemetry, and readiness artifacts."""
    resolved_repo_root = (
        Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parents[2]
    )
    resolved_output_dir = Path(output_dir).resolve()
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
    config_hash = compute_contract_hash(G11_CONTRACT)

    contract_dir = resolved_output_dir / "contract"
    contract_dir.mkdir(parents=True, exist_ok=True)
    contract_config_path = contract_dir / "g11_campaign_contract.toml"
    contract_config_path.write_text(
        render_contract_toml(G11_CONTRACT, config_hash, gate="HPC", status="NO_SUBMIT"),
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
            ("runtime_seconds", "float"),
            ("peak_memory_gb", "float"),
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
    readiness_blockers = (
        "scheduler_submission_permitted is false; package is NO-SUBMIT by construction",
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
    )

    campaign_envelope = _build_campaign_envelope(
        stages,
        cluster=cluster,
        retry_limit=G11_CONTRACT.retry_limit,
        allocation_quota_is_unknown=False,
    )
    try:
        allocation_quota = int(cluster.allocation_quota)
    except ValueError as exc:
        raise ValueError("allocation_quota must be a positive integer AU limit.") from exc
    if allocation_quota < campaign_envelope.requested_au_node_hours:
        raise ValueError(
            "allocation_quota is below the conservative requested AU envelope: "
            f"{allocation_quota} < {campaign_envelope.requested_au_node_hours}."
        )
    dag = CampaignDAG(
        run_id=raw_config["run_id"],
        repo_root=resolved_repo_root,
        output_dir=resolved_output_dir,
        cluster=cluster,
        scheduler_submission_permitted=scheduler_submission_permitted,
        source_hash=source_hash,
        config_hash=config_hash,
        lock_hash=lock_hash,
        contract_config_path=contract_config_path,
        readiness_blockers=readiness_blockers,
        telemetry_schema=telemetry_schema,
        pilot_matrix=pilot_matrix,
        post_pilot_selection=post_pilot_selection,
        campaign_envelope=campaign_envelope,
        stages=stages,
    )
    _write_package_metadata(dag)
    return dag


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
        cpu_cores_per_node=int(raw["cpu_cores_per_node"]),
        memory_per_node_gb=int(raw["memory_per_node_gb"]),
        partitions=tuple(str(value) for value in raw["partitions"]),
        project_root=str(raw["project_root"]),
        scratch_template=str(raw["scratch_template"]),
        kill_wait=str(raw["kill_wait"]),
        job_requeue=bool(raw["job_requeue"]),
        allocation_unit=str(raw["allocation_unit"]),
        allocation_quota=str(raw["allocation_quota"]),
    )
    if cluster.account != "bsm":
        raise ValueError("Kestrel package must target account=bsm")
    if cluster.max_array_size != 11000:
        raise ValueError("Kestrel package must target max_array_size=11000")
    if cluster.cpu_cores_per_node != 104 or cluster.memory_per_node_gb != 240:
        raise ValueError("Kestrel package must use the committed 104-core / 240-GB node shape")
    if cluster.allocation_unit != "node_hour":
        raise ValueError("Kestrel package allocation_unit must be node_hour")
    if not cluster.partitions:
        raise ValueError("Kestrel package must list at least one partition")
    return cluster


def _build_pilot_matrix(cluster: ClusterConfig) -> tuple[dict[str, Any], ...]:
    resolution_rows = [
        {
            "stage": "resolution",
            "fixture": fixture,
            "schedule_index": schedule_index,
            "partition": "debug",
            "cpus": 30,
            "memory_gb": 60,
            "walltime": "00:20:00",
            "schedule_hash": _stable_hash(
                {
                    "stage": "resolution",
                    "fixture": fixture,
                    "schedule_index": schedule_index,
                }
            ),
        }
        for fixture in _RESOLUTION_FIXTURES
        for schedule_index in range(G11_CONTRACT.resolution_schedules)
    ]
    gate_rows = [
        {
            "stage": "gate_b",
            "scenario_id": scenario.id,
            "replicate_index": 0,
            "partition": "short",
            "cpus": 80,
            "memory_gb": 160,
            "walltime": "02:30:00",
            "schedule_hash": _stable_hash({"stage": "gate_b", "scenario_id": scenario.id}),
        }
        for scenario in G11_CONTRACT.scenarios
        if scenario.kind == "null"
    ]
    recovery_rows = [
        {
            "stage": "recovery",
            "scenario_id": scenario.id,
            "replicate_index": 0,
            "partition": "short",
            "cpus": 60,
            "memory_gb": 120,
            "walltime": "02:00:00",
            "schedule_hash": _stable_hash({"stage": "recovery", "scenario_id": scenario.id}),
        }
        for scenario in G11_CONTRACT.scenarios
        if scenario.kind in {"strong", "stress"}
    ]
    supplement_rows = [
        {
            "stage": "fixed_family_supplement",
            "family_size": family_size,
            "schedule_index": schedule_index,
            "partition": "shared",
            "cpus": 20,
            "memory_gb": 40,
            "walltime": "00:40:00",
            "schedule_hash": _stable_hash(
                {
                    "stage": "fixed_family_supplement",
                    "family_size": family_size,
                    "schedule_index": schedule_index,
                }
            ),
        }
        for family_size in G11_CONTRACT.fixed_family_sizes
        for schedule_index in range(G11_CONTRACT.resolution_schedules)
    ]
    matrix = tuple(resolution_rows + gate_rows + recovery_rows + supplement_rows)
    for row in matrix:
        if row["partition"] not in cluster.partitions:
            raise ValueError(f"pilot matrix uses uncommitted partition {row['partition']}")
    return matrix


def _select_post_pilot_profiles(
    pilot_matrix: tuple[dict[str, Any], ...],
) -> dict[str, dict[str, Any]]:
    selection: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in pilot_matrix:
        grouped.setdefault(str(row["stage"]), []).append(row)

    for stage_name, rows in grouped.items():
        chosen = min(
            rows,
            key=lambda row: (
                str(row["partition"]),
                str(row.get("scenario_id", "")),
                int(row.get("family_size", 0)),
                int(row.get("schedule_index", -1)),
                str(row.get("fixture", "")),
            ),
        )
        selection[stage_name] = {
            "partition": chosen["partition"],
            "schedule_hash": chosen["schedule_hash"],
            "selector": "lexicographic-minimum-pilot-profile",
        }
    return selection


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
) -> tuple[StagePlan, ...]:
    stage_specs = (
        {
            "name": "resolution",
            "partition": "debug",
            "job_count": len(_RESOLUTION_FIXTURES) * G11_CONTRACT.resolution_schedules,
            "parent_stage_name": None,
            "worker_resources": _resource_envelope(24, 48, 900),
            "reducer_resources": _resource_envelope(8, 16, 300),
        },
        {
            "name": "gate_b",
            "partition": "short",
            "job_count": sum(
                scenario.n_replicates
                for scenario in G11_CONTRACT.scenarios
                if scenario.kind == "null"
            ),
            "parent_stage_name": "resolution",
            "worker_resources": _resource_envelope(64, 128, 7200),
            "reducer_resources": _resource_envelope(24, 96, 1800),
        },
        {
            "name": "recovery",
            "partition": "short",
            "job_count": sum(
                scenario.n_replicates
                for scenario in G11_CONTRACT.scenarios
                if scenario.kind in {"strong", "stress"}
            ),
            "parent_stage_name": "gate_b",
            "worker_resources": _resource_envelope(48, 96, 5400),
            "reducer_resources": _resource_envelope(24, 96, 1800),
        },
        {
            "name": "fixed_family_supplement",
            "partition": "shared",
            "job_count": len(G11_CONTRACT.fixed_family_sizes) * G11_CONTRACT.resolution_schedules,
            "parent_stage_name": "recovery",
            "worker_resources": _resource_envelope(16, 32, 1800),
            "reducer_resources": _resource_envelope(12, 24, 600),
        },
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
        output_root = stage_dir / "results"
        reducer_output_dir = stage_dir / "reducer"
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
            stage_name=str(spec["name"]),
            job_count=int(spec["job_count"]),
            output_root=output_root,
            parent_hash=parent_hash,
            source_hash=source_hash,
            config_hash=config_hash,
            lock_hash=lock_hash,
            worker_resources=spec["worker_resources"],
            contract_config_path=contract_config_path,
        )
        _write_jsonl(manifest_path, records)
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
            reducer_script_path=reducer_script_path,
            output_root=output_root,
            reducer_output_dir=reducer_output_dir,
            reducer_output_hash=reducer_output_hash,
            reducer_expected_range=(0, int(spec["job_count"])),
        )
        plans.append(plan)
        reducer_hash_by_stage[plan.name] = reducer_output_hash
    return tuple(plans)


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
) -> CampaignEnvelope:
    """Calculate a whole-package envelope without asserting a live quota.

    Each planned worker is capped at one GiB of durable artifact output and
    six durable files (result, telemetry, success marker, and bounded
    stage-native payload/metadata). These are deliberately package limits,
    not inferred Kestrel capacities.
    """
    stage_au_node_hours = tuple(
        _stage_allocation_estimate(stage, cluster=cluster) for stage in stages
    )
    estimated_au = sum(estimate.estimated_au_node_hours for estimate in stage_au_node_hours)
    requested_au = math.ceil(
        sum(estimate.requested_au_node_hours for estimate in stage_au_node_hours)
    )
    total_workers = sum(stage.job_count for stage in stages)
    estimated_storage_gb = total_workers + len(stages)
    estimated_inode_count = total_workers * 6 + len(stages) * 6
    estimated_retry_attempts = total_workers * retry_limit
    return CampaignEnvelope(
        estimated_au_node_hours=estimated_au,
        requested_au_node_hours=max(requested_au, math.ceil(estimated_au * 1.25)),
        stage_au_node_hours=stage_au_node_hours,
        estimated_storage_gb=estimated_storage_gb,
        requested_storage_gb=math.ceil(estimated_storage_gb * 1.25),
        estimated_inode_count=estimated_inode_count,
        requested_inode_count=math.ceil(estimated_inode_count * 1.25),
        estimated_retry_attempts=estimated_retry_attempts,
        requested_retry_attempts=math.ceil(estimated_retry_attempts * 1.25),
        allocation_quota_readiness_blocker=allocation_quota_is_unknown,
    )


def _stage_allocation_estimate(
    stage: StagePlan,
    *,
    cluster: ClusterConfig,
) -> StageAllocationEstimate:
    """Calculate Kestrel allocation usage from allocated node count and walltime."""
    worker_estimated_nodes = _allocated_node_count(stage.worker_resources, cluster=cluster)
    worker_requested_nodes = _allocated_node_count(
        stage.worker_resources,
        cluster=cluster,
        requested=True,
    )
    reducer_estimated_nodes = _allocated_node_count(stage.reducer_resources, cluster=cluster)
    reducer_requested_nodes = _allocated_node_count(
        stage.reducer_resources,
        cluster=cluster,
        requested=True,
    )
    return StageAllocationEstimate(
        stage_name=stage.name,
        estimated_au_node_hours=(
            stage.job_count
            * worker_estimated_nodes
            * stage.worker_resources.estimated_walltime_seconds
            / 3600
            + reducer_estimated_nodes * stage.reducer_resources.estimated_walltime_seconds / 3600
        ),
        requested_au_node_hours=(
            stage.job_count
            * worker_requested_nodes
            * stage.worker_resources.requested_walltime_seconds
            / 3600
            + reducer_requested_nodes * stage.reducer_resources.requested_walltime_seconds / 3600
        ),
    )


def _allocated_node_count(
    resources: ResourceEnvelope,
    *,
    cluster: ClusterConfig,
    requested: bool = False,
) -> int:
    """Return the whole Kestrel nodes reserved for a resource request."""
    cpu_cores = resources.requested_cpu_cores if requested else resources.estimated_cpu_cores
    memory_gb = resources.requested_memory_gb if requested else resources.estimated_memory_gb
    return max(
        math.ceil(cpu_cores / cluster.cpu_cores_per_node),
        math.ceil(memory_gb / cluster.memory_per_node_gb),
    )


def _build_manifest_records(
    *,
    stage_name: str,
    job_count: int,
    output_root: Path,
    parent_hash: str,
    source_hash: str,
    config_hash: str,
    lock_hash: str,
    worker_resources: ResourceEnvelope,
    contract_config_path: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if stage_name == "resolution":
        descriptors = [
            {
                "fixture": fixture,
                "schedule_index": schedule_index,
                "expected_range_start": index,
                "expected_range_end": index + 1,
            }
            for index, (fixture, schedule_index) in enumerate(
                (fixture, schedule_index)
                for fixture in _RESOLUTION_FIXTURES
                for schedule_index in range(G11_CONTRACT.resolution_schedules)
            )
        ]
    elif stage_name == "gate_b":
        descriptors = _replicate_descriptors({"null"})
    elif stage_name == "recovery":
        descriptors = _replicate_descriptors({"strong", "stress"})
    elif stage_name == "fixed_family_supplement":
        descriptors = [
            {
                "family_size": family_size,
                "schedule_index": schedule_index,
                "expected_range_start": index,
                "expected_range_end": index + 1,
            }
            for index, (family_size, schedule_index) in enumerate(
                (family_size, schedule_index)
                for family_size in G11_CONTRACT.fixed_family_sizes
                for schedule_index in range(G11_CONTRACT.resolution_schedules)
            )
        ]
    else:  # pragma: no cover - guarded by stage plan construction
        raise ValueError(f"unsupported stage {stage_name}")

    if len(descriptors) != job_count:
        raise ValueError(
            f"stage {stage_name} expected {job_count} jobs but built {len(descriptors)}"
        )

    for index, descriptor in enumerate(descriptors):
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
                "stage": stage_name,
                "shard_id": shard_id,
                "interaction_sharding": "draw-block",
                "status": "pending",
                "attempt": 0,
                "max_retries": G11_CONTRACT.retry_limit,
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
                "worker_resources": asdict(worker_resources),
                **descriptor,
            }
        )
    return rows


def _replicate_descriptors(kinds: set[str]) -> list[dict[str, Any]]:
    descriptors: list[dict[str, Any]] = []
    index = 0
    for scenario in G11_CONTRACT.scenarios:
        if scenario.kind not in kinds:
            continue
        for replicate_index in range(scenario.n_replicates):
            descriptors.append(
                {
                    "scenario_id": scenario.id,
                    "replicate_index": replicate_index,
                    "expected_range_start": index,
                    "expected_range_end": index + 1,
                }
            )
            index += 1
    return descriptors


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
) -> str:
    scratch_root = _resolved_scratch_template(cluster.scratch_template, run_id)
    return f"""#!/bin/bash
#SBATCH --job-name=g11-{stage_name}-worker
#SBATCH --account={cluster.account}
#SBATCH --partition={partition}
#SBATCH --time={resources.requested_walltime_hms}
#SBATCH --cpus-per-task={resources.requested_cpu_cores}
#SBATCH --mem={resources.requested_memory_gb}G
#SBATCH --array=0-{job_count - 1}%{min(job_count, cluster.max_array_size)}
#SBATCH --output={cluster.project_root}/{run_id}/logs/{stage_name}-%A_%a.out
#SBATCH --signal=USR1@{_kill_wait_seconds(cluster.kill_wait)}
#SBATCH --no-requeue
set -euo pipefail

# NO-SUBMIT package: script generation only; manual review required before any scheduler action.
RUN_ID="{run_id}"
PROJECT_ROOT="{cluster.project_root}"
SCRATCH_ROOT="{scratch_root}"
MANIFEST_PATH="{manifest_path}"
OUTPUT_ROOT="{output_root}"

pixi run python -m rfm_pipeline.hpc_campaign_package worker \\
  --manifest "$MANIFEST_PATH" \\
  --stage "{stage_name}" \\
  --task-id "${{SLURM_ARRAY_TASK_ID:-0}}" \\
  --output-root "$OUTPUT_ROOT" \\
  --scratch-root "$SCRATCH_ROOT" \\
  --contract-config "{contract_config_path}" \\
  --contract-hash "{contract_hash}" \\
  --expected-range-start 0 \\
  --expected-range-end "{job_count}"
"""


def _render_reducer_script(
    *,
    stage_name: str,
    partition: str,
    cluster: ClusterConfig,
    resources: ResourceEnvelope,
    manifest_path: Path,
    reducer_output_dir: Path,
    contract_config_path: Path,
    contract_hash: str,
    job_count: int,
    parent_hash: str,
    run_id: str,
) -> str:
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

pixi run python -m rfm_pipeline.hpc_campaign_package reduce \\
  --manifest "$MANIFEST_PATH" \\
  --stage "{stage_name}" \\
  --output-root "$REDUCER_OUTPUT_DIR" \\
  --contract-config "{contract_config_path}" \\
  --contract-hash "{contract_hash}" \\
  --expected-range-start 0 \\
  --expected-range-end "{job_count}" \\
  --expected-parent-hash "{parent_hash}"
"""


def _write_package_metadata(dag: CampaignDAG) -> None:
    package_summary = {
        "run_id": dag.run_id,
        "scheduler_submission_permitted": dag.scheduler_submission_permitted,
        "source_hash": dag.source_hash,
        "config_hash": dag.config_hash,
        "lock_hash": dag.lock_hash,
        "readiness_blockers": list(dag.readiness_blockers),
        "campaign_envelope": asdict(dag.campaign_envelope),
        "stages": [
            {
                "name": stage.name,
                "partition": stage.partition,
                "job_count": stage.job_count,
                "manifest_path": str(stage.manifest_path),
                "worker_script_path": str(stage.worker_script_path),
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


def _stage_by_name(dag: CampaignDAG, stage_name: str) -> StagePlan:
    for stage in dag.stages:
        if stage.name == stage_name:
            return stage
    raise ValueError(f"unknown stage {stage_name}")


def _resolved_scratch_template(template: str, run_id: str) -> str:
    return template.replace("{user}", "${USER}").replace("{run_id}", run_id)


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


def _stable_hash(payload: Any) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _cli_worker(args: argparse.Namespace) -> int:
    records = load_stage_manifest(args.manifest)
    task_id = int(args.task_id)
    if task_id < 0 or task_id >= len(records):
        raise ValueError(f"task_id {task_id} out of range for manifest with {len(records)} records")
    record = records[task_id]
    summary = {
        "stage": args.stage,
        "task_id": task_id,
        "shard_id": record["shard_id"],
        "output_root": args.output_root,
        "scratch_root": args.scratch_root,
        "contract_config": args.contract_config,
        "contract_hash": args.contract_hash,
        "expected_range": [args.expected_range_start, args.expected_range_end],
        "mode": "no_submit_validation_only",
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


def _cli_reduce(args: argparse.Namespace) -> int:
    records = load_stage_manifest(args.manifest)
    summary = {
        "stage": args.stage,
        "records": len(records),
        "output_root": args.output_root,
        "contract_config": args.contract_config,
        "contract_hash": args.contract_hash,
        "expected_range": [args.expected_range_start, args.expected_range_end],
        "expected_parent_hash": args.expected_parent_hash,
        "mode": "no_submit_validation_only",
    }
    print(json.dumps(summary, sort_keys=True))
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
    reducer.add_argument("--contract-config", required=True)
    reducer.add_argument("--contract-hash", required=True)
    reducer.add_argument("--expected-range-start", required=True, type=int)
    reducer.add_argument("--expected-range-end", required=True, type=int)
    reducer.add_argument("--expected-parent-hash", required=True)
    reducer.set_defaults(func=_cli_reduce)

    args = parser.parse_args(argv)
    return int(args.func(args))


__all__ = [
    "CampaignDAG",
    "CampaignEnvelope",
    "ClusterConfig",
    "ResourceEnvelope",
    "StagePlan",
    "generate_campaign_package",
    "load_stage_manifest",
    "main",
    "validate_resume_artifacts",
    "validate_stage_dependencies",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
