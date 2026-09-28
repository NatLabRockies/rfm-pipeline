"""Restartable, fail-closed controller for the last G11 Kestrel campaign.

The controller performs only short probes and ``sbatch`` calls on the login
node. Every scientific task and the final artifact compilation run under
Slurm. ``advance`` performs at most one state transition; ``watch`` repeats
that operation at a bounded cadence. No job can be submitted without the
literal ``--execute`` flag.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any, Callable

if __package__ in {
    None,
    "",
}:  # Support the documented ``python scripts/...`` entry point.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_g11_publication_artifacts import _artifact_inventory
from scripts.g11_campaign_workflow import (
    _sha256_path,
    _stable_hash,
    _remaining_confirmatory_estimated_au,
    _sacct_array_command,
    _scheduler_environment,
    _validate_development_completion,
    _write_new_json,
    build_campaign_budget_certificate,
    build_development_admission_guard,
    close_completed_pilot,
    load_packaged_dag,
    preflight_and_authorize_phase,
    prepare_development_package,
    prepare_final_package,
    submit_authorized_phase,
    validate_scheduler_completion,
    validate_phase_budget_guard,
    verify_phase_completion,
)


Runner = Callable[..., subprocess.CompletedProcess[str]]

_REQUIRED_RFM_RUNTIME_APIS = {
    "collect_pilot_accounting",
    "generate_campaign_package",
    "select_pilot_resources",
}


def _validate_rfm_runtime_binding(
    rfm_controller_root: str | Path,
    *,
    module_path: str | Path | None = None,
    available_names: set[str] | None = None,
) -> Path:
    """Require imports to resolve from the pinned RFM controller checkout."""
    expected = (
        Path(rfm_controller_root).resolve()
        / "src"
        / "rfm_pipeline"
        / "hpc_campaign_package.py"
    )
    if module_path is None or available_names is None:
        import rfm_pipeline.hpc_campaign_package as hpc_campaign_package

        module_path = Path(str(hpc_campaign_package.__file__))
        available_names = {
            name
            for name in _REQUIRED_RFM_RUNTIME_APIS
            if hasattr(hpc_campaign_package, name)
        }
    observed = Path(module_path).resolve()
    if observed != expected:
        raise RuntimeError(
            "final controller must run with the pinned RFM controller runtime: "
            f"expected {expected}, imported {observed}"
        )
    missing = sorted(_REQUIRED_RFM_RUNTIME_APIS - set(available_names))
    if missing:
        raise RuntimeError(
            "pinned RFM controller lacks required campaign API(s): "
            + ", ".join(missing)
        )
    return observed


def _activate_rfm_controller(rfm_controller_root: str | Path) -> Path:
    """Prepend controller code in-process without exporting it to Slurm jobs."""
    controller_src = Path(rfm_controller_root).resolve() / "src"
    expected_package = controller_src / "rfm_pipeline"
    loaded = [
        Path(str(module.__file__)).resolve()
        for name, module in sys.modules.items()
        if (name == "rfm_pipeline" or name.startswith("rfm_pipeline."))
        and getattr(module, "__file__", None)
    ]
    if any(expected_package not in path.parents for path in loaded):
        raise RuntimeError(
            "rfm_pipeline was imported before the pinned controller was activated"
        )
    controller_src_text = str(controller_src)
    if controller_src_text not in sys.path:
        sys.path.insert(0, controller_src_text)
    importlib.invalidate_caches()
    return _validate_rfm_runtime_binding(rfm_controller_root)


def _python_tree_sha256(root: str | Path) -> str:
    source_root = Path(root).resolve()
    files = sorted(path for path in source_root.rglob("*.py") if path.is_file())
    if not files:
        raise ValueError(f"Python source tree is empty: {source_root}")
    inventory = {
        str(path.relative_to(source_root)): _sha256_path(path) for path in files
    }
    return _stable_hash(inventory)


def _clean_git_commit(root: str | Path) -> str:
    repository = Path(root).resolve()
    status = subprocess.run(
        ["git", "-C", str(repository), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status.stdout.strip() or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError(f"runtime repository is not clean and pinned: {repository}")
    return commit


def render_final_campaign_config(
    *, base_config_path: str | Path, output_path: str | Path, run_id: str
) -> Path:
    """Copy the reviewed config while replacing exactly one placeholder run ID."""
    if (
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{2,127}", run_id) is None
        or "nosubmit" in run_id.lower()
    ):
        raise ValueError("run_id must be a unique final campaign ID, not a placeholder")
    source = Path(base_config_path).resolve()
    destination = Path(output_path).resolve()
    if destination.exists():
        raise ValueError(f"refusing to overwrite final campaign config: {destination}")
    text = source.read_text(encoding="utf-8")
    matches = list(re.finditer(r"(?m)^(  run_id:\s*)[^\n]+$", text))
    if len(matches) != 1:
        raise ValueError(
            "base campaign config must contain exactly one top-level run_id"
        )
    if not re.search(r"(?m)^  scheduler_submission_permitted:\s*false\s*$", text):
        raise ValueError("base campaign config is not an immutable NO-SUBMIT config")
    rendered = re.sub(r"(?m)^(  run_id:\s*)[^\n]+$", rf"\g<1>{run_id}", text, count=1)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(rendered, encoding="utf-8")
    return destination


def _load_submission(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    identity = {
        key: value
        for key, value in payload.items()
        if key != "submission_record_sha256"
    }
    if payload.get("submission_record_sha256") != _stable_hash(identity):
        raise ValueError("submission record self-hash differs")
    job_ids = payload.get("submission_job_ids")
    if (
        payload.get("status") != "SUBMITTED"
        or not isinstance(job_ids, dict)
        or not job_ids
    ):
        raise ValueError("submission record is not a nonempty SUBMITTED record")
    if any(re.fullmatch(r"[0-9]+", str(value)) is None for value in job_ids.values()):
        raise ValueError("submission record contains a malformed scheduler job ID")
    return payload


def inspect_submitted_phase(
    submission_record_path: str | Path,
    *,
    expected_array_task_counts: dict[str, int] | None = None,
    run_command: Runner = subprocess.run,
) -> dict[str, Any]:
    """Classify a submitted phase without creating premature completion evidence."""
    submission = _load_submission(Path(submission_record_path))
    job_ids = {
        str(key): str(value) for key, value in submission["submission_job_ids"].items()
    }
    joined = ",".join(sorted(job_ids.values(), key=int))
    queued = run_command(
        ["squeue", "-h", "-j", joined, "-o", "%i|%T"],
        check=True,
        capture_output=True,
        text=True,
    )
    if str(queued.stdout).strip():
        return {
            "status": "WAITING_FOR_SCHEDULER",
            "phase": submission.get("phase"),
            "job_count": len(job_ids),
            "active": str(queued.stdout).strip().splitlines(),
        }
    accounting = run_command(
        _sacct_array_command(job_ids, fields=("JobID", "State", "ExitCode")),
        check=True,
        capture_output=True,
        text=True,
    )
    raw = str(accounting.stdout)
    rows: dict[str, tuple[str, str]] = {}
    expected = set(job_ids.values())
    for line in raw.splitlines():
        values = line.split("|")
        if len(values) != 3:
            continue
        row_id = values[0]
        relevant = row_id in expected or any(
            re.fullmatch(rf"{re.escape(parent)}_[0-9]+", row_id) for parent in expected
        )
        if not relevant:
            continue
        if row_id in rows:
            raise RuntimeError(f"sacct duplicated allocation job {row_id}")
        rows[row_id] = (values[1], values[2])
    task_counts = (
        {step_id: 0 for step_id in job_ids}
        if expected_array_task_counts is None
        else expected_array_task_counts
    )
    if set(task_counts) != set(job_ids) or any(
        not isinstance(value, int) or value < 0 for value in task_counts.values()
    ):
        raise ValueError("phase probe array-task metadata differs from submission")
    missing = []
    failures = []
    for step_id, job_id in job_ids.items():
        task_count = task_counts[step_id]
        expected_rows = (
            {f"{job_id}_{index}" for index in range(task_count)}
            if task_count
            else {job_id}
        )
        missing.extend(sorted(expected_rows - set(rows)))
        for row_id in sorted(expected_rows & set(rows)):
            state, exit_code = rows[row_id]
            if state != "COMPLETED" or exit_code != "0:0":
                failures.append(f"{row_id}:{state}/{exit_code}")
        if task_count and job_id in rows:
            state, exit_code = rows[job_id]
            if state != "COMPLETED" or exit_code != "0:0":
                failures.append(f"{job_id}:{state}/{exit_code}")
    if failures:
        raise RuntimeError(
            "terminal scheduler failure; downstream locked: " + ", ".join(failures)
        )
    if missing:
        return {
            "status": "WAITING_FOR_SACCT",
            "phase": submission.get("phase"),
            "job_count": len(job_ids),
            "observed_job_count": len(rows),
            "missing_allocation_rows": missing,
        }
    return {
        "status": "READY_TO_VERIFY",
        "phase": submission.get("phase"),
        "job_count": len(job_ids),
    }


def initialize_campaign(
    *,
    campaign_root: str | Path,
    run_id: str,
    base_config_path: str | Path,
    pilot_package_root: str | Path,
    pilot_submission_record: str | Path,
    pilot_resource_freeze: str | Path,
    rfm_repo_root: str | Path,
    rfm_controller_root: str | Path,
    bsm_runtime_root: str | Path,
    prior_sunk_au: float,
    remaining_au: int,
    allocation_quota_au: float = 25_000.0,
    postprocessing_reserved_au: float = 5.0,
    resolution_cache_evidence_root: str | Path | None = None,
) -> dict[str, Any]:
    """Create one immutable control root; never submit or run scientific work."""
    root = Path(campaign_root).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("campaign control root must be new or empty")
    root.mkdir(parents=True, exist_ok=True)
    config = render_final_campaign_config(
        base_config_path=base_config_path,
        output_path=root / "final_campaign.yml",
        run_id=run_id,
    )
    paths = {
        "pilot_package_root": Path(pilot_package_root).resolve(),
        "pilot_submission_record": Path(pilot_submission_record).resolve(),
        "pilot_resource_freeze": Path(pilot_resource_freeze).resolve(),
        "rfm_repo_root": Path(rfm_repo_root).resolve(),
        "rfm_controller_root": Path(rfm_controller_root).resolve(),
        "bsm_runtime_root": Path(bsm_runtime_root).resolve(),
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise ValueError("campaign prerequisite path is missing: " + ", ".join(missing))
    if prior_sunk_au < 0 or remaining_au <= 0 or allocation_quota_au <= 0:
        raise ValueError("campaign allocation inputs must be positive/nonnegative")
    if postprocessing_reserved_au < 5.0:
        raise ValueError("postprocessing AU reserve must cover the bounded 5-AU job")
    controller_module = (
        paths["rfm_controller_root"]
        / "src"
        / "rfm_pipeline"
        / "hpc_campaign_package.py"
    )
    if not controller_module.is_file():
        raise ValueError(f"RFM controller module is missing: {controller_module}")
    runtime_commits = {
        "rfm_scientific_git_commit": _clean_git_commit(paths["rfm_repo_root"]),
        "rfm_controller_git_commit": _clean_git_commit(paths["rfm_controller_root"]),
        "bsm_runtime_git_commit": _clean_git_commit(paths["bsm_runtime_root"]),
    }
    identity = {
        "schema_version": 1,
        "status": "INITIALIZED_NO_SUBMIT",
        "run_id": run_id,
        "campaign_root": str(root),
        "final_config_path": str(config),
        "final_config_sha256": _sha256_path(config),
        **{name: str(path) for name, path in paths.items()},
        **runtime_commits,
        "rfm_controller_module_sha256": _sha256_path(controller_module),
        "rfm_controller_source_sha256": _python_tree_sha256(
            paths["rfm_controller_root"] / "src" / "rfm_pipeline"
        ),
        "bsm_scripts_source_sha256": _python_tree_sha256(
            paths["bsm_runtime_root"] / "scripts"
        ),
        "prior_sunk_au": float(prior_sunk_au),
        "remaining_au_for_live_preflight": int(remaining_au),
        "allocation_quota_au": float(allocation_quota_au),
        "postprocessing_reserved_au": float(postprocessing_reserved_au),
        "submission_requires_execute_flag": True,
    }
    if resolution_cache_evidence_root is not None:
        identity.update(
            _resolution_cache_evidence_identity(resolution_cache_evidence_root)
        )
    manifest = {**identity, "control_manifest_sha256": _stable_hash(identity)}
    _write_new_json(root / "control_manifest.json", manifest)
    return manifest


def _load_control(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = {
        key: value for key, value in payload.items() if key != "control_manifest_sha256"
    }
    if payload.get("control_manifest_sha256") != _stable_hash(identity):
        raise ValueError("campaign control manifest self-hash differs")
    config = Path(str(payload["final_config_path"]))
    if _sha256_path(config) != payload.get("final_config_sha256"):
        raise ValueError("final campaign config changed after initialization")
    controller_module = (
        Path(str(payload["rfm_controller_root"]))
        / "src"
        / "rfm_pipeline"
        / "hpc_campaign_package.py"
    )
    if _sha256_path(controller_module) != payload.get("rfm_controller_module_sha256"):
        raise ValueError("RFM controller module changed after initialization")
    for field, root_field in (
        ("rfm_scientific_git_commit", "rfm_repo_root"),
        ("rfm_controller_git_commit", "rfm_controller_root"),
        ("bsm_runtime_git_commit", "bsm_runtime_root"),
    ):
        if _clean_git_commit(payload[root_field]) != payload.get(field):
            raise ValueError(f"{field} changed after initialization")
    if _python_tree_sha256(
        Path(str(payload["rfm_controller_root"])) / "src" / "rfm_pipeline"
    ) != payload.get("rfm_controller_source_sha256"):
        raise ValueError("RFM controller source changed after initialization")
    if _python_tree_sha256(
        Path(str(payload["bsm_runtime_root"])) / "scripts"
    ) != payload.get("bsm_scripts_source_sha256"):
        raise ValueError("BSM controller source changed after initialization")
    if "resolution_cache_evidence_root" in payload:
        observed_cache = _resolution_cache_evidence_identity(
            payload["resolution_cache_evidence_root"]
        )
        for field, value in observed_cache.items():
            if payload.get(field) != value:
                raise ValueError(
                    "resolution cache evidence changed after initialization"
                )
    return payload


def _resolution_cache_evidence_identity(
    evidence_root: str | Path,
) -> dict[str, Any]:
    """Bind a preserved failure bundle without repeatedly hashing every artifact."""
    root = Path(evidence_root).resolve()
    sums = root / "SHA256SUMS"
    sums_checksum = root / "SHA256SUMS.sha256"
    if not root.is_dir() or not sums.is_file() or not sums_checksum.is_file():
        raise ValueError("resolution cache evidence lacks SHA256SUMS identity files")
    lines = [line for line in sums.read_text(encoding="utf-8").splitlines() if line]
    expected = _sha256_path(sums)
    checksum_fields = sums_checksum.read_text(encoding="utf-8").strip().split()
    if checksum_fields != [expected, "SHA256SUMS"]:
        raise ValueError("resolution cache evidence SHA256SUMS checksum differs")
    if not lines:
        raise ValueError("resolution cache evidence SHA256SUMS is empty")
    return {
        "resolution_cache_evidence_root": str(root),
        "resolution_cache_sha256s_sha256": expected,
        "resolution_cache_file_count": len(lines),
    }


def _load_resolution_cache_inventory(evidence_root: Path) -> dict[str, str]:
    """Parse the preserved relative-path inventory and reject ambiguity."""
    inventory: dict[str, str] = {}
    for line_number, line in enumerate(
        (evidence_root / "SHA256SUMS").read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line:
            continue
        fields = line.split(maxsplit=1)
        if (
            len(fields) != 2
            or re.fullmatch(r"[0-9a-f]{64}", fields[0]) is None
            or not fields[1].startswith("./")
        ):
            raise ValueError(
                f"resolution cache inventory line {line_number} is malformed"
            )
        relative = fields[1][2:]
        if relative in inventory:
            raise ValueError("resolution cache inventory contains a duplicate path")
        inventory[relative] = fields[0]
    return inventory


def _load_scientific_interaction_runtime(rfm_repo_root: str | Path) -> Any:
    """Load snapshot verification from the immutable scientific RFM checkout."""
    root = Path(rfm_repo_root).resolve()
    module_path = root / "src" / "rfm_pipeline" / "interaction_contract.py"
    lock_path = root / "pixi.lock"
    if not module_path.is_file() or not lock_path.is_file():
        raise ValueError(
            "scientific RFM runtime lacks interaction-contract or lock bytes"
        )
    module_name = "_g11_scientific_interaction_contract_" + _stable_hash(
        {
            "root": str(root),
            "module_sha256": _sha256_path(module_path),
            "lock_sha256": _sha256_path(lock_path),
        }
    )
    loaded = sys.modules.get(module_name)
    if loaded is not None:
        return loaded
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load scientific RFM interaction runtime")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def _verified_cache_file(
    evidence_root: Path, inventory: dict[str, str], relative_path: str
) -> Path:
    """Return one regular evidence file only after its preserved hash verifies."""
    if relative_path not in inventory:
        raise ValueError(f"resolution cache evidence omits {relative_path}")
    path = (evidence_root / relative_path).resolve()
    try:
        path.relative_to(evidence_root)
    except ValueError as exc:
        raise ValueError(
            "resolution cache inventory path escapes its evidence root"
        ) from exc
    if (
        not path.is_file()
        or path.is_symlink()
        or _sha256_path(path) != inventory[relative_path]
    ):
        raise ValueError(f"resolution cache evidence file differs: {relative_path}")
    return path


_RESOLUTION_CACHE_SCIENCE_FIELDS = (
    "stage",
    "operation",
    "fixture_kind",
    "schedule_index",
    "seed",
    "base_draws",
    "nested_schedule_draws",
    "family_size",
    "source_hash",
    "config_hash",
    "lock_hash",
    "family_order_file_sha256",
    "bsm_recovery_driver_sha256",
    "bsm_dgp_contract_sha256",
    "worker_resources",
)


def _promote_resolution_cache(
    *, control: dict[str, Any], development: Any
) -> dict[str, Any]:
    """Promote verified completed score blocks into a fresh authorized run.

    This performs no model fitting or permutation scoring. It reconstructs only
    terminal JSON and standard atomic success wrappers from self-verifying score
    artifacts whose complete scientific identity matches the fresh manifest.
    """
    from rfm_pipeline.campaign_contract import load_contract
    from rfm_pipeline.hpc_campaign_package import (
        load_stage_manifest,
        validate_resume_artifacts,
    )

    root = Path(str(control["campaign_root"]))
    promotion_path = root / "development" / "cache_promotion.json"
    resolution_stages = [
        stage for stage in development.stages if stage.name == "resolution"
    ]
    if len(resolution_stages) != 1:
        raise ValueError("development package does not uniquely contain resolution")
    stage = resolution_stages[0]
    if promotion_path.is_file():
        payload = json.loads(promotion_path.read_text(encoding="utf-8"))
        identity = {
            key: value for key, value in payload.items() if key != "promotion_sha256"
        }
        if payload.get("promotion_sha256") != _stable_hash(identity):
            raise ValueError("resolution cache promotion record self-hash differs")
        if validate_resume_artifacts(stage) != {
            "completed": stage.job_count,
            "failed": 0,
            "pending": 0,
        }:
            raise ValueError("resolution cache promotion no longer has exact coverage")
        return payload

    authorization = root / "development" / "authorization.json"
    submission = root / "development" / "submission.json"
    if not authorization.is_file() or submission.exists():
        raise ValueError(
            "resolution cache promotion requires authorization before submission"
        )
    target_records = load_stage_manifest(stage.manifest_path)
    if any(Path(record["output_dir"]).exists() for record in target_records):
        raise ValueError("resolution cache target output already exists")

    evidence_root = Path(str(control["resolution_cache_evidence_root"])).resolve()
    inventory = _load_resolution_cache_inventory(evidence_root)
    status_path = _verified_cache_file(evidence_root, inventory, "failure_status.json")
    diagnosis_path = _verified_cache_file(
        evidence_root, inventory, "failure_diagnosis.json"
    )
    accounting_path = _verified_cache_file(
        evidence_root, inventory, "failure_accounting.json"
    )
    failure_status = json.loads(status_path.read_text(encoding="utf-8"))
    diagnosis = json.loads(diagnosis_path.read_text(encoding="utf-8"))
    accounting = json.loads(accounting_path.read_text(encoding="utf-8"))
    if (
        failure_status.get("status") != "STOPPED_AFTER_SCIENTIFIC_FAILURE"
        or diagnosis.get("status") != "CONFIRMED_ROOT_CAUSE"
        or diagnosis.get("worker_failures_with_identical_signature") != 20
        or diagnosis.get("canonical_property") != "payload_sha256"
        or accounting.get("status") != "REJECTED_SCIENTIFIC_ADAPTER_FAILURE_ACCOUNTED"
        or accounting.get("telemetry_eligibility")
        != "DIAGNOSTIC_ONLY_NEVER_REUSE_FOR_RESOURCE_SELECTION"
    ):
        raise ValueError(
            "resolution cache evidence is not the accepted terminal failure bundle"
        )

    source_manifest_relative = (
        "control_root/development_package/stages/resolution/manifest.jsonl"
    )
    source_manifest = _verified_cache_file(
        evidence_root, inventory, source_manifest_relative
    )
    source_records = load_stage_manifest(source_manifest)
    if len(source_records) != len(target_records) or len(target_records) != 20:
        raise ValueError("resolution cache manifest coverage differs from the target")
    source_by_shard = {str(record["shard_id"]): record for record in source_records}
    target_by_shard = {str(record["shard_id"]): record for record in target_records}
    if set(source_by_shard) != set(target_by_shard):
        raise ValueError("resolution cache shard identities differ from the target")

    source_adapter = _verified_cache_file(
        evidence_root, inventory, "runtime_source/g11_campaign_adapter.py"
    )
    old_adapter_hashes = {
        str(record["scientific_adapter_sha256"]) for record in source_records
    }
    if old_adapter_hashes != {_sha256_path(source_adapter)}:
        raise ValueError("resolution cache source adapter differs from its manifest")
    target_adapter_paths = {
        Path(str(record["scientific_adapter_path"])) for record in target_records
    }
    target_adapter_hashes = {
        str(record["scientific_adapter_sha256"]) for record in target_records
    }
    if len(target_adapter_paths) != 1 or len(target_adapter_hashes) != 1:
        raise ValueError("target resolution manifest mixes adapter identities")
    target_adapter_path = target_adapter_paths.pop()
    target_adapter_hash = target_adapter_hashes.pop()
    if _sha256_path(target_adapter_path) != target_adapter_hash:
        raise ValueError("target resolution adapter differs from its manifest")
    adapter_spec = importlib.util.spec_from_file_location(
        f"_g11_resolution_cache_adapter_{target_adapter_hash}", target_adapter_path
    )
    if adapter_spec is None or adapter_spec.loader is None:
        raise RuntimeError("could not load target resolution adapter")
    adapter = importlib.util.module_from_spec(adapter_spec)
    adapter_spec.loader.exec_module(adapter)

    contract, contract_hash = load_contract(development.contract_config_path)
    if contract_hash != development.config_hash:
        raise ValueError("target resolution contract bytes differ")
    scientific_interaction = _load_scientific_interaction_runtime(
        control["rfm_repo_root"]
    )
    pending_root = stage.output_root.parent / ".resolution-cache-promotion.pending"
    if pending_root.exists():
        raise ValueError("resolution cache promotion has a stale pending directory")
    pending_root.mkdir(parents=True)
    promoted: list[dict[str, Any]] = []
    try:
        for shard_id in sorted(target_by_shard):
            source_record = source_by_shard[shard_id]
            target_record = target_by_shard[shard_id]
            differing = [
                field
                for field in _RESOLUTION_CACHE_SCIENCE_FIELDS
                if source_record.get(field) != target_record.get(field)
            ]
            if differing:
                raise ValueError(
                    f"resolution cache scientific identity differs for {shard_id}: {differing}"
                )
            relative_source = (
                f"scratch_run/stages/resolution/results/{shard_id}/"
                "interaction_score_blocks"
            )
            source_scores = _verified_cache_file(
                evidence_root,
                inventory,
                f"{relative_source}/score_only_interaction.npz",
            )
            source_metadata = _verified_cache_file(
                evidence_root,
                inventory,
                f"{relative_source}/score_only_interaction.json",
            )
            artifact = scientific_interaction.ScoreOnlyInteractionArtifact.read_from(
                source_scores.parent
            )
            worker_resources = target_record.get("worker_resources")
            if not isinstance(worker_resources, dict):
                raise ValueError(
                    f"target resolution worker resources are invalid for {shard_id}"
                )
            requested_cpu_cores = int(worker_resources.get("requested_cpu_cores", 0))
            if requested_cpu_cores <= 0:
                raise ValueError(
                    f"target resolution CPU request is invalid for {shard_id}"
                )
            interaction_spec = adapter._interaction_spec(
                contract,
                draws=int(target_record["nested_schedule_draws"]),
                seed=int(target_record["seed"]),
                n_jobs=requested_cpu_cores,
            )
            scientific_interaction.verify_control_snapshot(
                artifact.control_snapshot,
                scientific_interaction.canonical_execution_contract_from_specs(
                    interaction_spec
                ),
            )
            terminal = adapter._resolution_terminal_from_artifact(
                record=target_record,
                artifact=artifact,
                campaign_contract=contract,
            )

            pending_shard = pending_root / shard_id
            pending_scores = pending_shard / "interaction_score_blocks"
            pending_scores.mkdir(parents=True)
            shutil.copy2(source_scores, pending_scores / source_scores.name)
            shutil.copy2(source_metadata, pending_scores / source_metadata.name)
            copied = scientific_interaction.ScoreOnlyInteractionArtifact.read_from(
                pending_scores
            )
            if copied.payload_sha256 != artifact.payload_sha256:
                raise ValueError("copied resolution cache payload differs")
            target_shard = Path(str(target_record["output_dir"]))
            cache_identity = {
                "schema_version": 1,
                "status": "VERIFIED_SCIENTIFIC_CACHE_PROMOTION",
                "source_evidence_sha256s_sha256": control[
                    "resolution_cache_sha256s_sha256"
                ],
                "source_manifest_sha256": _sha256_path(source_manifest),
                "source_adapter_sha256": _sha256_path(source_adapter),
                "target_adapter_sha256": target_adapter_hash,
                "source_shard_id": shard_id,
                "target_shard_id": shard_id,
                "target_output_hash": target_record["output_hash"],
                "score_payload_sha256": artifact.payload_sha256,
                "source_npz_sha256": _sha256_path(source_scores),
                "source_metadata_sha256": _sha256_path(source_metadata),
                "scientific_compute_reused": True,
                "failed_scheduler_telemetry_reused": False,
            }
            cache_provenance = {
                **cache_identity,
                "cache_provenance_sha256": _stable_hash(cache_identity),
            }
            (pending_shard / "cache_provenance.json").write_text(
                json.dumps(cache_provenance, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (pending_shard / "terminal_record.json").write_text(
                json.dumps(terminal, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            artifact_rows = []
            for artifact_path in sorted(
                path for path in pending_shard.rglob("*") if path.is_file()
            ):
                relative = artifact_path.relative_to(pending_shard)
                artifact_rows.append(
                    {
                        "path": str(target_shard / relative),
                        "relative_path": relative.as_posix(),
                        "sha256": _sha256_path(artifact_path),
                        "bytes": artifact_path.stat().st_size,
                    }
                )
            result = {
                **terminal,
                "terminal_record": str(target_shard / "terminal_record.json"),
                "scientific_artifacts": artifact_rows,
                "schema_version": 2,
                "stage": target_record["stage"],
                "shard_id": shard_id,
                "input_hash": target_record["input_hash"],
                "schedule_hash": target_record["schedule_hash"],
                "elapsed_seconds": 0.0,
                "total_cpu_seconds": 0.0,
                "cpu_time_seconds": 0.0,
                "max_rss_bytes": 0,
                "bytes_read": 0,
                "bytes_written": 0,
                "task_size": int(target_record.get("task_size", 1)),
                "block_size": int(target_record.get("block_size", 1)),
                "executed_work_units": 0,
                "profile_id": "verified-cache-promotion",
                "partition": str(target_record.get("partition", "")),
                "requested_cpus": int(
                    target_record["worker_resources"]["requested_cpu_cores"]
                ),
                "requested_memory_gb": int(
                    target_record["worker_resources"]["requested_memory_gb"]
                ),
                "requested_walltime_seconds": int(
                    target_record["worker_resources"]["requested_walltime_seconds"]
                ),
                "status": "completed",
                "attempt": int(target_record["attempt"]) + 1,
                "hostname": "verified-cache-promotion",
                "slurm_job_id": None,
                "slurm_array_job_id": None,
                "slurm_array_task_id": None,
                "source_hash": target_record["source_hash"],
                "config_hash": target_record["config_hash"],
                "lock_hash": target_record["lock_hash"],
                "parent_hash": target_record["parent_hash"],
                "output_hash": target_record["output_hash"],
                "cache_provenance": str(target_shard / "cache_provenance.json"),
            }
            result_path = pending_shard / "result.json"
            result_path.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            marker = {
                "schema_version": 2,
                "stage": target_record["stage"],
                "shard_id": shard_id,
                "output_hash": target_record["output_hash"],
                "artifact_sha256": _sha256_path(result_path),
                "parent_hash": target_record["parent_hash"],
                "attempt": int(target_record["attempt"]) + 1,
                "status": "completed",
            }
            (pending_shard / "_SUCCESS.json").write_text(
                json.dumps(marker, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            promoted.append(
                {
                    "shard_id": shard_id,
                    "score_payload_sha256": artifact.payload_sha256,
                    "result_sha256": marker["artifact_sha256"],
                    "target_output_hash": target_record["output_hash"],
                }
            )
        stage.output_root.mkdir(parents=True, exist_ok=True)
        for shard_id in sorted(target_by_shard):
            (pending_root / shard_id).replace(
                Path(str(target_by_shard[shard_id]["output_dir"]))
            )
    finally:
        if pending_root.exists():
            shutil.rmtree(pending_root)

    summary = validate_resume_artifacts(stage)
    if summary != {"completed": 20, "failed": 0, "pending": 0}:
        raise ValueError("resolution cache promotion lacks exact completed coverage")
    promotion_identity = {
        "schema_version": 1,
        "status": "RESOLUTION_CACHE_PROMOTED",
        "run_id": development.run_id,
        "authorization_sha256": _sha256_path(authorization),
        "target_manifest_sha256": _sha256_path(stage.manifest_path),
        "source_evidence_sha256s_sha256": control["resolution_cache_sha256s_sha256"],
        "source_failure_accounting_sha256": _sha256_path(accounting_path),
        "scientific_compute_reused": True,
        "failed_scheduler_telemetry_reused": False,
        "coverage": summary,
        "promoted_shards": promoted,
    }
    payload = {
        **promotion_identity,
        "promotion_sha256": _stable_hash(promotion_identity),
    }
    _write_new_json(promotion_path, payload)
    return payload


def _phase_paths(root: Path, phase: str) -> dict[str, Path]:
    phase_root = root / phase
    return {
        "root": phase_root,
        "preflight_dir": phase_root / "preflight",
        "preflight": phase_root / "preflight" / "preflight.json",
        "authorization": phase_root / "authorization.json",
        "submission": phase_root / "submission.json",
        "completion_evidence": phase_root / "completion_evidence",
        "completion": phase_root / "completion.json",
        "budget_guard": phase_root / "budget_guard.json",
    }


def _stage_artifact(dag: Any, stage_name: str, filename: str) -> Path:
    matches = [stage for stage in dag.stages if stage.name == stage_name]
    if len(matches) != 1:
        raise ValueError(
            f"campaign package does not uniquely contain stage {stage_name}"
        )
    return matches[0].reducer_output_dir / filename


def _results_root(dag: Any) -> Path:
    roots = {stage.reducer_output_dir.parents[2] for stage in dag.stages}
    if len(roots) != 1:
        raise ValueError("campaign stages do not share one results root")
    return roots.pop()


def _validate_budget_certificate(
    *,
    dag: Any,
    certificate_path: Path,
    resource_freeze_path: Path,
    control: dict[str, Any],
) -> dict[str, Any]:
    observed = json.loads(certificate_path.read_text(encoding="utf-8"))
    freeze = json.loads(resource_freeze_path.read_text(encoding="utf-8"))
    amendment = json.loads(
        (dag.output_dir / "contract" / "fixed_family_amendment.json").read_text(
            encoding="utf-8"
        )
    )
    development_completion = json.loads(
        (
            Path(str(control["campaign_root"])) / "development" / "completion.json"
        ).read_text(encoding="utf-8")
    )
    remaining_estimated_au = _remaining_confirmatory_estimated_au(
        dag.campaign_envelope.stage_allocations
    )
    remaining_requested_au = math.ceil(remaining_estimated_au * 1.20)
    accepted_pilot_observed_au = float(
        freeze["allocation_accounting"]["accepted_pilot_observed_au"]
    )
    development_observed_au = _validate_development_completion(
        development_completion,
        expected_run_id=dag.run_id,
        resource_freeze=freeze,
    )
    expected = build_campaign_budget_certificate(
        run_id=dag.run_id,
        campaign_requested_au=(
            accepted_pilot_observed_au
            + development_observed_au
            + remaining_requested_au
        ),
        prior_rejected_attempt_au=float(
            freeze["allocation_accounting"]["prior_rejected_attempt_au"]
        ),
        postprocessing_reserved_au=float(control["postprocessing_reserved_au"]),
        allocation_quota_au=float(control["allocation_quota_au"]),
        resource_freeze_sha256=str(freeze["resource_freeze_sha256"]),
        campaign_inventory_hash=dag.campaign_inventory_hash,
        contract_amendment_sha256=str(amendment["contract_amendment_sha256"]),
        accepted_pilot_observed_au=accepted_pilot_observed_au,
        development_resolution_observed_au=development_observed_au,
        remaining_confirmatory_estimated_au=remaining_estimated_au,
        remaining_confirmatory_requested_au_with_shared_reserve=remaining_requested_au,
    )
    if observed != expected:
        raise ValueError(
            "budget certificate is missing, stale, or differs from the final package"
        )
    return observed


def _advance_phase(
    *,
    dag: Any,
    phase: str,
    root: Path,
    config: Path,
    repo: Path,
    freeze: Path,
    prerequisites: tuple[Path, ...],
    remaining_au: int,
    execute: bool,
) -> dict[str, Any]:
    paths = _phase_paths(root, phase)
    if not paths["authorization"].is_file():
        if not execute:
            return {
                "status": "EXECUTION_FLAG_REQUIRED",
                "phase": phase,
                "next_action": "same-day live preflight followed by submission",
            }
        preflight_and_authorize_phase(
            dag=dag,
            phase=phase,
            remaining_au=remaining_au,
            evidence_dir=paths["preflight_dir"],
            resource_freeze_path=freeze,
            prerequisite_paths=prerequisites,
            authorization_path=paths["authorization"],
        )
        return {"status": "PHASE_PREFLIGHTED", "phase": phase}
    if not paths["submission"].is_file():
        if not execute:
            return {"status": "EXECUTION_FLAG_REQUIRED", "phase": phase}
        submit_authorized_phase(
            dag=dag,
            phase=phase,
            preflight_path=paths["preflight"],
            authorization_path=paths["authorization"],
            submission_record_path=paths["submission"],
        )
        return {"status": "PHASE_SUBMITTED", "phase": phase}
    if not paths["completion"].is_file():
        from rfm_pipeline.hpc_campaign_package import build_campaign_phase_plan

        plan = build_campaign_phase_plan(dag, phase=phase)
        probe = inspect_submitted_phase(
            paths["submission"],
            expected_array_task_counts={
                str(step["step_id"]): int(step["array_task_count"])
                for step in plan["steps"]
            },
        )
        if probe["status"] != "READY_TO_VERIFY":
            return probe
        verify_phase_completion(
            dag=dag,
            phase=phase,
            submission_record_path=paths["submission"],
            evidence_dir=paths["completion_evidence"],
            completion_path=paths["completion"],
        )
        return {"status": "PHASE_VERIFIED", "phase": phase}
    _validate_phase_completion_record(
        dag=dag,
        phase=phase,
        completion_path=paths["completion"],
        submission_path=paths["submission"],
    )
    return {"status": "PHASE_COMPLETE", "phase": phase}


def _validate_phase_completion_record(
    *, dag: Any, phase: str, completion_path: Path, submission_path: Path
) -> dict[str, Any]:
    from rfm_pipeline.hpc_campaign_package import build_campaign_phase_plan

    payload = json.loads(completion_path.read_text(encoding="utf-8"))
    identity = {
        key: value for key, value in payload.items() if key != "completion_sha256"
    }
    if payload.get("completion_sha256") != _stable_hash(identity):
        raise ValueError(f"{phase} completion record self-hash differs")
    if float(payload.get("observed_total_au", -1.0)) < 0:
        raise ValueError(f"{phase} completion record lacks valid AU accounting")
    submission = _load_submission(submission_path)
    for field, expected in (
        ("status", "PHASE_COMPLETE"),
        ("phase", phase),
        ("run_id", dag.run_id),
        ("source_hash", dag.source_hash),
        ("config_hash", dag.config_hash),
        ("lock_hash", dag.lock_hash),
        ("campaign_inventory_hash", dag.campaign_inventory_hash),
        ("submission_record_sha256", submission["submission_record_sha256"]),
    ):
        if payload.get(field) != expected:
            raise ValueError(f"{phase} completion record {field} differs")
    stages = {stage.name: stage for stage in dag.stages}
    reducers = payload.get("reducers")
    plan = build_campaign_phase_plan(dag, phase=phase)
    expected_stages = {str(step["stage"]) for step in plan["steps"]}
    if not isinstance(reducers, dict) or set(reducers) != expected_stages:
        raise ValueError(f"{phase} completion lacks exact reducer coverage")
    for stage_name, recorded in reducers.items():
        stage = stages.get(stage_name)
        if stage is None:
            raise ValueError(f"{phase} completion names an unknown reducer stage")
        result = stage.reducer_output_dir / "reduced_result.json"
        if recorded.get(
            "reducer_output_hash"
        ) != stage.reducer_output_hash or recorded.get(
            "reduced_result_sha256"
        ) != _sha256_path(result):
            raise ValueError(f"{phase} reducer bytes changed after completion")
    return payload


def _confirmatory_phase_budget_guard(
    *,
    dag: Any,
    phase: str,
    root: Path,
    resource_freeze_path: Path,
    control: dict[str, Any],
) -> dict[str, Any]:
    """Reconcile completed actual AUs before authorizing another phase."""
    from rfm_pipeline.hpc_campaign_package import build_campaign_phase_plan

    freeze = json.loads(resource_freeze_path.read_text(encoding="utf-8"))
    development_completion = json.loads(
        _phase_paths(root, "development")["completion"].read_text(encoding="utf-8")
    )
    development_observed_au = _validate_development_completion(
        development_completion,
        expected_run_id=dag.run_id,
        resource_freeze=freeze,
    )
    completed_stage_names: set[str] = set()
    completed_confirmatory_observed_au = 0.0
    for prior_phase in ("gate_b", "gate_p", "gate_c"):
        paths = _phase_paths(root, prior_phase)
        if not paths["completion"].is_file():
            continue
        completion = _validate_phase_completion_record(
            dag=dag,
            phase=prior_phase,
            completion_path=paths["completion"],
            submission_path=paths["submission"],
        )
        completed_confirmatory_observed_au += float(completion["observed_total_au"])
        plan = build_campaign_phase_plan(dag, phase=prior_phase)
        completed_stage_names.update(str(step["stage"]) for step in plan["steps"])
    current_plan = build_campaign_phase_plan(dag, phase=phase)
    current_stage_names = {str(step["stage"]) for step in current_plan["steps"]}
    return validate_phase_budget_guard(
        stage_allocations=dag.campaign_envelope.stage_allocations,
        completed_stage_names=completed_stage_names,
        current_stage_names=current_stage_names,
        prior_rejected_attempt_au=float(
            freeze["allocation_accounting"]["prior_rejected_attempt_au"]
        ),
        accepted_pilot_observed_au=float(
            freeze["allocation_accounting"]["accepted_pilot_observed_au"]
        ),
        development_resolution_observed_au=development_observed_au,
        completed_confirmatory_observed_au=completed_confirmatory_observed_au,
        postprocessing_reserved_au=float(control["postprocessing_reserved_au"]),
        allocation_quota_au=float(control["allocation_quota_au"]),
    )


def _development_phase_budget_guard(
    *, dag: Any, resource_freeze_path: Path, control: dict[str, Any]
) -> dict[str, Any]:
    """Require the complete resolution request to preserve the B=999 branch."""
    freeze = json.loads(resource_freeze_path.read_text(encoding="utf-8"))
    return build_development_admission_guard(
        stage_allocations=dag.campaign_envelope.stage_allocations,
        resource_freeze=freeze,
        postprocessing_reserved_au=float(control["postprocessing_reserved_au"]),
        allocation_quota_au=float(control["allocation_quota_au"]),
    )


def _prepare_artifact_job(
    *, root: Path, dag: Any, bsm_runtime_root: Path
) -> dict[str, Any]:
    job_root = root / "publication_job"
    script = job_root / "build_publication.sbatch"
    inputs_path = job_root / "artifact_job_inputs.json"
    if script.exists() or inputs_path.exists():
        raise ValueError("publication job preparation is partial or already exists")
    job_root.mkdir(parents=True, exist_ok=True)
    results = _results_root(dag)
    output = results / "publication_artifacts"
    if output.exists() and any(output.iterdir()):
        raise ValueError("publication artifact output already contains files")
    python = Path(dag.repo_root) / ".pixi" / "envs" / "default" / "bin" / "python"
    builder = bsm_runtime_root / "scripts" / "build_g11_publication_artifacts.py"
    if not python.is_file() or not builder.is_file():
        raise ValueError(
            "immutable runtimes lack the publication compiler or scientific Python"
        )
    command = " ".join(
        shlex.quote(str(value))
        for value in (
            python,
            builder,
            "--package-root",
            dag.output_dir,
            "--results-root",
            results,
            "--output-root",
            output,
        )
    )
    lines = [
        "#!/bin/bash",
        "#SBATCH --account=nationalpfa",
        "#SBATCH --partition=shared",
        "#SBATCH --time=00:30:00",
        "#SBATCH --cpus-per-task=1",
        "#SBATCH --mem=8G",
        f"#SBATCH --job-name={dag.run_id}-publication",
        f"#SBATCH --output={job_root / 'slurm-%j.out'}",
        f"#SBATCH --error={job_root / 'slurm-%j.err'}",
        "set -euo pipefail",
        "unset PYTHONPATH PYTHONHOME",
        "export PYTHONNOUSERSITE=1",
        command,
    ]
    script.write_text("\n".join(lines) + "\n", encoding="utf-8")
    identity = {
        "schema_version": 1,
        "status": "NO_SUBMIT",
        "run_id": dag.run_id,
        "package_root": str(dag.output_dir),
        "campaign_inventory_hash": dag.campaign_inventory_hash,
        "results_root": str(results),
        "publication_root": str(output),
        "scientific_python_path": str(python),
        "scientific_python_sha256": _sha256_path(python),
        "builder_path": str(builder),
        "builder_sha256": _sha256_path(builder),
        "script_path": str(script),
        "script_sha256": _sha256_path(script),
        "resource_bound": {
            "partition": "shared",
            "cpus": 1,
            "memory_gb": 8,
            "minutes": 30,
        },
        "maximum_reserved_au": 5.0,
    }
    payload = {**identity, "artifact_job_inputs_sha256": _stable_hash(identity)}
    _write_new_json(inputs_path, payload)
    return payload


def _load_artifact_inputs(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    identity = {
        key: value
        for key, value in payload.items()
        if key != "artifact_job_inputs_sha256"
    }
    if payload.get("artifact_job_inputs_sha256") != _stable_hash(identity):
        raise ValueError("publication job inputs self-hash differs")
    for field in ("scientific_python_path", "builder_path", "script_path"):
        source = Path(str(payload[field]))
        hash_field = field.replace("_path", "_sha256")
        if not source.is_file() or _sha256_path(source) != payload.get(hash_field):
            raise ValueError(f"publication job {field} bytes changed after preparation")
    return payload


def _load_artifact_submission_preflight(
    path: Path, *, inputs: dict[str, Any]
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    identity = {
        key: value
        for key, value in payload.items()
        if key != "submission_preflight_sha256"
    }
    if (
        payload.get("submission_preflight_sha256") != _stable_hash(identity)
        or payload.get("status") != "SBATCH_TEST_ONLY_PASSED"
        or payload.get("script_sha256") != inputs.get("script_sha256")
        or int(payload.get("returncode", -1)) != 0
    ):
        raise ValueError("publication submission preflight is stale or invalid")
    return payload


def _submit_artifact_job(root: Path) -> dict[str, Any]:
    job_root = root / "publication_job"
    inputs = _load_artifact_inputs(job_root / "artifact_job_inputs.json")
    script = Path(str(inputs["script_path"]))
    if _sha256_path(script) != inputs.get("script_sha256"):
        raise ValueError("publication job script changed after preparation")
    preflight_path = job_root / "submission_preflight.json"
    journal_path = job_root / "submission_journal.jsonl"
    if preflight_path.exists() or journal_path.exists():
        raise ValueError(
            "publication preflight or submission journal already exists; refusing "
            "to risk stale evidence or a duplicate job"
        )
    preflight_command = ["sbatch", "--test-only", str(script)]
    preflight_result = subprocess.run(
        preflight_command,
        check=False,
        capture_output=True,
        text=True,
        env=_scheduler_environment(),
    )
    preflight_identity = {
        "schema_version": 1,
        "status": (
            "SBATCH_TEST_ONLY_PASSED"
            if preflight_result.returncode == 0
            else "SBATCH_TEST_ONLY_FAILED"
        ),
        "observed_date": date.today().isoformat(),
        "script_sha256": inputs["script_sha256"],
        "command": preflight_command,
        "returncode": int(preflight_result.returncode),
        "stdout": str(preflight_result.stdout),
        "stderr": str(preflight_result.stderr),
    }
    preflight = {
        **preflight_identity,
        "submission_preflight_sha256": _stable_hash(preflight_identity),
    }
    _write_new_json(preflight_path, preflight)
    if preflight_result.returncode != 0:
        raise subprocess.CalledProcessError(
            preflight_result.returncode,
            preflight_command,
            output=preflight_result.stdout,
            stderr=preflight_result.stderr,
        )
    command = ["sbatch", "--parsable", str(script)]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=_scheduler_environment(),
    )
    job_id = str(completed.stdout).strip().split(";", maxsplit=1)[0]
    entry = {
        "sequence": 0,
        "step_id": "publication-artifacts",
        "command": command,
        "returncode": int(completed.returncode),
        "job_id": job_id if re.fullmatch(r"[0-9]+", job_id) else None,
        "stdout": str(completed.stdout),
        "stderr": str(completed.stderr),
    }
    with journal_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode,
            command,
            output=completed.stdout,
            stderr=completed.stderr,
        )
    if re.fullmatch(r"[0-9]+", job_id) is None:
        raise RuntimeError("scheduler returned no valid publication job ID")
    identity = {
        "schema_version": 1,
        "status": "SUBMITTED",
        "phase": "publication_artifacts",
        "run_id": inputs["run_id"],
        "artifact_job_inputs_sha256": inputs["artifact_job_inputs_sha256"],
        "submission_preflight_sha256": preflight["submission_preflight_sha256"],
        "submission_journal_sha256": _sha256_path(journal_path),
        "submission_job_ids": {"publication-artifacts": job_id},
    }
    payload = {**identity, "submission_record_sha256": _stable_hash(identity)}
    _write_new_json(job_root / "submission.json", payload)
    return payload


def _verify_publication_job(root: Path) -> dict[str, Any]:
    job_root = root / "publication_job"
    inputs = _load_artifact_inputs(job_root / "artifact_job_inputs.json")
    preflight = _load_artifact_submission_preflight(
        job_root / "submission_preflight.json", inputs=inputs
    )
    submission_path = job_root / "submission.json"
    submission = _load_submission(submission_path)
    if submission.get("submission_preflight_sha256") != preflight.get(
        "submission_preflight_sha256"
    ):
        raise ValueError("publication submission does not bind its preflight")
    job_ids = {
        str(key): str(value) for key, value in submission["submission_job_ids"].items()
    }
    completed = subprocess.run(
        _sacct_array_command(job_ids, fields=("JobID", "State", "ExitCode")),
        check=True,
        capture_output=True,
        text=True,
    )
    raw_path = job_root / "sacct_raw.psv"
    if raw_path.exists():
        raise ValueError("refusing to overwrite publication-job scheduler evidence")
    raw_path.write_text(str(completed.stdout), encoding="utf-8")
    validate_scheduler_completion(job_ids, str(completed.stdout))
    publication = Path(str(inputs["publication_root"]))
    manifest_path = publication / "publication_artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "PUBLICATION_ARTIFACTS_COMPLETE" or manifest.get(
        "run_id"
    ) != inputs.get("run_id"):
        raise ValueError("publication artifact manifest is incomplete or stale")
    inventory = _artifact_inventory(publication)
    if inventory != manifest.get("artifacts"):
        raise ValueError("publication artifact inventory differs from current files")
    if _stable_hash(inventory) != manifest.get("artifact_set_sha256"):
        raise ValueError("publication artifact-set hash differs")
    identity = {
        "schema_version": 1,
        "status": "HPC_AND_PUBLICATION_BUNDLE_COMPLETE",
        "run_id": inputs["run_id"],
        "artifact_job_inputs_sha256": inputs["artifact_job_inputs_sha256"],
        "submission_record_sha256": submission["submission_record_sha256"],
        "sacct_raw_sha256": _sha256_path(raw_path),
        "publication_manifest_sha256": _sha256_path(manifest_path),
        "publication_root": str(publication),
    }
    payload = {**identity, "completion_sha256": _stable_hash(identity)}
    _write_new_json(job_root / "completion.json", payload)
    return payload


def _validate_publication_completion(root: Path) -> dict[str, Any]:
    job_root = root / "publication_job"
    inputs = _load_artifact_inputs(job_root / "artifact_job_inputs.json")
    preflight = _load_artifact_submission_preflight(
        job_root / "submission_preflight.json", inputs=inputs
    )
    submission = _load_submission(job_root / "submission.json")
    if submission.get("submission_preflight_sha256") != preflight.get(
        "submission_preflight_sha256"
    ):
        raise ValueError("publication submission does not bind its preflight")
    completion_path = job_root / "completion.json"
    payload = json.loads(completion_path.read_text(encoding="utf-8"))
    identity = {
        key: value for key, value in payload.items() if key != "completion_sha256"
    }
    if payload.get("completion_sha256") != _stable_hash(identity):
        raise ValueError("publication completion record self-hash differs")
    manifest_path = Path(str(inputs["publication_root"])) / (
        "publication_artifact_manifest.json"
    )
    if (
        payload.get("status") != "HPC_AND_PUBLICATION_BUNDLE_COMPLETE"
        or payload.get("run_id") != inputs.get("run_id")
        or payload.get("artifact_job_inputs_sha256")
        != inputs.get("artifact_job_inputs_sha256")
        or payload.get("submission_record_sha256")
        != submission.get("submission_record_sha256")
        or payload.get("publication_manifest_sha256") != _sha256_path(manifest_path)
    ):
        raise ValueError("publication completion record is stale or differs")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    publication = manifest_path.parent
    inventory = _artifact_inventory(publication)
    if (
        manifest.get("status") != "PUBLICATION_ARTIFACTS_COMPLETE"
        or inventory != manifest.get("artifacts")
        or _stable_hash(inventory) != manifest.get("artifact_set_sha256")
    ):
        raise ValueError("publication artifacts changed after terminal completion")
    return payload


def advance_campaign(
    control_manifest_path: str | Path,
    *,
    execute: bool,
    remaining_au: int | None = None,
) -> dict[str, Any]:
    """Perform exactly one safe state transition, or report a waiting state."""
    control = _load_control(control_manifest_path)
    live_remaining_au = int(
        control["remaining_au_for_live_preflight"]
        if remaining_au is None
        else remaining_au
    )
    if live_remaining_au <= 0:
        raise ValueError("live remaining AU must be positive")
    root = Path(str(control["campaign_root"]))
    config = Path(str(control["final_config_path"]))
    repo = Path(str(control["rfm_repo_root"]))
    freeze = root / "resource_freeze.json"
    if not freeze.is_file():
        pilot = load_packaged_dag(
            package_root=control["pilot_package_root"],
            config_path=config,
            repo_root=repo,
        )
        close_completed_pilot(
            dag=pilot,
            submission_record_path=Path(str(control["pilot_submission_record"])),
            accepted_resource_freeze_path=Path(str(control["pilot_resource_freeze"])),
            accounting_output_dir=root / "pilot_accounting",
            resource_freeze_path=freeze,
            prior_sunk_au=float(control["prior_sunk_au"]),
        )
        return {"status": "PILOT_CLOSED_AND_RESOURCES_FROZEN"}

    development_package = root / "development_package"
    if not (development_package / "package_summary.json").is_file():
        prepare_development_package(
            output_dir=development_package,
            resource_freeze_path=freeze,
            repo_root=repo,
            config_path=config,
        )
        return {"status": "DEVELOPMENT_PACKAGE_READY"}
    development = load_packaged_dag(
        package_root=development_package, config_path=config, repo_root=repo
    )
    development_paths = _phase_paths(root, "development")
    development_guard = _development_phase_budget_guard(
        dag=development,
        resource_freeze_path=freeze,
        control=control,
    )
    if development_paths["budget_guard"].is_file():
        recorded_guard = json.loads(
            development_paths["budget_guard"].read_text(encoding="utf-8")
        )
        if recorded_guard != development_guard:
            raise ValueError("development budget guard is stale or differs")
    else:
        _write_new_json(development_paths["budget_guard"], development_guard)
    if (
        "resolution_cache_evidence_root" in control
        and development_paths["authorization"].is_file()
        and not development_paths["submission"].exists()
    ):
        promotion_was_absent = not (
            development_paths["root"] / "cache_promotion.json"
        ).is_file()
        promotion = _promote_resolution_cache(
            control=control,
            development=development,
        )
        if promotion_was_absent:
            return {
                "status": "RESOLUTION_CACHE_PROMOTED",
                "promoted_shards": len(promotion["promoted_shards"]),
            }
    development_result = _advance_phase(
        dag=development,
        phase="development",
        root=root,
        config=config,
        repo=repo,
        freeze=freeze,
        prerequisites=(),
        remaining_au=live_remaining_au,
        execute=execute,
    )
    if development_result["status"] != "PHASE_COMPLETE":
        return development_result

    resolution = _stage_artifact(development, "resolution", "resolution_decision.json")
    final_package = root / "final_confirmatory_package"
    budget = root / "budget_certificate.json"
    if not (final_package / "package_summary.json").is_file():
        prepare_final_package(
            output_dir=final_package,
            resource_freeze_path=freeze,
            resolution_decision_path=resolution,
            development_completion_path=_phase_paths(root, "development")["completion"],
            repo_root=repo,
            config_path=config,
            allocation_quota_au=float(control["allocation_quota_au"]),
            postprocessing_reserved_au=float(control["postprocessing_reserved_au"]),
            budget_certificate_path=budget,
        )
        return {"status": "FINAL_CONFIRMATORY_PACKAGE_BUDGET_CERTIFIED"}
    final = load_packaged_dag(
        package_root=final_package, config_path=config, repo_root=repo
    )
    if not budget.is_file():
        raise ValueError(
            "final package exists without its budget certificate; downstream remains locked"
        )
    _validate_budget_certificate(
        dag=final,
        certificate_path=budget,
        resource_freeze_path=freeze,
        control=control,
    )
    phase_prerequisites = {
        "gate_b": (resolution,),
        "gate_p": (
            _stage_artifact(final, "gate_b", "gate_b_decision.json"),
            _stage_artifact(
                final, "fixed_family_supplement", "fixed_family_decision.json"
            ),
        ),
        "gate_c": (
            _stage_artifact(final, "gate_b", "gate_b_decision.json"),
            _stage_artifact(
                final, "fixed_family_supplement", "fixed_family_decision.json"
            ),
            _stage_artifact(
                final, "applied_bootstrap", "scientific_stage_summary.json"
            ),
        ),
    }
    for phase in ("gate_b", "gate_p", "gate_c"):
        paths = _phase_paths(root, phase)
        if not paths["completion"].is_file():
            guard = _confirmatory_phase_budget_guard(
                dag=final,
                phase=phase,
                root=root,
                resource_freeze_path=freeze,
                control=control,
            )
            if paths["budget_guard"].is_file():
                recorded_guard = json.loads(
                    paths["budget_guard"].read_text(encoding="utf-8")
                )
                if recorded_guard != guard:
                    raise ValueError(f"{phase} budget guard is stale or differs")
            else:
                _write_new_json(paths["budget_guard"], guard)
        result = _advance_phase(
            dag=final,
            phase=phase,
            root=root,
            config=config,
            repo=repo,
            freeze=freeze,
            prerequisites=phase_prerequisites[phase],
            remaining_au=live_remaining_au,
            execute=execute,
        )
        if result["status"] != "PHASE_COMPLETE":
            return result

    job_root = root / "publication_job"
    if not (job_root / "artifact_job_inputs.json").is_file():
        _prepare_artifact_job(
            root=root,
            dag=final,
            bsm_runtime_root=Path(str(control["bsm_runtime_root"])),
        )
        return {"status": "PUBLICATION_JOB_PREPARED_NO_SUBMIT"}
    if not (job_root / "submission.json").is_file():
        if not execute:
            return {
                "status": "EXECUTION_FLAG_REQUIRED",
                "phase": "publication_artifacts",
            }
        _submit_artifact_job(root)
        return {"status": "PUBLICATION_JOB_SUBMITTED"}
    if not (job_root / "completion.json").is_file():
        probe = inspect_submitted_phase(job_root / "submission.json")
        if probe["status"] != "READY_TO_VERIFY":
            return probe
        return _verify_publication_job(root)
    return _validate_publication_completion(root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    initialize = commands.add_parser(
        "initialize", help="create an immutable NO-SUBMIT root"
    )
    initialize.add_argument("--campaign-root", required=True)
    initialize.add_argument("--run-id", required=True)
    initialize.add_argument("--base-config", required=True)
    initialize.add_argument("--pilot-package-root", required=True)
    initialize.add_argument("--pilot-submission-record", required=True)
    initialize.add_argument("--pilot-resource-freeze", required=True)
    initialize.add_argument("--rfm-repo-root", required=True)
    initialize.add_argument("--rfm-controller-root", required=True)
    initialize.add_argument("--bsm-runtime-root", required=True)
    initialize.add_argument("--prior-sunk-au", required=True, type=float)
    initialize.add_argument("--remaining-au", required=True, type=int)
    initialize.add_argument("--allocation-quota-au", type=float, default=25_000.0)
    initialize.add_argument("--postprocessing-reserved-au", type=float, default=5.0)
    initialize.add_argument(
        "--resolution-cache-evidence-root",
        help=(
            "preserved failed-run bundle containing self-verifying completed "
            "resolution score artifacts"
        ),
    )

    advance = commands.add_parser("advance", help="perform at most one safe transition")
    advance.add_argument("--control-manifest", required=True)
    advance.add_argument(
        "--remaining-au",
        type=int,
        help="fresh same-day remaining AU for any live preflight in this invocation",
    )
    advance.add_argument("--execute", action="store_true")

    watch = commands.add_parser("watch", help="advance until complete or failed")
    watch.add_argument("--control-manifest", required=True)
    watch.add_argument(
        "--remaining-au",
        type=int,
        help="fresh same-day remaining AU for any live preflight in this invocation",
    )
    watch.add_argument("--poll-seconds", type=int, default=300)
    watch.add_argument("--execute", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "initialize":
        rfm_controller_root = Path(args.rfm_controller_root)
    else:
        raw_control = json.loads(
            Path(args.control_manifest).read_text(encoding="utf-8")
        )
        rfm_controller_root = Path(str(raw_control.get("rfm_controller_root", "")))
    _activate_rfm_controller(rfm_controller_root)

    if args.command == "initialize":
        result = initialize_campaign(
            campaign_root=args.campaign_root,
            run_id=args.run_id,
            base_config_path=args.base_config,
            pilot_package_root=args.pilot_package_root,
            pilot_submission_record=args.pilot_submission_record,
            pilot_resource_freeze=args.pilot_resource_freeze,
            rfm_repo_root=args.rfm_repo_root,
            rfm_controller_root=args.rfm_controller_root,
            bsm_runtime_root=args.bsm_runtime_root,
            prior_sunk_au=args.prior_sunk_au,
            remaining_au=args.remaining_au,
            allocation_quota_au=args.allocation_quota_au,
            postprocessing_reserved_au=args.postprocessing_reserved_au,
            resolution_cache_evidence_root=args.resolution_cache_evidence_root,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.command == "advance":
        print(
            json.dumps(
                advance_campaign(
                    args.control_manifest,
                    execute=args.execute,
                    remaining_au=args.remaining_au,
                ),
                sort_keys=True,
            )
        )
        return 0
    if args.poll_seconds < 60:
        raise ValueError("watch polling cadence must be at least 60 seconds")
    while True:
        result = advance_campaign(
            args.control_manifest,
            execute=args.execute,
            remaining_au=args.remaining_au,
        )
        print(json.dumps(result, sort_keys=True), flush=True)
        if result["status"] in {
            "HPC_AND_PUBLICATION_BUNDLE_COMPLETE",
            "EXECUTION_FLAG_REQUIRED",
        }:
            return 0
        if result["status"] in {"WAITING_FOR_SCHEDULER", "WAITING_FOR_SACCT"}:
            time.sleep(args.poll_seconds)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
