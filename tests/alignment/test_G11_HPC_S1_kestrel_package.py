"""G11-HPC-S1: content-addressed NO-SUBMIT Kestrel HPC package."""

from __future__ import annotations

import hashlib
import json
import math
import re
from argparse import Namespace
from dataclasses import replace
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from rfm_pipeline.campaign_contract import G11_CONTRACT, load_contract
from rfm_pipeline.hpc_campaign_package import (
    _cli_audit,
    _cli_reduce,
    _cli_worker,
    _execute_scientific_reduce,
    _execute_worker_operation,
    _pilot_sparse_inputs,
    build_campaign_phase_plan,
    build_submission_plan,
    collect_pilot_accounting,
    execute_campaign_phase_plan,
    execute_retry_submission_plan,
    execute_submission_plan,
    finalize_post_resolution_package,
    generate_campaign_package,
    load_stage_manifest,
    main,
    plan_retry_attempts,
    prepare_stage_retry_package,
    run_hpc_live_smoke,
    select_pilot_resources,
    validate_campaign_table,
    validate_hpc_preflight,
    validate_resume_artifacts,
    validate_stage_dependencies,
    write_phase_authorization,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "hpc" / "g11_kestrel_campaign.yml"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_PILOT_STAGES_FOR_TEST = {
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
}


def _stage(dag, name: str):  # noqa: ANN001
    return next(stage for stage in dag.stages if stage.name == name)


def _config_with_quota(tmp_path: Path, quota: int) -> Path:
    config_path = tmp_path / "g11_kestrel_campaign.yml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        re.sub(
            r'allocation_quota: "(?:UNKNOWN|[0-9]+)"',
            f'allocation_quota: "{quota}"',
            CONFIG_PATH.read_text(encoding="utf-8"),
        ).replace("/scratch/{user}/bsm_runs/{run_id}", f"{tmp_path}/scratch/{{run_id}}"),
        encoding="utf-8",
    )
    return config_path


def _pilot_preflight(dag) -> dict[str, object]:  # noqa: ANN001
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "HPC_SUBMISSION_READY",
        "target_phase": "pilot",
        "run_id": dag.run_id,
        "source_hash": dag.source_hash,
        "config_hash": dag.config_hash,
        "lock_hash": dag.lock_hash,
        "campaign_inventory_hash": dag.campaign_inventory_hash,
        "resource_freeze_sha256": "PENDING",
        "observed_date": date.today().isoformat(),
        "remaining_au": 1_000_000,
        "required_au": 1,
        "required_storage_gb": 1,
        "required_inodes": 1,
        "rfm_git_commit": "1" * 40,
        "bsm_git_commit": "2" * 40,
        "evidence_sha256": "3" * 64,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["preflight_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return payload


def _phase_preflight(
    dag,
    phase: str,
    resource_freeze_sha256: str,  # noqa: ANN001
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "HPC_SUBMISSION_READY",
        "target_phase": phase,
        "run_id": dag.run_id,
        "source_hash": dag.source_hash,
        "config_hash": dag.config_hash,
        "lock_hash": dag.lock_hash,
        "campaign_inventory_hash": dag.campaign_inventory_hash,
        "readiness_blockers": list(dag.readiness_blockers),
        "resource_freeze_sha256": resource_freeze_sha256,
        "observed_date": date.today().isoformat(),
        "remaining_au": 1_000_000,
        "required_au": 1,
        "required_storage_gb": 1,
        "required_inodes": 1,
        "rfm_git_commit": "1" * 40,
        "bsm_git_commit": "2" * 40,
        "evidence_sha256": "3" * 64,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["preflight_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return payload


def _signed_authorization(
    dag,
    phase: str,
    resource_freeze_sha256: str,  # noqa: ANN001
    *,
    preflight_sha256: str,
) -> dict[str, object]:
    identity: dict[str, object] = {
        "schema_version": 2,
        "status": "ACCEPTED",
        "phase": phase,
        "run_id": dag.run_id,
        "contract_hash": dag.config_hash,
        "source_hash": dag.source_hash,
        "lock_hash": dag.lock_hash,
        "campaign_inventory_hash": dag.campaign_inventory_hash,
        "preflight_sha256": preflight_sha256,
        "prerequisite_sha256": {},
        "execution_permitted": True,
        "resource_freeze_sha256": resource_freeze_sha256,
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return {
        **identity,
        "authorization_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
    }


def _final_package(tmp_path: Path, *, quota: int = 25_000):
    base = _package(tmp_path / "base")
    freeze = select_pilot_resources(
        pilot_matrix=base.pilot_matrix,
        telemetry=_accepted_pilot_telemetry(base),
        cluster=base.cluster,
        source_hash=base.source_hash,
        config_hash=base.config_hash,
        lock_hash=base.lock_hash,
    )
    freeze_path = tmp_path / "resource_freeze.json"
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    decision = {
        "operation": "resolution",
        "status": "completed",
        "decision": "ACCEPTED",
        "contract_hash": base.config_hash,
        "terminal_record_count": 20,
        "selected_B_interaction": 999,
    }
    decision_path = tmp_path / "resolution_decision.json"
    decision_path.write_text(json.dumps(decision, sort_keys=True), encoding="utf-8")
    final = finalize_post_resolution_package(
        output_dir=tmp_path / "final" / "package",
        resource_freeze_path=freeze_path,
        resolution_decision_path=decision_path,
        repo_root=REPO_ROOT,
        config_path=_config_with_quota(tmp_path / "final", quota),
    )
    return base, freeze, freeze_path, decision_path, final


def _package(tmp_path: Path):
    return generate_campaign_package(
        output_dir=tmp_path / "package",
        repo_root=REPO_ROOT,
        config_path=_config_with_quota(tmp_path, 25_000),
    )


def test_full_package_requires_an_empty_destination(tmp_path: Path) -> None:
    package_root = tmp_path / "package"
    package_root.mkdir(parents=True)
    (package_root / "stale.txt").write_text("stale", encoding="utf-8")

    with pytest.raises(ValueError, match="output directory must be empty"):
        generate_campaign_package(
            output_dir=package_root,
            repo_root=REPO_ROOT,
            config_path=_config_with_quota(tmp_path, 1_000_000),
        )


def test_malformed_manifest_rejection(tmp_path: Path) -> None:
    manifest_path = tmp_path / "bad_manifest.jsonl"
    manifest_path.write_text(json.dumps({"stage": "gate_b"}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="manifest"):
        load_stage_manifest(manifest_path)


def test_resume_artifact_validation_requires_atomic_success_marker(tmp_path: Path) -> None:
    dag = _package(tmp_path)
    resolution = _stage(dag, "resolution")
    record = load_stage_manifest(resolution.manifest_path)[0]
    shard_dir = Path(record["output_dir"])
    shard_dir.mkdir(parents=True, exist_ok=True)
    result_path = shard_dir / "result.json"
    nested = shard_dir / "terminal_record.json"
    nested.write_text("{}", encoding="utf-8")
    result_path.write_text(
        json.dumps(
            {
                "scientific_artifacts": [
                    {
                        "path": str(nested),
                        "relative_path": nested.name,
                        "sha256": hashlib.sha256(nested.read_bytes()).hexdigest(),
                        "bytes": nested.stat().st_size,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    pending_summary = validate_resume_artifacts(resolution)
    assert pending_summary["completed"] == 0
    assert pending_summary["pending"] == resolution.job_count

    (shard_dir / "_SUCCESS.json").write_text(
        json.dumps(
            {
                "stage": record["stage"],
                "shard_id": record["shard_id"],
                "output_hash": record["output_hash"],
                "artifact_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
                "parent_hash": record["parent_hash"],
                "attempt": 1,
                "status": "completed",
                "source_hash": dag.source_hash,
                "config_hash": dag.config_hash,
                "lock_hash": dag.lock_hash,
            }
        ),
        encoding="utf-8",
    )

    resumed_summary = validate_resume_artifacts(resolution)
    assert resumed_summary["completed"] == 1
    assert resumed_summary["pending"] == resolution.job_count - 1
    nested.write_text('{"tampered": true}', encoding="utf-8")
    with pytest.raises(ValueError, match="scientific artifact byte"):
        validate_resume_artifacts(resolution)


def test_dependency_failure_propagation(tmp_path: Path) -> None:
    dag = _package(tmp_path)
    resolution = _stage(dag, "resolution")
    gate_b = _stage(dag, "gate_b")
    resolution.reducer_output_dir.mkdir(parents=True, exist_ok=True)
    (resolution.reducer_output_dir / "_FAILED.json").write_text(
        json.dumps(
            {
                "stage": resolution.name,
                "status": "failed",
                "output_hash": resolution.reducer_output_hash,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="resolution"):
        validate_stage_dependencies(dag, gate_b.name)


def test_no_pair_sharding_assertion(tmp_path: Path) -> None:
    dag = _package(tmp_path)

    for stage in dag.stages:
        records = load_stage_manifest(stage.manifest_path)
        assert records
        assert {record["interaction_sharding"] for record in records} == {"draw-block"}
        assert all("pair_start" not in record and "pair_end" not in record for record in records)


def test_resource_arithmetic_keeps_at_least_25_percent_headroom(tmp_path: Path) -> None:
    dag = _package(tmp_path)

    for stage in dag.stages:
        for envelope in (stage.worker_resources, stage.reducer_resources):
            assert envelope.requested_cpu_cores >= math.ceil(envelope.estimated_cpu_cores * 1.25)
            assert envelope.requested_memory_gb >= math.ceil(envelope.estimated_memory_gb * 1.25)
            assert envelope.requested_walltime_seconds >= math.ceil(
                envelope.estimated_walltime_seconds * 1.25
            )
            assert envelope.requested_cpu_cores <= dag.cluster.cpu_cores_per_node
            assert envelope.requested_memory_gb <= dag.cluster.memory_per_node_gb


def test_campaign_envelope_covers_au_storage_inode_and_retry_headroom(tmp_path: Path) -> None:
    """The budget envelope reserves 20% for overrun/retries without doubling all work."""
    dag = _package(tmp_path)
    envelope = dag.campaign_envelope

    assert envelope.estimated_node_hours > 0
    assert envelope.estimated_au > 0
    assert envelope.estimated_storage_gb > 0
    assert envelope.estimated_inode_count > 0
    assert envelope.estimated_retry_attempts == 0
    assert envelope.requested_retry_attempts == math.ceil(
        sum(stage.job_count for stage in dag.stages) * 0.20
    )
    assert envelope.requested_au == math.ceil(envelope.estimated_au * 1.20)
    assert envelope.requested_storage_gb == math.ceil(envelope.estimated_storage_gb * 1.20)
    assert envelope.requested_inode_count == math.ceil(envelope.estimated_inode_count * 1.20)
    assert envelope.allocation_quota_readiness_blocker is False
    assert any("provisional pre-pilot forecast" in blocker for blocker in dag.readiness_blockers)


def test_campaign_envelope_applies_kestrel_cpu_charge_factor(tmp_path: Path) -> None:
    """Kestrel CPU work is charged at ten AUs per allocated node-hour."""
    dag = _package(tmp_path)
    envelope = dag.campaign_envelope

    assert dag.cluster.allocation_unit == "AU"
    assert dag.cluster.cpu_charge_factor == 10.0
    assert envelope.estimated_au >= 10.0 * envelope.estimated_node_hours
    assert envelope.requested_au == math.ceil(envelope.estimated_au * 1.20)


def test_campaign_envelope_counts_worker_audit_and_reducer_jobs(tmp_path: Path) -> None:
    dag = _package(tmp_path)
    stage = _stage(dag, "scheduler_diagnostic")
    estimate = next(
        row for row in dag.campaign_envelope.stage_allocations if row.stage_name == stage.name
    )

    # debug is exclusive: one worker plus one audit plus one reducer, each one node.
    assert estimate.estimated_node_hours == pytest.approx((180 + 300 + 300) / 3600)
    assert estimate.requested_node_hours == pytest.approx((225 + 375 + 375) / 3600)


def test_pre_pilot_campaign_envelope_exposes_full_provisional_cost(tmp_path: Path) -> None:
    """A provisional pre-pilot forecast is diagnostic, not a quota admission gate."""
    dag = generate_campaign_package(
        output_dir=tmp_path / "package",
        repo_root=REPO_ROOT,
        config_path=CONFIG_PATH,
    )

    assert dag.campaign_envelope.requested_au == 167_593
    assert dag.campaign_envelope.requested_au > 25_000
    gate_b = next(
        estimate
        for estimate in dag.campaign_envelope.stage_allocations
        if estimate.stage_name == "gate_b"
    )
    gate_c = next(
        estimate
        for estimate in dag.campaign_envelope.stage_allocations
        if estimate.stage_name == "recovery"
    )
    assert math.ceil(gate_b.requested_au) == 140_003
    assert math.ceil(gate_c.requested_au) == 30_003
    assert any(
        "production resources and block sizes are not frozen" in blocker
        for blocker in dag.readiness_blockers
    )


def test_configured_allocation_quota_is_the_25k_campaign_ceiling(tmp_path: Path) -> None:
    dag = generate_campaign_package(
        output_dir=tmp_path / "package",
        repo_root=REPO_ROOT,
        config_path=CONFIG_PATH,
    )
    assert dag.campaign_envelope.allocation_quota_readiness_blocker is False
    assert int(dag.cluster.allocation_quota) == 25_000
    assert any("provisional pre-pilot forecast" in blocker for blocker in dag.readiness_blockers)


def test_pre_pilot_package_accepts_25k_budget_pending_live_telemetry(tmp_path: Path) -> None:
    dag = generate_campaign_package(
        output_dir=tmp_path / "package",
        repo_root=REPO_ROOT,
        config_path=_config_with_quota(tmp_path, 25_000),
    )

    assert int(dag.cluster.allocation_quota) == 25_000
    assert dag.campaign_envelope.requested_au > 25_000
    assert any("provisional pre-pilot forecast" in blocker for blocker in dag.readiness_blockers)


def test_scripts_bind_kestrel_account_partition_walltime_without_sbatch_execution(
    tmp_path: Path,
) -> None:
    dag = _package(tmp_path)

    for stage in dag.stages:
        for script_path in (stage.worker_script_path, stage.reducer_script_path):
            text = script_path.read_text(encoding="utf-8")
            assert "#SBATCH --account=nationalpfa" in text
            assert f"#SBATCH --partition={stage.partition}" in text
            assert re.search(r"^#SBATCH --time=\d{2}:\d{2}:\d{2}$", text, flags=re.MULTILINE)
            assert not re.search(r"^\s*sbatch\b", text, flags=re.MULTILINE)


def test_pilot_smoke_uses_debug_but_interaction_sizing_uses_short(tmp_path: Path) -> None:
    dag = _package(tmp_path)

    assert dag.cluster.account == "nationalpfa"
    for stage in (stage for stage in dag.stages if stage.name in _PILOT_STAGES_FOR_TEST):
        expected_partition = "short" if stage.name == "pilot_interaction_score" else "debug"
        maximum_walltime = 4 * 3600 if expected_partition == "short" else 3600
        assert stage.partition == expected_partition
        for script_path in (
            *stage.worker_script_paths,
            stage.audit_script_path,
            stage.reducer_script_path,
        ):
            text = script_path.read_text(encoding="utf-8")
            assert "#SBATCH --account=nationalpfa" in text
            assert f"#SBATCH --partition={expected_partition}" in text
            match = re.search(
                r"^#SBATCH --time=(\d{2}):(\d{2}):(\d{2})$",
                text,
                flags=re.MULTILINE,
            )
            assert match is not None
            hours, minutes, seconds = (int(value) for value in match.groups())
            assert hours * 3600 + minutes * 60 + seconds <= maximum_walltime

    interaction = _stage(dag, "pilot_interaction_score")
    assert interaction.worker_resources.estimated_walltime_seconds == 3 * 3600
    assert interaction.worker_resources.requested_walltime_seconds == 3 * 3600 + 45 * 60


def test_scheduler_diagnostic_can_start_the_pinned_environment(tmp_path: Path) -> None:
    dag = _package(tmp_path)
    stage = next(stage for stage in dag.stages if stage.name == "scheduler_diagnostic")

    assert stage.worker_resources.requested_memory_gb >= 10
    assert 60 <= stage.worker_resources.requested_walltime_seconds <= 300


def test_every_campaign_script_uses_the_direct_interpreter_from_fast_runtime_storage(
    tmp_path: Path,
) -> None:
    """Array tasks must not import Python modules from metadata-heavy project storage."""
    dag = _package(tmp_path)

    assert dag.cluster.rfm_repository_root.startswith("/scratch/")
    assert dag.cluster.bsm_repository_root.startswith("/scratch/")
    for stage in dag.stages:
        for script_path in (
            *stage.worker_script_paths,
            stage.audit_script_path,
            stage.reducer_script_path,
        ):
            text = script_path.read_text(encoding="utf-8")
            assert "pixi run python" not in text
            assert f'cd "{dag.cluster.rfm_repository_root}"' in text
            assert (
                f'RUNTIME_PYTHON="{dag.cluster.rfm_repository_root}/.pixi/envs/default/bin/python"'
                in text
            )
            assert '"$RUNTIME_PYTHON" -m rfm_pipeline.hpc_campaign_package' in text


def test_hash_completeness_is_fail_closed(tmp_path: Path) -> None:
    dag = _package(tmp_path)
    required_hashes = {
        "source_hash",
        "config_hash",
        "lock_hash",
        "input_hash",
        "schedule_hash",
        "parent_hash",
        "output_hash",
    }

    for stage in dag.stages:
        for record in load_stage_manifest(stage.manifest_path):
            for field in required_hashes:
                value = record[field]
                assert value
                assert _HEX64.fullmatch(value)
                assert "TODO" not in value
                assert "TBD" not in value


def test_pilot_matrix_correctness_and_selector_determinism(tmp_path: Path) -> None:
    dag = _package(tmp_path)
    rerendered = _package(tmp_path / "rerun")
    assert {row["stage"] for row in dag.pilot_matrix} == {
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
    }
    assert len(dag.pilot_matrix) == 1 + 12 * 3
    for stage_name in {
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
    }:
        rows = [row for row in dag.pilot_matrix if row["stage"] == stage_name]
        assert len(rows) == 3
        assert {row["profile_id"] for row in rows} == {"p0", "p1", "p2"}
        assert len({(row["cpus"], row["block_size"]) for row in rows}) >= 2
        assert all(row["timing_repetitions"] == 1 for row in rows)
    assert all(row["block_size"] <= row["task_size"] for row in dag.pilot_matrix)
    assert all(
        profile["status"] == "PENDING_TELEMETRY" for profile in dag.post_pilot_selection.values()
    )
    assert dag.post_pilot_selection == rerendered.post_pilot_selection


def test_pilot_manifest_rows_bind_distinct_scheduler_resource_profiles(tmp_path: Path) -> None:
    """Pilot alternatives must be real scheduler requests, not labels on one array."""
    dag = _package(tmp_path)
    stage = _stage(dag, "pilot_interaction_score")
    records = load_stage_manifest(stage.manifest_path)

    assert len(records) == 3
    assert {record["profile_id"] for record in records} == {"p0", "p1", "p2"}
    assert (
        len(
            {
                (
                    record["worker_resources"]["requested_cpu_cores"],
                    record["block_size"],
                )
                for record in records
            }
        )
        >= 2
    )
    assert len(stage.worker_script_paths) == 3
    for task_id, script_path in enumerate(stage.worker_script_paths):
        text = script_path.read_text(encoding="utf-8")
        assert f'--task-id "{task_id}"' in text
        assert "#SBATCH --array" not in text
        assert (
            f"#SBATCH --cpus-per-task={records[task_id]['worker_resources']['requested_cpu_cores']}"
        ) in text


def test_pilot_work_units_match_production_dimensions(tmp_path: Path) -> None:
    dag = _package(tmp_path)
    bootstrap = [row for row in dag.pilot_matrix if row["stage"] == "pilot_bootstrap"]
    assert [row["executed_work_units"] for row in bootstrap] == [125_000, 500_000, 2_000_000]

    telemetry = _accepted_pilot_telemetry(dag)
    freeze = select_pilot_resources(
        pilot_matrix=dag.pilot_matrix,
        telemetry=telemetry,
        cluster=dag.cluster,
        source_hash=dag.source_hash,
        config_hash=dag.config_hash,
        lock_hash=dag.lock_hash,
    )
    selected = freeze["selections"]["pilot_bootstrap"]
    assert selected["production_target_work_units"] == selected["block_size"] * 23_495 * 5


def test_holdout_and_bootstrap_pilot_repetitions_consume_matching_upstream_shards(
    tmp_path: Path,
) -> None:
    """Each timing chain must stay isolated from the other pilot repetitions."""
    dag = _package(tmp_path)
    holdout_records = load_stage_manifest(_stage(dag, "pilot_holdout_predict").manifest_path)
    bootstrap_records = load_stage_manifest(_stage(dag, "pilot_bootstrap").manifest_path)

    assert [record["upstream_shard_id"] for record in holdout_records] == [
        "task-0000",
        "task-0001",
        "task-0002",
    ]
    assert [record["upstream_shard_id"] for record in bootstrap_records] == [
        "task-0000",
        "task-0001",
        "task-0002",
    ]


def test_sparse_resample_reconstructs_matching_upstream_sparse_fixture(tmp_path: Path) -> None:
    """A sparse resample must score the exact fixture used by its frozen full fit."""
    dag = _package(tmp_path)
    full_records = load_stage_manifest(_stage(dag, "pilot_sparse_full").manifest_path)
    resample_records = load_stage_manifest(_stage(dag, "pilot_sparse_resample").manifest_path)

    fixtures = []
    for full_record, resample_record in zip(full_records, resample_records, strict=True):
        full_fixture = _pilot_sparse_inputs(full_record, n_rows=32, n_components=3)
        resample_fixture = _pilot_sparse_inputs(resample_record, n_rows=32, n_components=3)
        assert all(
            left.equals(right)
            for left, right in zip(full_fixture[:4], resample_fixture[:4], strict=True)
        )
        fixtures.append(full_fixture)

    assert not fixtures[0][0].equals(fixtures[1][0])


def _accepted_pilot_telemetry(dag) -> list[dict[str, object]]:  # noqa: ANN001
    elapsed_by_profile = {"p0": 400.0, "p1": 100.0, "p2": 300.0}
    execution_schedule_hashes = {
        (stage.name, str(record["profile_id"])): str(record["schedule_hash"])
        for stage in dag.stages
        if stage.name in _PILOT_STAGES_FOR_TEST
        for record in load_stage_manifest(stage.manifest_path)
    }
    rows: list[dict[str, object]] = []
    for profile in dag.pilot_matrix:
        identity = (str(profile["stage"]), str(profile["profile_id"]))
        rows.append(
            {
                "stage": profile["stage"],
                "profile_id": profile["profile_id"],
                "schedule_hash": execution_schedule_hashes[identity],
                "profile_schedule_hash": profile["schedule_hash"],
                "status": "completed",
                "source_hash": dag.source_hash,
                "config_hash": dag.config_hash,
                "lock_hash": dag.lock_hash,
                "elapsed_seconds": (
                    10.0
                    if profile["stage"] == "scheduler_diagnostic"
                    else elapsed_by_profile[str(profile["profile_id"])]
                ),
                "total_cpu_seconds": 10.0,
                "cpu_time_seconds": 10.0,
                "max_rss_bytes": 2 * 1024**3,
                "bytes_read": 1024,
                "bytes_written": 2048,
                "task_size": profile["task_size"],
                "block_size": profile["block_size"],
                "executed_work_units": profile["executed_work_units"],
                "requested_cpus": profile["cpus"],
                "requested_memory_gb": profile["memory_gb"],
                "requested_walltime_seconds": profile["walltime_seconds"],
                "partition": profile["partition"],
                "hostname": "x1000c0s0b0n0",
                "slurm_job_id": str(100000 + len(rows)),
                "slurm_array_job_id": None,
                "slurm_array_task_id": None,
                "scheduler_state": "COMPLETED",
                "scheduler_exit_code": "0:0",
                "scheduler_elapsed_seconds": int(
                    10.0
                    if profile["stage"] == "scheduler_diagnostic"
                    else elapsed_by_profile[str(profile["profile_id"])]
                ),
                "allocated_nodes": 1,
                "allocated_cpus": 104,
                "allocated_tres": "billing=1024,cpu=104,node=1",
                "scheduler_max_rss_bytes": 2 * 1024**3,
            }
        )
    return rows


def test_pilot_resource_selector_requires_complete_telemetry_and_minimizes_projected_au(
    tmp_path: Path,
) -> None:
    dag = _package(tmp_path)
    telemetry = _accepted_pilot_telemetry(dag)

    freeze = select_pilot_resources(
        pilot_matrix=dag.pilot_matrix,
        telemetry=telemetry,
        cluster=dag.cluster,
        source_hash=dag.source_hash,
        config_hash=dag.config_hash,
        lock_hash=dag.lock_hash,
    )

    assert freeze["status"] == "ACCEPTED"
    assert _HEX64.fullmatch(freeze["telemetry_sha256"])
    for stage_name, selection in freeze["selections"].items():
        if stage_name == "scheduler_diagnostic":
            continue
        # The middle profile has the lowest elapsed time per completed block
        # under this fixture, hence the lowest projected AU per work unit.
        expected_profile = "p2" if stage_name == "pilot_bootstrap" else "p1"
        assert selection["selected_profile_id"] == expected_profile
        assert selection["requested_memory_gb"] >= 5
        assert selection["requested_walltime_seconds"] >= 450

    with pytest.raises(ValueError, match="missing pilot telemetry"):
        select_pilot_resources(
            pilot_matrix=dag.pilot_matrix,
            telemetry=telemetry[:-1],
            cluster=dag.cluster,
            source_hash=dag.source_hash,
            config_hash=dag.config_hash,
            lock_hash=dag.lock_hash,
        )


def test_pilot_resource_selector_prices_exclusive_profiles_as_full_nodes(
    tmp_path: Path,
) -> None:
    """Kestrel debug/short/standard jobs are exclusive whole-node charges."""
    dag = _package(tmp_path)
    telemetry = _accepted_pilot_telemetry(dag)
    for row in telemetry:
        if row["stage"] != "pilot_recovery":
            continue
        elapsed = {
            "p0": 100.0,
            "p1": 60.0,
            "p2": 50.0,
        }[str(row["profile_id"])]
        row["elapsed_seconds"] = elapsed
        row["scheduler_elapsed_seconds"] = int(elapsed)

    freeze = select_pilot_resources(
        pilot_matrix=dag.pilot_matrix,
        telemetry=telemetry,
        cluster=dag.cluster,
        source_hash=dag.source_hash,
        config_hash=dag.config_hash,
        lock_hash=dag.lock_hash,
    )

    # Fractional CPU pricing incorrectly favors p0.  All three requests occupy
    # one exclusive node, so p2 has the lowest elapsed node-seconds per run.
    assert freeze["selections"]["pilot_recovery"]["selected_profile_id"] == "p2"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"status": "failed"}, "did not complete"),
        ({"profile_schedule_hash": "0" * 64}, "profile schedule hash"),
        ({"schedule_hash": "invalid"}, "execution schedule hash"),
        ({"block_size": 999999}, "block size"),
        ({"max_rss_bytes": 10**15}, "memory"),
    ],
)
def test_pilot_resource_selector_rejects_invalid_evidence(
    tmp_path: Path,
    mutation: dict[str, object],
    message: str,
) -> None:
    dag = _package(tmp_path)
    telemetry = _accepted_pilot_telemetry(dag)
    telemetry[1] = {**telemetry[1], **mutation}

    with pytest.raises(ValueError, match=message):
        select_pilot_resources(
            pilot_matrix=dag.pilot_matrix,
            telemetry=telemetry,
            cluster=dag.cluster,
            source_hash=dag.source_hash,
            config_hash=dag.config_hash,
            lock_hash=dag.lock_hash,
        )


def test_submission_plan_uses_afterany_only_for_audit_and_afterok_for_science(
    tmp_path: Path,
) -> None:
    dag = _package(tmp_path)
    plan = build_submission_plan(dag)
    pilot_steps = plan["pilot_steps"]

    assert plan["status"] == "NO_SUBMIT"
    assert all(not step["locked"] for step in pilot_steps)
    for step in pilot_steps:
        if step["action"] == "audit":
            assert step["dependency_type"] == "afterany"
            assert step["depends_on"]
        elif step["action"] == "reduce":
            assert step["dependency_type"] == "afterok"
            assert any(dependency.endswith(":audit") for dependency in step["depends_on"])
            assert any(":worker" in dependency for dependency in step["depends_on"])
        elif step["depends_on"]:
            assert step["dependency_type"] == "afterok"
    assert all(step["locked"] for step in plan["production_steps"])
    assert all(step["unlock_requirement"] for step in plan["production_steps"])


def test_submission_client_fails_closed_on_scheduler_error_or_missing_job_id(
    tmp_path: Path,
) -> None:
    dag = _package(tmp_path)
    plan = build_submission_plan(dag)
    calls: list[list[str]] = []

    def missing_job_id(command: list[str]):
        calls.append(command)
        return Namespace(returncode=0, stdout="", stderr="")

    with pytest.raises(RuntimeError, match="job ID"):
        execute_submission_plan(
            plan,
            run_command=missing_job_id,
            authorize=True,
            preflight=_pilot_preflight(dag),
        )
    assert len(calls) == 1

    calls.clear()

    def scheduler_failure(command: list[str]):
        calls.append(command)
        return Namespace(returncode=1, stdout="", stderr="account rejected")

    with pytest.raises(RuntimeError, match="account rejected"):
        execute_submission_plan(
            plan,
            run_command=scheduler_failure,
            authorize=True,
            preflight=_pilot_preflight(dag),
        )
    assert len(calls) == 1

    with pytest.raises(PermissionError, match="authorization"):
        execute_submission_plan(plan, run_command=missing_job_id, authorize=False)
    assert len(calls) == 1


def test_submission_client_materializes_exact_dependency_job_ids(tmp_path: Path) -> None:
    dag = _package(tmp_path)
    plan = build_submission_plan(dag)
    calls: list[list[str]] = []

    def success(command: list[str]):
        calls.append(command)
        return Namespace(returncode=0, stdout=f"{1000 + len(calls)}\n", stderr="")

    job_ids = execute_submission_plan(
        plan,
        run_command=success,
        authorize=True,
        preflight=_pilot_preflight(dag),
    )

    assert len(job_ids) == len(plan["pilot_steps"])
    audit_index = next(
        index for index, step in enumerate(plan["pilot_steps"]) if step["action"] == "audit"
    )
    audit_step = plan["pilot_steps"][audit_index]
    dependency_ids = [job_ids[step_id] for step_id in audit_step["depends_on"]]
    assert f"--dependency=afterany:{':'.join(dependency_ids)}" in calls[audit_index]


def test_pilot_accounting_joins_scheduler_rows_to_profile_telemetry(tmp_path: Path) -> None:
    dag = _package(tmp_path)
    stage = _stage(dag, "scheduler_diagnostic")
    record = load_stage_manifest(stage.manifest_path)[0]
    monkey_result = {
        "schema_version": 2,
        "stage": stage.name,
        "shard_id": record["shard_id"],
        "status": "completed",
        "attempt": 1,
        "elapsed_seconds": 0.25,
        "total_cpu_seconds": 0.2,
        "cpu_time_seconds": 0.2,
        "max_rss_bytes": 1024,
        "bytes_read": 0,
        "bytes_written": 0,
        "task_size": 1,
        "block_size": 1,
        "executed_work_units": 1,
        "profile_id": "p0",
        "partition": "debug",
        "requested_cpus": 2,
        "requested_memory_gb": 10,
        "requested_walltime_seconds": 225,
        "hostname": "x1000c0s0b0n0",
        "slurm_job_id": "123456",
        "slurm_array_job_id": None,
        "slurm_array_task_id": None,
        **{
            field: record[field]
            for field in (
                "source_hash",
                "config_hash",
                "lock_hash",
                "input_hash",
                "schedule_hash",
                "parent_hash",
                "output_hash",
            )
        },
    }
    result_path = Path(record["output_dir"]) / "result.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(json.dumps(monkey_result), encoding="utf-8")
    marker = {
        "schema_version": 2,
        "stage": stage.name,
        "shard_id": record["shard_id"],
        "output_hash": record["output_hash"],
        "artifact_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
        "parent_hash": record["parent_hash"],
        "attempt": 1,
        "status": "completed",
    }
    (result_path.parent / "_SUCCESS.json").write_text(json.dumps(marker), encoding="utf-8")
    submission = {
        "scheduler_diagnostic:worker": "123456",
        "scheduler_diagnostic:audit": "123457",
        "scheduler_diagnostic:reduce": "123458",
    }

    def fake_sacct(command: list[str], **_: object) -> Namespace:
        assert command[0] == "sacct"
        return Namespace(
            returncode=0,
            stdout=(
                "123456|g11-scheduler_diagnostic-worker|debug|COMPLETED|0:0|8|1|104|"
                "2|10G|billing=1024,cpu=104,node=1|832|00:00:01||1K|2K\n"
                "123456.batch|batch||COMPLETED|0:0|8|1|104|2|10G|"
                "billing=1024,cpu=104,node=1|832|00:00:01|180M|1K|2K\n"
                "123457|g11-scheduler_diagnostic-audit|debug|COMPLETED|0:0|5|1|104|"
                "5|10G|billing=1024,cpu=104,node=1|520|00:00:01|||\n"
                "123458|g11-scheduler_diagnostic-reduce|debug|COMPLETED|0:0|7|1|104|"
                "5|10G|billing=1024,cpu=104,node=1|728|00:00:01|||\n"
            ),
            stderr="",
        )

    evidence = collect_pilot_accounting(
        dag,
        submission_job_ids=submission,
        output_dir=tmp_path / "accounting",
        run_command=fake_sacct,
        stage_names=("scheduler_diagnostic",),
    )

    assert evidence["status"] == "COMPLETE"
    assert evidence["observed_worker_au"] == pytest.approx(8 / 3600 * 10)
    assert evidence["observed_total_au"] == pytest.approx((8 + 5 + 7) / 3600 * 10)
    assert evidence["pilot_telemetry"][0]["scheduler_elapsed_seconds"] == 8
    assert evidence["pilot_telemetry"][0]["allocated_nodes"] == 1
    assert evidence["pilot_telemetry"][0]["allocated_cpus"] == 104
    assert evidence["pilot_telemetry"][0]["scheduler_max_rss_bytes"] == 180 * 1024**2
    assert evidence["pilot_telemetry"][0]["schedule_hash"] == record["schedule_hash"]
    assert evidence["pilot_telemetry"][0]["profile_schedule_hash"] == next(
        row["schedule_hash"]
        for row in dag.pilot_matrix
        if row["stage"] == stage.name and row["profile_id"] == "p0"
    )
    assert (tmp_path / "accounting" / "sacct_raw.psv").is_file()
    assert (tmp_path / "accounting" / "pilot_accounting.json").is_file()


def test_phase_submission_rejects_source_or_lock_drift_before_sbatch(
    tmp_path: Path,
) -> None:
    _, freeze, _, _, dag = _final_package(tmp_path)
    plan = build_campaign_phase_plan(dag, phase="gate_b")

    for field in ("source_hash", "lock_hash"):
        preflight = _phase_preflight(dag, "gate_b", str(freeze["resource_freeze_sha256"]))
        authorization = _signed_authorization(
            dag,
            "gate_b",
            str(freeze["resource_freeze_sha256"]),
            preflight_sha256=str(preflight["preflight_sha256"]),
        )
        preflight[field] = "0" * 64
        identity = {key: value for key, value in preflight.items() if key != "preflight_sha256"}
        canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        preflight["preflight_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
        calls: list[list[str]] = []

        def must_not_submit(command: list[str], call_log: list[list[str]] = calls):
            call_log.append(command)
            return Namespace(returncode=0, stdout="1001\n", stderr="")

        with pytest.raises(PermissionError, match="preflight"):
            execute_campaign_phase_plan(
                plan,
                run_command=must_not_submit,
                authorize=True,
                preflight=preflight,
                phase_authorization=authorization,
            )
        assert calls == []


def test_preflight_accepts_current_remaining_allocation_above_phase_envelope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import rfm_pipeline.hpc_campaign_package as hpc

    dag = _package(tmp_path)
    selected = [stage for stage in dag.stages if stage.name in _PILOT_STAGES_FOR_TEST]
    script_hashes = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for stage in selected
        for path in (
            *stage.worker_script_paths,
            stage.audit_script_path,
            stage.reducer_script_path,
        )
    }

    def fake_git(command: list[str], **_: object) -> Namespace:
        root = str(command[2])
        if command[-2:] == ["status", "--porcelain"]:
            return Namespace(returncode=0, stdout="", stderr="")
        commit = "1" * 40 if root == dag.cluster.rfm_repository_root else "2" * 40
        return Namespace(returncode=0, stdout=f"{commit}\n", stderr="")

    real_hash_file = hpc._hash_file
    bound_hashes = {
        dag.cluster.scientific_adapter_relative_path: dag.cluster.scientific_adapter_sha256,
        dag.cluster.bsm_recovery_driver_relative_path: dag.cluster.bsm_recovery_driver_sha256,
        dag.cluster.bsm_dgp_contract_relative_path: dag.cluster.bsm_dgp_contract_sha256,
        dag.cluster.applied_data_preparer_relative_path: (dag.cluster.applied_data_preparer_sha256),
    }

    def fake_hash_file(path: Path) -> str:
        value = Path(path)
        if value.name == "pixi.lock":
            return dag.lock_hash
        for suffix, expected in bound_hashes.items():
            if str(value).endswith(suffix):
                return expected
        return real_hash_file(value)

    monkeypatch.setattr(hpc.subprocess, "run", fake_git)
    monkeypatch.setattr(hpc, "_hash_python_tree", lambda _: dag.source_hash)
    monkeypatch.setattr(hpc, "_hash_file", fake_hash_file)
    evidence = {
        "observed_date": date.today().isoformat(),
        "account": dag.cluster.account,
        "project_root": dag.cluster.project_root,
        "rfm_git_clean": True,
        "rfm_git_commit": "1" * 40,
        "bsm_git_clean": True,
        "bsm_git_commit": "2" * 40,
        "remaining_au": 900_000,
        "project_available_bytes": 10 * 1024**3,
        "project_available_inodes": 100_000,
        "scratch_available_bytes": 10_000 * 1024**3,
        "scratch_available_inodes": 1_000_000,
        "sbatch_test_only": script_hashes,
    }

    preflight = validate_hpc_preflight(dag, evidence=evidence, target_phase="pilot")

    assert preflight["status"] == "HPC_SUBMISSION_READY"
    assert preflight["remaining_au"] == 900_000
    assert preflight["required_au"] <= preflight["remaining_au"]


def test_pilot_preflight_rejects_stale_bsm_execution_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import rfm_pipeline.hpc_campaign_package as hpc

    dag = _package(tmp_path)
    selected = [stage for stage in dag.stages if stage.name in _PILOT_STAGES_FOR_TEST]
    script_hashes = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for stage in selected
        for path in (
            *stage.worker_script_paths,
            stage.audit_script_path,
            stage.reducer_script_path,
        )
    }

    def fake_git(command: list[str], **_: object) -> Namespace:
        root = str(command[2])
        if command[-2:] == ["status", "--porcelain"]:
            return Namespace(returncode=0, stdout="", stderr="")
        commit = "1" * 40 if root == dag.cluster.rfm_repository_root else "2" * 40
        return Namespace(returncode=0, stdout=f"{commit}\n", stderr="")

    real_hash_file = hpc._hash_file

    def stale_adapter_hash(path: Path) -> str:
        value = Path(path)
        if value.name == "pixi.lock":
            return dag.lock_hash
        if str(value).endswith(dag.cluster.scientific_adapter_relative_path):
            return "0" * 64
        return real_hash_file(value)

    monkeypatch.setattr(hpc.subprocess, "run", fake_git)
    monkeypatch.setattr(hpc, "_hash_python_tree", lambda _: dag.source_hash)
    monkeypatch.setattr(hpc, "_hash_file", stale_adapter_hash)
    evidence = {
        "observed_date": date.today().isoformat(),
        "account": dag.cluster.account,
        "project_root": dag.cluster.project_root,
        "rfm_git_clean": True,
        "rfm_git_commit": "1" * 40,
        "bsm_git_clean": True,
        "bsm_git_commit": "2" * 40,
        "remaining_au": 900_000,
        "project_available_bytes": 10 * 1024**3,
        "project_available_inodes": 100_000,
        "scratch_available_bytes": 10_000 * 1024**3,
        "scratch_available_inodes": 1_000_000,
        "sbatch_test_only": script_hashes,
    }

    with pytest.raises(ValueError, match="Kestrel BSM file bytes differ"):
        validate_hpc_preflight(dag, evidence=evidence, target_phase="pilot")


def test_live_smoke_collects_current_evidence_without_submitting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import rfm_pipeline.hpc_campaign_package as hpc

    dag = _package(tmp_path)
    commands: list[list[str]] = []

    def fake_read_only_command(command: list[str], **_: object) -> Namespace:
        commands.append(command)
        if command == ["aus_report"]:
            return Namespace(
                returncode=0,
                stdout="allocation bsm remaining 5183 AU\n",
                stderr="",
            )
        if command[0] == "sacctmgr":
            return Namespace(
                returncode=0,
                stdout="kestrel|nationalpfa|test-user\n",
                stderr="",
            )
        if command[0] == "git" and command[-2:] == ["status", "--porcelain"]:
            return Namespace(returncode=0, stdout="", stderr="")
        if command[0] == "git" and command[-2:] == ["rev-parse", "HEAD"]:
            commit = "1" * 40 if command[2] == dag.cluster.rfm_repository_root else "2" * 40
            return Namespace(returncode=0, stdout=f"{commit}\n", stderr="")
        if command[:2] == ["lfs", "project"]:
            return Namespace(
                returncode=0,
                stdout=f"110255 P {dag.cluster.project_root}\n",
                stderr="",
            )
        if command[:2] == ["lfs", "quota"]:
            return Namespace(
                returncode=0,
                stdout=(
                    "Disk quotas for prj 110255 (pid 110255):\n"
                    "Filesystem used quota limit grace files quota limit grace\n"
                    f"{dag.cluster.project_root}\n"
                    "1G 20G 20G - 100 100000 100000 -\n"
                ),
                stderr="",
            )
        if command[:2] == ["sbatch", "--test-only"]:
            return Namespace(returncode=0, stdout="Batch job submission test passed\n", stderr="")
        raise AssertionError(f"unexpected command: {command}")

    real_hash_file = hpc._hash_file
    bound_hashes = {
        dag.cluster.scientific_adapter_relative_path: dag.cluster.scientific_adapter_sha256,
        dag.cluster.bsm_recovery_driver_relative_path: dag.cluster.bsm_recovery_driver_sha256,
        dag.cluster.bsm_dgp_contract_relative_path: dag.cluster.bsm_dgp_contract_sha256,
        dag.cluster.applied_data_preparer_relative_path: (dag.cluster.applied_data_preparer_sha256),
    }

    def fake_hash_file(path: Path) -> str:
        value = Path(path)
        if value.name == "pixi.lock":
            return dag.lock_hash
        for suffix, expected in bound_hashes.items():
            if str(value).endswith(suffix):
                return expected
        return real_hash_file(value)

    monkeypatch.setattr(hpc, "_hash_python_tree", lambda _: dag.source_hash)
    monkeypatch.setattr(hpc, "_hash_file", fake_hash_file)
    monkeypatch.setattr(
        hpc,
        "_filesystem_availability",
        lambda _: (20_000 * 1024**3, 2_000_000),
    )

    result = run_hpc_live_smoke(
        dag,
        target_phase="pilot",
        remaining_au=25_000,
        evidence_dir=tmp_path / "live-smoke",
        run_command=fake_read_only_command,
    )

    assert result["preflight"]["status"] == "HPC_SUBMISSION_READY"
    assert (tmp_path / "live-smoke" / "preflight.json").is_file()
    assert (tmp_path / "live-smoke" / "aus_report.txt").is_file()
    assert any(command[0] == "sacctmgr" for command in commands)
    sbatch_commands = [command for command in commands if command[0] == "sbatch"]
    assert sbatch_commands
    assert all(command[1] == "--test-only" and len(command) == 3 for command in sbatch_commands)


def test_project_capacity_uses_filesystem_when_kestrel_has_no_project_quota(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rfm_pipeline.hpc_campaign_package as hpc

    commands: list[list[str]] = []

    def fake_command(command: list[str], **_: object) -> Namespace:
        commands.append(command)
        assert command == ["lfs", "project", "-d", "/projects/bsm"]
        return Namespace(returncode=0, stdout="    0 - /projects/bsm\n", stderr="")

    monkeypatch.setattr(hpc, "_filesystem_availability", lambda _: (123_456, 789))

    available_bytes, available_inodes, audit = hpc._project_quota_availability(
        "/projects/bsm",
        run_command=fake_command,
    )

    assert (available_bytes, available_inodes) == (123_456, 789)
    assert audit["quota_mode"] == "filesystem_available_no_project_quota"
    assert commands == [["lfs", "project", "-d", "/projects/bsm"]]


def test_live_smoke_rejects_unknown_allocation_before_any_probe(tmp_path: Path) -> None:
    dag = _package(tmp_path)
    dag = replace(dag, cluster=replace(dag.cluster, allocation_quota="UNKNOWN"))
    commands: list[list[str]] = []

    def must_not_run(command: list[str], **_: object) -> Namespace:
        commands.append(command)
        raise AssertionError("live probe must not run with an unresolved allocation")

    with pytest.raises(ValueError, match="numeric allocation_quota"):
        run_hpc_live_smoke(
            dag,
            target_phase="pilot",
            remaining_au=900_000,
            evidence_dir=tmp_path / "live-smoke",
            run_command=must_not_run,
        )

    assert commands == []
    assert not (tmp_path / "live-smoke").exists()


def test_post_pilot_projection_fits_the_25k_campaign_budget(tmp_path: Path) -> None:
    _, _, _, _, final = _final_package(tmp_path)

    assert final.campaign_envelope.requested_au == 10_322
    assert final.campaign_envelope.requested_au <= 25_000
    assert {"pilot_conditioning", "resolution"} <= {
        estimate.stage_name for estimate in final.campaign_envelope.stage_allocations
    }


def test_post_pilot_gate_counts_pilot_and_resolution_against_whole_campaign_cap(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="telemetry-based whole-campaign projection exceeds allocation_quota: 10322 > 10000",
    ):
        _final_package(tmp_path, quota=10_000)


def test_live_smoke_cli_builds_a_fresh_pilot_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import rfm_pipeline.hpc_campaign_package as hpc

    sentinel_dag = object()
    observed: dict[str, object] = {}

    def fake_generate(**kwargs: object) -> object:
        observed["generate"] = kwargs
        return sentinel_dag

    def fake_live_smoke(dag: object, **kwargs: object) -> dict[str, object]:
        observed["dag"] = dag
        observed["smoke"] = kwargs
        return {
            "preflight": {
                "status": "HPC_SUBMISSION_READY",
                "target_phase": "pilot",
                "preflight_sha256": "a" * 64,
                "required_au": 1315,
            }
        }

    monkeypatch.setattr(hpc, "generate_campaign_package", fake_generate)
    monkeypatch.setattr(hpc, "run_hpc_live_smoke", fake_live_smoke)

    exit_code = main(
        [
            "live-smoke",
            "--package-root",
            str(tmp_path / "package"),
            "--evidence-root",
            str(tmp_path / "evidence"),
            "--repo-root",
            str(REPO_ROOT),
            "--config",
            str(CONFIG_PATH),
            "--remaining-au",
            "900000",
        ]
    )

    assert exit_code == 0
    assert observed["dag"] is sentinel_dag
    assert observed["generate"] == {
        "output_dir": str(tmp_path / "package"),
        "repo_root": str(REPO_ROOT),
        "config_path": str(CONFIG_PATH),
        "package_mode": "full",
    }
    assert observed["smoke"] == {
        "target_phase": "pilot",
        "remaining_au": 900000,
        "evidence_dir": str(tmp_path / "evidence"),
    }


def test_preflight_scopes_confirmatory_scripts_and_data_checks_to_exact_phase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import rfm_pipeline.hpc_campaign_package as hpc

    _, freeze, _, _, dag = _final_package(tmp_path)

    def fake_git(command: list[str], **_: object) -> Namespace:
        root = str(command[2])
        if command[-2:] == ["status", "--porcelain"]:
            return Namespace(returncode=0, stdout="", stderr="")
        commit = "1" * 40 if root == dag.cluster.rfm_repository_root else "2" * 40
        return Namespace(returncode=0, stdout=f"{commit}\n", stderr="")

    real_hash_file = hpc._hash_file
    bound_hashes = {
        dag.cluster.scientific_adapter_relative_path: dag.cluster.scientific_adapter_sha256,
        dag.cluster.bsm_recovery_driver_relative_path: (dag.cluster.bsm_recovery_driver_sha256),
        dag.cluster.bsm_dgp_contract_relative_path: dag.cluster.bsm_dgp_contract_sha256,
        dag.cluster.applied_data_preparer_relative_path: (dag.cluster.applied_data_preparer_sha256),
        dag.cluster.applied_config_relative_path: dag.cluster.applied_config_sha256,
    }

    def fake_hash_file(path: Path) -> str:
        value = Path(path)
        if value.name == "pixi.lock":
            return dag.lock_hash
        for suffix, expected in bound_hashes.items():
            if str(value).endswith(suffix):
                return expected
        return real_hash_file(value)

    monkeypatch.setattr(hpc.subprocess, "run", fake_git)
    monkeypatch.setattr(hpc, "_hash_python_tree", lambda _: dag.source_hash)
    monkeypatch.setattr(hpc, "_hash_file", fake_hash_file)

    for phase, stage_predicate in (
        ("gate_b", lambda name: name in {"gate_b", "fixed_family_supplement"}),
    ):
        selected = [stage for stage in dag.stages if stage_predicate(stage.name)]
        scripts = {
            str(path): real_hash_file(path)
            for stage in selected
            for path in (
                *stage.worker_script_paths,
                stage.audit_script_path,
                stage.reducer_script_path,
            )
        }
        evidence = {
            "observed_date": date.today().isoformat(),
            "account": dag.cluster.account,
            "project_root": dag.cluster.project_root,
            "rfm_git_clean": True,
            "rfm_git_commit": "1" * 40,
            "bsm_git_clean": True,
            "bsm_git_commit": "2" * 40,
            "remaining_au": 2_000_000,
            "project_available_bytes": 10 * 1024**3,
            "project_available_inodes": 100_000,
            "scratch_available_bytes": 30_000 * 1024**3,
            "scratch_available_inodes": 1_000_000,
            "sbatch_test_only": scripts,
        }

        preflight = validate_hpc_preflight(dag, evidence=evidence, target_phase=phase)

        assert preflight["target_phase"] == phase
        assert preflight["resource_freeze_sha256"] == freeze["resource_freeze_sha256"]

    gate_p_stages = [stage for stage in dag.stages if stage.name.startswith("applied_")]
    gate_p_scripts = {
        str(path): real_hash_file(path)
        for stage in gate_p_stages
        for path in (
            *stage.worker_script_paths,
            stage.audit_script_path,
            stage.reducer_script_path,
        )
    }
    gate_p_evidence = {**evidence, "sbatch_test_only": gate_p_scripts}
    with pytest.raises(ValueError, match="dataset_manifest|applied-data manifest"):
        validate_hpc_preflight(dag, evidence=gate_p_evidence, target_phase="gate_p")


def test_phase_authorization_rejects_stale_inventory_and_unbound_prerequisites(
    tmp_path: Path,
) -> None:
    _, freeze, freeze_path, _, dag = _final_package(tmp_path)
    preflight = _phase_preflight(dag, "gate_p", str(freeze["resource_freeze_sha256"]))
    preflight["campaign_inventory_hash"] = "0" * 64
    preflight_identity = {
        key: value for key, value in preflight.items() if key != "preflight_sha256"
    }
    canonical = json.dumps(preflight_identity, sort_keys=True, separators=(",", ":"))
    preflight["preflight_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    preflight_path = tmp_path / "preflight.json"
    preflight_path.write_text(json.dumps(preflight), encoding="utf-8")

    with pytest.raises(ValueError, match="preflight identity"):
        write_phase_authorization(
            dag,
            phase="gate_p",
            resource_freeze_path=freeze_path,
            preflight_path=preflight_path,
            prerequisite_paths=(),
            output_path=tmp_path / "authorization-stale.json",
        )

    preflight = _phase_preflight(dag, "gate_p", str(freeze["resource_freeze_sha256"]))
    preflight_path.write_text(json.dumps(preflight), encoding="utf-8")
    forged_gate_b = tmp_path / "forged_gate_b.json"
    forged_gate_b.write_text(
        json.dumps(
            {
                "operation": "gate_b",
                "status": "completed",
                "decision": "PASS",
                "contract_hash": dag.config_hash,
            }
        ),
        encoding="utf-8",
    )
    forged_fixed = tmp_path / "forged_fixed.json"
    forged_fixed.write_text(
        json.dumps(
            {
                "operation": "fixed_family_supplement",
                "status": "completed",
                "decision": "PASS",
                "contract_hash": dag.config_hash,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="reducer output"):
        write_phase_authorization(
            dag,
            phase="gate_p",
            resource_freeze_path=freeze_path,
            preflight_path=preflight_path,
            prerequisite_paths=(forged_gate_b, forged_fixed),
            output_path=tmp_path / "authorization-forged.json",
        )


def test_generated_scripts_have_zero_unresolved_placeholders(tmp_path: Path) -> None:
    dag = _package(tmp_path)
    forbidden_tokens = ("{TODO", "TBD", "<PLACEHOLDER>", "{user}", "{run_id}")

    for stage in dag.stages:
        for script_path in (stage.worker_script_path, stage.reducer_script_path):
            text = script_path.read_text(encoding="utf-8")
            assert all(token not in text for token in forbidden_tokens)


def test_reducer_arguments_include_contract_config_hash_and_expected_ranges(
    tmp_path: Path,
) -> None:
    dag = _package(tmp_path)
    recovery = _stage(dag, "recovery")
    reducer_text = recovery.reducer_script_path.read_text(encoding="utf-8")

    assert f'--contract-config "{dag.contract_config_path}"' in reducer_text
    assert f'--contract-hash "{dag.config_hash}"' in reducer_text
    assert "--expected-range-start 0" in reducer_text
    assert f'--expected-range-end "{recovery.job_count}"' in reducer_text


def test_campaign_contains_complete_prespecified_stage_dag(tmp_path: Path) -> None:
    dag = _package(tmp_path)
    names = tuple(stage.name for stage in dag.stages)
    assert names == (
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
        "resolution",
        "gate_b",
        "fixed_family_supplement",
        "applied_conditioning",
        "applied_screening",
        "applied_interaction",
        "applied_nonlinear",
        "applied_sparse_full",
        "applied_sparse_resample",
        "applied_terminal_fit",
        "applied_ablation_fit",
        "applied_holdout_authorize",
        "applied_holdout_predict",
        "applied_eligibility",
        "applied_bootstrap",
        "recovery",
    )


def test_one_campaign_table_proves_exact_worker_reducer_and_scientific_coverage(
    tmp_path: Path,
) -> None:
    dag = _package(tmp_path)
    rows = [
        json.loads(line)
        for line in dag.campaign_inventory_path.read_text(encoding="utf-8").splitlines()
    ]

    assert _HEX64.fullmatch(dag.campaign_inventory_hash)
    assert hashlib.sha256(dag.campaign_inventory_path.read_bytes()).hexdigest() == (
        dag.campaign_inventory_hash
    )
    assert sum(row["record_type"] == "reducer" for row in rows) == len(dag.stages)
    scientific = [
        row
        for row in rows
        if row["record_type"] == "worker" and row["stage"] in {"gate_b", "recovery"}
    ]
    assert len(scientific) == 6400
    assert sum(row["scenario_kind"] == "null" for row in scientific) == 5000
    assert sum(row["scenario_kind"] in {"strong", "stress"} for row in scientific) == 1400
    assert (
        sum(row["stage"] == "gate_b" and row["scenario_kind"] == "strong" for row in scientific)
        == 600
    )
    assert (
        sum(row["stage"] == "recovery" and row["scenario_kind"] == "stress" for row in scientific)
        == 800
    )
    comparator_population = [
        row for row in scientific if row["scenario_kind"] in {"strong", "stress"}
    ]
    calibration_population = [row for row in scientific if row["scenario_kind"] == "null"]
    assert len(comparator_population) == 1400
    assert len(calibration_population) == 5000
    assert all(
        row["recovery_comparators"]
        == [
            "proposed_terminal_workflow",
            "oracle_ols",
            "elastic_net_algebraic_library",
            "raw_input_boosted_tree",
        ]
        for row in comparator_population
    )
    assert all(row["recovery_comparators"] == [] for row in calibration_population)
    assert all(
        "comparator_predictions.npz" in row["expected_artifacts"] for row in comparator_population
    )
    assert all(
        "comparator_predictions.npz" not in row["expected_artifacts"]
        for row in calibration_population
    )
    fixed_family = [
        row
        for row in rows
        if row["record_type"] == "worker" and row["stage"] == "fixed_family_supplement"
    ]
    assert len({row["family_order_sha256"] for row in fixed_family}) == 1
    fixed_order_path = Path(fixed_family[0]["family_order_path"])
    assert fixed_order_path.is_file()
    assert (
        hashlib.sha256(fixed_order_path.read_bytes()).hexdigest()
        == fixed_family[0]["family_order_file_sha256"]
    )
    validate_campaign_table(dag)


def test_campaign_table_validator_rejects_missing_duplicate_and_zero_width_units(
    tmp_path: Path,
) -> None:
    dag = _package(tmp_path)
    rows = [
        json.loads(line)
        for line in dag.campaign_inventory_path.read_text(encoding="utf-8").splitlines()
    ]

    with pytest.raises(ValueError, match="inventory row count"):
        validate_campaign_table(dag, rows=rows[:-1])
    with pytest.raises(ValueError, match="duplicate analysis unit"):
        validate_campaign_table(dag, rows=[*rows, rows[0]])

    mutated = [dict(row) for row in rows]
    block_index = next(
        index
        for index, row in enumerate(mutated)
        if row.get("record_type") == "worker" and "block_start" in row
    )
    mutated[block_index]["block_end"] = mutated[block_index]["block_start"]
    with pytest.raises(ValueError, match="zero-width"):
        validate_campaign_table(dag, rows=mutated)


def test_telemetry_schema_supports_resource_selection(tmp_path: Path) -> None:
    dag = _package(tmp_path)
    names = {field["name"] for field in dag.telemetry_schema}
    assert {
        "elapsed_seconds",
        "total_cpu_seconds",
        "cpu_time_seconds",
        "max_rss_bytes",
        "bytes_read",
        "bytes_written",
        "task_size",
        "block_size",
    }.issubset(names)


def test_worker_writes_content_verified_result_and_success_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dag = _package(tmp_path)
    stage = _stage(dag, "scheduler_diagnostic")
    record = load_stage_manifest(stage.manifest_path)[0]
    monkeypatch.setenv("SLURM_JOB_ID", "123456")
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "123450")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "6")
    result = _cli_worker(
        Namespace(
            manifest=stage.manifest_path,
            stage=stage.name,
            task_id=0,
            output_root=str(stage.output_root),
            scratch_root=str(tmp_path / "scratch"),
            contract_config=str(dag.contract_config_path),
            contract_hash=dag.config_hash,
            expected_range_start=0,
            expected_range_end=stage.job_count,
        )
    )
    assert result == 0
    shard_dir = Path(record["output_dir"])
    assert (shard_dir / "result.json").is_file()
    assert (shard_dir / "_SUCCESS.json").is_file()
    telemetry = json.loads((shard_dir / "result.json").read_text(encoding="utf-8"))
    assert telemetry["status"] == "completed"
    assert telemetry["attempt"] == 1
    assert telemetry["source_hash"] == dag.source_hash
    assert telemetry["config_hash"] == dag.config_hash
    assert telemetry["lock_hash"] == dag.lock_hash
    assert telemetry["parent_hash"] == record["parent_hash"]
    assert telemetry["output_hash"] == record["output_hash"]
    assert telemetry["slurm_job_id"] == "123456"
    assert telemetry["slurm_array_job_id"] == "123450"
    assert telemetry["slurm_array_task_id"] == "6"
    assert telemetry["hostname"]
    assert validate_resume_artifacts(stage)["completed"] == 1


def test_screening_pilot_executes_and_persists_the_stage_native_kernel(tmp_path: Path) -> None:
    record = {
        "stage": "pilot_screening",
        "operation": "pilot_screening",
        "schedule_hash": "1" * 64,
        "config_hash": "2" * 64,
        "block_size": 2,
    }
    result = _execute_worker_operation(record, shard_dir=tmp_path)
    assert result["kernel"] == "score_screening_draw_block"
    assert result["draw_start"] == 0
    assert result["draw_end"] == 2
    assert (tmp_path / "screening_block" / "screening_block.npz").is_file()


def test_interaction_score_pilot_reports_canonical_payload_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The worker must not reference a nonexistent compatibility checksum."""
    import rfm_pipeline.hpc_campaign_package as campaign
    import rfm_pipeline.interaction_contract as interaction_contract
    import rfm_pipeline.manuscript_stages as manuscript_stages

    payload_sha256 = "a" * 64

    class PayloadOnlyArtifact:
        pair_names = ("x000:x001",)

        def write_to(self, directory: Path) -> None:
            directory.mkdir(parents=True)

        @property
        def payload_sha256(self) -> str:
            return payload_sha256

    monkeypatch.setattr(
        campaign,
        "_pilot_training_tables",
        lambda record: (None, None, None, None),
    )
    monkeypatch.setattr(
        campaign,
        "_pilot_interaction_spec",
        lambda record: SimpleNamespace(n_tree_estimators=250),
    )
    monkeypatch.setattr(
        interaction_contract,
        "canonical_execution_contract_from_specs",
        lambda spec: object(),
    )
    monkeypatch.setattr(
        manuscript_stages,
        "score_interaction_draw_block",
        lambda *args, **kwargs: PayloadOnlyArtifact(),
    )
    record = {
        "stage": "pilot_interaction_score",
        "operation": "pilot_interaction_score",
        "schedule_hash": "1" * 64,
        "config_hash": "2" * 64,
        "block_size": 2,
    }

    result = _execute_worker_operation(record, shard_dir=tmp_path)

    assert result["artifact_checksum"] == payload_sha256
    json.dumps(result, allow_nan=False)


def test_resume_rejects_marker_that_does_not_hash_result_bytes(tmp_path: Path) -> None:
    dag = _package(tmp_path)
    stage = _stage(dag, "scheduler_diagnostic")
    record = load_stage_manifest(stage.manifest_path)[0]
    shard_dir = Path(record["output_dir"])
    shard_dir.mkdir(parents=True)
    (shard_dir / "result.json").write_text('{"tampered": true}\n', encoding="utf-8")
    (shard_dir / "_SUCCESS.json").write_text(
        json.dumps(
            {
                "stage": stage.name,
                "shard_id": record["shard_id"],
                "output_hash": record["output_hash"],
                "parent_hash": record["parent_hash"],
                "attempt": 1,
                "status": "completed",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="artifact"):
        validate_resume_artifacts(stage)


def test_reducer_requires_and_hashes_every_worker_artifact(tmp_path: Path) -> None:
    dag = _package(tmp_path)
    stage = _stage(dag, "scheduler_diagnostic")
    worker_args = Namespace(
        manifest=stage.manifest_path,
        stage=stage.name,
        task_id=0,
        output_root=str(stage.output_root),
        scratch_root=str(tmp_path / "scratch"),
        contract_config=str(dag.contract_config_path),
        contract_hash=dag.config_hash,
        expected_range_start=0,
        expected_range_end=1,
    )
    assert _cli_worker(worker_args) == 0
    reduce_args = Namespace(
        manifest=stage.manifest_path,
        stage=stage.name,
        output_root=str(stage.reducer_output_dir),
        contract_config=str(dag.contract_config_path),
        contract_hash=dag.config_hash,
        expected_range_start=0,
        expected_range_end=1,
        expected_parent_hash=load_stage_manifest(stage.manifest_path)[0]["parent_hash"],
        audit_success=str(stage.audit_output_dir / "_SUCCESS.json"),
    )
    with pytest.raises(ValueError, match="audit"):
        _cli_reduce(reduce_args)
    assert (
        _cli_audit(
            Namespace(
                manifest=stage.manifest_path,
                stage=stage.name,
                output_root=str(stage.audit_output_dir),
                contract_hash=dag.config_hash,
            )
        )
        == 0
    )
    assert _cli_reduce(reduce_args) == 0
    assert (stage.reducer_output_dir / "reduced_result.json").is_file()
    assert (stage.reducer_output_dir / "_SUCCESS.json").is_file()


def test_production_worker_dispatches_only_through_content_hashed_adapter(
    tmp_path: Path,
) -> None:
    adapter = tmp_path / "adapter.py"
    adapter.write_text(
        "import hashlib\n"
        "def execute_scientific_work_unit(record, shard_dir):\n"
        "    shard_dir.mkdir(parents=True, exist_ok=True)\n"
        "    path = shard_dir / 'terminal_record.json'\n"
        "    path.write_text('{}')\n"
        "    artifact = {'path': str(path), 'relative_path': path.name, "
        "'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}\n"
        "    return {'operation': record['operation'], 'status': 'completed', "
        "'scientific_artifacts': [artifact]}\n",
        encoding="utf-8",
    )
    record = {
        "stage": "gate_b",
        "operation": "gate_b",
        "scientific_adapter_path": str(adapter),
        "scientific_adapter_sha256": hashlib.sha256(adapter.read_bytes()).hexdigest(),
        "resource_freeze_sha256": "1" * 64,
    }

    result = _execute_worker_operation(record, shard_dir=tmp_path / "result")
    assert result["operation"] == "gate_b"
    assert result["status"] == "completed"
    assert len(result["scientific_artifacts"]) == 1
    with pytest.raises(ValueError, match="adapter hash"):
        _execute_worker_operation(
            {**record, "scientific_adapter_sha256": "0" * 64},
            shard_dir=tmp_path / "tampered",
        )


def test_production_reducer_dispatches_through_the_same_hashed_adapter(
    tmp_path: Path,
) -> None:
    adapter = tmp_path / "adapter.py"
    adapter.write_text(
        "import hashlib\n"
        "def reduce_scientific_stage(records, output_dir):\n"
        "    output_dir.mkdir(parents=True, exist_ok=True)\n"
        "    path = output_dir / 'decision.json'\n"
        "    path.write_text('{}')\n"
        "    artifact = {'path': str(path), 'relative_path': path.name, "
        "'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}\n"
        "    return {'operation': records[0]['operation'], 'status': 'completed', "
        "'scientific_artifacts': [artifact]}\n",
        encoding="utf-8",
    )
    record = {
        "operation": "resolution",
        "scientific_adapter_path": str(adapter),
        "scientific_adapter_sha256": hashlib.sha256(adapter.read_bytes()).hexdigest(),
    }
    result = _execute_scientific_reduce([record], output_dir=tmp_path / "reduced")
    assert result["operation"] == "resolution"
    assert result["status"] == "completed"
    assert len(result["scientific_artifacts"]) == 1


def test_every_production_record_binds_adapter_dgp_data_and_phase_authorization(
    tmp_path: Path,
) -> None:
    dag = _package(tmp_path)
    production = [stage for stage in dag.stages if not stage.name.startswith("pilot_")]
    production = [stage for stage in production if stage.name != "scheduler_diagnostic"]

    for stage in production:
        for record in load_stage_manifest(stage.manifest_path):
            assert record["scientific_adapter_path"].endswith("g11_campaign_adapter.py")
            assert _HEX64.fullmatch(record["scientific_adapter_sha256"])
            assert record["bsm_recovery_driver_path"].endswith("run_bsm_recovery_study.py")
            assert _HEX64.fullmatch(record["bsm_recovery_driver_sha256"])
            assert record["bsm_dgp_contract_path"].endswith("bsm_dgp_contract.yaml")
            assert _HEX64.fullmatch(record["bsm_dgp_contract_sha256"])
            assert record["applied_data_preparer_path"].endswith("prepare_g11_applied_data.py")
            assert _HEX64.fullmatch(record["applied_data_preparer_sha256"])
            assert record["execution_authorization_path"].endswith(
                f"/{dag.run_id}/{record['phase']}.json"
            )
            assert record["rfm_repository_root"] == dag.cluster.rfm_repository_root
            assert record["bsm_repository_root"] == dag.cluster.bsm_repository_root


def test_worker_scripts_enter_the_pinned_rfm_checkout(tmp_path: Path) -> None:
    dag = _package(tmp_path)
    for stage in dag.stages:
        for script_path in stage.worker_script_paths:
            text = script_path.read_text(encoding="utf-8")
            assert f'cd "{dag.cluster.rfm_repository_root}"' in text


def test_retry_planner_allows_one_identical_attempt_only_for_infrastructure_states() -> None:
    record = {
        "stage": "gate_b",
        "shard_id": "task-0000",
        "attempt": 0,
        "max_retries": 1,
        "input_hash": "1" * 64,
        "schedule_hash": "2" * 64,
        "output_hash": "3" * 64,
        "retryable_scheduler_states": ["BOOT_FAIL", "NODE_FAIL", "PREEMPTED"],
    }

    retries = plan_retry_attempts([record], {"task-0000": "NODE_FAIL"})
    assert len(retries) == 1
    assert retries[0]["attempt"] == 1
    assert retries[0]["input_hash"] == record["input_hash"]
    assert retries[0]["schedule_hash"] == record["schedule_hash"]
    assert retries[0]["output_hash"] == record["output_hash"]

    assert plan_retry_attempts([record], {"task-0000": "FAILED"}) == []
    assert plan_retry_attempts([{**record, "attempt": 1}], {"task-0000": "NODE_FAIL"}) == []


def test_gate_b_retry_preserves_strong_resource_class_dispatch_range(
    tmp_path: Path,
) -> None:
    _, _, _, _, final = _final_package(tmp_path)
    payload = prepare_stage_retry_package(
        final,
        stage_name="gate_b",
        scheduler_states={"task-5000": "NODE_FAIL"},
        output_dir=tmp_path / "retry",
    )

    script = Path(payload["retry_scripts"][0]).read_text(encoding="utf-8")
    assert '--task-id "5000"' in script
    assert '--expected-range-start "5000"' in script
    assert '--expected-range-end "5600"' in script


def test_retry_package_rebuilds_exact_coverage_audit_and_reducer_chain(
    tmp_path: Path,
) -> None:
    _, _, _, _, final = _final_package(tmp_path)
    payload = prepare_stage_retry_package(
        final,
        stage_name="gate_b",
        scheduler_states={"task-5000": "NODE_FAIL"},
        output_dir=tmp_path / "retry",
    )

    steps = payload["submission_steps"]
    workers = [step for step in steps if step["action"] == "worker"]
    audit = next(step for step in steps if step["action"] == "audit")
    reducer = next(step for step in steps if step["action"] == "reduce")

    assert workers
    assert audit["dependency_type"] == "afterany"
    assert audit["depends_on"] == [step["step_id"] for step in workers]
    assert reducer["dependency_type"] == "afterok"
    assert reducer["depends_on"] == [
        *[step["step_id"] for step in workers],
        audit["step_id"],
    ]
    assert str(tmp_path / "retry" / "audit") in Path(payload["audit_script"]).read_text(
        encoding="utf-8"
    )
    assert payload["submission_script_sha256"] == {
        str(Path(step["command"][-1])): hashlib.sha256(
            Path(step["command"][-1]).read_bytes()
        ).hexdigest()
        for step in steps
    }


def test_retry_submission_runs_test_only_before_rebuilt_dependency_chain(
    tmp_path: Path,
) -> None:
    _, freeze, _, _, final = _final_package(tmp_path)
    payload = prepare_stage_retry_package(
        final,
        stage_name="gate_b",
        scheduler_states={"task-5000": "NODE_FAIL"},
        output_dir=tmp_path / "retry",
    )
    preflight = _phase_preflight(final, "gate_b", str(freeze["resource_freeze_sha256"]))
    authorization = _signed_authorization(
        final,
        "gate_b",
        str(freeze["resource_freeze_sha256"]),
        preflight_sha256=str(preflight["preflight_sha256"]),
    )
    calls: list[list[str]] = []

    def success(command: list[str]) -> Namespace:
        calls.append(command)
        if command[:2] == ["sbatch", "--test-only"]:
            return Namespace(returncode=0, stdout="test accepted\n", stderr="")
        return Namespace(returncode=0, stdout=f"{1000 + len(calls)}\n", stderr="")

    job_ids = execute_retry_submission_plan(
        payload,
        run_command=success,
        authorize=True,
        preflight=preflight,
        phase_authorization=authorization,
    )

    script_count = len(payload["submission_script_sha256"])
    assert all(command[:2] == ["sbatch", "--test-only"] for command in calls[:script_count])
    submitted = calls[script_count:]
    assert len(submitted) == len(payload["submission_steps"])
    audit_index = next(
        index for index, step in enumerate(payload["submission_steps"]) if step["action"] == "audit"
    )
    audit = payload["submission_steps"][audit_index]
    dependency_ids = [job_ids[step_id] for step_id in audit["depends_on"]]
    assert f"--dependency=afterany:{':'.join(dependency_ids)}" in submitted[audit_index]


def test_resolution_selected_draw_count_regenerates_contract_manifests_and_fresh_seeds(
    tmp_path: Path,
) -> None:
    base = _package(tmp_path / "base")
    selected_contract = replace(G11_CONTRACT, B_interaction=1998)
    selected = generate_campaign_package(
        output_dir=tmp_path / "selected" / "package",
        repo_root=REPO_ROOT,
        config_path=_config_with_quota(tmp_path / "selected", 2_000_000),
        contract=selected_contract,
    )

    loaded, loaded_hash = load_contract(selected.contract_config_path)
    assert loaded.B_interaction == 1998
    assert loaded_hash == selected.config_hash != base.config_hash
    base_gate_b = load_stage_manifest(_stage(base, "gate_b").manifest_path)
    selected_gate_b = load_stage_manifest(_stage(selected, "gate_b").manifest_path)
    assert [row["seed"] for row in base_gate_b] != [row["seed"] for row in selected_gate_b]
    assert _stage(selected, "applied_interaction").job_count == math.ceil(1998 / 50)


def test_post_resolution_finalizer_excludes_development_and_refreshes_seeds(
    tmp_path: Path,
) -> None:
    base = _package(tmp_path / "base")
    freeze = select_pilot_resources(
        pilot_matrix=base.pilot_matrix,
        telemetry=_accepted_pilot_telemetry(base),
        cluster=base.cluster,
        source_hash=base.source_hash,
        config_hash=base.config_hash,
        lock_hash=base.lock_hash,
    )
    freeze_path = tmp_path / "resource_freeze.json"
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    decision = {
        "operation": "resolution",
        "status": "completed",
        "decision": "ACCEPTED",
        "contract_hash": base.config_hash,
        "terminal_record_count": 20,
        "selected_B_interaction": 999,
    }
    decision_path = tmp_path / "resolution_decision.json"
    decision_path.write_text(json.dumps(decision, sort_keys=True), encoding="utf-8")

    final = finalize_post_resolution_package(
        output_dir=tmp_path / "final" / "package",
        resource_freeze_path=freeze_path,
        resolution_decision_path=decision_path,
        repo_root=REPO_ROOT,
        config_path=_config_with_quota(tmp_path / "final", 2_000_000),
    )

    names = {stage.name for stage in final.stages}
    assert final.package_mode == "confirmatory"
    assert "resolution" not in names
    assert not names.intersection(_PILOT_STAGES_FOR_TEST)
    assert _stage(final, "gate_b").parent_stage_name is None
    assert _stage(final, "applied_sparse_resample").job_count == 10
    base_seeds = [row["seed"] for row in load_stage_manifest(_stage(base, "gate_b").manifest_path)]
    final_records = load_stage_manifest(_stage(final, "gate_b").manifest_path)
    assert [row["seed"] for row in final_records] != base_seeds
    assert {row["resource_freeze_sha256"] for row in final_records} == {
        freeze["resource_freeze_sha256"]
    }


def test_selected_double_interaction_schedule_scales_simulation_walltime(
    tmp_path: Path,
) -> None:
    base = _package(tmp_path / "base")
    freeze = select_pilot_resources(
        pilot_matrix=base.pilot_matrix,
        telemetry=_accepted_pilot_telemetry(base),
        cluster=base.cluster,
        source_hash=base.source_hash,
        config_hash=base.config_hash,
        lock_hash=base.lock_hash,
    )
    freeze_path = tmp_path / "resource_freeze.json"
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    decision = {
        "operation": "resolution",
        "status": "completed",
        "decision": "ACCEPTED",
        "contract_hash": base.config_hash,
        "terminal_record_count": 20,
        "selected_B_interaction": 1998,
    }
    decision_path = tmp_path / "resolution_decision.json"
    decision_path.write_text(json.dumps(decision, sort_keys=True), encoding="utf-8")

    final = finalize_post_resolution_package(
        output_dir=tmp_path / "final" / "package",
        resource_freeze_path=freeze_path,
        resolution_decision_path=decision_path,
        repo_root=REPO_ROOT,
        config_path=_config_with_quota(tmp_path / "final", 2_000_000),
    )

    gate_b_records = load_stage_manifest(_stage(final, "gate_b").manifest_path)
    null_record = next(row for row in gate_b_records if row["scenario_kind"] == "null")
    strong_record = next(row for row in gate_b_records if row["scenario_kind"] == "strong")
    measured_null = int(freeze["selections"]["pilot_gate_b_null"]["requested_walltime_seconds"])
    measured_strong = int(freeze["selections"]["pilot_recovery"]["requested_walltime_seconds"])
    assert null_record["worker_resources"]["requested_walltime_seconds"] == 2 * measured_null
    assert strong_record["worker_resources"]["requested_walltime_seconds"] == 2 * measured_strong
    assert _stage(final, "recovery").worker_resources.requested_walltime_seconds == (
        2 * measured_strong
    )


def test_gate_b_uses_separate_pilot_frozen_null_and_strong_resource_arrays(
    tmp_path: Path,
) -> None:
    _, freeze, _, _, final = _final_package(tmp_path)
    gate_b = _stage(final, "gate_b")
    records = load_stage_manifest(gate_b.manifest_path)
    null_records = [row for row in records if row["scenario_kind"] == "null"]
    strong_records = [row for row in records if row["scenario_kind"] == "strong"]

    assert len(null_records) == 5000
    assert len(strong_records) == 600
    assert len(gate_b.worker_script_paths) == 2
    assert {row["worker_resources"]["requested_cpu_cores"] for row in null_records} == {
        freeze["selections"]["pilot_gate_b_null"]["requested_cpus"]
    }
    assert {row["worker_resources"]["requested_cpu_cores"] for row in strong_records} == {
        freeze["selections"]["pilot_recovery"]["requested_cpus"]
    }
    null_script, strong_script = (
        path.read_text(encoding="utf-8") for path in gate_b.worker_script_paths
    )
    assert "#SBATCH --array=0-4999%64" in null_script
    assert '--expected-range-start "0"' in null_script
    assert '--expected-range-end "5000"' in null_script
    assert "#SBATCH --array=0-599%64" in strong_script
    assert "$((SLURM_ARRAY_TASK_ID + 5000))" in strong_script
    assert '--expected-range-start "5000"' in strong_script
    assert '--expected-range-end "5600"' in strong_script
