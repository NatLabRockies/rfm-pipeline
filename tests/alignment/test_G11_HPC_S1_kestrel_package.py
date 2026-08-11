"""G11-HPC-S1: content-addressed NO-SUBMIT Kestrel HPC package."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest

from rfm_pipeline.hpc_campaign_package import (
    generate_campaign_package,
    load_stage_manifest,
    validate_resume_artifacts,
    validate_stage_dependencies,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "hpc" / "g11_kestrel_campaign.yml"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _stage(dag, name: str):  # noqa: ANN001
    return next(stage for stage in dag.stages if stage.name == name)


def _config_with_quota(tmp_path: Path, quota: int) -> Path:
    config_path = tmp_path / "g11_kestrel_campaign.yml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        CONFIG_PATH.read_text(encoding="utf-8").replace(
            'allocation_quota: "16000"',
            f'allocation_quota: "{quota}"',
        ),
        encoding="utf-8",
    )
    return config_path


def _package(tmp_path: Path):
    return generate_campaign_package(
        output_dir=tmp_path / "package",
        repo_root=REPO_ROOT,
        config_path=_config_with_quota(tmp_path, 20_000),
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
    (shard_dir / "worker_result.json").write_text("{}", encoding="utf-8")

    pending_summary = validate_resume_artifacts(resolution)
    assert pending_summary["completed"] == 0
    assert pending_summary["pending"] == resolution.job_count

    (shard_dir / "_SUCCESS.json").write_text(
        json.dumps(
            {
                "stage": record["stage"],
                "shard_id": record["shard_id"],
                "output_hash": record["output_hash"],
                "parent_hash": record["parent_hash"],
                "attempt": 1,
                "status": "completed",
            }
        ),
        encoding="utf-8",
    )

    resumed_summary = validate_resume_artifacts(resolution)
    assert resumed_summary["completed"] == 1
    assert resumed_summary["pending"] == resolution.job_count - 1


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
    """The package must expose a conservative whole-campaign capacity envelope."""
    dag = _package(tmp_path)
    envelope = dag.campaign_envelope

    assert envelope.estimated_au_node_hours > 0
    assert envelope.estimated_storage_gb > 0
    assert envelope.estimated_inode_count > 0
    assert envelope.estimated_retry_attempts >= 0
    assert envelope.requested_au_node_hours >= math.ceil(envelope.estimated_au_node_hours * 1.25)
    assert envelope.requested_storage_gb >= math.ceil(envelope.estimated_storage_gb * 1.25)
    assert envelope.requested_inode_count >= math.ceil(envelope.estimated_inode_count * 1.25)
    assert envelope.requested_retry_attempts >= math.ceil(envelope.estimated_retry_attempts * 1.25)
    assert envelope.allocation_quota_readiness_blocker is False
    assert int(dag.cluster.allocation_quota) >= envelope.requested_au_node_hours


def test_campaign_envelope_charges_allocation_node_hours(tmp_path: Path) -> None:
    """Allocation AUs are Kestrel node-hours, not CPU-core-hours."""
    dag = _package(tmp_path)
    envelope = dag.campaign_envelope

    assert dag.cluster.allocation_unit == "node_hour"
    assert envelope.estimated_au_node_hours == pytest.approx(12_121.25)
    assert envelope.requested_au_node_hours == 15_152
    assert {
        estimate.stage_name: estimate.requested_au_node_hours
        for estimate in envelope.stage_au_node_hours
    } == {
        "resolution": pytest.approx(6.354_166_666_666_667),
        "gate_b": pytest.approx(12_500.625),
        "recovery": pytest.approx(2_625.625),
        "fixed_family_supplement": pytest.approx(18.958_333_333_333_332),
    }


def test_configured_allocation_quota_accepts_node_hour_envelope(tmp_path: Path) -> None:
    dag = generate_campaign_package(
        output_dir=tmp_path / "package",
        repo_root=REPO_ROOT,
        config_path=CONFIG_PATH,
    )

    assert dag.cluster.allocation_quota == "16000"
    assert dag.campaign_envelope.requested_au_node_hours == 15_152


def test_campaign_rejects_quota_below_node_hour_envelope(tmp_path: Path) -> None:
    with pytest.raises(
        ValueError,
        match=r"allocation_quota is below the conservative requested AU envelope: 15151 < 15152",
    ):
        generate_campaign_package(
            output_dir=tmp_path / "package",
            repo_root=REPO_ROOT,
            config_path=_config_with_quota(tmp_path, 15_151),
        )


def test_scripts_bind_kestrel_account_partition_walltime_without_sbatch_execution(
    tmp_path: Path,
) -> None:
    dag = _package(tmp_path)

    for stage in dag.stages:
        for script_path in (stage.worker_script_path, stage.reducer_script_path):
            text = script_path.read_text(encoding="utf-8")
            assert "#SBATCH --account=bsm" in text
            assert f"#SBATCH --partition={stage.partition}" in text
            assert re.search(r"^#SBATCH --time=\d{2}:\d{2}:\d{2}$", text, flags=re.MULTILINE)
            assert not re.search(r"^\s*sbatch\b", text, flags=re.MULTILINE)


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
    resolution_rows = [row for row in dag.pilot_matrix if row["stage"] == "resolution"]
    fixed_family_rows = [
        row for row in dag.pilot_matrix if row["stage"] == "fixed_family_supplement"
    ]

    assert len(resolution_rows) == 20
    assert {row["fixture"] for row in resolution_rows} == {"B", "2B"}
    assert {row["schedule_index"] for row in resolution_rows} == set(range(10))
    assert {row["family_size"] for row in fixed_family_rows} == {10, 100, 1000}
    assert dag.post_pilot_selection == rerendered.post_pilot_selection


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
