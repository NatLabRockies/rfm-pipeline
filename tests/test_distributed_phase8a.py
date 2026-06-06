"""Tests for rfm_pipeline.distributed — Phase 8a: SLURM array baseline.

Covers:
- Config loading and validation
- ShardManifest JSONL round-trip
- CheckpointManager: mark_running, promote, is_complete, skip behavior
- SlurmArrayRunner: script generation, write_scripts
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rfm_pipeline.distributed.checkpoint import CheckpointManager
from rfm_pipeline.distributed.config_distributed import (
    DistributedConfig,
    KestrelConfig,
    SlurmConfig,
    SpillConfig,
    load_distributed_config,
)
from rfm_pipeline.distributed.manifest import (
    build_manifest,
    load_manifest,
    manifest_summary,
    save_manifest,
    update_shard_status,
)
from rfm_pipeline.distributed.slurm_array_runner import SlurmArrayRunner

# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestDistributedConfig:
    def test_defaults_are_valid(self):
        cfg = DistributedConfig()
        errors = cfg.validate()
        assert errors == [], f"Default config has validation errors: {errors}"
        assert cfg.run_id == "rfm_run"
        assert cfg.pixi_env_path == "/projects/rfm/.pixi"
        assert cfg.slurm.account == "rfm"
        assert cfg.slurm.log_dir == "/scratch/${USER}/rfm/${RUN_ID}/logs"
        assert cfg.kestrel.projects_root == "/projects/rfm"
        assert cfg.kestrel.pixi_cache_dir == "/projects/rfm/.cache/pixi"

    def test_invalid_backend(self):
        cfg = DistributedConfig(backend="not_a_backend")
        errors = cfg.validate()
        assert any("backend" in e for e in errors)

    def test_empty_run_id_invalid(self):
        cfg = DistributedConfig(run_id="")
        errors = cfg.validate()
        assert any("run_id" in e for e in errors)

    def test_negative_memory_invalid(self):
        cfg = DistributedConfig(slurm=SlurmConfig(memory_gb=-1))
        errors = cfg.validate()
        assert any("memory_gb" in e for e in errors)

    def test_load_from_yaml(self, tmp_path):
        yaml_content = """
distributed:
  enabled: true
  backend: slurm_array
  run_id: test_run
  pixi_env_path: /projects/bsm/.pixi
  slurm:
    account: bsm
    partition: debug
    walltime: "00:30:00"
    memory_gb: 16
    cpus_per_task: 4
    max_concurrent_array_tasks: 2
  spill:
    min_free_gb: 5.0
"""
        config_path = tmp_path / "test_config.yml"
        config_path.write_text(yaml_content)

        cfg = load_distributed_config(config_path)
        assert cfg.enabled is True
        assert cfg.run_id == "test_run"
        assert cfg.slurm.account == "bsm"
        assert cfg.slurm.partition == "debug"
        assert cfg.slurm.memory_gb == 16
        assert cfg.spill.min_free_gb == 5.0

    def test_load_standalone_yaml(self, tmp_path):
        """Standalone YAML (no 'distributed' top-level key)."""
        yaml_content = """
enabled: true
backend: slurm_array
run_id: standalone_run
"""
        config_path = tmp_path / "standalone.yml"
        config_path.write_text(yaml_content)
        cfg = load_distributed_config(config_path)
        assert cfg.run_id == "standalone_run"

    def test_unknown_keys_ignored(self, tmp_path):
        yaml_content = """
distributed:
  enabled: true
  run_id: test
  unknown_field: should_be_ignored
"""
        config_path = tmp_path / "config.yml"
        config_path.write_text(yaml_content)
        cfg = load_distributed_config(config_path)
        assert cfg.run_id == "test"

    def test_spill_resolve_scratch(self, monkeypatch):
        monkeypatch.setenv("USER", "testuser")
        spill = SpillConfig(scratch_root="/scratch/${USER}/bsm")
        resolved = spill.resolve_scratch("my_run")
        assert "testuser" in resolved
        assert resolved.endswith("/my_run")


# ---------------------------------------------------------------------------
# Manifest tests
# ---------------------------------------------------------------------------


class TestShardManifest:
    def test_build_manifest_basic(self):
        shards = build_manifest(
            stage="interaction_discovery",
            input_paths=["data/X.parquet", "data/Y.parquet"],
            output_root="/tmp/outputs",
            n_shards=4,
            expected_rows=3000,
            expected_columns=100,
        )
        assert len(shards) == 4
        assert all(s.stage == "interaction_discovery" for s in shards)
        assert all(s.status == "pending" for s in shards)
        assert shards[0].shard_id == "task-0000"
        assert shards[3].shard_id == "task-0003"
        # Feature ranges should be set when expected_columns > 0
        assert shards[0].feature_start_idx == 0
        assert shards[3].feature_end_idx == 100

    def test_build_manifest_no_columns(self):
        shards = build_manifest(
            stage="sparse_selection",
            input_paths=["data/X.parquet"],
            output_root="/tmp/outputs",
            n_shards=3,
        )
        assert len(shards) == 3
        assert shards[0].feature_start_idx is None

    def test_jsonl_round_trip(self, tmp_path):
        shards = build_manifest(
            stage="interaction_discovery",
            input_paths=["data/X.parquet"],
            output_root=str(tmp_path / "out"),
            n_shards=5,
            expected_rows=100,
            expected_columns=50,
        )
        manifest_path = tmp_path / "manifest.jsonl"
        save_manifest(shards, manifest_path)
        loaded = load_manifest(manifest_path)

        assert len(loaded) == len(shards)
        for orig, loaded_shard in zip(shards, loaded, strict=True):
            assert orig.shard_id == loaded_shard.shard_id
            assert orig.stage == loaded_shard.stage
            assert orig.status == loaded_shard.status
            assert orig.feature_start_idx == loaded_shard.feature_start_idx

    def test_update_shard_status(self, tmp_path):
        shards = build_manifest(
            stage="interaction_discovery",
            input_paths=[],
            output_root=str(tmp_path / "out"),
            n_shards=3,
        )
        manifest_path = tmp_path / "manifest.jsonl"
        save_manifest(shards, manifest_path)

        update_shard_status(manifest_path, "task-0001", "running")
        loaded = load_manifest(manifest_path)
        assert loaded[1].status == "running"
        assert loaded[1].attempt == 1
        assert loaded[0].status == "pending"

        update_shard_status(manifest_path, "task-0001", "completed")
        loaded = load_manifest(manifest_path)
        assert loaded[1].status == "completed"
        assert loaded[1].completed_at is not None

    def test_update_shard_status_failed(self, tmp_path):
        shards = build_manifest(
            stage="interaction_discovery",
            input_paths=[],
            output_root=str(tmp_path / "out"),
            n_shards=2,
        )
        manifest_path = tmp_path / "manifest.jsonl"
        save_manifest(shards, manifest_path)
        update_shard_status(manifest_path, "task-0000", "failed", error_message="OOM")
        loaded = load_manifest(manifest_path)
        assert loaded[0].status == "failed"
        assert loaded[0].error_message == "OOM"

    def test_manifest_summary(self):
        shards = build_manifest(
            stage="test_stage",
            input_paths=[],
            output_root="/tmp",
            n_shards=4,
        )
        shards[0].status = "completed"
        shards[1].status = "running"
        shards[2].status = "failed"
        # shards[3] stays pending
        summary = manifest_summary(shards)
        assert summary["completed"] == 1
        assert summary["running"] == 1
        assert summary["failed"] == 1
        assert summary["pending"] == 1

    def test_build_manifest_limits_shards_when_columns_smaller_than_requested(self):
        shards = build_manifest(
            stage="interaction_discovery",
            input_paths=["data/X.parquet"],
            output_root="/tmp/outputs",
            n_shards=10,
            expected_rows=100,
            expected_columns=3,
        )
        assert len(shards) == 3
        assert shards[0].feature_start_idx == 0
        assert shards[0].feature_end_idx == 1
        assert shards[1].feature_start_idx == 1
        assert shards[1].feature_end_idx == 2
        assert shards[2].feature_start_idx == 2
        assert shards[2].feature_end_idx == 3


# ---------------------------------------------------------------------------
# Checkpoint tests
# ---------------------------------------------------------------------------


class TestCheckpointManager:
    def test_is_complete_false_initially(self, tmp_path):
        cm = CheckpointManager(str(tmp_path), "task-0000")
        assert cm.is_complete() is False

    def test_mark_running_creates_staging_dir(self, tmp_path):
        cm = CheckpointManager(str(tmp_path), "task-0000")
        cm.mark_running()
        assert cm.staging_dir.exists()
        running_marker = cm.staging_dir / "_RUNNING.json"
        assert running_marker.exists()
        data = json.loads(running_marker.read_text())
        assert data["shard_id"] == "task-0000"
        assert data["attempt"] == 0

    def test_validate_and_promote_success(self, tmp_path):
        cm = CheckpointManager(str(tmp_path), "task-0000")
        cm.mark_running()
        # Write expected output
        (cm.staging_dir / "results.parquet").write_bytes(b"fake parquet")
        (cm.staging_dir / "shard_result.json").write_text('{"status": "ok"}')
        cm.validate_and_promote(expected_files=["results.parquet", "shard_result.json"])
        assert cm.is_complete()
        assert cm.success_path.exists()
        success = json.loads(cm.success_path.read_text())
        assert success["shard_id"] == "task-0000"

    def test_validate_and_promote_missing_file_raises(self, tmp_path):
        cm = CheckpointManager(str(tmp_path), "task-0000")
        cm.mark_running()
        with pytest.raises(FileNotFoundError, match="missing expected output files"):
            cm.validate_and_promote(expected_files=["results.parquet"])

    def test_skip_on_is_complete(self, tmp_path):
        cm = CheckpointManager(str(tmp_path), "task-0000")
        cm.mark_running()
        (cm.staging_dir / "shard_result.json").write_text("{}")
        cm.validate_and_promote(expected_files=["shard_result.json"])
        assert cm.is_complete()

        # Second attempt: is_complete should short-circuit
        cm2 = CheckpointManager(str(tmp_path), "task-0000")
        assert cm2.is_complete() is True

    def test_mark_failed_writes_failure_json(self, tmp_path):
        cm = CheckpointManager(str(tmp_path), "task-0000")
        cm.mark_running()
        cm.mark_failed("Some error occurred")
        failure_path = cm.staging_dir / "_FAILURE.json"
        assert failure_path.exists()
        data = json.loads(failure_path.read_text())
        assert "Some error occurred" in data["error_message"]

    def test_all_complete_helper(self, tmp_path):
        for shard_id in ["task-0000", "task-0001", "task-0002"]:
            cm = CheckpointManager(str(tmp_path), shard_id)
            cm.mark_running()
            (cm.staging_dir / "shard_result.json").write_text("{}")
            cm.validate_and_promote(expected_files=["shard_result.json"])

        all_ids = ["task-0000", "task-0001", "task-0002"]
        assert CheckpointManager.all_complete(str(tmp_path), all_ids)
        assert not CheckpointManager.all_complete(str(tmp_path), ["task-0000", "task-9999"])

    def test_completion_summary(self, tmp_path):
        # Complete one, leave one pending
        cm = CheckpointManager(str(tmp_path), "task-0000")
        cm.mark_running()
        (cm.staging_dir / "shard_result.json").write_text("{}")
        cm.validate_and_promote(expected_files=["shard_result.json"])

        summary = CheckpointManager.completion_summary(str(tmp_path), ["task-0000", "task-0001"])
        assert summary["completed"] == 1
        assert summary["pending"] == 1


# ---------------------------------------------------------------------------
# SlurmArrayRunner tests
# ---------------------------------------------------------------------------


class TestSlurmArrayRunner:
    def _make_runner(self, tmp_path: Path, n_shards: int = 4) -> SlurmArrayRunner:
        cfg = DistributedConfig(
            enabled=True,
            backend="slurm_array",
            run_id="test_run",
            pixi_env_path="/projects/bsm/.pixi",
            slurm=SlurmConfig(
                account="bsm",
                partition="debug",
                walltime="01:00:00",
                memory_gb=16,
                cpus_per_task=4,
                max_concurrent_array_tasks=2,
                log_dir=str(tmp_path / "logs"),
            ),
            kestrel=KestrelConfig(
                projects_root="/projects/bsm",
                pixi_cache_dir="/projects/bsm/.cache/pixi",
            ),
            spill=SpillConfig(scratch_root=str(tmp_path / "scratch")),
        )
        shards = build_manifest(
            stage="interaction_discovery",
            input_paths=["data/X.parquet"],
            output_root=str(tmp_path / "outputs"),
            n_shards=n_shards,
            expected_rows=300,
            expected_columns=20,
        )
        manifest_path = tmp_path / "manifest.jsonl"
        save_manifest(shards, manifest_path)
        return SlurmArrayRunner(
            config=cfg,
            manifest_path=manifest_path,
            output_root=str(tmp_path / "outputs"),
            repo_root=tmp_path,
        )

    def test_generate_stage_script_content(self, tmp_path):
        runner = self._make_runner(tmp_path)
        script = runner.generate_stage_script("interaction_discovery")
        assert "#!/bin/bash" in script
        assert "#SBATCH --account=bsm" in script
        assert "#SBATCH --partition=debug" in script
        assert "#SBATCH --job-name=rfm_interaction_discovery_test_run" in script
        assert "#SBATCH --output=" in script
        assert "rfm_interaction_discovery_%A_%a.out" in script
        assert "rfm_interaction_discovery_%A_%a.err" in script
        assert "#SBATCH --array=0-3%2" in script  # 4 shards, max 2 concurrent
        assert "interaction_discovery" in script
        assert "test_run" in script
        assert "hpc_shard_worker.py" in script
        # Bash-level success check must be present
        assert "_SUCCESS.json" in script

    def test_generate_stage_script_sparse_task_ids(self, tmp_path):
        runner = self._make_runner(tmp_path, n_shards=10)
        # Simulate a restart where only tasks 1, 3, 5 remain
        script = runner.generate_stage_script("interaction_discovery", task_ids=[1, 3, 5])
        assert "#SBATCH --array=1,3,5%2" in script

    def test_generate_stage_script_consecutive_ranges(self, tmp_path):
        runner = self._make_runner(tmp_path, n_shards=10)
        script = runner.generate_stage_script("interaction_discovery", task_ids=[0, 1, 2, 5, 6])
        assert "#SBATCH --array=0-2,5-6%2" in script

    def test_generate_reduce_script_uses_afterany(self, tmp_path):
        runner = self._make_runner(tmp_path)
        script = runner.generate_reduce_script("interaction_discovery", after_job_id=12345)
        assert "#!/bin/bash" in script
        assert "#SBATCH --job-name=rfm_reduce_interaction_discovery_test_run" in script
        assert "rfm_reduce_interaction_discovery_%j.out" in script
        assert "rfm_reduce_interaction_discovery_%j.err" in script
        assert "--dependency=afterany:12345" in script
        assert "hpc_reduce.py" in script
        # Self-guard completeness check
        assert "_SUCCESS.json" in script
        assert "N_EXPECTED" in script

    def test_generate_reduce_script_no_dependency(self, tmp_path):
        runner = self._make_runner(tmp_path)
        script = runner.generate_reduce_script("interaction_discovery", after_job_id=None)
        assert "--dependency=" not in script

    def test_generate_diagnostic_script(self, tmp_path):
        runner = self._make_runner(tmp_path)
        script = runner.generate_diagnostic_script()
        assert "#!/bin/bash" in script
        assert "import rfm_pipeline" in script
        assert "debug" in script

    def test_write_scripts_creates_files(self, tmp_path):
        runner = self._make_runner(tmp_path)
        script_dir = tmp_path / "scripts"
        scripts = runner.write_scripts(script_dir, stage="interaction_discovery")

        assert scripts["stage"].exists()
        assert scripts["reduce"].exists()
        assert scripts["diagnostic"].exists()
        assert scripts["submit_all"].exists()

        # Scripts must be executable
        import stat

        for path in scripts.values():
            mode = path.stat().st_mode
            assert mode & stat.S_IXUSR, f"{path.name} not executable"

    def test_write_scripts_reduce_only_when_no_incomplete(self, tmp_path):
        """task_ids=[] means all shards done — no stage script, only reduce + diagnostic."""
        runner = self._make_runner(tmp_path)
        script_dir = tmp_path / "scripts"
        scripts = runner.write_scripts(script_dir, stage="interaction_discovery", task_ids=[])
        assert "stage" not in scripts
        assert scripts["reduce"].exists()
        assert scripts["diagnostic"].exists()

    def test_write_scripts_sparse_task_ids(self, tmp_path):
        """task_ids=[1,3] produces a sparse array spec in the stage script."""
        runner = self._make_runner(tmp_path, n_shards=4)
        script_dir = tmp_path / "scripts"
        scripts = runner.write_scripts(script_dir, stage="interaction_discovery", task_ids=[1, 3])
        assert scripts["stage"].exists()
        content = scripts["stage"].read_text()
        assert "#SBATCH --array=1,3%2" in content

    def test_submit_all_uses_afterany(self, tmp_path):
        runner = self._make_runner(tmp_path)
        script_dir = tmp_path / "scripts"
        scripts = runner.write_scripts(script_dir, stage="interaction_discovery")
        submit_all = scripts["submit_all"].read_text()
        assert "afterany" in submit_all
        assert "afterok" not in submit_all

    def test_invalid_config_raises(self, tmp_path):
        cfg = DistributedConfig(backend="invalid_backend")
        manifest_path = tmp_path / "manifest.jsonl"
        save_manifest([], manifest_path)
        with pytest.raises(ValueError, match="Invalid DistributedConfig"):
            SlurmArrayRunner(cfg, manifest_path, str(tmp_path / "out"))

    def test_status_returns_summary(self, tmp_path):
        runner = self._make_runner(tmp_path, n_shards=3)
        summary = runner.status()
        assert summary["pending"] == 3
        assert summary.get("completed", 0) == 0

    def test_no_shards_raises_on_generate(self, tmp_path):
        cfg = DistributedConfig(run_id="test")
        manifest_path = tmp_path / "empty.jsonl"
        save_manifest([], manifest_path)
        runner = SlurmArrayRunner(cfg, manifest_path, str(tmp_path / "out"), repo_root=tmp_path)
        with pytest.raises(ValueError, match="No shards"):
            runner.generate_stage_script("interaction_discovery")


# ---------------------------------------------------------------------------
# _task_ids_to_array_spec helper tests
# ---------------------------------------------------------------------------


class TestTaskIdsToArraySpec:
    from rfm_pipeline.distributed.slurm_array_runner import _task_ids_to_array_spec

    def test_single_id(self):
        from rfm_pipeline.distributed.slurm_array_runner import _task_ids_to_array_spec

        assert _task_ids_to_array_spec([5], 10) == "5%10"

    def test_consecutive_range(self):
        from rfm_pipeline.distributed.slurm_array_runner import _task_ids_to_array_spec

        assert _task_ids_to_array_spec([0, 1, 2, 3], 500) == "0-3%500"

    def test_sparse_list(self):
        from rfm_pipeline.distributed.slurm_array_runner import _task_ids_to_array_spec

        assert _task_ids_to_array_spec([0, 1, 2, 5, 6, 10], 50) == "0-2,5-6,10%50"

    def test_full_range_1000(self):
        from rfm_pipeline.distributed.slurm_array_runner import _task_ids_to_array_spec

        spec = _task_ids_to_array_spec(list(range(1000)), 500)
        assert spec == "0-999%500"

    def test_deduplicates_ids(self):
        from rfm_pipeline.distributed.slurm_array_runner import _task_ids_to_array_spec

        assert _task_ids_to_array_spec([3, 3, 3], 1) == "3%1"

    def test_empty_raises(self):
        from rfm_pipeline.distributed.slurm_array_runner import _task_ids_to_array_spec

        with pytest.raises(ValueError):
            _task_ids_to_array_spec([], 10)


# ---------------------------------------------------------------------------
# Restart idempotency helpers
# ---------------------------------------------------------------------------


class TestFindIncompleteTaskIds:
    """Tests for the _find_incomplete_task_ids helper in bsm_hpc_submit."""

    def _write_success(self, output_root: Path, task_id: int) -> None:
        shard_dir = output_root / f"task-{task_id:04d}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        (shard_dir / "_SUCCESS.json").write_text('{"shard_id": "ok"}')

    def test_all_incomplete(self, tmp_path):
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
        from rfm_hpc_submit import _find_incomplete_task_ids

        result = _find_incomplete_task_ids(tmp_path / "shards", 5)
        assert result == [0, 1, 2, 3, 4]

    def test_some_complete(self, tmp_path):
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
        from rfm_hpc_submit import _find_incomplete_task_ids

        root = tmp_path / "shards"
        self._write_success(root, 0)
        self._write_success(root, 2)
        result = _find_incomplete_task_ids(root, 4)
        assert result == [1, 3]

    def test_all_complete(self, tmp_path):
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
        from rfm_hpc_submit import _find_incomplete_task_ids

        root = tmp_path / "shards"
        for i in range(3):
            self._write_success(root, i)
        result = _find_incomplete_task_ids(root, 3)
        assert result == []


class TestReconcileManifestFromFs:
    def test_reconciles_stale_statuses(self, tmp_path):
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
        from rfm_hpc_submit import _reconcile_manifest_from_fs

        shards = build_manifest(
            stage="output_conditioning",
            input_paths=[],
            output_root=str(tmp_path / "shards"),
            n_shards=3,
        )
        # Simulate shard 0 having completed on disk but manifest shows pending
        shard_dir = tmp_path / "shards" / "task-0000"
        shard_dir.mkdir(parents=True)
        (shard_dir / "_SUCCESS.json").write_text("{}")

        updated, n_reconciled = _reconcile_manifest_from_fs(shards, tmp_path / "shards")
        assert n_reconciled == 1
        assert updated[0].status == "completed"
        assert updated[1].status == "pending"
        assert updated[2].status == "pending"
