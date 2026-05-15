"""Tests for unified HPC workflow orchestration config and command builders."""

from __future__ import annotations

from pathlib import Path

import yaml

from bsm_rfm.hpc_workflow_config import (
    build_collect_command,
    build_remote_status_command,
    build_remote_submit_commands,
    load_hpc_workflow_config,
    resolved_cpu_suite_output_root,
    ssh_target,
)


def test_committed_kestrel_orchestration_config_loads() -> None:
    config_path = Path("configs/hpc/kestrel_workflow_orchestration.yml")
    cfg = load_hpc_workflow_config(config_path)
    assert cfg.cluster.host
    assert cfg.execution.local_cores >= 1
    assert len(cfg.execution.cpu_tiers) == 3
    assert cfg.execution.cpu_tiers[0].nodes == 2


def test_committed_small_distributed_orchestration_config_loads() -> None:
    config_path = Path("configs/hpc/kestrel_workflow_small_distributed.yml")
    cfg = load_hpc_workflow_config(config_path)
    assert cfg.cluster.host
    assert len(cfg.execution.cpu_tiers) == 1
    assert cfg.execution.cpu_tiers[0].nodes == 2
    assert cfg.execution.cpu_tiers[0].config_path == "configs/hpc/kestrel_cpu_scale_2_smoke.yml"
    assert cfg.execution.prepare_interaction_inputs is True
    assert cfg.gpu.enabled is False
    assert cfg.pullback.mode == "manifest_only"


def test_committed_cpu_scale_2_smoke_config_is_lightweight() -> None:
    cfg = yaml.safe_load(
        Path("configs/hpc/kestrel_cpu_scale_2_smoke.yml").read_text(encoding="utf-8")
    )
    assert cfg["dataset"]["type"] == "synthetic_300_sample"
    assert cfg["validation"]["fast_mode"] is True
    assert cfg["distributed"]["slurm"]["walltime"] == "00:30:00"
    assert cfg["distributed"]["slurm"]["max_concurrent_array_tasks"] == 2


def test_small_distributed_local_script_calls_unified_runner() -> None:
    script = Path("scripts/kestrel/run_small_distributed_test_local.sh").read_text(encoding="utf-8")
    assert "set -euo pipefail" in script
    assert "hpc-workflow" in script
    assert "--action" in script
    assert "run_step submit" in script
    assert "run_step status" in script
    assert "run_step collect" in script
    assert "kestrel_workflow_small_distributed.yml" in script
    assert "if (( ${#common_flags[@]} > 0 )); then" in script
    assert 'cmd+=("${common_flags[@]}")' in script


def test_load_hpc_workflow_config_with_root_key(tmp_path: Path) -> None:
    config_path = tmp_path / "hpc.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "hpc_workflow": {
                    "cluster": {"host": "kl1.hpc.nrel.gov", "user": "alice"},
                    "execution": {
                        "stage": "interaction_discovery",
                        "local_cores": 4,
                        "cpu_tiers": [
                            {"nodes": 2, "config_path": "configs/hpc/kestrel_cpu_scale_2.yml"},
                            {"nodes": 10, "config_path": "configs/hpc/kestrel_cpu_scale_10.yml"},
                        ],
                    },
                    "gpu": {"enabled": True, "n_shards": 12},
                }
            }
        ),
        encoding="utf-8",
    )

    cfg = load_hpc_workflow_config(config_path)
    assert cfg.cluster.user == "alice"
    assert cfg.execution.local_cores == 4
    assert [tier.nodes for tier in cfg.execution.cpu_tiers] == [2, 10]
    assert cfg.gpu.enabled is True
    assert cfg.gpu.n_shards == 12


def test_submit_commands_include_tiers_and_gpu(tmp_path: Path) -> None:
    config_path = tmp_path / "hpc.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "cluster": {"host": "kl1.hpc.nrel.gov", "user": "alice"},
                "paths": {
                    "remote_artifacts_root": "/scratch/alice/bsm/bsm-public-rf/artifacts",
                },
                "execution": {
                    "stage": "interaction_discovery",
                    "run_diagnostic": True,
                    "cpu_tiers": [
                        {"nodes": 2, "config_path": "configs/hpc/kestrel_cpu_scale_2.yml"},
                        {"nodes": 10, "config_path": "configs/hpc/kestrel_cpu_scale_10.yml"},
                    ],
                },
                "gpu": {
                    "enabled": True,
                    "config_path": "configs/hpc/kestrel_gpu_h100.yml",
                    "n_shards": 10,
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = load_hpc_workflow_config(config_path)

    commands = build_remote_submit_commands(cfg, submit=True, dry_run=False)
    assert any("--diagnostic-only" in command for command in commands)
    assert any("--n-shards 2" in command for command in commands)
    assert any("--n-shards 10" in command for command in commands)
    assert any("kestrel_gpu_h100.yml" in command for command in commands)
    assert all("--submit" in command for command in commands)


def test_submit_commands_prepare_interaction_inputs_when_enabled(tmp_path: Path) -> None:
    config_path = tmp_path / "hpc.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "cluster": {"host": "kl1.hpc.nrel.gov", "user": "alice"},
                "execution": {
                    "stage": "interaction_discovery",
                    "prepare_interaction_inputs": True,
                    "cpu_tiers": [
                        {"nodes": 2, "config_path": "configs/hpc/kestrel_cpu_scale_2_smoke.yml"},
                    ],
                },
                "gpu": {"enabled": False},
            }
        ),
        encoding="utf-8",
    )
    cfg = load_hpc_workflow_config(config_path)

    commands = build_remote_submit_commands(cfg, submit=True, dry_run=False)
    assert commands
    assert commands[0].startswith(
        "pixi run python tools/run_manuscript_pipeline.py configs/hpc/kestrel_cpu_scale_2_smoke.yml"
    )
    assert "--start-stage output_conditioning" in commands[0]
    assert "--stop-stage empirical_null_screen" in commands[0]
    assert any("--diagnostic-only" in command for command in commands)
    assert any("--stage interaction_discovery" in command for command in commands)


def test_status_and_collect_commands_bind_configured_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "hpc.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "cluster": {"host": "kl1.hpc.nrel.gov", "user": "alice"},
                "paths": {
                    "remote_repo_root": "/projects/bsm/bsm-public-rf",
                    "remote_artifacts_root": "/scratch/alice/bsm/bsm-public-rf/artifacts",
                    "remote_logs_root": "/scratch/alice/bsm",
                    "remote_snapshot_root": "/scratch/alice/bsm/kestrel_hpc_snapshots",
                    "local_bundle_dir": "./artifacts/kestrel_collected_bundles",
                },
                "execution": {
                    "cpu_tiers": [
                        {"nodes": 2, "config_path": "configs/hpc/kestrel_cpu_scale_2.yml"},
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = load_hpc_workflow_config(config_path)

    status_cmd = build_remote_status_command(cfg)
    assert "ARTIFACTS_ROOT=/scratch/alice/bsm/bsm-public-rf/artifacts" in status_cmd
    assert "LOGS_ROOT=/scratch/alice/bsm" in status_cmd
    assert "STATUS_CPU_TIERS=2" in status_cmd
    assert "STATUS_INCLUDE_GPU=0" in status_cmd
    assert "STATUS_GPU_SHARDS=10" in status_cmd
    assert "status_all_tests.sh" in status_cmd

    collect_cmd = build_collect_command(cfg, repo_root=Path.cwd())
    collect_str = " ".join(collect_cmd)
    assert "--hpc-host alice@kl1.hpc.nrel.gov" in collect_str
    assert "--hpc-repo-root /projects/bsm/bsm-public-rf" in collect_str
    assert "--hpc-artifacts-root /scratch/alice/bsm/bsm-public-rf/artifacts" in collect_str
    assert "--remote-snapshot-root /scratch/alice/bsm/kestrel_hpc_snapshots" in collect_str
    assert "--pullback-mode reporting_bundle" in collect_str


def test_default_cpu_suite_output_root_uses_artifacts_root(tmp_path: Path) -> None:
    config_path = tmp_path / "hpc.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "cluster": {"host": "kl1.hpc.nrel.gov", "user": "alice"},
                "paths": {
                    "remote_artifacts_root": "/scratch/alice/bsm/bsm-public-rf/artifacts",
                },
                "execution": {
                    "cpu_tiers": [
                        {"nodes": 2, "config_path": "configs/hpc/kestrel_cpu_scale_2.yml"},
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = load_hpc_workflow_config(config_path)
    assert (
        resolved_cpu_suite_output_root(cfg)
        == "/scratch/alice/bsm/bsm-public-rf/artifacts/kestrel_cpu_scaling_suite"
    )
    assert ssh_target(cfg) == "alice@kl1.hpc.nrel.gov"


def test_config_rejects_home_scoped_runtime_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "hpc.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "cluster": {"host": "kl1.hpc.nrel.gov", "user": "alice"},
                "paths": {
                    "remote_artifacts_root": "/home/alice/bsm/artifacts",
                    "remote_logs_root": "/home/alice/bsm/logs",
                    "remote_snapshot_root": "/home/alice/bsm/snapshots",
                },
                "execution": {
                    "cpu_tiers": [
                        {"nodes": 2, "config_path": "configs/hpc/kestrel_cpu_scale_2.yml"},
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    try:
        load_hpc_workflow_config(config_path)
        raise AssertionError(
            "expected load_hpc_workflow_config to reject home-scoped runtime paths"
        )
    except ValueError as exc:
        message = str(exc)
        assert "cannot point under /home/alice" in message
