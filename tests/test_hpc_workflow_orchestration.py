"""Tests for unified HPC workflow orchestration config and command builders."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from rfm_pipeline.hpc_workflow_config import (
    build_collect_command,
    build_remote_status_command,
    build_remote_submit_commands,
    load_hpc_workflow_config,
    resolved_cpu_suite_output_root,
    ssh_target,
)


@pytest.mark.skipif(
    not Path("configs/hpc/kestrel_workflow_orchestration.yml").exists(),
    reason="BSM Kestrel configs not present in this repo",
)
def test_committed_kestrel_orchestration_config_loads() -> None:
    config_path = Path("configs/hpc/kestrel_workflow_orchestration.yml")
    cfg = load_hpc_workflow_config(config_path)
    assert cfg.cluster.host
    assert cfg.execution.local_cores >= 1
    assert len(cfg.execution.cpu_tiers) == 3
    assert cfg.execution.cpu_tiers[0].nodes == 2


@pytest.mark.skipif(
    not Path("configs/hpc/kestrel_workflow_small_distributed.yml").exists(),
    reason="BSM Kestrel configs not present in this repo",
)
def test_committed_small_distributed_orchestration_config_loads() -> None:
    config_path = Path("configs/hpc/kestrel_workflow_small_distributed.yml")
    cfg = load_hpc_workflow_config(config_path)
    assert cfg.cluster.host
    assert len(cfg.execution.cpu_tiers) == 1
    assert cfg.execution.cpu_tiers[0].nodes == 2
    assert cfg.execution.cpu_tiers[0].config_path == "configs/hpc/kestrel_cpu_scale_2_smoke.yml"
    assert cfg.execution.prepare_interaction_inputs is False
    assert cfg.execution.prepare_full_pipeline_artifacts is True
    assert cfg.gpu.enabled is False
    assert cfg.pullback.mode == "study_package"


@pytest.mark.skipif(
    not Path("configs/hpc/kestrel_cpu_scale_2_smoke.yml").exists(),
    reason="BSM Kestrel configs not present in this repo",
)
def test_committed_cpu_scale_2_smoke_config_is_lightweight() -> None:
    cfg = yaml.safe_load(
        Path("configs/hpc/kestrel_cpu_scale_2_smoke.yml").read_text(encoding="utf-8")
    )
    assert cfg["dataset"]["type"] == "synthetic_300_sample"
    assert cfg["validation"]["fast_mode"] is True
    assert cfg["distributed"]["slurm"]["walltime"] == "00:30:00"
    assert cfg["distributed"]["slurm"]["max_concurrent_array_tasks"] == 2


@pytest.mark.skipif(
    not Path("scripts/kestrel/run_small_distributed_test_local.sh").exists(),
    reason="BSM Kestrel scripts not present in this repo",
)
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


def test_submit_commands_prepare_full_pipeline_when_enabled(tmp_path: Path) -> None:
    config_path = tmp_path / "hpc.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "cluster": {"host": "kl1.hpc.nrel.gov", "user": "alice"},
                "execution": {
                    "stage": "interaction_discovery",
                    "prepare_full_pipeline_artifacts": True,
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
    assert "--stop-stage final_manuscript_artifacts" in commands[0]
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

    # build_collect_command now requires the configured collect script
    # to exist locally. Materialize a stub so this test covers only the
    # CLI-binding contract (a separate test covers the missing-script
    # FileNotFoundError path).
    stub_script = tmp_path / "scripts" / "kestrel" / "pull_hpc_artifacts_bundle.sh"
    stub_script.parent.mkdir(parents=True, exist_ok=True)
    stub_script.write_text("#!/bin/bash\n:\n", encoding="utf-8")
    collect_cmd = build_collect_command(cfg, repo_root=tmp_path)
    collect_str = " ".join(collect_cmd)
    assert "--hpc-host alice@kl1.hpc.nrel.gov" in collect_str
    assert "--hpc-repo-root /projects/bsm/bsm-public-rf" in collect_str
    assert "--hpc-artifacts-root /scratch/alice/bsm/bsm-public-rf/artifacts" in collect_str
    assert "--remote-snapshot-root /scratch/alice/bsm/kestrel_hpc_snapshots" in collect_str
    assert "--pullback-mode reporting_bundle" in collect_str
    assert "--cpu-tier-specs 2=configs/hpc/kestrel_cpu_scale_2.yml" in collect_str
    assert "--include-gpu 0" in collect_str
    assert "--gpu-config configs/hpc/kestrel_gpu_h100.yml" in collect_str


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


def test_submit_commands_cascade_over_stages_list(tmp_path: Path) -> None:
    """When execution.stages is set, one rfm-hpc-submit per stage per tier."""
    config_path = tmp_path / "hpc.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "cluster": {"host": "kl1.hpc.nrel.gov", "user": "alice"},
                "execution": {
                    "stages": [
                        "output_conditioning",
                        "empirical_null_screening",
                        "interaction_discovery",
                        "nonlinear_discovery",
                        "sparse_selection",
                        "final_manuscript_artifacts",
                    ],
                    "run_diagnostic": False,
                    "cpu_tiers": [
                        {"nodes": 2, "config_path": "configs/hpc/kestrel_cpu_scale_2.yml"},
                    ],
                },
                "gpu": {"enabled": False},
            }
        ),
        encoding="utf-8",
    )
    cfg = load_hpc_workflow_config(config_path)
    assert cfg.execution.effective_stages() == [
        "output_conditioning",
        "empirical_null_screening",
        "interaction_discovery",
        "nonlinear_discovery",
        "sparse_selection",
        "final_manuscript_artifacts",
    ]

    commands = build_remote_submit_commands(cfg, submit=True, dry_run=False)
    for stage in cfg.execution.effective_stages():
        assert any(f"--stage {stage}" in c for c in commands), (
            f"missing submit command for stage {stage}: {commands}"
        )
        # Per-stage output dirs avoid clobbering between stages.
        assert any(f"/cpu_nodes_2_{stage}/hpc_scripts" in c for c in commands)


def test_submit_commands_dry_run_propagates_to_rfm_hpc_submit(tmp_path: Path) -> None:
    """--dry-run on the orchestrator must reach the per-stage rfm-hpc-submit calls
    so the remote command actually generates SLURM scripts for validation."""
    config_path = tmp_path / "hpc.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "cluster": {"host": "kl1.hpc.nrel.gov", "user": "alice"},
                "execution": {
                    "stages": ["output_conditioning", "interaction_discovery"],
                    "run_diagnostic": False,
                    "cpu_tiers": [
                        {"nodes": 2, "config_path": "configs/hpc/kestrel_cpu_scale_2.yml"},
                    ],
                },
                "gpu": {"enabled": False},
            }
        ),
        encoding="utf-8",
    )
    cfg = load_hpc_workflow_config(config_path)

    commands = build_remote_submit_commands(cfg, submit=False, dry_run=True)
    assert commands
    for c in commands:
        assert "--dry-run" in c
        assert "--submit" not in c


def test_invalid_stage_in_stages_list_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "hpc.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "cluster": {"host": "kl1.hpc.nrel.gov", "user": "alice"},
                "execution": {
                    "stages": ["output_conditioning", "not_a_real_stage"],
                    "run_diagnostic": False,
                    "cpu_tiers": [
                        {"nodes": 2, "config_path": "configs/hpc/kestrel_cpu_scale_2.yml"},
                    ],
                },
                "gpu": {"enabled": False},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not_a_real_stage"):
        load_hpc_workflow_config(config_path)


def test_submit_command_groups_separates_prep_diagnostic_and_per_stage(
    tmp_path: Path,
) -> None:
    """build_remote_submit_command_groups returns labeled cascade groups so
    orchestrators can capture each stage's reduce job id and chain the
    next stage with --depends-on-job-id."""
    from rfm_pipeline.hpc_workflow_config import build_remote_submit_command_groups

    config_path = tmp_path / "hpc.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "cluster": {"host": "kl1.hpc.nrel.gov", "user": "alice"},
                "execution": {
                    "stages": ["output_conditioning", "interaction_discovery"],
                    "run_diagnostic": True,
                    "prepare_interaction_inputs": True,
                    "cpu_tiers": [
                        {"nodes": 2, "config_path": "configs/hpc/kestrel_cpu_scale_2.yml"},
                    ],
                },
                "gpu": {"enabled": False},
            }
        ),
        encoding="utf-8",
    )
    cfg = load_hpc_workflow_config(config_path)
    groups = build_remote_submit_command_groups(cfg, submit=True, dry_run=False)

    stage_groups = [(s, c) for s, c in groups if s is not None]
    none_groups = [c for s, c in groups if s is None]

    # prep + diagnostic appear in None-marked groups
    assert any(any("run_manuscript_pipeline" in c for c in g) for g in none_groups)
    assert any(any("--diagnostic-only" in c for c in g) for g in none_groups)

    # Two stage groups in cascade order
    assert [s for s, _ in stage_groups] == [
        "output_conditioning",
        "interaction_discovery",
    ]
    for stage_name, cmds in stage_groups:
        assert all(f"--stage {stage_name}" in c for c in cmds)


def test_build_collect_command_raises_when_script_missing(tmp_path: Path) -> None:
    config_path = tmp_path / "hpc.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "cluster": {"host": "kl1.hpc.nrel.gov", "user": "alice"},
                "paths": {
                    "remote_repo_root": "/projects/bsm/bsm-public-rf",
                    "remote_artifacts_root": "/scratch/alice/artifacts",
                    "remote_logs_root": "/scratch/alice",
                    "remote_snapshot_root": "/scratch/alice/snap",
                    "local_bundle_dir": "./artifacts/collected",
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
    # Empty tmp_path repo root => default script path does not exist.
    with pytest.raises(FileNotFoundError, match="HPC collect script not found"):
        build_collect_command(cfg, repo_root=tmp_path)


def test_build_status_and_collect_honor_custom_script_paths(tmp_path: Path) -> None:
    custom_status = "ops/hpc/status_custom.sh"
    custom_collect = "ops/hpc/collect_custom.sh"
    config_path = tmp_path / "hpc.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "cluster": {"host": "kl1.hpc.nrel.gov", "user": "alice"},
                "paths": {
                    "remote_repo_root": "/projects/x",
                    "remote_artifacts_root": "/scratch/alice/artifacts",
                    "remote_logs_root": "/scratch/alice",
                    "remote_snapshot_root": "/scratch/alice/snap",
                    "local_bundle_dir": "./artifacts/collected",
                    "remote_status_script": custom_status,
                    "local_collect_script": custom_collect,
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
    assert custom_status in status_cmd
    assert "status_all_tests.sh" not in status_cmd

    stub = tmp_path / custom_collect
    stub.parent.mkdir(parents=True, exist_ok=True)
    stub.write_text("#!/bin/bash\n:\n", encoding="utf-8")
    collect_cmd = build_collect_command(cfg, repo_root=tmp_path)
    assert str(stub) in collect_cmd
