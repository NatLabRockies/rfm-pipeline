"""Tests for tool-level HPC workflow runner command construction."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from tools import run_hpc_workflow


def test_run_remote_shell_preserves_full_bash_lc_command() -> None:
    with patch("tools.run_hpc_workflow._run_command") as run_mock:
        run_hpc_workflow._run_remote_shell(
            "alice@kl1.hpc.nrel.gov",
            "/projects/bsm/bsm-public-rf",
            (
                "pixi run bsm-hpc-submit --config "
                "configs/hpc/kestrel_cpu_scale_2_smoke.yml --diagnostic-only"
            ),
            dry_run=True,
        )

    args, kwargs = run_mock.call_args
    command = args[0]
    assert command[:3] == ["ssh", "-T", "alice@kl1.hpc.nrel.gov"]
    assert len(command) == 4
    remote_invocation = command[3]
    assert remote_invocation.startswith("bash -lc ")
    assert (
        "cd /projects/bsm/bsm-public-rf && pixi run bsm-hpc-submit --config "
        "configs/hpc/kestrel_cpu_scale_2_smoke.yml --diagnostic-only"
    ) in remote_invocation
    assert kwargs["dry_run"] is True


def test_resolve_dataset_sync_specs_for_synthetic_dataset(tmp_path: Path) -> None:
    tier_cfg = tmp_path / "tier.yml"
    tier_cfg.write_text("dataset:\n  type: synthetic_300_sample\n", encoding="utf-8")

    specs = run_hpc_workflow._resolve_dataset_sync_specs(
        "/home/alice/src/bsm-public-rf",
        [str(tier_cfg)],
    )
    assert specs
    local, remote = specs[0]
    assert local == run_hpc_workflow.REPO_ROOT / "artifacts" / "test_dataset_300"
    assert remote == "/home/alice/src/bsm-public-rf/artifacts/test_dataset_300"


def test_sync_required_datasets_runs_mkdir_and_scp_when_missing(tmp_path: Path) -> None:
    local_dataset = tmp_path / "test_dataset_300"
    local_dataset.mkdir(parents=True)
    (local_dataset / "X.parquet").write_text("stub", encoding="utf-8")

    with (
        patch(
            "tools.run_hpc_workflow._resolve_dataset_sync_specs",
            return_value=[
                (
                    local_dataset,
                    "/home/alice/src/bsm-public-rf/artifacts/test_dataset_300",
                )
            ],
        ),
        patch("tools.run_hpc_workflow._remote_path_exists", return_value=False),
        patch("tools.run_hpc_workflow._run_command") as run_mock,
    ):
        run_hpc_workflow._sync_required_datasets(
            ssh_dest="alice@kl1.hpc.nrel.gov",
            remote_repo_root="/home/alice/src/bsm-public-rf",
            cpu_tier_configs=["configs/hpc/kestrel_cpu_scale_2_smoke.yml"],
            dry_run=False,
        )

    assert run_mock.call_count == 2
    mkdir_call = run_mock.call_args_list[0][0][0]
    scp_call = run_mock.call_args_list[1][0][0]
    assert mkdir_call[:3] == ["ssh", "-T", "alice@kl1.hpc.nrel.gov"]
    assert scp_call[:2] == ["scp", "-r"]
    assert str(local_dataset) in scp_call
