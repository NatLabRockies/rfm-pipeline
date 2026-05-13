"""Tests for CPU scaling stress-suite assets (2 -> 10 -> 1000 node tiers)."""

from __future__ import annotations

from pathlib import Path

import pytest

from bsm_rfm.distributed.config_distributed import load_distributed_config
from bsm_rfm.distributed.manifest import build_manifest, save_manifest
from bsm_rfm.distributed.slurm_array_runner import SlurmArrayRunner

_CPU_SCALING_CASES = [
    ("configs/hpc/kestrel_cpu_scale_2.yml", 2, "debug"),
    ("configs/hpc/kestrel_cpu_scale_10.yml", 10, "shared"),
    ("configs/hpc/kestrel_cpu_scale_1000.yml", 1000, "shared"),
]


@pytest.mark.parametrize(
    ("config_path", "expected_nodes", "expected_partition"),
    _CPU_SCALING_CASES,
)
def test_cpu_scaling_configs_have_expected_slurm_concurrency(
    config_path: str, expected_nodes: int, expected_partition: str
) -> None:
    cfg = load_distributed_config(config_path)
    assert cfg.enabled is True
    assert cfg.backend == "slurm_array"
    assert cfg.slurm.partition == expected_partition
    assert cfg.slurm.max_concurrent_array_tasks == expected_nodes
    assert cfg.slurm.cpus_per_task == 104


@pytest.mark.parametrize(("config_path", "expected_nodes", "_"), _CPU_SCALING_CASES)
def test_cpu_scaling_stage_script_renders_expected_array_throttle(
    tmp_path: Path, config_path: str, expected_nodes: int, _: str
) -> None:
    cfg = load_distributed_config(config_path)
    shards = build_manifest(
        stage="interaction_discovery",
        input_paths=["data/X.parquet"],
        output_root=str(tmp_path / "outputs"),
        n_shards=expected_nodes,
    )
    manifest_path = tmp_path / f"manifest_{expected_nodes}.jsonl"
    save_manifest(shards, manifest_path)

    runner = SlurmArrayRunner(
        config=cfg,
        manifest_path=manifest_path,
        output_root=str(tmp_path / "outputs"),
        repo_root=tmp_path,
    )
    stage_script = runner.generate_stage_script("interaction_discovery")
    assert f"#SBATCH --array=0-{expected_nodes - 1}%{expected_nodes}" in stage_script
    assert f"#SBATCH --partition={cfg.slurm.partition}" in stage_script
    assert "#SBATCH --cpus-per-task=104" in stage_script


def test_cpu_scaling_suite_script_targets_all_tiers() -> None:
    script_path = Path("scripts/kestrel/submit_cpu_scaling_suite.sh")
    text = script_path.read_text(encoding="utf-8")

    assert "set -euo pipefail" in text
    assert "pixi run bsm-hpc-submit" in text
    assert "--diagnostic-only" in text

    assert "configs/hpc/kestrel_cpu_scale_2.yml" in text
    assert "configs/hpc/kestrel_cpu_scale_10.yml" in text
    assert "configs/hpc/kestrel_cpu_scale_1000.yml" in text

    assert '--n-shards "${nodes}"' in text
    assert "2:configs/hpc/kestrel_cpu_scale_2.yml" in text
    assert "10:configs/hpc/kestrel_cpu_scale_10.yml" in text
    assert "1000:configs/hpc/kestrel_cpu_scale_1000.yml" in text
