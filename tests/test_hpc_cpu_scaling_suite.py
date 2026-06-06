"""Tests for CPU scaling stress-suite assets (2 -> 10 -> 1000 node tiers)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rfm_pipeline.config import load_config
from rfm_pipeline.distributed.config_distributed import load_distributed_config
from rfm_pipeline.distributed.manifest import build_manifest, save_manifest
from rfm_pipeline.distributed.slurm_array_runner import SlurmArrayRunner

_FIXTURES = Path(__file__).parent / "fixtures" / "hpc"

_CPU_SCALING_CASES = [
    (str(_FIXTURES / "cpu_scale_2.yml"), 2, "shared", "./artifacts/cpu_scale_2_run"),
    (str(_FIXTURES / "cpu_scale_10.yml"), 10, "shared", "./artifacts/cpu_scale_10_run"),
    (str(_FIXTURES / "cpu_scale_1000.yml"), 1000, "shared", "./artifacts/cpu_scale_1000_run"),
]


@pytest.mark.parametrize(
    ("config_path", "expected_nodes", "expected_partition", "expected_artifact_dir"),
    _CPU_SCALING_CASES,
)
def test_cpu_scaling_configs_have_expected_slurm_concurrency(
    config_path: str,
    expected_nodes: int,
    expected_partition: str,
    expected_artifact_dir: str,
) -> None:
    workflow_cfg = load_config(config_path)
    cfg = load_distributed_config(config_path)
    assert workflow_cfg.output.artifact_dir == expected_artifact_dir
    assert cfg.enabled is True
    assert cfg.backend == "slurm_array"
    assert cfg.slurm.partition == expected_partition
    assert cfg.slurm.max_concurrent_array_tasks == expected_nodes


@pytest.mark.parametrize(("config_path", "expected_nodes", "_", "__"), _CPU_SCALING_CASES)
def test_cpu_scaling_stage_script_renders_expected_array_throttle(
    tmp_path: Path, config_path: str, expected_nodes: int, _: str, __: str
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


@pytest.mark.skipif(
    not Path("scripts/kestrel/submit_cpu_scaling_suite.sh").exists(),
    reason="BSM Kestrel scripts not present in this repo",
)
def test_cpu_scaling_suite_script_targets_all_tiers() -> None:
    script_path = Path("scripts/kestrel/submit_cpu_scaling_suite.sh")
    text = script_path.read_text(encoding="utf-8")

    assert "set -euo pipefail" in text
    assert "rfm-hpc-submit" in text
    assert "--diagnostic-only" in text

    assert "configs/hpc/kestrel_cpu_scale_2.yml" in text
    assert "configs/hpc/kestrel_cpu_scale_10.yml" in text
    assert "configs/hpc/kestrel_cpu_scale_1000.yml" in text


@pytest.mark.skipif(
    not Path("scripts/kestrel/submit_cpu_scale_2_live.sh").exists(),
    reason="BSM Kestrel scripts not present in this repo",
)
def test_cpu_scaling_live_submit_scripts_exist_with_expected_defaults() -> None:
    script_2 = Path("scripts/kestrel/submit_cpu_scale_2_live.sh").read_text(encoding="utf-8")
    script_10 = Path("scripts/kestrel/submit_cpu_scale_10_live.sh").read_text(encoding="utf-8")
    script_1000 = Path("scripts/kestrel/submit_cpu_scale_1000_live.sh").read_text(encoding="utf-8")
    script_chain = Path("scripts/kestrel/submit_cpu_scale_chain_live.sh").read_text(
        encoding="utf-8"
    )

    assert "set -euo pipefail" in script_2
    assert 'PARTITION="${PARTITION:-debug}"' in script_2
    assert 'WALLTIME="${WALLTIME:-01:00:00}"' in script_2
    assert "sbatch --parsable --partition=" in script_2
    assert "--dependency=afterok:${ARRAY_JOB_ID}" in script_2

    assert "set -euo pipefail" in script_10
    assert 'PARTITION="${PARTITION:-shared}"' in script_10
    assert 'WALLTIME="${WALLTIME:-04:00:00}"' in script_10
    assert "--dependency=afterok:${AFTER_JOB_ID}" in script_10

    assert "set -euo pipefail" in script_1000
    assert 'PARTITION="${PARTITION:-shared}"' in script_1000
    assert 'WALLTIME="${WALLTIME:-08:00:00}"' in script_1000
    assert "--dependency=afterok:${AFTER_JOB_ID}" in script_1000

    assert "set -euo pipefail" in script_chain
    assert "--partition=debug --time=01:00:00" in script_chain
    assert "--partition=shared --time=04:00:00 --dependency=afterok:${R2}" in script_chain
    assert "--partition=shared --time=08:00:00 --dependency=afterok:${R10}" in script_chain


@pytest.mark.skipif(
    not Path("scripts/kestrel/collect_cpu_scaling_results.sh").exists(),
    reason="BSM Kestrel scripts not present in this repo",
)
def test_cpu_scaling_collection_and_monitor_scripts_exist() -> None:
    collector = Path("scripts/kestrel/collect_cpu_scaling_results.sh").read_text(encoding="utf-8")
    monitor = Path("scripts/kestrel/watch_cpu_scaling_queue.sh").read_text(encoding="utf-8")
    status = Path("scripts/kestrel/status_all_tests.sh").read_text(encoding="utf-8")

    assert "set -euo pipefail" in collector
    assert "interaction_discovery_merged.json" in collector
    assert "retained_interaction_pairs_merged.csv" in collector
    assert "cpu_scaling_results_summary.csv" in collector
    assert "common_paths.sh" in collector

    assert "set -euo pipefail" in monitor
    assert "squeue -u" in monitor
    assert "sacct -u" in monitor
    assert "collect_cpu_scaling_results.sh" in monitor
    assert "common_paths.sh" in monitor
    assert "common_paths.sh" in status
