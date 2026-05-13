"""Tests for Phase 8b/8c/GPU distributed execution additions.

Covers:
- GpuConfig and DaskConfig dataclasses
- GPU config validation in DistributedConfig
- GPU script generation by SlurmArrayRunner (when gpu.enabled=True)
- gpu_scoring.py: detect_device with CPU-only machine (CUDA unavailable)
- DaskRunner: local client modes (threads/processes)
- MPI runner: assign_shards, get_rank_size (no MPI installed)
- RayRunner: gated behind BSM_ENABLE_RAY_EXPERIMENTAL env var
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from bsm_rfm.distributed.config_distributed import (
    DaskConfig,
    DistributedConfig,
    GpuConfig,
    load_distributed_config,
)
from bsm_rfm.distributed.manifest import ShardManifest, save_manifest
from bsm_rfm.distributed.slurm_array_runner import SlurmArrayRunner

# ---------------------------------------------------------------------------
# GpuConfig defaults and validation
# ---------------------------------------------------------------------------


def test_gpu_config_defaults():
    gpu = GpuConfig()
    assert gpu.enabled is False
    assert gpu.device == "auto"
    assert gpu.n_gpus == 1
    assert gpu.gpu_partition == "gpu-h100s"
    assert gpu.xgboost_tree_method == "hist"


def test_gpu_config_invalid_device():
    cfg = DistributedConfig()
    cfg.gpu.device = "tpu"  # not valid
    errors = cfg.validate()
    assert any("gpu.device" in e for e in errors)


def test_gpu_config_invalid_n_gpus():
    cfg = DistributedConfig()
    cfg.gpu.n_gpus = 0
    errors = cfg.validate()
    assert any("gpu.n_gpus" in e for e in errors)


def test_gpu_config_valid():
    cfg = DistributedConfig()
    cfg.gpu.enabled = True
    cfg.gpu.device = "cuda"
    cfg.gpu.n_gpus = 4
    errors = cfg.validate()
    assert errors == []


# ---------------------------------------------------------------------------
# DaskConfig defaults
# ---------------------------------------------------------------------------


def test_dask_config_defaults():
    dask = DaskConfig()
    assert dask.scheduler == "slurm"
    assert dask.n_workers == 4
    assert dask.memory_per_worker_gb == 8
    assert dask.network_interface == "hsn0"
    assert dask.dashboard_port == 8787


def test_dask_config_embedded_in_distributed_config():
    cfg = DistributedConfig()
    cfg.dask.n_workers = 16
    cfg.dask.scheduler = "processes"
    assert cfg.dask.n_workers == 16
    assert cfg.validate() == []


# ---------------------------------------------------------------------------
# GPU config YAML roundtrip
# ---------------------------------------------------------------------------


def test_gpu_config_yaml_load(tmp_path):
    yml = tmp_path / "gpu_test.yml"
    yml.write_text(
        """
distributed:
  enabled: true
  backend: slurm_array
  run_id: test_gpu_run
  gpu:
    enabled: true
    device: cuda
    n_gpus: 4
    gpu_partition: gpu-h100s
    gpu_walltime: "04:00:00"
  dask:
    n_workers: 10
    scheduler: slurm
"""
    )
    cfg = load_distributed_config(yml)
    assert cfg.gpu.enabled is True
    assert cfg.gpu.device == "cuda"
    assert cfg.gpu.n_gpus == 4
    assert cfg.dask.n_workers == 10
    assert cfg.validate() == []


def test_kestrel_gpu_h100_config_loads():
    """Verify the committed kestrel_gpu_h100.yml parses cleanly."""
    config_path = Path(__file__).parent.parent / "configs" / "hpc" / "kestrel_gpu_h100.yml"
    if not config_path.exists():
        pytest.skip("kestrel_gpu_h100.yml not present")
    cfg = load_distributed_config(config_path)
    assert cfg.gpu.enabled is True
    assert cfg.gpu.device == "cuda"
    assert cfg.slurm.partition == "gpu-h100s"
    assert cfg.validate() == []


# ---------------------------------------------------------------------------
# GPU script generation
# ---------------------------------------------------------------------------


def _make_gpu_runner(tmp_path) -> SlurmArrayRunner:
    """Build a SlurmArrayRunner with GPU enabled and a populated manifest."""
    cfg = DistributedConfig()
    cfg.run_id = "test_gpu"
    cfg.gpu.enabled = True
    cfg.gpu.device = "cuda"
    cfg.gpu.n_gpus = 1

    manifest_path = tmp_path / "manifest.jsonl"
    shards = [
        ShardManifest(
            shard_id="shard_0000",
            stage="interaction_discovery",
            input_paths=[str(tmp_path / "data.parquet")],
            output_path=str(tmp_path / "output" / "shard_0000"),
        )
    ]
    save_manifest(shards, manifest_path)

    return SlurmArrayRunner(
        config=cfg,
        manifest_path=manifest_path,
        output_root=tmp_path / "output",
        repo_root=tmp_path,
    )


def test_generate_gpu_stage_script_contains_cuda_env(tmp_path):
    runner = _make_gpu_runner(tmp_path)
    script = runner.generate_gpu_stage_script("interaction_discovery")
    assert "BSM_INTERACTION_DEVICE" in script
    assert "gpu-h100s" in script
    assert "gpus-per-node" in script
    assert "hpc_shard_worker.py" in script


def test_generate_gpu_diagnostic_script(tmp_path):
    runner = _make_gpu_runner(tmp_path)
    script = runner.generate_gpu_diagnostic_script()
    assert "gpu_diag" in script
    assert "nvidia-smi" in script
    assert "detect_device" in script


def test_generate_gpu_stage_script_raises_when_gpu_disabled(tmp_path):
    cfg = DistributedConfig()
    cfg.run_id = "test_nogpu"
    cfg.gpu.enabled = False

    manifest_path = tmp_path / "manifest.jsonl"
    shards = [
        ShardManifest(
            shard_id="shard_0000",
            stage="interaction_discovery",
            input_paths=[str(tmp_path / "data.parquet")],
            output_path=str(tmp_path / "output" / "shard_0000"),
        )
    ]
    save_manifest(shards, manifest_path)

    runner = SlurmArrayRunner(
        config=cfg,
        manifest_path=manifest_path,
        output_root=tmp_path / "output",
        repo_root=tmp_path,
    )
    with pytest.raises(ValueError, match="gpu.enabled is False"):
        runner.generate_gpu_stage_script("interaction_discovery")


def test_write_scripts_includes_gpu_scripts_when_enabled(tmp_path):
    runner = _make_gpu_runner(tmp_path)
    scripts = runner.write_scripts(tmp_path / "scripts", stage="interaction_discovery")
    assert "gpu_stage" in scripts
    assert "gpu_diagnostic" in scripts
    assert scripts["gpu_stage"].exists()
    assert scripts["gpu_diagnostic"].exists()


def test_write_scripts_no_gpu_scripts_when_disabled(tmp_path):
    cfg = DistributedConfig()
    cfg.run_id = "test_nogpu"
    cfg.gpu.enabled = False

    manifest_path = tmp_path / "manifest.jsonl"
    shards = [
        ShardManifest(
            shard_id="shard_0000",
            stage="interaction_discovery",
            input_paths=[str(tmp_path / "data.parquet")],
            output_path=str(tmp_path / "output" / "shard_0000"),
        )
    ]
    save_manifest(shards, manifest_path)

    runner = SlurmArrayRunner(
        config=cfg,
        manifest_path=manifest_path,
        output_root=tmp_path / "output",
        repo_root=tmp_path,
    )
    scripts = runner.write_scripts(tmp_path / "scripts", stage="interaction_discovery")
    assert "gpu_stage" not in scripts
    assert "gpu_diagnostic" not in scripts


# ---------------------------------------------------------------------------
# gpu_scoring: detect_device (CPU-only machine)
# ---------------------------------------------------------------------------


def test_detect_device_cpu_explicit():
    from bsm_rfm.distributed.gpu_scoring import detect_device

    with patch.dict(os.environ, {"BSM_INTERACTION_DEVICE": "cpu"}):
        assert detect_device("auto") == "cpu"


def test_detect_device_cuda_forced_raises_when_no_cuda():
    from bsm_rfm.distributed.gpu_scoring import detect_device

    # On a CPU-only machine (CI), requesting cuda should raise RuntimeError
    with patch.dict(os.environ, {"BSM_INTERACTION_DEVICE": "cuda"}):
        with patch("bsm_rfm.distributed.gpu_scoring._cuda_available", return_value=False):
            with pytest.raises(RuntimeError, match="CUDA is not available"):
                detect_device("auto")


def test_detect_device_auto_falls_back_to_cpu():
    from bsm_rfm.distributed.gpu_scoring import detect_device

    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("BSM_INTERACTION_DEVICE", None)
        with patch("bsm_rfm.distributed.gpu_scoring._cuda_available", return_value=False):
            assert detect_device("auto") == "cpu"


def test_detect_device_auto_uses_cuda_when_available():
    from bsm_rfm.distributed.gpu_scoring import detect_device

    os.environ.pop("BSM_INTERACTION_DEVICE", None)
    with patch("bsm_rfm.distributed.gpu_scoring._cuda_available", return_value=True):
        assert detect_device("auto") == "cuda"


def test_is_gpu_available_returns_bool():
    from bsm_rfm.distributed.gpu_scoring import is_gpu_available

    # Should return a bool regardless of CUDA availability
    result = is_gpu_available()
    assert isinstance(result, bool)


def test_gpu_score_interaction_pair_cpu_path():
    """Smoke test: CPU path produces a valid float."""
    from bsm_rfm.distributed.gpu_scoring import gpu_score_interaction_pair

    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 5))
    y = rng.standard_normal(50)

    with patch.dict(os.environ, {"BSM_INTERACTION_DEVICE": "cpu"}):
        score = gpu_score_interaction_pair(X, y, 0, 1, n_tree_estimators=10, seed=0)

    assert isinstance(score, float)
    assert score >= 0.0


def test_gpu_score_interaction_batch_cpu_path():
    """Smoke test: batch scoring returns significant pairs on structured data."""
    from bsm_rfm.distributed.gpu_scoring import gpu_score_interaction_batch

    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 4))
    # Make feature 0 and 1 interact strongly
    y = X[:, 0] * X[:, 1] + 0.1 * rng.standard_normal(50)

    with patch.dict(os.environ, {"BSM_INTERACTION_DEVICE": "cpu"}):
        results = gpu_score_interaction_batch(
            X,
            y,
            feature_pairs=[(0, 1), (2, 3)],
            n_permutations=5,
            n_tree_estimators=10,
            device="cpu",
        )

    # Results is a list of significant pairs; may be empty on tiny data but
    # should be a list without error
    assert isinstance(results, list)
    for r in results:
        assert "feature_i" in r
        assert "p_value" in r
        assert r["device"] == "cpu"


# ---------------------------------------------------------------------------
# MPI runner: assign_shards + get_rank_size (no mpi4py)
# ---------------------------------------------------------------------------


def test_get_rank_size_no_mpi():
    from bsm_rfm.distributed.mpi_runner import get_rank_size

    rank, size = get_rank_size()
    # Without mpi4py, should return (0, 1)
    assert rank == 0
    assert size == 1


def test_assign_shards_round_robin():
    from bsm_rfm.distributed.mpi_runner import assign_shards

    # 10 shards, 4 ranks
    assert assign_shards(10, rank=0, size=4) == [0, 4, 8]
    assert assign_shards(10, rank=1, size=4) == [1, 5, 9]
    assert assign_shards(10, rank=2, size=4) == [2, 6]
    assert assign_shards(10, rank=3, size=4) == [3, 7]


def test_assign_shards_single_rank():
    from bsm_rfm.distributed.mpi_runner import assign_shards

    assert assign_shards(5, rank=0, size=1) == [0, 1, 2, 3, 4]


def test_assign_shards_more_ranks_than_shards():
    from bsm_rfm.distributed.mpi_runner import assign_shards

    # 2 shards, 8 ranks: some ranks get nothing
    result = [assign_shards(2, rank=r, size=8) for r in range(8)]
    all_assigned = [s for r in result for s in r]
    assert sorted(all_assigned) == [0, 1]
    assert result[0] == [0]
    assert result[1] == [1]
    assert result[2] == []


def test_require_mpi4py_raises_without_mpi():
    from bsm_rfm.distributed.mpi_runner import _MPI4PY_AVAILABLE, _require_mpi4py

    if not _MPI4PY_AVAILABLE:
        with pytest.raises(ImportError, match="mpi4py"):
            _require_mpi4py()
    else:
        # mpi4py is installed; _require_mpi4py should succeed silently
        _require_mpi4py()


# ---------------------------------------------------------------------------
# Ray runner: gating
# ---------------------------------------------------------------------------


def test_ray_runner_raises_without_env_var():
    from bsm_rfm.distributed.ray_runner_experimental import RayRunner

    cfg = DistributedConfig()
    os.environ.pop("BSM_ENABLE_RAY_EXPERIMENTAL", None)
    with pytest.raises(RuntimeError, match="BSM_ENABLE_RAY_EXPERIMENTAL"):
        RayRunner(cfg)


def test_ray_runner_is_enabled_false_by_default():
    from bsm_rfm.distributed.ray_runner_experimental import RayRunner

    os.environ.pop("BSM_ENABLE_RAY_EXPERIMENTAL", None)
    assert RayRunner.is_enabled() is False


def test_ray_runner_is_enabled_true_with_env():
    from bsm_rfm.distributed.ray_runner_experimental import RayRunner

    with patch.dict(os.environ, {"BSM_ENABLE_RAY_EXPERIMENTAL": "1"}):
        assert RayRunner.is_enabled() is True


def test_ray_runner_is_available_returns_bool():
    from bsm_rfm.distributed.ray_runner_experimental import RayRunner

    assert isinstance(RayRunner.is_available(), bool)


# ---------------------------------------------------------------------------
# DaskRunner: availability and local modes
# ---------------------------------------------------------------------------


def test_dask_runner_is_available():
    from bsm_rfm.distributed.dask_runner import DaskRunner

    assert isinstance(DaskRunner.is_available(), bool)


def test_dask_runner_is_slurm_available():
    from bsm_rfm.distributed.dask_runner import DaskRunner

    assert isinstance(DaskRunner.is_slurm_available(), bool)


def test_dask_runner_raises_without_dask():
    from bsm_rfm.distributed.dask_runner import DaskRunner

    cfg = DistributedConfig()
    with patch("bsm_rfm.distributed.dask_runner._DASK_AVAILABLE", False):
        with pytest.raises(ImportError, match="Dask is required"):
            DaskRunner(cfg)


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("distributed"),
    reason="dask[distributed] not installed",
)
def test_dask_runner_threads_map():
    """Smoke test: local threads DaskRunner maps over a list."""
    from bsm_rfm.distributed.dask_runner import DaskRunner

    cfg = DistributedConfig()
    cfg.dask.scheduler = "threads"
    cfg.dask.n_workers = 2

    runner = DaskRunner(cfg)
    results = runner.map(lambda x: x * 2, [1, 2, 3])
    assert results == [2, 4, 6]
