"""Distributed and HPC execution adapters for the BSM manuscript pipeline.

Phase 8a: SLURM array baseline — config schema, manifest, checkpoint, sbatch generation.
Phase 8b: Out-of-core chunk processing + checkpoint/recovery.
Phase 8c: Optional Dask/MPI/Ray adapters.
"""

from rfm_pipeline.distributed.checkpoint import CheckpointManager, ShardStatus
from rfm_pipeline.distributed.config_distributed import (
    DaskConfig,
    DistributedConfig,
    GpuConfig,
    KestrelConfig,
    SlurmConfig,
    SpillConfig,
    load_distributed_config,
)
from rfm_pipeline.distributed.dask_runner import DaskRunner
from rfm_pipeline.distributed.gpu_scoring import detect_device, is_gpu_available
from rfm_pipeline.distributed.manifest import (
    ShardManifest,
    build_manifest,
    load_manifest,
    save_manifest,
)
from rfm_pipeline.distributed.mpi_runner import assign_shards, get_rank_size

__all__ = [
    "CheckpointManager",
    "DaskConfig",
    "DaskRunner",
    "DistributedConfig",
    "GpuConfig",
    "KestrelConfig",
    "ShardManifest",
    "ShardStatus",
    "SlurmConfig",
    "SpillConfig",
    "assign_shards",
    "build_manifest",
    "detect_device",
    "get_rank_size",
    "is_gpu_available",
    "load_distributed_config",
    "load_manifest",
    "save_manifest",
]
