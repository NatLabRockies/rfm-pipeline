"""Distributed and HPC execution adapters for the BSM manuscript pipeline.

Phase 8a: SLURM array baseline — config schema, manifest, checkpoint, sbatch generation.
Phase 8b: Out-of-core chunk processing + checkpoint/recovery.
Phase 8c: Optional Dask/MPI/Ray adapters.
"""

from bsm_rfm.distributed.checkpoint import CheckpointManager, ShardStatus
from bsm_rfm.distributed.config_distributed import (
    DistributedConfig,
    KestrelConfig,
    SlurmConfig,
    SpillConfig,
    load_distributed_config,
)
from bsm_rfm.distributed.manifest import ShardManifest, build_manifest, load_manifest, save_manifest

__all__ = [
    "CheckpointManager",
    "DistributedConfig",
    "KestrelConfig",
    "ShardManifest",
    "ShardStatus",
    "SlurmConfig",
    "SpillConfig",
    "build_manifest",
    "load_distributed_config",
    "load_manifest",
    "save_manifest",
]
