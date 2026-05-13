"""Typed configuration schema for distributed and HPC execution.

Supports SLURM array jobs on NREL Kestrel and generic HPC clusters.
All fields have sensible defaults that work on Kestrel with account=bsm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class SpillConfig:
    """Spill-to-disk and scratch filesystem configuration."""

    scratch_root: str = "/scratch/${USER}/bsm"
    """Scratch directory root for temporary distributed working sets."""
    tmpdir_spill_enabled: bool = False
    """Whether to use $TMPDIR for spill (disabled by default: Kestrel $TMPDIR can be RAM-backed)."""
    min_free_gb: float = 10.0
    """Minimum free space (GB) required on scratch before writing."""
    spill_chunk_size_mb: int = 256
    """Chunk size (MB) when flushing in-memory buffers to spill directory."""

    def resolve_scratch(self, run_id: str) -> str:
        """Expand environment variables and append run_id to scratch root."""
        import os

        base = os.path.expandvars(self.scratch_root)
        return str(Path(base) / run_id)


@dataclass
class SlurmConfig:
    """SLURM job submission parameters for array-based execution."""

    account: str = "bsm"
    """SLURM account (--account=)."""
    partition: str = "shared"
    """SLURM partition (--partition=). Use 'debug' for smoke tests, 'shared' for production."""
    walltime: str = "04:00:00"
    """Job walltime (--time=). Format HH:MM:SS."""
    memory_gb: int = 64
    """Memory per node in GB (--mem=). Default is conservative for shared partition."""
    cpus_per_task: int = 36
    """CPUs per SLURM task (--cpus-per-task=)."""
    max_concurrent_array_tasks: int = 50
    """Maximum simultaneously running array tasks (controls %N throttle)."""
    log_dir: str = "/scratch/${USER}/bsm/${RUN_ID}/logs"
    """SLURM stdout/stderr log directory."""
    requeue: bool = False
    """Whether to requeue on node failure. Kestrel PreemptMode=OFF so this is usually false."""
    signal_seconds_before_timeout: int = 60
    """Send USR1 signal this many seconds before walltime to trigger checkpoint."""

    def resolve_log_dir(self, run_id: str) -> str:
        """Expand environment variables and append run_id to log directory path."""
        import os

        return os.path.expandvars(self.log_dir.replace("${RUN_ID}", run_id))


@dataclass
class KestrelConfig:
    """NREL Kestrel-specific HPC settings (validated against live system probe)."""

    projects_root: str = "/projects/bsm"
    """Durable project root for Pixi cache, configs, and final artifacts."""
    pixi_cache_dir: str = "/projects/bsm/.cache/pixi"
    """Shared Pixi package cache (avoids redundant downloads on each node)."""
    dask_network_interface: str = "hsn0"
    """High-speed network interface for Dask cluster communication."""
    max_array_size: int = 11000
    """Kestrel MaxArraySize (from live system probe: 2026-05-10)."""
    debug_partition: str = "debug"
    """Partition for quick smoke tests (<= 1 hour)."""
    gpu_partition: str = "gpu-h100s"
    """GPU partition for GPU-accelerated stages (optional)."""
    production_partition: str = "shared"
    """Default production partition."""


@dataclass
class DistributedConfig:
    """Complete distributed execution configuration.

    Can be embedded in a WorkflowConfig YAML or loaded standalone.

    Example YAML::

        distributed:
          enabled: true
          backend: slurm_array
          run_id: bsm_2026_full_30k
          pixi_env_path: /projects/bsm/.pixi
          slurm:
            account: bsm
            partition: shared
            walltime: "08:00:00"
            memory_gb: 128
            cpus_per_task: 104
          kestrel:
            projects_root: /projects/bsm
          spill:
            scratch_root: /scratch/${USER}/bsm
            min_free_gb: 20.0
    """

    enabled: bool = False
    """Enable distributed/HPC execution. When false, runs locally as usual."""
    backend: str = "slurm_array"
    """Execution backend: slurm_array | dask_slurm | mpi | ray_experimental."""
    run_id: str = "bsm_run"
    """Unique run identifier used for artifact and scratch directory naming."""
    pixi_env_path: str = "/projects/bsm/.pixi"
    """Path to Pixi environment on shared filesystem (all nodes must see this)."""
    slurm: SlurmConfig = field(default_factory=SlurmConfig)
    kestrel: KestrelConfig = field(default_factory=KestrelConfig)
    spill: SpillConfig = field(default_factory=SpillConfig)

    def validate(self) -> list[str]:
        """Return a list of validation errors; empty list means valid."""
        errors = []
        if self.backend not in {"slurm_array", "dask_slurm", "mpi", "ray_experimental"}:
            errors.append(
                f"Unknown backend '{self.backend}'. Must be one of: "
                "slurm_array, dask_slurm, mpi, ray_experimental"
            )
        if not self.run_id or not self.run_id.strip():
            errors.append("run_id must be a non-empty string")
        if self.slurm.memory_gb < 1:
            errors.append("slurm.memory_gb must be >= 1")
        if self.slurm.cpus_per_task < 1:
            errors.append("slurm.cpus_per_task must be >= 1")
        if self.slurm.max_concurrent_array_tasks < 1:
            errors.append("slurm.max_concurrent_array_tasks must be >= 1")
        if self.spill.min_free_gb < 0:
            errors.append("spill.min_free_gb must be >= 0")
        return errors


def _dict_to_dataclass(cls, data: dict):
    """Recursively convert a dict to a nested dataclass, ignoring unknown keys."""
    import dataclasses

    if not dataclasses.is_dataclass(cls):
        return data

    field_types = {f.name: f.type for f in dataclasses.fields(cls)}
    kwargs = {}
    for key, value in data.items():
        if key not in field_types:
            continue  # ignore unknown config keys
        # resolve string annotations
        field_type = field_types[key]
        if isinstance(field_type, str):
            import sys

            frame_globals = sys.modules[cls.__module__].__dict__
            field_type = eval(field_type, frame_globals)  # noqa: S307
        if dataclasses.is_dataclass(field_type) and isinstance(value, dict):
            kwargs[key] = _dict_to_dataclass(field_type, value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def load_distributed_config(path: str | Path) -> DistributedConfig:
    """Load a DistributedConfig from a YAML file.

    The YAML may contain a top-level 'distributed' key (embedded in WorkflowConfig)
    or be a standalone distributed config file.

    Parameters
    ----------
    path
        Path to YAML file.

    Returns
    -------
    DistributedConfig
    """
    raw = yaml.safe_load(Path(path).read_text())
    if "distributed" in raw:
        raw = raw["distributed"]
    return _dict_to_dataclass(DistributedConfig, raw)
