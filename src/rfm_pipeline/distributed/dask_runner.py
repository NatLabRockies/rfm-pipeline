r"""Dask distributed execution adapter for the rfm-pipeline.

Provides a SLURMCluster-backed Dask client for Kestrel and generic HPC.
Used when SLURM arrays are insufficient (dynamic task scheduling needed).

Configuration via DistributedConfig.dask + DistributedConfig.slurm.

Usage (standalone)::

    from rfm_pipeline.distributed.dask_runner import DaskRunner
    runner = DaskRunner(config)
    with runner.client() as client:
        futures = [client.submit(score_fn, shard) for shard in shards]
        results = client.gather(futures)

Usage (pipeline)::

    pixi run rfm-hpc-submit --config configs/hpc/kestrel_30k.yml \
        --stage interaction_discovery --backend dask_slurm --submit

Backend: 'dask_slurm' in DistributedConfig.backend.
Requires: dask, distributed, dask-jobqueue (optional: available on Kestrel).
Falls back to local Dask ThreadPoolExecutor when dask-jobqueue is not available.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rfm_pipeline.distributed.config_distributed import DistributedConfig

logger = logging.getLogger(__name__)

_DASK_AVAILABLE = False
_DASK_JOBQUEUE_AVAILABLE = False

try:
    import dask  # noqa: F401
    import distributed  # noqa: F401

    _DASK_AVAILABLE = True
except ImportError:
    pass

try:
    from dask_jobqueue import SLURMCluster  # noqa: F401

    _DASK_JOBQUEUE_AVAILABLE = True
except ImportError:
    pass


class DaskRunner:
    """Dask distributed runner for rfm-pipeline stages.

    Supports three scheduler modes controlled by DistributedConfig.dask.scheduler:
      - 'slurm':     SLURMCluster via dask-jobqueue (HPC production)
      - 'processes': Local multiprocessing cluster (workstation)
      - 'threads':   Local threading cluster (debug/CI)

    Parameters
    ----------
    config
        Distributed execution configuration.
    """

    def __init__(self, config: DistributedConfig):
        self.config = config
        self._scheduler = config.dask.scheduler

        if not _DASK_AVAILABLE:
            raise ImportError(
                "Dask is required for DaskRunner. "
                "Install with: pip install dask[distributed] dask-jobqueue"
            )

        if self._scheduler == "slurm" and not _DASK_JOBQUEUE_AVAILABLE:
            logger.warning(
                "[dask_runner] dask-jobqueue not available; falling back to local cluster. "
                "Install dask-jobqueue for SLURM-backed Dask: pip install dask-jobqueue"
            )
            self._scheduler = "threads"

    @contextmanager
    def client(self):
        """Context manager that yields a connected Dask Client.

        Yields
        ------
        distributed.Client
            Connected Dask client.
        """
        if self._scheduler == "slurm":
            yield from self._slurm_client()
        elif self._scheduler == "processes":
            yield from self._local_client(processes=True)
        else:
            yield from self._local_client(processes=False)

    def _slurm_client(self):
        """Create a SLURMCluster-backed Dask client."""
        import distributed
        from dask_jobqueue import SLURMCluster

        cfg = self.config
        dask_cfg = cfg.dask
        slurm_cfg = cfg.slurm

        log_dir = slurm_cfg.resolve_log_dir(cfg.run_id)
        os.makedirs(log_dir, exist_ok=True)

        cluster = SLURMCluster(
            account=slurm_cfg.account,
            queue=slurm_cfg.partition,
            walltime=dask_cfg.walltime,
            memory=f"{dask_cfg.memory_per_worker_gb}GB",
            cores=slurm_cfg.cpus_per_task,
            interface=dask_cfg.network_interface,
            log_directory=log_dir,
            python=_pixi_python_path(cfg.pixi_env_path),
            env_extra=[
                f"export PIXI_HOME={cfg.pixi_env_path}",
                f"export PIXI_CACHE_DIR={cfg.kestrel.pixi_cache_dir}",
            ],
        )
        cluster.scale(dask_cfg.n_workers)
        logger.info(
            "[dask_runner] SLURMCluster: %d workers, %d GB/worker, partition=%s",
            dask_cfg.n_workers,
            dask_cfg.memory_per_worker_gb,
            slurm_cfg.partition,
        )
        client = distributed.Client(cluster)
        logger.info(
            "[dask_runner] Dashboard: %s (forward port %d via SSH)",
            client.dashboard_link,
            dask_cfg.dashboard_port,
        )
        try:
            yield client
        finally:
            client.close()
            cluster.close()
            logger.info("[dask_runner] SLURMCluster shut down")

    def _local_client(self, processes: bool = False):
        """Create a local Dask client (threads or processes)."""
        import distributed

        n_workers = self.config.dask.n_workers
        client = distributed.Client(
            n_workers=n_workers,
            threads_per_worker=1,
            processes=processes,
        )
        logger.info("[dask_runner] local client: %d workers, processes=%s", n_workers, processes)
        try:
            yield client
        finally:
            client.close()
            logger.info("[dask_runner] local client shut down")

    def map(
        self,
        fn: Callable,
        items: list[Any],
        **kwargs: Any,
    ) -> list[Any]:
        """Map fn over items using Dask, blocking until all complete.

        Parameters
        ----------
        fn
            Callable to apply to each item.
        items
            List of inputs.
        **kwargs
            Extra keyword arguments passed to fn.

        Returns
        -------
        list[Any]
            Results in input order.
        """
        with self.client() as client:
            futures = [client.submit(fn, item, **kwargs) for item in items]
            results = client.gather(futures)
        return list(results)

    @staticmethod
    def is_available() -> bool:
        """Return True if Dask is installed."""
        return _DASK_AVAILABLE

    @staticmethod
    def is_slurm_available() -> bool:
        """Return True if dask-jobqueue is installed."""
        return _DASK_JOBQUEUE_AVAILABLE


def _pixi_python_path(pixi_env_path: str) -> str:
    """Return the Python executable path inside a Pixi environment."""
    import sys
    from pathlib import Path

    # On HPC, Pixi stores the env under <pixi_env_path>/envs/default/bin/python
    candidate = Path(pixi_env_path) / "envs" / "default" / "bin" / "python"
    if candidate.exists():
        return str(candidate)
    # Fall back to current Python (useful when running inside the env already)
    return sys.executable
