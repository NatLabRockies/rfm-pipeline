"""Ray-based distributed execution — EXPERIMENTAL.

This module provides a Ray actor pool adapter for the rfm-pipeline.
It is EXPERIMENTAL and disabled by default.

To enable: set env var BSM_ENABLE_RAY_EXPERIMENTAL=1

Why experimental?
  - Ray on Kestrel requires the Ray cluster to be started manually before
    the SLURM job begins, or via ray.init() with specific Kestrel networking.
  - SLURM array jobs and Dask are more battle-tested on Kestrel.
  - Ray adds a persistent head-node process with port requirements that
    conflict with Kestrel's shared-node networking policies.
  - This module exists for workstations and cloud environments (AWS/GCP)
    where Ray's task autoscaling is superior.

Recommended alternatives on Kestrel:
  - SLURM array jobs (Phase 8a): rfm-hpc-submit + array tasks
  - Dask-SLURM (Phase 8c): DaskRunner with scheduler='slurm'
  - MPI (Phase 8c): mpi_runner.py via srun

Usage (if BSM_ENABLE_RAY_EXPERIMENTAL=1)::

    from rfm_pipeline.distributed.ray_runner_experimental import RayRunner

    runner = RayRunner(config)
    results = runner.map(score_fn, shards)

Ray installation::

    pip install ray[default]    # CPU
    pip install ray[default] cupy-cuda12x  # GPU

Reference: https://docs.ray.io/en/latest/cluster/getting-started.html
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rfm_pipeline.distributed.config_distributed import DistributedConfig

logger = logging.getLogger(__name__)

_RAY_AVAILABLE = False
try:
    import ray

    _RAY_AVAILABLE = True
except ImportError:
    pass


def _check_experimental() -> None:
    """Raise if BSM_ENABLE_RAY_EXPERIMENTAL is not set."""
    if not os.environ.get("BSM_ENABLE_RAY_EXPERIMENTAL"):
        raise RuntimeError(
            "Ray runner is experimental. Set BSM_ENABLE_RAY_EXPERIMENTAL=1 to enable. "
            "See src/rfm_pipeline/distributed/ray_runner_experimental.py for caveats."
        )


def _require_ray() -> None:
    """Raise ImportError if ray is not installed."""
    if not _RAY_AVAILABLE:
        raise ImportError("Ray is required for RayRunner. Install with: pip install ray[default]")


class RayRunner:
    """Ray actor pool adapter for rfm-pipeline stages.

    EXPERIMENTAL — see module docstring.

    Parameters
    ----------
    config
        Distributed execution configuration.
    address
        Ray head address. None initialises a local cluster (development).
    num_cpus
        CPUs to claim when starting a local cluster (ignored if address given).
    """

    def __init__(
        self,
        config: DistributedConfig,
        address: str | None = None,
        num_cpus: int | None = None,
    ):
        _check_experimental()
        _require_ray()

        self.config = config
        self._address = address
        self._num_cpus = num_cpus
        self._initialized = False

    def init(self) -> None:
        """Initialise Ray (connect to cluster or start local)."""
        if self._initialized:
            return
        kwargs: dict[str, Any] = {"ignore_reinit_error": True}
        if self._address:
            kwargs["address"] = self._address
            logger.info("[ray_runner] connecting to cluster at %s", self._address)
        else:
            if self._num_cpus:
                kwargs["num_cpus"] = self._num_cpus
            logger.info("[ray_runner] starting local Ray cluster (num_cpus=%s)", self._num_cpus)

        ray.init(**kwargs)
        self._initialized = True
        logger.info("[ray_runner] Ray initialised: %s", ray.cluster_resources())

    def shutdown(self) -> None:
        """Shut down Ray."""
        if self._initialized:
            ray.shutdown()
            self._initialized = False
            logger.info("[ray_runner] Ray shut down")

    def map(
        self,
        fn: Callable,
        items: list[Any],
        **kwargs: Any,
    ) -> list[Any]:
        """Map fn over items using Ray remote tasks, blocking until all complete.

        Parameters
        ----------
        fn
            Callable to apply to each item. Must be importable (not a closure).
        items
            List of inputs.
        **kwargs
            Extra keyword arguments passed to fn.

        Returns
        -------
        list[Any]
            Results in input order.
        """
        self.init()

        remote_fn = ray.remote(fn)
        futures = [remote_fn.remote(item, **kwargs) for item in items]
        results = ray.get(futures)
        return list(results)

    def map_gpu(
        self,
        fn: Callable,
        items: list[Any],
        num_gpus: float = 1.0,
        **kwargs: Any,
    ) -> list[Any]:
        """Map fn over items with GPU resources requested per task.

        Parameters
        ----------
        fn
            Callable to apply to each item.
        items
            List of inputs.
        num_gpus
            Number of GPUs to request per task (can be fractional).
        **kwargs
            Extra keyword arguments passed to fn.

        Returns
        -------
        list[Any]
            Results in input order.
        """
        self.init()

        remote_fn = ray.remote(num_gpus=num_gpus)(fn)
        futures = [remote_fn.remote(item, **kwargs) for item in items]
        results = ray.get(futures)
        return list(results)

    def __enter__(self) -> RayRunner:
        """Enter context manager — initialise Ray."""
        self.init()
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit context manager — shut down Ray."""
        self.shutdown()

    @staticmethod
    def is_available() -> bool:
        """Return True if Ray is installed."""
        return _RAY_AVAILABLE

    @staticmethod
    def is_enabled() -> bool:
        """Return True if BSM_ENABLE_RAY_EXPERIMENTAL=1 is set."""
        return bool(os.environ.get("BSM_ENABLE_RAY_EXPERIMENTAL"))
