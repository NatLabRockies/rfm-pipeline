"""Base executor interface for parallel execution across backends."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from typing import Any, Generic, Literal, TypeVar

logger = logging.getLogger("bsm.parallel.executor")

T = TypeVar("T")
R = TypeVar("R")


class ParallelExecutor(ABC, Generic[T, R]):
    """Abstract base class for parallel execution backends."""

    backend: Literal["joblib", "dask", "ray", "mpi"]

    @abstractmethod
    def map(self, func: Callable[[T], R], items: Iterable[T], **opts: Any) -> list[R]:
        """Apply function to items in parallel."""

    @abstractmethod
    def close(self) -> None:
        """Clean up backend resources."""


class JobLibExecutor(ParallelExecutor):
    """Joblib-based executor (default, single-node)."""

    backend = "joblib"

    def __init__(self, n_jobs: int = -1, timeout: int | None = None):
        """Initialize joblib executor."""
        self.n_jobs = n_jobs
        self.timeout = timeout

    def map(self, func: Callable[[T], R], items: Iterable[T], **opts: Any) -> list[R]:
        """Execute via joblib.Parallel."""
        from joblib import Parallel, delayed

        n_jobs = opts.get("n_jobs", self.n_jobs)
        timeout = opts.get("timeout", self.timeout)

        return Parallel(n_jobs=n_jobs, timeout=timeout)(delayed(func)(item) for item in items)

    def close(self) -> None:
        """Joblib requires no explicit cleanup."""
        pass


class DaskExecutor(ParallelExecutor):
    """Dask-based executor for multi-node HPC."""

    backend = "dask"

    def __init__(
        self,
        n_workers: int = 1,
        cores_per_worker: int = 1,
        memory_per_worker: str = "4 GB",
        timeout: int = 3600,
    ):
        """Initialize Dask executor."""
        self.n_workers = n_workers
        self.cores_per_worker = cores_per_worker
        self.memory_per_worker = memory_per_worker
        self.timeout = timeout
        self.cluster = None
        self.client = None

    def _setup_cluster(self) -> None:
        """Lazy initialization of Dask cluster."""
        if self.cluster is not None:
            return

        try:
            from dask.distributed import Client, LocalCluster

            logger.info(
                f"Starting Dask cluster: {self.n_workers} workers, "
                f"{self.cores_per_worker} cores/worker"
            )
            self.cluster = LocalCluster(
                n_workers=self.n_workers,
                threads_per_worker=self.cores_per_worker,
                memory_limit=self.memory_per_worker,
            )
            self.client = Client(self.cluster)
        except ImportError as e:
            msg = "Dask not installed. Install with: pip install dask[distributed]"
            raise ImportError(msg) from e

    def map(self, func: Callable[[T], R], items: Iterable[T], **opts: Any) -> list[R]:
        """Execute via Dask delayed."""
        from dask import delayed

        self._setup_cluster()
        items_list = list(items)
        logger.info(f"Submitting {len(items_list)} tasks to Dask")
        futures = [delayed(func, pure=True)(item) for item in items_list]
        results = [f.compute(scheduler="distributed") for f in futures]
        return results

    def close(self) -> None:
        """Shut down Dask cluster."""
        if self.client:
            self.client.close()
        if self.cluster:
            self.cluster.close()
        logger.info("Dask cluster closed")


def get_executor(
    backend: Literal["joblib", "dask"] = "joblib",
    **opts: Any,
) -> ParallelExecutor:
    """Get configured executor for specified backend."""
    if backend == "joblib":
        return JobLibExecutor(**opts)
    elif backend == "dask":
        return DaskExecutor(**opts)
    else:
        raise ValueError(f"Unknown backend: {backend}")
