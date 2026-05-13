"""Parallel execution abstraction layer for distributed computing."""

from .executor import DaskExecutor, JobLibExecutor, ParallelExecutor, get_executor

__all__ = [
    "ParallelExecutor",
    "JobLibExecutor",
    "DaskExecutor",
    "get_executor",
]
