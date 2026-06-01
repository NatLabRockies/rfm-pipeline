"""Tests for parallel executor abstraction."""

import pytest

from rfm_pipeline.parallel import DaskExecutor, JobLibExecutor, get_executor


def test_get_executor_joblib():
    """Test joblib executor factory."""
    executor = get_executor("joblib", n_jobs=2)
    assert isinstance(executor, JobLibExecutor)
    assert executor.n_jobs == 2
    executor.close()


def test_get_executor_dask():
    """Test Dask executor factory."""
    executor = get_executor("dask", n_workers=2, cores_per_worker=2)
    assert isinstance(executor, DaskExecutor)
    assert executor.n_workers == 2
    executor.close()


def test_joblib_executor_serial():
    """Test joblib executor in serial mode."""
    executor = JobLibExecutor(n_jobs=1)

    def square(x):
        return x**2

    results = executor.map(square, [1, 2, 3, 4])
    assert results == [1, 4, 9, 16]
    executor.close()


def test_joblib_executor_parallel():
    """Test joblib executor in parallel mode."""
    executor = JobLibExecutor(n_jobs=2)

    def square(x):
        return x**2

    results = executor.map(square, [1, 2, 3, 4])
    assert results == [1, 4, 9, 16]
    executor.close()


def test_executor_unknown_backend():
    """Test factory rejects unknown backend."""
    with pytest.raises(ValueError, match="Unknown backend"):
        get_executor("unknown_backend")


def test_dask_executor_lazy_init():
    """Test Dask executor doesn't create cluster until needed."""
    executor = DaskExecutor(n_workers=1)
    assert executor.cluster is None
    assert executor.client is None
    executor.close()


@pytest.mark.slow
def test_dask_executor_map():
    """Test Dask executor can execute tasks (requires dask installed)."""
    try:
        executor = DaskExecutor(n_workers=1, cores_per_worker=1, memory_per_worker="1 GB")

        def square(x):
            return x**2

        results = executor.map(square, [1, 2, 3])
        assert results == [1, 4, 9]
        executor.close()
    except ImportError:
        pytest.skip("Dask not available")
