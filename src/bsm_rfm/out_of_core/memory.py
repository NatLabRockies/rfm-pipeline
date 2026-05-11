"""Memory management and temp directory selection for out-of-core processing."""

import logging
import os
import shutil
import tempfile

import psutil

logger = logging.getLogger(__name__)


class MemoryBudget:
    """
    Monitor and enforce memory budget during streaming operations.

    Tracks actual memory usage using psutil and can indicate when spilling
    to disk is recommended.

    Usage:
        budget = MemoryBudget(budget_mb=8000, reserve_mb=500)
        if budget.should_spill():
            save_to_disk(data)
        available = budget.available_mb()
    """

    def __init__(self, budget_mb: int, reserve_mb: int = 500):
        """
        Initialize memory budget tracker.

        Args:
            budget_mb: Total memory budget in MB
            reserve_mb: Reserved memory to keep free (prevents total exhaustion).
        """
        self.budget_mb = budget_mb
        self.reserve_mb = reserve_mb
        self.process = psutil.Process(os.getpid())

        logger.info(f"MemoryBudget initialized: budget={budget_mb} MB, reserve={reserve_mb} MB")

    def current_usage_mb(self) -> float:
        """Return current process memory usage in MB."""
        return self.process.memory_info().rss / (1024 * 1024)

    def available_mb(self) -> float:
        """Return available memory before hitting budget + reserve."""
        usage = self.current_usage_mb()
        return max(0, self.budget_mb - usage - self.reserve_mb)

    def should_spill(self) -> bool:
        """Return True if memory usage exceeds threshold (recommend spilling to disk)."""
        return self.available_mb() < 100  # Spill when <100 MB remaining

    def check_fit(self, estimated_mb: float) -> bool:
        """Check if estimated_mb would fit within budget."""
        return self.available_mb() >= estimated_mb

    def report(self) -> dict:
        """Return memory usage report."""
        usage = self.current_usage_mb()
        return {
            "budget_mb": self.budget_mb,
            "usage_mb": round(usage, 1),
            "available_mb": round(self.available_mb(), 1),
            "pct_used": round(100 * usage / self.budget_mb, 1),
        }


def choose_temp_dir(preferred_root: str = None, fallback_root: str = None) -> str:
    """
    Intelligently select temp directory for spill operations.

    Prefers:
    1. preferred_root if provided and has space
    2. $TMPDIR if it's backed by real disk (not tmpfs/RAM)
    3. fallback_root if provided
    4. system temp directory

    Rejects tmpfs/RAM-backed directories to avoid memory exhaustion.

    Args:
        preferred_root: Preferred directory (e.g., /scratch/$USER/bsm_run)
        fallback_root: Fallback directory (e.g., /projects/bsm/temp)

    Returns
    -------
        Path to temp directory
    """
    candidates = []

    # 1. Preferred root
    if preferred_root:
        candidates.append(("preferred", preferred_root))

    # 2. $TMPDIR environment variable
    env_tmp = os.getenv("TMPDIR")
    if env_tmp:
        candidates.append(("env_tmpdir", env_tmp))

    # 3. Fallback root
    if fallback_root:
        candidates.append(("fallback", fallback_root))

    # 4. System temp
    candidates.append(("system_temp", tempfile.gettempdir()))

    for name, tmpdir in candidates:
        # Check if directory exists or can be created
        try:
            os.makedirs(tmpdir, exist_ok=True)
        except Exception as e:
            logger.debug(f"Cannot create {name} ({tmpdir}): {e}")
            continue

        # Check if it's RAM-backed (tmpfs, /dev/shm)
        try:
            result = shutil.disk_usage(tmpdir)
            os.statvfs(tmpdir)

            # Simple heuristic: if free space == total space, likely tmpfs/RAM
            if result.free >= result.total * 0.9:  # Very loose heuristic
                logger.debug(
                    f"Rejecting {name} ({tmpdir}): "
                    f"appears to be RAM-backed (free={result.free}, total={result.total})"
                )
                continue

            logger.info(
                f"Selected temp dir: {name}={tmpdir} "
                f"(free={result.free / 1024 / 1024 / 1024:.1f} GB)"
            )
            return tmpdir

        except Exception as e:
            logger.debug(f"Error checking {name} ({tmpdir}): {e}")
            continue

    # Fallback to system temp
    result = tempfile.gettempdir()
    logger.warning(f"Fell back to system temp: {result}")
    return result


def get_disk_free_mb(path: str) -> float:
    """Get free disk space at path in MB."""
    try:
        usage = shutil.disk_usage(path)
        return usage.free / (1024 * 1024)
    except Exception as e:
        logger.warning(f"Cannot get disk usage for {path}: {e}")
        return 0
