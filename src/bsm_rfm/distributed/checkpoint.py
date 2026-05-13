"""Idempotent checkpoint and success-marker management for distributed runs.

Each shard writes to a staging directory, validates its output, then
atomically promotes results to the final output directory and writes
_SUCCESS.json. A re-run skips shards whose _SUCCESS.json already exists.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class ShardStatus(Enum):
    """Shard execution status values."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class CheckpointManager:
    """Manage idempotent shard output layout with atomic promotion.

    Layout::

        <output_root>/
            <shard_id>/               ← final promoted output (read by reduce)
                _SUCCESS.json         ← written ONLY after validation passes
                <stage artifacts>
            <shard_id>_attempt_<N>/   ← staging area during execution
                <stage artifacts>

    Usage::

        cm = CheckpointManager("/scratch/bsm_run/work", "task-0042")
        if cm.is_complete():
            # skip
            return
        cm.mark_running()
        try:
            write_outputs(cm.staging_dir)
            cm.validate_and_promote(expected_files=["results.parquet"])
        except Exception as e:
            cm.mark_failed(str(e))
            raise
    """

    SUCCESS_FILENAME = "_SUCCESS.json"
    FAILURE_FILENAME = "_FAILURE.json"

    def __init__(self, output_root: str, shard_id: str):
        self.output_root = Path(output_root)
        self.shard_id = shard_id
        self._attempt = self._detect_current_attempt()

    @property
    def final_dir(self) -> Path:
        """Final promoted output directory path."""
        return self.output_root / self.shard_id

    @property
    def staging_dir(self) -> Path:
        """Staging directory path for current attempt."""
        return self.output_root / f"{self.shard_id}_attempt_{self._attempt}"

    @property
    def success_path(self) -> Path:
        """Path to the _SUCCESS.json marker file."""
        return self.final_dir / self.SUCCESS_FILENAME

    def _detect_current_attempt(self) -> int:
        """Return the next unused attempt number."""
        attempt = 0
        while (self.output_root / f"{self.shard_id}_attempt_{attempt}").exists():
            attempt += 1
        return attempt

    def is_complete(self) -> bool:
        """Return True if _SUCCESS.json exists in the final output directory."""
        return self.success_path.exists()

    def mark_running(self) -> None:
        """Create staging directory and write a running marker."""
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        running_marker = self.staging_dir / "_RUNNING.json"
        running_marker.write_text(
            json.dumps(
                {
                    "shard_id": self.shard_id,
                    "attempt": self._attempt,
                    "pid": os.getpid(),
                    "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                    "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
                },
                indent=2,
            )
        )
        logger.info(
            "[checkpoint] shard=%s attempt=%d staging=%s",
            self.shard_id,
            self._attempt,
            self.staging_dir,
        )

    def validate_and_promote(
        self,
        expected_files: list[str] | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Validate staging outputs and atomically promote to final directory.

        Parameters
        ----------
        expected_files
            File names (relative to staging_dir) that must exist.
        metadata
            Extra fields written into _SUCCESS.json.

        Raises
        ------
        FileNotFoundError
            If any expected file is missing from staging directory.
        RuntimeError
            If promotion fails after retries.
        """
        missing = []
        if expected_files:
            for fname in expected_files:
                if not (self.staging_dir / fname).exists():
                    missing.append(fname)
        if missing:
            raise FileNotFoundError(
                f"[checkpoint] shard={self.shard_id}: missing expected output files: {missing}"
            )

        # Atomic promotion: rename staging dir to final dir
        if self.final_dir.exists():
            # If a previous partial final exists (e.g. from aborted promotion), remove it
            shutil.rmtree(self.final_dir)

        # os.rename is atomic on POSIX within the same filesystem
        self.staging_dir.rename(self.final_dir)
        logger.info("[checkpoint] shard=%s promoted to %s", self.shard_id, self.final_dir)

        # Write _SUCCESS.json after promotion
        success = {
            "shard_id": self.shard_id,
            "attempt": self._attempt,
            "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        }
        if metadata:
            success.update(metadata)
        self.success_path.write_text(json.dumps(success, indent=2))
        logger.info("[checkpoint] shard=%s _SUCCESS written", self.shard_id)

    def mark_failed(self, error_message: str) -> None:
        """Write a failure marker in the staging directory."""
        failure_path = self.staging_dir / self.FAILURE_FILENAME
        try:
            failure_path.write_text(
                json.dumps(
                    {
                        "shard_id": self.shard_id,
                        "attempt": self._attempt,
                        "failed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "error_message": error_message,
                    },
                    indent=2,
                )
            )
        except Exception:
            pass  # best-effort; don't mask original error
        logger.error("[checkpoint] shard=%s FAILED: %s", self.shard_id, error_message)

    def cleanup_staging(self) -> None:
        """Remove staging directory (after successful promotion)."""
        if self.staging_dir.exists():
            shutil.rmtree(self.staging_dir)

    @staticmethod
    def all_complete(shards_output_root: str, shard_ids: list[str]) -> bool:
        """Return True only if every shard has a _SUCCESS.json marker."""
        root = Path(shards_output_root)
        return all((root / sid / CheckpointManager.SUCCESS_FILENAME).exists() for sid in shard_ids)

    @staticmethod
    def completion_summary(shards_output_root: str, shard_ids: list[str]) -> dict[str, int]:
        """Return counts of completed vs pending vs failed shards."""
        root = Path(shards_output_root)
        counts = {"completed": 0, "failed": 0, "pending": 0}
        for sid in shard_ids:
            if (root / sid / CheckpointManager.SUCCESS_FILENAME).exists():
                counts["completed"] += 1
            elif (root / sid).exists():
                # has output dir but no _SUCCESS → failed or in-progress
                failure_markers = list((root / sid).glob("_FAILURE.json"))
                attempt_dirs = list(root.glob(f"{sid}_attempt_*"))
                if failure_markers or (not attempt_dirs):
                    counts["failed"] += 1
                else:
                    counts["pending"] += 1
            else:
                counts["pending"] += 1
        return counts
