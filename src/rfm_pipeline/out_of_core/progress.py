"""Progress telemetry for streaming operations."""

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ChunkProgress:
    """
    Track and report progress of chunk-based operations.

    Supports live monitoring via JSON telemetry files.

    Usage:
        progress = ChunkProgress(total_chunks=30, output_file="progress.json")
        for i, chunk in enumerate(reader):
            process(chunk)
            progress.update(chunks_completed=i+1, bytes_read=chunk_bytes)
            # Can watch with: watch -n 1 cat progress.json
    """

    def __init__(
        self,
        total_chunks: int | None = None,
        output_file: str | None = None,
    ):
        """
        Initialize chunk progress tracker.

        Args:
            total_chunks: Total number of chunks (for ETA calculation)
            output_file: Optional path to write JSON progress updates.
        """
        self.total_chunks = total_chunks
        self.output_file = output_file

        self._chunks_completed = 0
        self._bytes_read = 0
        self._start_time = time.time()

        if output_file:
            self.output_path = Path(output_file)
            self.output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"ChunkProgress initialized: total_chunks={total_chunks}, output_file={output_file}"
        )

    def update(
        self,
        chunks_completed: int,
        bytes_read: int = 0,
    ) -> None:
        """
        Update progress.

        Args:
            chunks_completed: Number of chunks processed so far
            bytes_read: Total bytes read so far
        """
        self._chunks_completed = chunks_completed
        self._bytes_read = bytes_read

        # Write telemetry
        if self.output_file:
            self._write_telemetry()

    def _write_telemetry(self) -> None:
        """Write progress telemetry to JSON file."""
        elapsed = time.time() - self._start_time

        report = {
            "chunks_completed": self._chunks_completed,
            "total_chunks": self.total_chunks,
            "bytes_read": self._bytes_read,
            "elapsed_seconds": round(elapsed, 1),
        }

        # Calculate ETA if total known
        if self.total_chunks and self._chunks_completed > 0:
            rate = self._chunks_completed / elapsed  # chunks per second
            remaining = self.total_chunks - self._chunks_completed
            eta_seconds = remaining / rate if rate > 0 else None

            report["rate_chunks_per_sec"] = round(rate, 2)
            report["eta_seconds"] = round(eta_seconds, 1) if eta_seconds else None
            report["pct_complete"] = round(100 * self._chunks_completed / self.total_chunks, 1)

        # Write with atomic rename to avoid partial reads
        tmp_file = str(self.output_path) + ".tmp"
        with open(tmp_file, "w") as f:
            json.dump(report, f, indent=2)

        Path(tmp_file).replace(self.output_path)

    def report(self) -> dict[str, Any]:
        """Return current progress report as dictionary."""
        elapsed = time.time() - self._start_time

        report = {
            "chunks_completed": self._chunks_completed,
            "total_chunks": self.total_chunks,
            "bytes_read": self._bytes_read,
            "elapsed_seconds": round(elapsed, 1),
        }

        if self.total_chunks and self._chunks_completed > 0:
            rate = self._chunks_completed / elapsed
            remaining = self.total_chunks - self._chunks_completed
            eta_seconds = remaining / rate if rate > 0 else None

            report["rate_chunks_per_sec"] = round(rate, 2)
            report["eta_seconds"] = round(eta_seconds, 1) if eta_seconds else None
            report["pct_complete"] = round(100 * self._chunks_completed / self.total_chunks, 1)

        return report
