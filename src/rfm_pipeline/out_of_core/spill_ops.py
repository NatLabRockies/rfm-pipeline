"""Spill-to-disk operations for out-of-core processing."""

import logging
import os
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class SpillToDiskBuffer:
    """
    Accumulate chunks to disk when memory threshold is exceeded.

    Writes chunks to temporary Parquet files and reads back on finalize(),
    supporting operations on datasets larger than available RAM.

    Usage:
        buffer = SpillToDiskBuffer(temp_dir="/scratch/tmp", max_memory_mb=8000)
        for chunk in large_reader:
            buffer.add_chunk(chunk)
        final_df = buffer.get_final_dataframe()
    """

    def __init__(
        self,
        temp_dir: str,
        max_memory_mb: int = 8000,
        schema: pd.DataFrame | None = None,
    ):
        """
        Initialize spill-to-disk buffer.

        Args:
            temp_dir: Directory for temporary Parquet files
            max_memory_mb: Maximum memory before spilling to disk
            schema: Optional schema DataFrame (for validation).
        """
        self.temp_dir = temp_dir
        self.max_memory_mb = max_memory_mb
        self.schema = schema

        # Create temp directory
        os.makedirs(temp_dir, exist_ok=True)
        self.spill_dir = Path(temp_dir) / f"spill_{id(self)}"
        self.spill_dir.mkdir(exist_ok=True)

        # In-memory buffer
        self._buffer = []
        self._buffer_memory_mb = 0
        self._spill_count = 0
        self._total_rows = 0

        logger.info(
            f"SpillToDiskBuffer initialized: temp_dir={temp_dir}, max_memory={max_memory_mb} MB"
        )

    def add_chunk(self, chunk: pd.DataFrame) -> None:
        """Add a chunk of data, spilling to disk if needed."""
        if chunk.empty:
            return

        self._total_rows += len(chunk)
        chunk_memory_mb = chunk.memory_usage(deep=True).sum() / (1024 * 1024)

        # Check if we need to spill
        if self._buffer_memory_mb + chunk_memory_mb > self.max_memory_mb and self._buffer:
            self._spill_to_disk()

        self._buffer.append(chunk)
        self._buffer_memory_mb += chunk_memory_mb

        logger.debug(
            f"Added chunk: {len(chunk)} rows, "
            f"{chunk_memory_mb:.1f} MB (buffer: {self._buffer_memory_mb:.1f} MB)"
        )

    def _spill_to_disk(self) -> None:
        """Write buffered data to disk as Parquet."""
        if not self._buffer:
            return

        # Concatenate buffer
        combined = pd.concat(self._buffer, ignore_index=True)

        # Write to Parquet
        spill_path = self.spill_dir / f"chunk_{self._spill_count:06d}.parquet"
        combined.to_parquet(spill_path, index=False)

        spill_mb = combined.memory_usage(deep=True).sum() / (1024 * 1024)
        logger.info(f"Spilled to disk: {spill_path.name} ({len(combined)} rows, {spill_mb:.1f} MB)")

        # Clear buffer
        self._buffer = []
        self._buffer_memory_mb = 0
        self._spill_count += 1

    def get_final_dataframe(self) -> pd.DataFrame:
        """
        Return combined data from memory + disk.

        Reads back any spilled files and concatenates with buffered data.
        """
        dfs = []

        # Read spilled files
        spill_files = sorted(self.spill_dir.glob("chunk_*.parquet"))
        for spill_file in spill_files:
            df = pd.read_parquet(spill_file)
            dfs.append(df)
            logger.debug(f"Read spill file: {spill_file.name}")

        # Add buffered data
        if self._buffer:
            dfs.append(pd.concat(self._buffer, ignore_index=True))

        # Combine
        if not dfs:
            return pd.DataFrame()

        result = pd.concat(dfs, ignore_index=True)
        logger.info(
            f"SpillToDiskBuffer finalized: {self._spill_count} spilled, {len(result)} total rows"
        )

        return result

    def cleanup(self) -> None:
        """Remove temporary spill directory."""
        import shutil

        if self.spill_dir.exists():
            shutil.rmtree(self.spill_dir)
            logger.info(f"Cleaned up spill directory: {self.spill_dir}")

    def __del__(self):
        """Ensure cleanup on object destruction."""
        try:
            self.cleanup()
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")


class LargeArrayWriter:
    """
    Write large arrays in chunks (Parquet-backed).

    Useful for operations that produce data larger than memory,
    e.g., model coefficient matrices, covariance matrices.

    Usage:
        writer = LargeArrayWriter("output.parquet", dtype='float64', shape=(30000, 5000))
        for chunk_rows in range(0, 30000, 10000):
            writer.write_chunk(compute(chunk_rows), offset=chunk_rows)
        writer.finalize()
    """

    def __init__(
        self,
        output_path: str,
        dtype: str = "float64",
        shape: tuple = None,
        chunk_size_mb: int = 512,
    ):
        """
        Initialize large array writer.

        Args:
            output_path: Output Parquet file
            dtype: Data type
            shape: Expected shape (rows, cols)
            chunk_size_mb: Max chunk size before writing.
        """
        self.output_path = output_path
        self.dtype = dtype
        self.shape = shape
        self.chunk_size_mb = chunk_size_mb

        self._chunks = []
        self._rows_written = 0

        logger.info(
            f"LargeArrayWriter initialized: output={output_path}, dtype={dtype}, shape={shape}"
        )

    def write_chunk(self, chunk: pd.DataFrame, row_offset: int) -> None:
        """Write a chunk at the specified row offset."""
        self._chunks.append((row_offset, chunk))
        self._rows_written = max(self._rows_written, row_offset + len(chunk))

    def finalize(self) -> None:
        """Write all chunks to output file."""
        if not self._chunks:
            logger.warning("No chunks written")
            return

        # Sort by row offset
        self._chunks.sort(key=lambda x: x[0])

        # Extract data
        dfs = [chunk for _, chunk in self._chunks]
        combined = pd.concat(dfs, ignore_index=True)

        combined.to_parquet(self.output_path, index=False)

        logger.info(f"LargeArrayWriter finalized: {self.output_path} ({len(combined)} rows)")
