"""Chunked I/O operations for streaming large Parquet/CSV files without full materialization."""

import logging
from collections.abc import Iterator

import pandas as pd
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


class ChunkedParquetReader:
    """
    Stream large Parquet files in chunks without loading entire file into memory.

    Respects Parquet row group boundaries for efficient chunking.
    Supports column selection and filtering for further memory efficiency.

    Usage:
        reader = ChunkedParquetReader("data.parquet", chunk_size_mb=512, columns=["col1", "col2"])
        for chunk_df in reader:
            process(chunk_df)
    """

    def __init__(
        self,
        path: str,
        chunk_size_mb: int = 512,
        columns: list[str] | None = None,
    ):
        """
        Initialize Parquet chunked reader.

        Args:
            path: Path to Parquet file
            chunk_size_mb: Target chunk size in MB (may not be exact due to row-group boundaries)
            columns: Optional list of columns to read. If None, reads all columns.
        """
        self.path = path
        self.chunk_size_mb = chunk_size_mb
        self.columns = columns

        # Read parquet metadata to get row group information
        parquet_file = pq.ParquetFile(path)
        self.num_row_groups = parquet_file.num_row_groups
        self.metadata = parquet_file.metadata

        # Compute row group sizes
        self._row_group_sizes = []
        for i in range(self.num_row_groups):
            rg = self.metadata.row_group(i)
            size_bytes = rg.total_byte_size
            self._row_group_sizes.append(size_bytes)

        logger.info(
            f"ChunkedParquetReader initialized: {path}, "
            f"{self.num_row_groups} row groups, "
            f"target chunk {chunk_size_mb} MB"
        )

    def __iter__(self) -> Iterator[pd.DataFrame]:
        """Yield chunks of the Parquet file."""
        parquet_file = pq.ParquetFile(self.path)

        current_rgs = []
        current_size_bytes = 0
        target_size_bytes = self.chunk_size_mb * 1024 * 1024

        for rg_idx in range(self.num_row_groups):
            rg_size = self._row_group_sizes[rg_idx]
            current_rgs.append(rg_idx)
            current_size_bytes += rg_size

            # Flush chunk if we exceed target size or reach end
            if current_size_bytes >= target_size_bytes or rg_idx == self.num_row_groups - 1:
                # Read row groups as a single chunk
                table = parquet_file.read_row_groups(current_rgs, columns=self.columns)
                df = table.to_pandas()

                current_size_bytes / (1024 * 1024)
                logger.debug(
                    f"Yielding chunk with {len(df)} rows, "
                    f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB"
                )

                yield df

                current_rgs = []
                current_size_bytes = 0

    def __len__(self) -> int:
        """Return total number of rows in the file (reads metadata only)."""
        parquet_file = pq.ParquetFile(self.path)
        return parquet_file.metadata.num_rows

    @property
    def num_chunks(self) -> int:
        """Approximate number of chunks we will yield."""
        if not self._row_group_sizes:
            return 1

        total_bytes = sum(self._row_group_sizes)
        target_bytes = self.chunk_size_mb * 1024 * 1024

        return max(1, int((total_bytes + target_bytes - 1) // target_bytes))


class ChunkedCSVReader:
    """
    Stream large CSV files in chunks.

    Note: Parquet is generally preferred for efficiency. CSV is fallback option.

    Usage:
        reader = ChunkedCSVReader("data.csv", chunk_size=10000)
        for chunk_df in reader:
            process(chunk_df)
    """

    def __init__(self, path: str, chunk_size: int = 10000, sep: str = ",", **read_csv_kwargs):
        """
        Initialize CSV chunked reader.

        Args:
            path: Path to CSV file
            chunk_size: Number of rows per chunk
            sep: CSV separator
            **read_csv_kwargs: Additional arguments passed to pd.read_csv().
        """
        self.path = path
        self.chunk_size = chunk_size
        self.sep = sep
        self.read_csv_kwargs = read_csv_kwargs

        logger.info(f"ChunkedCSVReader initialized: {path}, chunk_size={chunk_size} rows")

    def __iter__(self) -> Iterator[pd.DataFrame]:
        """Yield chunks of the CSV file."""
        chunks = pd.read_csv(
            self.path, sep=self.sep, chunksize=self.chunk_size, **self.read_csv_kwargs
        )

        for i, chunk in enumerate(chunks):
            logger.debug(
                f"Yielding CSV chunk {i}: {len(chunk)} rows, "
                f"{chunk.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB"
            )
            yield chunk

    @property
    def num_chunks(self) -> int:
        """Approximate number of chunks (requires reading first row to estimate)."""
        pd.read_csv(self.path, sep=self.sep, nrows=self.chunk_size)
        total_rows = sum(1 for _ in open(self.path)) - 1  # -1 for header
        return max(1, int((total_rows + self.chunk_size - 1) // self.chunk_size))
