"""Out-of-core processing module for handling datasets larger than memory.

This module provides streaming I/O, chunked aggregation, spill-to-disk operations,
and memory budget tracking to enable processing of arbitrarily large datasets
on machines with limited RAM.

Main components:
- ChunkedParquetReader / ChunkedCSVReader: Stream large files in chunks
- StreamingAggregation / StreamingQuantile: Compute aggregations on chunks
- SpillToDiskBuffer / LargeArrayWriter: Write overflow data to disk
- MemoryBudget / choose_temp_dir: Memory and storage management
- ChunkProgress: Live progress telemetry
"""

from .chunked_io import ChunkedCSVReader, ChunkedParquetReader
from .memory import MemoryBudget, choose_temp_dir, get_disk_free_mb
from .progress import ChunkProgress
from .spill_ops import LargeArrayWriter, SpillToDiskBuffer
from .streaming_ops import StreamingAggregation, StreamingQuantile

__all__ = [
    "ChunkedParquetReader",
    "ChunkedCSVReader",
    "StreamingAggregation",
    "StreamingQuantile",
    "SpillToDiskBuffer",
    "LargeArrayWriter",
    "MemoryBudget",
    "choose_temp_dir",
    "get_disk_free_mb",
    "ChunkProgress",
]
