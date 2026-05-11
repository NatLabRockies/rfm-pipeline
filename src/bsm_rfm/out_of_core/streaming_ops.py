"""Streaming aggregation operations that work with chunked data without full materialization."""

import logging
from typing import Any, Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class StreamingAggregation:
    """
    Streaming reduction for operations like sum, mean, count, concat.

    Accumulates statistics incrementally across chunks, allowing operations
    on datasets larger than memory.

    Usage:
        agg = StreamingAggregation("sum")
        for chunk in reader:
            agg.add_chunk(chunk)
        result = agg.finalize()
    """

    def __init__(
        self,
        operation: Literal["sum", "mean", "count", "concat"],
        output_dtype: str | None = None,
    ):
        """
        Initialize streaming aggregation.

        Args:
            operation: Type of aggregation (sum, mean, count, concat)
            output_dtype: Optional dtype for output (e.g., 'float64').
        """
        self.operation = operation
        self.output_dtype = output_dtype

        # State for different aggregations
        self._sum = None
        self._mean_sum = None
        self._mean_count = None
        self._count = 0
        self._concat_list = []
        self._dtypes = None

        logger.info(f"StreamingAggregation initialized: operation={operation}")

    def add_chunk(self, chunk: pd.DataFrame) -> None:
        """Add a chunk of data to the aggregation."""
        if chunk.empty:
            return

        if self.operation == "sum":
            self._add_sum(chunk)
        elif self.operation == "mean":
            self._add_mean(chunk)
        elif self.operation == "count":
            self._add_count(chunk)
        elif self.operation == "concat":
            self._add_concat(chunk)
        else:
            raise ValueError(f"Unknown operation: {self.operation}")

    def _add_sum(self, chunk: pd.DataFrame) -> None:
        """Accumulate sum."""
        if self._sum is None:
            self._sum = chunk.sum()
            self._dtypes = chunk.dtypes
        else:
            self._sum = self._sum.add(chunk.sum(), fill_value=0)

    def _add_mean(self, chunk: pd.DataFrame) -> None:
        """Accumulate statistics for mean calculation."""
        if self._mean_sum is None:
            self._mean_sum = chunk.sum()
            self._mean_count = len(chunk)
            self._dtypes = chunk.dtypes
        else:
            self._mean_sum = self._mean_sum.add(chunk.sum(), fill_value=0)
            self._mean_count += len(chunk)

    def _add_count(self, chunk: pd.DataFrame) -> None:
        """Accumulate count."""
        self._count += len(chunk)

    def _add_concat(self, chunk: pd.DataFrame) -> None:
        """Accumulate for concatenation."""
        self._concat_list.append(chunk)
        if self._dtypes is None:
            self._dtypes = chunk.dtypes

    def finalize(self) -> Any:
        """Return the final aggregation result."""
        if self.operation == "sum":
            result = self._sum
        elif self.operation == "mean":
            if self._mean_count == 0:
                result = pd.Series()
            else:
                result = self._mean_sum / self._mean_count
        elif self.operation == "count":
            result = self._count
        elif self.operation == "concat":
            if not self._concat_list:
                result = pd.DataFrame()
            else:
                result = pd.concat(self._concat_list, ignore_index=True)
        else:
            raise ValueError(f"Unknown operation: {self.operation}")

        # Apply output dtype if specified
        if self.output_dtype and isinstance(result, pd.Series):
            result = result.astype(self.output_dtype)

        logger.info(f"StreamingAggregation finalized: operation={self.operation}")
        return result


class StreamingQuantile:
    """
    Compute approximate quantiles from streaming chunks.

    Uses a simple algorithm: accumulate sorted chunks and compute quantiles
    at the end. For exact quantiles on smaller datasets; for approximate
    quantiles on very large streams, consider t-digest or similar.

    Usage:
        sq = StreamingQuantile([0.25, 0.5, 0.75])
        for chunk in reader:
            sq.add_chunk(chunk)
        quantiles_df = sq.finalize()
    """

    def __init__(self, percentiles: list[float], column: str | None = None):
        """
        Initialize streaming quantile calculator.

        Args:
            percentiles: List of percentiles to compute (e.g., [0.25, 0.5, 0.75])
            column: If specified, compute quantiles for this column only.
        """
        self.percentiles = percentiles
        self.column = column
        self._data = []

        logger.info(f"StreamingQuantile initialized: percentiles={percentiles}, column={column}")

    def add_chunk(self, chunk: pd.DataFrame) -> None:
        """Add chunk of data."""
        if self.column:
            if self.column in chunk.columns:
                self._data.extend(chunk[self.column].dropna().tolist())
        else:
            # For non-column data, assume chunk is numeric
            if isinstance(chunk, pd.Series):
                self._data.extend(chunk.dropna().tolist())
            else:
                raise ValueError("StreamingQuantile requires column or Series input")

    def finalize(self) -> pd.DataFrame:
        """Return quantiles as a DataFrame."""
        if not self._data:
            return pd.DataFrame({f"q{int(p * 100)}": [] for p in self.percentiles})

        # Sort all data (this loads entire dataset into memory at finalize step)
        sorted_data = sorted(self._data)

        result = {f"q{int(p * 100)}": np.quantile(sorted_data, p) for p in self.percentiles}

        logger.info(
            f"StreamingQuantile finalized: {len(self._data)} total values, "
            f"computed {len(self.percentiles)} quantiles"
        )

        return pd.DataFrame([result])
