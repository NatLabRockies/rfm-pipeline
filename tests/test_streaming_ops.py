"""Tests for streaming aggregations."""

import numpy as np
import pandas as pd
import pytest

from rfm_pipeline.out_of_core import StreamingAggregation, StreamingQuantile


class TestStreamingAggregation:
    """Test StreamingAggregation for computing stats on chunks."""

    @pytest.fixture
    def sample_data_chunks(self):
        """Create sample data as list of chunks."""
        return [
            pd.DataFrame({"value": np.arange(0, 10), "group": "A"}),
            pd.DataFrame({"value": np.arange(10, 20), "group": "A"}),
            pd.DataFrame({"value": np.arange(20, 30), "group": "B"}),
        ]

    def test_sum_operation(self, sample_data_chunks):
        """Test sum aggregation."""
        agg = StreamingAggregation(operation="sum")

        for chunk in sample_data_chunks:
            agg.add_chunk(chunk)

        result = agg.finalize()

        # Sum of values column
        expected_sum = sum(c["value"].sum() for c in sample_data_chunks)
        assert result["value"] == expected_sum

    def test_mean_operation(self, sample_data_chunks):
        """Test mean aggregation."""
        # Only use numeric columns
        numeric_chunks = [c[["value"]] for c in sample_data_chunks]

        agg = StreamingAggregation(operation="mean")

        for chunk in numeric_chunks:
            agg.add_chunk(chunk)

        result = agg.finalize()

        # Compare to pandas
        all_data = pd.concat(numeric_chunks, ignore_index=True)
        expected = all_data["value"].mean()

        assert np.isclose(result["value"], expected, rtol=1e-10)

    def test_count_operation(self, sample_data_chunks):
        """Test count aggregation."""
        agg = StreamingAggregation(operation="count")

        for chunk in sample_data_chunks:
            agg.add_chunk(chunk)

        result = agg.finalize()
        expected = sum(len(c) for c in sample_data_chunks)

        assert result == expected

    def test_concat_operation(self, sample_data_chunks):
        """Test concatenation operation."""
        agg = StreamingAggregation(operation="concat")

        for chunk in sample_data_chunks:
            agg.add_chunk(chunk)

        result = agg.finalize()

        # Verify shape and content
        assert len(result) == 30
        assert list(result.columns) == ["value", "group"]

    def test_empty_chunks(self):
        """Test with empty chunks."""
        agg = StreamingAggregation(operation="sum")
        agg.add_chunk(pd.DataFrame({"value": []}))
        result = agg.finalize()

        # Empty series for sum
        assert result is None or len(result) == 0

    def test_single_chunk(self):
        """Test with single chunk."""
        chunk = pd.DataFrame({"value": [1, 2, 3, 4, 5]})

        agg = StreamingAggregation(operation="sum")
        agg.add_chunk(chunk)
        result = agg.finalize()

        assert result["value"] == 15


class TestStreamingQuantile:
    """Test StreamingQuantile for computing quantiles on chunks."""

    @pytest.fixture
    def sample_data_chunks(self):
        """Create sample data as list of chunks."""
        return [
            pd.DataFrame({"value": np.arange(0, 100, 10)}),
            pd.DataFrame({"value": np.arange(100, 200, 10)}),
            pd.DataFrame({"value": np.arange(200, 300, 10)}),
        ]

    def test_percentiles(self, sample_data_chunks):
        """Test various percentiles."""
        sq = StreamingQuantile(percentiles=[0.25, 0.5, 0.75], column="value")

        for chunk in sample_data_chunks:
            sq.add_chunk(chunk)

        result = sq.finalize()

        # Result should be a DataFrame with percentile columns
        assert "q25" in result.columns
        assert "q50" in result.columns
        assert "q75" in result.columns

    def test_median(self, sample_data_chunks):
        """Test median calculation."""
        sq = StreamingQuantile(percentiles=[0.5], column="value")

        for chunk in sample_data_chunks:
            sq.add_chunk(chunk)

        result = sq.finalize()

        # Compare to pandas
        all_data = pd.concat(sample_data_chunks, ignore_index=True)
        expected = all_data["value"].median()

        assert np.isclose(result["q50"].iloc[0], expected, rtol=1e-10)

    def test_extreme_quantiles(self, sample_data_chunks):
        """Test min and max (0.0 and 1.0 quantiles)."""
        sq = StreamingQuantile(percentiles=[0.0, 1.0], column="value")

        for chunk in sample_data_chunks:
            sq.add_chunk(chunk)

        result = sq.finalize()

        all_data = pd.concat(sample_data_chunks, ignore_index=True)
        expected_min = all_data["value"].min()
        expected_max = all_data["value"].max()

        assert result["q0"].iloc[0] == expected_min
        assert result["q100"].iloc[0] == expected_max

    def test_empty_chunks(self):
        """Test with empty chunks."""
        sq = StreamingQuantile(percentiles=[0.5], column="value")

        sq.add_chunk(pd.DataFrame({"value": []}))
        result = sq.finalize()

        # Should return empty DataFrame with expected columns
        assert len(result) == 0


class TestStreamingEquivalence:
    """Test that streaming results match non-streaming pandas."""

    def test_large_synthetic_dataset(self):
        """Test streaming vs pandas on larger dataset."""
        # Create synthetic data
        np.random.seed(42)
        n = 10000
        data = pd.DataFrame(
            {
                "value": np.random.normal(loc=100, scale=15, size=n),
                "group": np.random.choice(["A", "B", "C"], size=n),
            }
        )

        # Only use numeric column
        numeric_data = data[["value"]]

        # Streaming approach
        chunks = [numeric_data.iloc[i : i + 1000] for i in range(0, len(numeric_data), 1000)]

        stream_sum = StreamingAggregation(operation="sum")
        stream_mean = StreamingAggregation(operation="mean")
        stream_quantile = StreamingQuantile(percentiles=[0.5], column="value")

        for chunk in chunks:
            stream_sum.add_chunk(chunk)
            stream_mean.add_chunk(chunk)
            stream_quantile.add_chunk(pd.DataFrame({"value": chunk["value"]}))

        # Pandas approach
        pandas_sum = numeric_data["value"].sum()
        pandas_mean = numeric_data["value"].mean()
        pandas_median = numeric_data["value"].median()

        # Check results
        assert np.isclose(stream_sum.finalize()["value"], pandas_sum, rtol=1e-10)
        assert np.isclose(stream_mean.finalize()["value"], pandas_mean, rtol=1e-10)
        assert np.isclose(stream_quantile.finalize()["q50"].iloc[0], pandas_median, rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
