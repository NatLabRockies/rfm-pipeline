"""Tests for out-of-core chunked I/O."""

import os
import tempfile

import pandas as pd
import pytest

from rfm_pipeline.out_of_core import ChunkedCSVReader, ChunkedParquetReader


class TestChunkedParquetReader:
    """Test ChunkedParquetReader with small Parquet files."""

    @pytest.fixture
    def sample_parquet_file(self):
        """Create a temporary Parquet file with sample data."""
        df = pd.DataFrame(
            {
                "id": range(1000),
                "value": [i * 1.5 for i in range(1000)],
                "category": ["A" if i % 2 == 0 else "B" for i in range(1000)],
            }
        )

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            df.to_parquet(f.name, index=False)
            yield f.name

        os.unlink(f.name)

    def test_basic_reading(self, sample_parquet_file):
        """Test basic chunked reading."""
        reader = ChunkedParquetReader(sample_parquet_file, chunk_size_mb=1)
        chunks = list(reader)

        # Should chunk at MB boundaries, concatenate and compare to original
        result = pd.concat(chunks, ignore_index=True)
        original = pd.read_parquet(sample_parquet_file)
        pd.testing.assert_frame_equal(result, original)

    def test_small_chunk_size(self, sample_parquet_file):
        """Test with very small chunk size."""
        reader = ChunkedParquetReader(sample_parquet_file, chunk_size_mb=0.001)
        chunks = list(reader)

        # Verify all data is preserved (chunking happens at row group boundaries)
        result = pd.concat(chunks, ignore_index=True)
        original = pd.read_parquet(sample_parquet_file)
        pd.testing.assert_frame_equal(result, original)

    def test_large_chunk_size(self, sample_parquet_file):
        """Test when chunk size exceeds file size."""
        reader = ChunkedParquetReader(sample_parquet_file, chunk_size_mb=1000)
        chunks = list(reader)

        # Should return single chunk with all rows
        assert len(chunks) == 1
        assert len(chunks[0]) == 1000

    def test_empty_file(self):
        """Test with empty Parquet file."""
        df = pd.DataFrame({"id": [], "value": []})

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            df.to_parquet(f.name, index=False)
            empty_file = f.name

        try:
            reader = ChunkedParquetReader(empty_file, chunk_size_mb=1)
            chunks = list(reader)
            # Empty file returns single empty chunk
            assert len(chunks) <= 1
            if chunks:
                assert len(chunks[0]) == 0
        finally:
            os.unlink(empty_file)

    def test_column_selection(self, sample_parquet_file):
        """Test reading specific columns only."""
        reader = ChunkedParquetReader(sample_parquet_file, chunk_size_mb=1, columns=["id", "value"])
        chunks = list(reader)

        result = pd.concat(chunks, ignore_index=True)
        assert list(result.columns) == ["id", "value"]


class TestChunkedCSVReader:
    """Test ChunkedCSVReader with CSV files."""

    @pytest.fixture
    def sample_csv_file(self):
        """Create a temporary CSV file with sample data."""
        df = pd.DataFrame(
            {
                "id": range(500),
                "value": [i * 2.0 for i in range(500)],
                "name": [f"item_{i}" for i in range(500)],
            }
        )

        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            df.to_csv(f.name, index=False)
            yield f.name

        os.unlink(f.name)

    def test_basic_reading(self, sample_csv_file):
        """Test basic CSV chunked reading."""
        reader = ChunkedCSVReader(sample_csv_file, chunk_size=50)
        chunks = list(reader)

        # Should have 10 chunks of 50 rows each
        assert len(chunks) == 10
        assert all(len(c) == 50 for c in chunks)

    def test_chunk_size_larger_than_file(self, sample_csv_file):
        """Test when chunk size exceeds file size."""
        reader = ChunkedCSVReader(sample_csv_file, chunk_size=1000)
        chunks = list(reader)

        # Should return single chunk
        assert len(chunks) == 1
        assert len(chunks[0]) == 500

    def test_data_types_preserved(self, sample_csv_file):
        """Test that data types are correctly inferred."""
        reader = ChunkedCSVReader(sample_csv_file, chunk_size=100)
        chunks = list(reader)

        # Check types in first chunk
        chunk = chunks[0]
        assert pd.api.types.is_integer_dtype(chunk["id"])
        assert pd.api.types.is_float_dtype(chunk["value"])
        # String columns are str or object dtype depending on pandas version
        is_string = pd.api.types.is_string_dtype(chunk["name"])
        is_object = pd.api.types.is_object_dtype(chunk["name"])
        assert is_string or is_object


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
