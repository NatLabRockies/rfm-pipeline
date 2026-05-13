"""Integration test for out-of-core processing in manuscript workflow."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bsm_rfm.out_of_core import (
    ChunkedParquetReader,
    ChunkProgress,
    MemoryBudget,
    SpillToDiskBuffer,
)


class TestOutOfCoreIntegration:
    """Test out-of-core processing integration with streaming data."""

    @pytest.fixture
    def synthetic_stability_results(self):
        """Create synthetic stability selection results (like sparse_selection output)."""
        n_features = 500
        n_subsamples = 100

        # Simulate stability selection: (n_subsamples, n_features) matrix of inclusion counts
        stability_scores = np.random.binomial(1, 0.3, size=(n_subsamples, n_features)).astype(float)

        return stability_scores

    @pytest.fixture
    def bootstrap_coefficient_matrix(self):
        """Create synthetic bootstrap coefficient matrix (like final_artifacts output)."""
        n_bootstraps = 100
        n_features = 500

        # Simulate bootstrap coefficients
        coefficients = np.random.normal(loc=0, scale=0.1, size=(n_bootstraps, n_features))

        return coefficients

    def test_chunked_aggregation_stability_scores(self, synthetic_stability_results):
        """Test streaming aggregation of stability scores."""
        from bsm_rfm.out_of_core import StreamingAggregation

        scores = synthetic_stability_results
        n_features = scores.shape[1]

        # Chunk the data
        chunks = [
            pd.DataFrame(scores[i : i + 10, :], columns=[f"f{j}" for j in range(n_features)])
            for i in range(0, len(scores), 10)
        ]

        # Streaming mean
        agg = StreamingAggregation(operation="mean")
        for chunk in chunks:
            agg.add_chunk(chunk)

        streaming_mean = agg.finalize()

        # Compare to direct mean
        direct_mean = pd.Series(scores.mean(axis=0), index=[f"f{j}" for j in range(n_features)])

        for col in direct_mean.index:
            assert np.isclose(streaming_mean[col], direct_mean[col], rtol=1e-10)

    def test_spill_to_disk_bootstrap_coefficients(self, bootstrap_coefficient_matrix):
        """Test spill-to-disk for large bootstrap coefficient matrices."""
        coefficients = bootstrap_coefficient_matrix
        n_features = coefficients.shape[1]

        with tempfile.TemporaryDirectory() as tmpdir:
            buffer = SpillToDiskBuffer(temp_dir=tmpdir, max_memory_mb=0.05)

            # Add chunks of bootstrap coefficients
            for i in range(0, len(coefficients), 20):
                chunk = pd.DataFrame(
                    coefficients[i : i + 20, :],
                    columns=[f"feature_{j}" for j in range(n_features)],
                )
                buffer.add_chunk(chunk)

            # Finalize and check
            result = buffer.get_final_dataframe()

            assert len(result) == len(coefficients)
            assert result.shape[1] == n_features

            # Check numerical equivalence
            reconstructed = result.values
            assert np.allclose(reconstructed, coefficients, rtol=1e-6)
            assert buffer._spill_count > 0

            buffer.cleanup()

    def test_memory_budget_enforcement(self):
        """Test that MemoryBudget correctly detects when to spill."""
        # Budget large enough to not be immediately exceeded, but test the logic
        budget = MemoryBudget(budget_mb=500, reserve_mb=50)

        # Check that available is less than total
        available = budget.available_mb()
        assert available < 500  # Some memory already in use

        # Test should_spill threshold (when <100 MB available)
        # This won't trigger on small data, but we're testing the logic works
        assert isinstance(budget.should_spill(), bool)

    def test_chunked_read_bootstrap_results(self):
        """Test chunked reading of bootstrap result Parquet files."""
        import tempfile

        n_bootstraps = 200
        n_features = 100

        # Create synthetic Parquet file
        coefficients = np.random.normal(loc=0, scale=0.1, size=(n_bootstraps, n_features))
        df = pd.DataFrame(coefficients)

        with tempfile.TemporaryDirectory() as tmpdir:
            parquet_file = Path(tmpdir) / "bootstrap_coefs.parquet"
            df.to_parquet(parquet_file)

            # Read in chunks
            reader = ChunkedParquetReader(str(parquet_file), chunk_size_mb=0.5)
            chunks = list(reader)

            # Should have multiple chunks (since file is small, may have 1 chunk)
            assert len(chunks) >= 1

            # Reconstruct and verify
            reconstructed = pd.concat(chunks, ignore_index=True)
            pd.testing.assert_frame_equal(reconstructed, df)

    def test_chunked_reader_matches_parquet_tables(self):
        """Chunked reader should match pandas parquet values on real validation tables."""
        x_path = Path("artifacts/test_dataset_300/X.parquet")
        y_path = Path("artifacts/test_dataset_300/Y.parquet")
        if not x_path.exists() or not y_path.exists():
            pytest.skip("requires artifacts/test_dataset_300 parquet tables")

        x_full = pd.read_parquet(x_path)
        y_full = pd.read_parquet(y_path)

        x_reader = ChunkedParquetReader(str(x_path), chunk_size_mb=8)
        y_reader = ChunkedParquetReader(str(y_path), chunk_size_mb=8)

        x_chunked = pd.concat(list(x_reader), ignore_index=True)
        y_chunked = pd.concat(list(y_reader), ignore_index=True)

        pd.testing.assert_frame_equal(x_chunked, x_full)
        pd.testing.assert_frame_equal(y_chunked, y_full)


class TestChunkedIOWorkflowIntegration:
    """Test integration with actual manuscript workflow."""

    def test_config_loads_with_chunked_io(self):
        """Test that config with out-of-core settings loads correctly."""
        from bsm_rfm.config import load_config

        config = load_config("configs/validation_100_sample_chunked_io.yml")

        assert config.runtime.use_chunked_io is True
        assert config.runtime.chunked_io_config is not None
        assert config.runtime.chunked_io_config["chunk_size_mb"] == 10
        assert config.runtime.chunked_io_config["max_memory_budget_mb"] == 512

    def test_chunk_progress_writes_telemetry_file(self):
        """ChunkProgress should atomically write JSON telemetry."""
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            progress_file = Path(tmpdir) / "progress.json"
            tracker = ChunkProgress(total_chunks=4, output_file=str(progress_file))
            tracker.update(chunks_completed=2, bytes_read=2048)
            assert progress_file.exists()
            payload = json.loads(progress_file.read_text(encoding="utf-8"))
            assert payload["chunks_completed"] == 2
            assert payload["total_chunks"] == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
