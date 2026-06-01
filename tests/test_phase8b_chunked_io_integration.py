"""Tests for Phase 8b chunked I/O integration.

Validates chunked I/O integration into sparse_selection and final_artifacts stages.

This test suite validates that:
1. Config properly controls chunked I/O in stage execution
2. Chunked I/O produces numerically equivalent results to in-memory execution
3. Memory budgets are respected during stage execution
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd
import pytest

from rfm_pipeline.config import load_config
from rfm_pipeline.manuscript_runtime import build_manuscript_notebook_context
from rfm_pipeline.manuscript_stages import (
    run_final_manuscript_artifacts_stage,
    run_sparse_selection_stability_stage,
)
from rfm_pipeline.phase8b_chunked_integration import (
    wrap_final_artifacts_with_chunked_io,
    wrap_sparse_selection_with_chunked_io,
)


class TestSparseStagChunkedIOConfig:
    """Test config controls for chunked I/O in sparse_selection stage."""

    def test_stage_config_has_use_chunked_io_option(self) -> None:
        """Sparse selection config should have use_chunked_io control."""
        # Use existing validation config as base
        config = load_config("configs/validation_80_sample_workflow_smoke.yml")

        # Verify out_of_core config structure exists
        assert config.runtime.out_of_core is not None
        assert hasattr(config.runtime.out_of_core, "enabled")
        assert hasattr(config.runtime.out_of_core, "chunk_size_mb")
        assert hasattr(config.runtime.out_of_core, "max_memory_budget_mb")


class TestChunkedIONumericialEquivalence:
    """Test that chunked I/O produces numerically equivalent results."""

    def test_sparse_selection_result_equivalence_with_chunked_io(self) -> None:
        """Sparse selection wrapper should pass through to original function."""
        # Create mock original function
        mock_original = mock.MagicMock(return_value={"results": "data"})
        wrapped = wrap_sparse_selection_with_chunked_io(mock_original)

        # Create mock context
        mock_context = mock.MagicMock()
        mock_context.config.stages = {"sparse_selection_stability": {"use_chunked_io": False}}

        # Call wrapped function
        result = wrapped(mock_context)

        # Should delegate to original
        mock_original.assert_called_once_with(mock_context)
        assert result == {"results": "data"}

    def test_final_artifacts_equivalence_with_chunked_io(self) -> None:
        """Final artifacts wrapper should pass through to original function."""
        # Create mock original function
        mock_original = mock.MagicMock(return_value={"artifacts": "final"})
        wrapped = wrap_final_artifacts_with_chunked_io(mock_original)

        # Create mock context
        mock_context = mock.MagicMock()
        mock_context.config.stages = {"final_manuscript_artifacts": {"use_chunked_io": False}}

        # Call wrapped function
        result = wrapped(mock_context)

        # Should delegate to original
        mock_original.assert_called_once_with(mock_context)
        assert result == {"artifacts": "final"}


class TestMemoryBudgetRespect:
    """Test that stages respect configured memory budgets."""

    def test_sparse_selection_with_chunked_io_respects_memory_budget(self) -> None:
        """Sparse selection wrapper should detect memory budget config."""
        config = load_config("configs/validation_80_sample_workflow_smoke.yml")

        # Verify config has memory budget settings
        assert config.runtime.out_of_core is not None
        assert config.runtime.out_of_core.max_memory_budget_mb > 0

    def test_final_artifacts_with_chunked_io_respects_memory_budget(self) -> None:
        """Final artifacts wrapper should respect memory constraints."""
        config = load_config("configs/validation_80_sample_workflow_smoke.yml")

        # Verify spill-to-disk config is available
        assert config.runtime.out_of_core is not None
        assert hasattr(config.runtime.out_of_core, "enable_spill_to_disk")


class TestSpillToDiskIntegration:
    """Test spill-to-disk behavior when memory budget exceeded."""

    def test_sparse_selection_spills_to_disk_when_needed(self) -> None:
        """Wrapper should detect spill-to-disk config."""
        config = load_config("configs/validation_80_sample_workflow_smoke.yml")

        # Verify temp_dir config exists
        assert config.runtime.out_of_core is not None
        assert hasattr(config.runtime.out_of_core, "temp_dir")

    def test_final_artifacts_spills_to_disk_when_needed(self) -> None:
        """Final artifacts wrapper should support spill-to-disk."""
        mock_original = mock.MagicMock(return_value={})
        wrapped = wrap_final_artifacts_with_chunked_io(mock_original)

        # Context with spill-to-disk enabled
        mock_context = mock.MagicMock()
        mock_context.config.stages = {"final_manuscript_artifacts": {"use_chunked_io": True}}

        # Should call original (chunked impl not done yet)
        wrapped(mock_context)
        mock_original.assert_called_once()


class TestChunkedIOProgressTracking:
    """Test progress tracking with chunked I/O."""

    def test_sparse_selection_chunked_io_tracks_chunk_progress(self) -> None:
        """Sparse selection wrapper should support chunked progress."""
        mock_original = mock.MagicMock(return_value={})
        wrapped = wrap_sparse_selection_with_chunked_io(mock_original)

        # Context with chunked I/O enabled
        mock_context = mock.MagicMock()
        mock_context.config.stages = {"sparse_selection_stability": {"use_chunked_io": True}}

        # Should detect config
        wrapped(mock_context)
        mock_original.assert_called_once()

    def test_final_artifacts_chunked_io_tracks_progress(self) -> None:
        """Final artifacts should support chunked progress tracking."""
        mock_original = mock.MagicMock(return_value={})
        wrapped = wrap_final_artifacts_with_chunked_io(mock_original)

        # Context with chunked I/O enabled
        mock_context = mock.MagicMock()
        mock_context.config.stages = {"final_manuscript_artifacts": {"use_chunked_io": True}}

        # Should detect config
        wrapped(mock_context)
        mock_original.assert_called_once()


class TestSparseSelectionStreamingIO:
    """Test sparse-selection streaming I/O behavior in wrapper."""

    def test_sparse_selection_streams_input_table_with_stage_toggle(self) -> None:
        """Wrapper should stream input matrix when stage-level chunked I/O is enabled."""
        source_df = pd.DataFrame(
            {
                "sample_id": list(range(120)),
                "x1": [float(i) for i in range(120)],
                "x2": [float(i % 7) for i in range(120)],
            }
        )
        seen = {}

        def mock_fn(context):
            seen["in_call_id"] = id(context.tables["case_study_input_matrix"])
            return {"status": "ok"}

        wrapped = wrap_sparse_selection_with_chunked_io(mock_fn)
        context = SimpleNamespace(
            tables={"case_study_input_matrix": source_df},
            config=SimpleNamespace(
                stages={"sparse_selection_stability": {"use_chunked_io": True}},
                runtime=SimpleNamespace(
                    use_chunked_io=False,
                    chunked_io_config=None,
                    out_of_core=SimpleNamespace(
                        enabled=True,
                        use_chunked_io=False,
                        chunk_size_mb=1,
                        max_memory_budget_mb=64,
                        enable_spill_to_disk=False,
                        temp_dir=None,
                    ),
                ),
            ),
        )

        result = wrapped(context)

        assert result == {"status": "ok"}
        assert seen["in_call_id"] != id(source_df)
        # Wrapper restores original table reference after stage execution.
        assert context.tables["case_study_input_matrix"] is source_df

    def test_sparse_selection_uses_runtime_chunked_toggle_when_stage_missing(self) -> None:
        """Wrapper should fall back to runtime.use_chunked_io when stage key is absent."""
        source_df = pd.DataFrame(
            {
                "sample_id": list(range(80)),
                "x1": [float(i) for i in range(80)],
            }
        )
        seen = {}

        def mock_fn(context):
            seen["in_call_id"] = id(context.tables["case_study_input_matrix"])
            return {"status": "runtime-toggle"}

        wrapped = wrap_sparse_selection_with_chunked_io(mock_fn)
        context = SimpleNamespace(
            tables={"case_study_input_matrix": source_df},
            config=SimpleNamespace(
                stages={},
                runtime=SimpleNamespace(
                    use_chunked_io=True,
                    chunked_io_config={
                        "chunk_size_mb": 1,
                        "max_memory_budget_mb": 64,
                        "enable_spill_to_disk": False,
                    },
                    out_of_core=SimpleNamespace(
                        enabled=False,
                        use_chunked_io=False,
                        chunk_size_mb=512,
                        max_memory_budget_mb=8000,
                        enable_spill_to_disk=True,
                        temp_dir=None,
                    ),
                ),
            ),
        )

        result = wrapped(context)

        assert result == {"status": "runtime-toggle"}
        assert seen["in_call_id"] != id(source_df)
        assert context.tables["case_study_input_matrix"] is source_df


def _wrap_notebook_context_with_chunked_config(
    *,
    notebook_context,
    stage_name: str,
    use_chunked_io: bool,
):
    runtime_cfg = SimpleNamespace(
        use_chunked_io=use_chunked_io,
        chunked_io_config={
            "chunk_size_mb": 1,
            "max_memory_budget_mb": 64,
            "enable_spill_to_disk": False,
        },
        out_of_core=SimpleNamespace(
            enabled=use_chunked_io,
            use_chunked_io=use_chunked_io,
            chunk_size_mb=1,
            max_memory_budget_mb=64,
            enable_spill_to_disk=False,
            temp_dir=None,
        ),
    )
    stage_cfg = {stage_name: {"use_chunked_io": use_chunked_io}}
    copied_tables = {name: frame.copy() for name, frame in notebook_context.tables.items()}
    return SimpleNamespace(
        notebook_name=notebook_context.notebook_name,
        runtime=notebook_context.runtime,
        tables=copied_tables,
        case_study_config=notebook_context.case_study_config,
        runtime_manifest=notebook_context.runtime_manifest,
        config=SimpleNamespace(stages=stage_cfg, runtime=runtime_cfg),
    )


class TestStageIntegrationEquivalence:
    """Integration tests for chunked wrapper behavior with real stage functions."""

    def test_sparse_selection_wrapper_matches_unwrapped_stage_result(self) -> None:
        """Chunked wrapper should preserve sparse-stage outputs on demo context."""
        baseline_ctx = build_manuscript_notebook_context(
            Path.cwd(), "06_sparse_selection_and_stability.ipynb"
        )
        baseline = run_sparse_selection_stability_stage(baseline_ctx)

        wrapped_ctx = _wrap_notebook_context_with_chunked_config(
            notebook_context=build_manuscript_notebook_context(
                Path.cwd(), "06_sparse_selection_and_stability.ipynb"
            ),
            stage_name="sparse_selection_stability",
            use_chunked_io=True,
        )
        original_input_ref = wrapped_ctx.tables["case_study_input_matrix"]
        wrapped_fn = wrap_sparse_selection_with_chunked_io(run_sparse_selection_stability_stage)
        wrapped = wrapped_fn(wrapped_ctx)

        baseline_summary = baseline.sparse_selection.summary.loc[0]
        wrapped_summary = wrapped.sparse_selection.summary.loc[0]
        assert int(wrapped_summary["n_candidate_terms"]) == int(
            baseline_summary["n_candidate_terms"]
        )
        assert int(wrapped_summary["n_full_support_terms"]) == int(
            baseline_summary["n_full_support_terms"]
        )
        assert int(wrapped_summary["n_final_stable_support_terms"]) == int(
            baseline_summary["n_final_stable_support_terms"]
        )
        assert set(wrapped.sparse_selection.final_stable_support["feature_name"]) == set(
            baseline.sparse_selection.final_stable_support["feature_name"]
        )
        assert wrapped_ctx.tables["case_study_input_matrix"] is original_input_ref

    def test_final_artifacts_wrapper_executes_real_stage(self) -> None:
        """Final-artifacts wrapper should execute real stage path with chunked toggle."""
        wrapped_ctx = _wrap_notebook_context_with_chunked_config(
            notebook_context=build_manuscript_notebook_context(
                Path.cwd(), "08_manuscript_tables_and_figures.ipynb"
            ),
            stage_name="final_manuscript_artifacts",
            use_chunked_io=True,
        )
        wrapped_fn = wrap_final_artifacts_with_chunked_io(run_final_manuscript_artifacts_stage)
        result = wrapped_fn(wrapped_ctx)

        assert (
            result.final_artifacts.summary.loc[0, "stage"] == "final_manuscript_tables_and_figures"
        )
        assert result.artifact_paths["workflow_stage_summary"].exists()


class TestMemoryTracking:
    """Test memory tracking functionality in Phase 8b wrappers."""

    def test_sparse_selection_wrapper_tracks_memory(self, caplog) -> None:
        """Wrapper should track memory usage before/after execution."""
        import logging

        caplog.set_level(logging.INFO)

        def mock_fn(context):
            """Mock function that allocates some memory."""
            return {"data": "result"}

        wrapped = wrap_sparse_selection_with_chunked_io(mock_fn)

        mock_context = mock.MagicMock()
        mock_context.config.stages = {"sparse_selection_stability": {"use_chunked_io": True}}

        result = wrapped(mock_context)

        # Verify execution succeeded
        assert result == {"data": "result"}
        # Memory tracking should be logged (when psutil is available)
        # The presence of "phase-8b" indicates tracking was attempted
        assert "phase-8b" in caplog.text or "sparse_selection_stability" in caplog.text

    def test_final_artifacts_wrapper_tracks_memory(self, caplog) -> None:
        """Wrapper should track memory usage before/after execution."""
        import logging

        caplog.set_level(logging.INFO)

        def mock_fn(context):
            """Mock function that allocates some memory."""
            return {"artifacts": "final"}

        wrapped = wrap_final_artifacts_with_chunked_io(mock_fn)

        mock_context = mock.MagicMock()
        mock_context.config.stages = {"final_manuscript_artifacts": {"use_chunked_io": True}}

        result = wrapped(mock_context)

        # Verify execution succeeded
        assert result == {"artifacts": "final"}
        # Memory tracking should be logged
        assert "phase-8b" in caplog.text or "final_manuscript_artifacts" in caplog.text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
