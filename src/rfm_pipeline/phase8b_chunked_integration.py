"""Phase 8b chunked I/O integration for sparse_selection and final_artifacts stages.

This module provides wrappers around stage execution functions that optionally
enable chunked I/O based on config settings. When enabled, stages respect memory
budgets by streaming data and results.

Key strategy:
- Detect config.stages.sparse_selection_stability.use_chunked_io
- When True, use out_of_core module components for data I/O
- Delegate core computation to existing stage functions
- Validate numerical equivalence between chunked and in-memory paths

Implementation progression:
1. Phase 8b Slice 1 (complete): Wrapper architecture + detection logic
2. Phase 8b Slice 2 (complete): Memory tracking + config logging
3. Phase 8b Slice 3 (complete): Sparse-selection input streaming + spill-aware wrapper path
4. Phase 8b Slice 4: Integration testing + validation
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import pandas as pd
import psutil

from rfm_pipeline.out_of_core import SpillToDiskBuffer, StreamingAggregation
from rfm_pipeline.out_of_core.memory import choose_temp_dir

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def should_use_chunked_io_for_stage(stage_config: dict | None) -> bool:
    """Check if chunked I/O should be used for a stage.

    Parameters
    ----------
    stage_config
        Stage-specific config dict (e.g., from config.stages.sparse_selection_stability).

    Returns
    -------
    bool
        True if use_chunked_io is set in stage config.
    """
    if stage_config is None:
        return False
    return bool(stage_config.get("use_chunked_io", False))


def _runtime_prefers_chunked_io(context: Any) -> bool:
    """Return True when runtime-level config enables chunked I/O."""
    cfg = getattr(context, "config", None)
    runtime = getattr(cfg, "runtime", None) if cfg is not None else None
    if runtime is None:
        return False

    out_of_core = getattr(runtime, "out_of_core", None)
    if out_of_core is not None:
        enabled = out_of_core.enabled if hasattr(out_of_core, "enabled") else False
        if isinstance(enabled, bool) and enabled:
            return True
        use_chunked = (
            out_of_core.use_chunked_io if hasattr(out_of_core, "use_chunked_io") else False
        )
        if isinstance(use_chunked, bool) and use_chunked:
            return True

    runtime_use_chunked = runtime.use_chunked_io if hasattr(runtime, "use_chunked_io") else False
    if isinstance(runtime_use_chunked, bool) and runtime_use_chunked:
        return True

    legacy_chunked_cfg = runtime.chunked_io_config if hasattr(runtime, "chunked_io_config") else {}
    legacy_chunked_cfg = legacy_chunked_cfg or {}
    return isinstance(legacy_chunked_cfg, dict) and bool(legacy_chunked_cfg)


def _resolve_stage_config(context: Any, stage_name: str) -> dict[str, Any]:
    """Resolve stage config from dict-style or dataclass-style stage containers."""
    cfg = getattr(context, "config", None)
    stages = getattr(cfg, "stages", None) if cfg is not None else None
    if stages is None:
        return {}

    if isinstance(stages, dict):
        stage_cfg = stages.get(stage_name, {})
        return stage_cfg if isinstance(stage_cfg, dict) else {}

    stage_attr = {
        "sparse_selection_stability": "sparse_selection",
        "final_manuscript_artifacts": "final_artifacts",
    }.get(stage_name, stage_name)
    stage_obj = getattr(stages, stage_attr, None)
    if stage_obj is None:
        return {}
    if isinstance(stage_obj, dict):
        return dict(stage_obj)
    if hasattr(stage_obj, "__dict__"):
        return dict(vars(stage_obj))
    return {}


def _resolve_out_of_core_settings(context: Any) -> dict[str, Any]:
    """Resolve out-of-core settings from runtime config with legacy fallbacks."""
    settings: dict[str, Any] = {
        "chunk_size_mb": 512,
        "max_memory_budget_mb": 8000,
        "enable_spill_to_disk": True,
        "temp_dir": None,
    }

    cfg = getattr(context, "config", None)
    runtime = getattr(cfg, "runtime", None) if cfg is not None else None
    if runtime is None:
        return settings

    out_of_core = getattr(runtime, "out_of_core", None)
    if out_of_core is not None:
        chunk_size = (
            out_of_core.chunk_size_mb
            if hasattr(out_of_core, "chunk_size_mb")
            else settings["chunk_size_mb"]
        )
        if isinstance(chunk_size, int):
            settings["chunk_size_mb"] = max(1, chunk_size)

        budget_mb = (
            out_of_core.max_memory_budget_mb
            if hasattr(out_of_core, "max_memory_budget_mb")
            else settings["max_memory_budget_mb"]
        )
        if isinstance(budget_mb, int):
            settings["max_memory_budget_mb"] = max(64, budget_mb)

        spill_enabled = (
            out_of_core.enable_spill_to_disk
            if hasattr(out_of_core, "enable_spill_to_disk")
            else settings["enable_spill_to_disk"]
        )
        if isinstance(spill_enabled, bool):
            settings["enable_spill_to_disk"] = spill_enabled

        temp_dir = (
            out_of_core.temp_dir if hasattr(out_of_core, "temp_dir") else settings["temp_dir"]
        )
        if isinstance(temp_dir, str) or temp_dir is None:
            settings["temp_dir"] = temp_dir

    legacy_chunked_cfg = runtime.chunked_io_config if hasattr(runtime, "chunked_io_config") else {}
    legacy_chunked_cfg = legacy_chunked_cfg or {}
    if isinstance(legacy_chunked_cfg, dict) and legacy_chunked_cfg:
        settings["chunk_size_mb"] = int(
            max(1, legacy_chunked_cfg.get("chunk_size_mb", settings["chunk_size_mb"]))
        )
        settings["max_memory_budget_mb"] = int(
            max(
                64,
                legacy_chunked_cfg.get("max_memory_budget_mb", settings["max_memory_budget_mb"]),
            )
        )
        settings["enable_spill_to_disk"] = bool(
            legacy_chunked_cfg.get("enable_spill_to_disk", settings["enable_spill_to_disk"])
        )
        settings["temp_dir"] = legacy_chunked_cfg.get("temp_dir", settings["temp_dir"])

    return settings


def _estimate_rows_per_chunk(frame: pd.DataFrame, chunk_size_mb: int) -> int:
    """Estimate row chunk size from frame footprint and target MB per chunk."""
    if frame.empty:
        return 1
    target_bytes = max(1, int(chunk_size_mb) * 1024 * 1024)
    bytes_per_row = max(1, int(frame.memory_usage(index=True, deep=True).sum() / len(frame)))
    return max(1, target_bytes // bytes_per_row)


def _iter_frame_chunks(frame: pd.DataFrame, rows_per_chunk: int) -> Iterator[pd.DataFrame]:
    """Yield row chunks from a DataFrame."""
    if frame.empty:
        yield frame.copy()
        return
    for start in range(0, len(frame), rows_per_chunk):
        stop = min(start + rows_per_chunk, len(frame))
        yield frame.iloc[start:stop].copy()


def _stream_dataframe(frame: pd.DataFrame, *, settings: dict[str, Any]) -> tuple[pd.DataFrame, int]:
    """Stream a DataFrame through chunked aggregation/spill and return equivalent content."""
    rows_per_chunk = _estimate_rows_per_chunk(frame, int(settings["chunk_size_mb"]))
    if rows_per_chunk >= len(frame):
        return frame.copy(), 1

    chunk_count = 0
    if bool(settings["enable_spill_to_disk"]):
        temp_root = choose_temp_dir(preferred_root=settings["temp_dir"])
        buffer = SpillToDiskBuffer(
            temp_dir=temp_root,
            max_memory_mb=int(settings["max_memory_budget_mb"]),
        )
        try:
            for chunk in _iter_frame_chunks(frame, rows_per_chunk):
                chunk_count += 1
                buffer.add_chunk(chunk)
            streamed = buffer.get_final_dataframe()
        finally:
            buffer.cleanup()
        return streamed, chunk_count

    agg = StreamingAggregation(operation="concat")
    for chunk in _iter_frame_chunks(frame, rows_per_chunk):
        chunk_count += 1
        agg.add_chunk(chunk)
    streamed = agg.finalize()
    if not isinstance(streamed, pd.DataFrame):
        return frame.copy(), chunk_count
    return streamed, chunk_count


def _log_memory_config(context: Any, stage_name: str) -> None:
    """Log memory configuration for debugging.

    Parameters
    ----------
    context
        Manuscript runtime context.
    stage_name
        Name of the stage (e.g., "sparse_selection_stability").
    """
    cfg = getattr(context, "config", None)
    if cfg is None:
        return
    if not hasattr(cfg, "runtime") or not hasattr(cfg.runtime, "out_of_core"):
        return

    ooc = cfg.runtime.out_of_core
    if ooc.enabled:
        logger.info(
            f"[phase-8b] {stage_name}: out_of_core enabled | "
            f"chunk_size={ooc.chunk_size_mb}MB, "
            f"budget={ooc.max_memory_budget_mb}MB, "
            f"spill={ooc.enable_spill_to_disk}, "
            f"temp_dir={ooc.temp_dir}"
        )


def _get_current_memory_mb() -> float:
    """Get current process memory usage in MB."""
    try:
        process = psutil.Process()
        rss_bytes = process.memory_info().rss
        return rss_bytes / (1024 * 1024)
    except Exception as e:
        logger.debug(f"Could not read memory: {e}")
        return -1.0


def _log_memory_usage(stage_name: str, prefix: str, memory_mb: float) -> None:
    """Log memory usage for a stage.

    Parameters
    ----------
    stage_name
        Name of the stage.
    prefix
        Prefix for the log message (e.g., "before", "after").
    memory_mb
        Memory usage in MB.
    """
    if memory_mb >= 0:
        logger.info(f"[phase-8b] {stage_name} {prefix}: {memory_mb:.1f} MB")
    else:
        logger.debug(f"[phase-8b] {stage_name} {prefix}: memory tracking unavailable")


def wrap_sparse_selection_with_chunked_io(
    original_fn: callable,
) -> callable:
    """Wrap sparse_selection_stability stage to optionally use chunked I/O.

    Parameters
    ----------
    original_fn
        The original run_sparse_selection_stability_stage function.

    Returns
    -------
    callable
        Wrapped function that checks config and delegates appropriately.

    Notes
    -----
    Current implementation:
    - Detects chunked I/O via stage and runtime config
    - Logs memory budget settings
    - Streams sparse-selection input matrix through out-of-core helpers when enabled
    - Delegates to original function
    """

    def wrapped(context: Any) -> Any:
        # Check if chunked I/O is enabled for this stage
        stage_cfg = _resolve_stage_config(context, "sparse_selection_stability")
        use_chunked = should_use_chunked_io_for_stage(stage_cfg) or _runtime_prefers_chunked_io(
            context
        )
        out_of_core_settings = _resolve_out_of_core_settings(context)

        if use_chunked:
            logger.info("[phase-8b] sparse_selection_stability: chunked I/O enabled")
            _log_memory_config(context, "sparse_selection_stability")

        original_input_matrix = None
        if use_chunked and hasattr(context, "tables"):
            input_matrix = context.tables.get("case_study_input_matrix")
            if isinstance(input_matrix, pd.DataFrame):
                streamed_input, chunk_count = _stream_dataframe(
                    input_matrix,
                    settings=out_of_core_settings,
                )
                original_input_matrix = input_matrix
                context.tables["case_study_input_matrix"] = streamed_input
                logger.info(
                    "[phase-8b] sparse_selection_stability streamed input matrix: "
                    f"{len(streamed_input)} rows in {chunk_count} chunk(s)"
                )

        # Track memory usage (Phase 8b Slice 2: memory tracking)
        mem_before = _get_current_memory_mb()
        if mem_before >= 0:
            _log_memory_usage("sparse_selection_stability", "before", mem_before)

        # Delegate to original function after optional streaming input preparation
        try:
            result = original_fn(context)
        finally:
            if original_input_matrix is not None:
                context.tables["case_study_input_matrix"] = original_input_matrix

        # Track memory usage after execution
        mem_after = _get_current_memory_mb()
        if mem_after >= 0:
            _log_memory_usage("sparse_selection_stability", "after", mem_after)
            if mem_before >= 0:
                delta = mem_after - mem_before
                logger.info(f"[phase-8b] sparse_selection_stability memory delta: {delta:+.1f} MB")

        return result

    return wrapped


def wrap_final_artifacts_with_chunked_io(
    original_fn: callable,
) -> callable:
    """Wrap final_manuscript_artifacts stage to optionally use chunked I/O.

    Parameters
    ----------
    original_fn
        The original run_final_manuscript_artifacts_stage function.

    Returns
    -------
    callable
        Wrapped function that checks config and delegates appropriately.

    Notes
    -----
    Current implementation:
    - Detects chunked I/O config via stage config
    - Logs memory budget settings
    - Delegates to original function
    - Future: implement streaming during OLS fitting and bootstrap aggregation
    """

    def wrapped(context: Any) -> Any:
        # Check if chunked I/O is enabled for this stage
        stage_cfg = _resolve_stage_config(context, "final_manuscript_artifacts")
        use_chunked = should_use_chunked_io_for_stage(stage_cfg) or _runtime_prefers_chunked_io(
            context
        )

        if use_chunked:
            logger.info("[phase-8b] final_manuscript_artifacts: chunked I/O enabled")
            _log_memory_config(context, "final_manuscript_artifacts")

        # Track memory usage (Phase 8b Slice 2: memory tracking)
        mem_before = _get_current_memory_mb()
        if mem_before >= 0:
            _log_memory_usage("final_manuscript_artifacts", "before", mem_before)

        # Delegate to original function
        # Phase 8b Slice 2: Added memory tracking
        # TODO (Phase 8b Slice 4): Implement final-artifacts streaming path
        #   - Stream large intermediate dataframes during OLS fitting
        #   - Use streaming aggregation for bootstrap statistics
        #   - Spill coefficient matrices to disk if memory budget exceeded
        result = original_fn(context)

        # Track memory usage after execution
        mem_after = _get_current_memory_mb()
        if mem_after >= 0:
            _log_memory_usage("final_manuscript_artifacts", "after", mem_after)
            if mem_before >= 0:
                delta = mem_after - mem_before
                logger.info(f"[phase-8b] final_manuscript_artifacts memory delta: {delta:+.1f} MB")

        return result

    return wrapped


__all__ = [
    "should_use_chunked_io_for_stage",
    "wrap_sparse_selection_with_chunked_io",
    "wrap_final_artifacts_with_chunked_io",
]
