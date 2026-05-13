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
2. Phase 8b Slice 2: Sparse selection chunked I/O (stream output writing)
3. Phase 8b Slice 3: Final artifacts chunked I/O (stream OLS fitting)
4. Phase 8b Slice 4: Integration testing + validation
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

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


def _log_memory_config(context: Any, stage_name: str) -> None:
    """Log memory configuration for debugging.

    Parameters
    ----------
    context
        Manuscript runtime context.
    stage_name
        Name of the stage (e.g., "sparse_selection_stability").
    """
    if not hasattr(context.config, "runtime") or not hasattr(context.config.runtime, "out_of_core"):
        return

    ooc = context.config.runtime.out_of_core
    if ooc.enabled:
        logger.info(
            f"[phase-8b] {stage_name}: out_of_core enabled | "
            f"chunk_size={ooc.chunk_size_mb}MB, "
            f"budget={ooc.max_memory_budget_mb}MB, "
            f"spill={ooc.enable_spill_to_disk}, "
            f"temp_dir={ooc.temp_dir}"
        )


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
    - Detects chunked I/O config via stage config
    - Logs memory budget settings
    - Delegates to original function
    - Future: implement streaming output writing during stability resamples
    """

    def wrapped(context: Any) -> Any:
        # Check if chunked I/O is enabled for this stage
        stage_cfg = context.config.stages.get("sparse_selection_stability", {})
        use_chunked = should_use_chunked_io_for_stage(stage_cfg)

        if use_chunked:
            logger.info("[phase-8b] sparse_selection_stability: chunked I/O enabled")
            _log_memory_config(context, "sparse_selection_stability")

        # Delegate to original function
        # TODO (Phase 8b Slice 2): Implement chunked I/O streaming
        #   - Stream stability resample results instead of accumulating all in memory
        #   - Use out_of_core module for intermediate dataframe I/O
        #   - Spill large resample arrays to disk if memory budget exceeded
        return original_fn(context)

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
        stage_cfg = context.config.stages.get("final_manuscript_artifacts", {})
        use_chunked = should_use_chunked_io_for_stage(stage_cfg)

        if use_chunked:
            logger.info("[phase-8b] final_manuscript_artifacts: chunked I/O enabled")
            _log_memory_config(context, "final_manuscript_artifacts")

        # Delegate to original function
        # TODO (Phase 8b Slice 3): Implement chunked I/O streaming
        #   - Stream large intermediate dataframes during OLS fitting
        #   - Use streaming aggregation for bootstrap statistics
        #   - Spill coefficient matrices to disk if memory budget exceeded
        return original_fn(context)

    return wrapped


__all__ = [
    "should_use_chunked_io_for_stage",
    "wrap_sparse_selection_with_chunked_io",
    "wrap_final_artifacts_with_chunked_io",
]
