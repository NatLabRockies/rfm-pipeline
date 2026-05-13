"""Phase 8b chunked I/O integration for sparse_selection and final_artifacts stages.

This module provides wrappers around stage execution functions that optionally
enable chunked I/O based on config settings. When enabled, stages stream data in
chunks to respect memory budgets.

Key strategy:
- Detect config.stages.sparse_selection_stability.use_chunked_io
- When True, use out_of_core module components for data I/O
- Delegate core computation to existing stage functions
- Validate numerical equivalence between chunked and in-memory paths
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
    Current implementation delegates to original function. Future enhancement:
    - Detect when chunked I/O is enabled in config
    - Stream input data in chunks
    - Accumulate sparse selection results across chunks
    - Use spill-to-disk if memory budget exceeded
    """

    def wrapped(context: Any) -> Any:
        # Check if chunked I/O is enabled for this stage
        stage_cfg = context.config.stages.get("sparse_selection_stability", {})
        if should_use_chunked_io_for_stage(stage_cfg):
            logger.info("[phase-8b] sparse_selection with chunked I/O enabled")
            # TODO: Implement chunked I/O path
            # For now, delegate to original (backward compatible)
            logger.warning(
                "[phase-8b] chunked I/O wrapper not yet implemented; using in-memory path"
            )

        # Delegate to original function
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
    Current implementation delegates to original function. Future enhancement:
    - Detect when chunked I/O is enabled in config
    - Stream large intermediate dataframes in chunks during OLS fitting
    - Use streaming aggregation for statistics computation
    - Use spill-to-disk for intermediate coefficient matrices
    """

    def wrapped(context: Any) -> Any:
        # Check if chunked I/O is enabled for this stage
        stage_cfg = context.config.stages.get("final_manuscript_artifacts", {})
        if should_use_chunked_io_for_stage(stage_cfg):
            logger.info("[phase-8b] final_artifacts with chunked I/O enabled")
            # TODO: Implement chunked I/O path
            # For now, delegate to original (backward compatible)
            logger.warning(
                "[phase-8b] chunked I/O wrapper not yet implemented; using in-memory path"
            )

        # Delegate to original function
        return original_fn(context)

    return wrapped


__all__ = [
    "should_use_chunked_io_for_stage",
    "wrap_sparse_selection_with_chunked_io",
    "wrap_final_artifacts_with_chunked_io",
]
