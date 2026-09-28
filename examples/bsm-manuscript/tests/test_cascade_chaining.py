"""Cascade-chaining smoke tests for scripts/hpc_workflow.py.

The orchestrator depends on:
1. ``parse_reduce_job_id`` and ``inject_dependency_flag`` from the
   shared ``rfm_pipeline.hpc_cascade`` helpers (round 24).
2. The grouped builder ``build_remote_submit_command_groups`` from
   ``rfm_pipeline.hpc_workflow_config`` that exposes per-stage command
   batches.

These tests cover the bsm-side integration points only; the helpers'
own edge-case coverage lives in
``rfm-pipeline/tests/test_hpc_cascade.py``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
HPC_WORKFLOW = REPO_ROOT / "scripts" / "hpc_workflow.py"


def _load_hpc_workflow():
    spec = importlib.util.spec_from_file_location("hpc_workflow", HPC_WORKFLOW)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_orchestrator_imports_shared_cascade_helpers() -> None:
    """Orchestrator must rely on rfm_pipeline.hpc_cascade so both the rfm
    and bsm orchestrators stay in sync. A regression where the bsm
    script re-implements its own parser would silently desync."""
    module = _load_hpc_workflow()
    from rfm_pipeline.hpc_cascade import (
        CascadeChainError,
        inject_dependency_flag,
        parse_reduce_job_id,
    )

    assert module.parse_reduce_job_id is parse_reduce_job_id
    assert module.inject_dependency_flag is inject_dependency_flag
    assert module.CascadeChainError is CascadeChainError


def test_parse_reduce_job_id_marker_round_trip() -> None:
    module = _load_hpc_workflow()
    assert module.parse_reduce_job_id(
        "noise\nRFM_HPC_SUBMIT_REDUCE_JOB_ID=98765\nmore noise\n"
    ) == 98765


def test_parse_reduce_job_id_returns_last_for_multiple_markers() -> None:
    """Sparse re-submit can emit the marker more than once. The LAST
    marker reflects the actually-submitted reduce job."""
    module = _load_hpc_workflow()
    stdout = (
        "RFM_HPC_SUBMIT_REDUCE_JOB_ID=100\n"
        "RFM_HPC_SUBMIT_REDUCE_JOB_ID=200\n"
    )
    assert module.parse_reduce_job_id(stdout) == 200


def test_inject_dependency_flag_chains_next_stage() -> None:
    module = _load_hpc_workflow()
    cmd = "pixi run rfm-hpc-submit --config foo.yml --stage interaction_discovery"
    chained = module.inject_dependency_flag(cmd, 12345)
    assert chained == cmd + " --depends-on-job-id 12345"


def test_inject_dependency_flag_idempotent_when_already_chained() -> None:
    module = _load_hpc_workflow()
    cmd = "pixi run rfm-hpc-submit --depends-on-job-id 999 --stage foo"
    assert module.inject_dependency_flag(cmd, 12345) == cmd


def test_cascade_chain_error_is_runtime_error_subclass() -> None:
    """Orchestrator raises CascadeChainError when a stage was submitted
    but produced no parseable marker — runtime failure, not silent
    chain-onto-stale-id."""
    module = _load_hpc_workflow()
    assert issubclass(module.CascadeChainError, RuntimeError)
    with pytest.raises(module.CascadeChainError):
        raise module.CascadeChainError("test")
