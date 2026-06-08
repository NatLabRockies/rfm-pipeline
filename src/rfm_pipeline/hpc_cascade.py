"""Cascade-chaining helpers for HPC orchestrators.

Both ``rfm-pipeline/tools/run_hpc_workflow.py`` and
``bsm-public-rf/scripts/hpc_workflow.py`` consume
``build_remote_submit_command_groups`` and need identical logic to:

1. capture each cascade stage's per-tier reduce job ids from the
   ``RFM_HPC_SUBMIT_REDUCE_JOB_ID=<id>`` marker emitted on stdout by
   ``rfm-hpc-submit`` (one per per-tier submission within the stage);
2. rewrite the next stage's commands to append
   ``--depends-on-job-id <id>[:<id>...]`` so SLURM enforces the
   cascade order across every upstream tier (not just the last one).

Pulling the logic here keeps both orchestrators in sync and makes the
behavior testable from the rfm-pipeline test suite (bsm has no
package-level tests for this).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

_MARKER = "RFM_HPC_SUBMIT_REDUCE_JOB_ID="


def parse_reduce_job_id(stdout: str) -> int | None:
    """Return the LAST ``RFM_HPC_SUBMIT_REDUCE_JOB_ID=<id>`` marker.

    Returning the last (not the first) match matters when a single
    ``rfm-hpc-submit`` invocation prints the marker more than once
    (e.g. re-submit after a sparse-shard restart) — the most recent
    marker is the one that reflects the actually-submitted reduce job.
    Returns ``None`` when no marker is present or the value is not an
    int.
    """
    last: int | None = None
    for raw in stdout.splitlines():
        line = raw.strip()
        if not line.startswith(_MARKER):
            continue
        tail = line[len(_MARKER) :].strip()
        try:
            last = int(tail)
        except ValueError:
            # Skip malformed lines; do not silently overwrite a good id.
            continue
    return last


def parse_all_reduce_job_ids(stdout: str) -> list[int]:
    """Return every ``RFM_HPC_SUBMIT_REDUCE_JOB_ID=<id>`` marker in order.

    For a single ``rfm-hpc-submit`` invocation that re-submits sparse
    shards, multiple markers may be emitted; the last is canonical for
    THAT invocation (see ``parse_reduce_job_id``). For orchestrators
    chaining a whole stage group (multiple tier invocations), call this
    helper on each invocation's stdout separately and collect the LAST
    id of each (use ``parse_reduce_job_id``) — DO NOT pool all markers
    across invocations into a flat list, because intermediate sparse
    re-submit markers from one invocation would then become spurious
    dependency ids for the next stage. Returns an empty list when no
    valid marker is present.
    """
    ids: list[int] = []
    for raw in stdout.splitlines():
        line = raw.strip()
        if not line.startswith(_MARKER):
            continue
        tail = line[len(_MARKER) :].strip()
        try:
            ids.append(int(tail))
        except ValueError:
            continue
    return ids


def _format_upstream(upstream: int | str | Sequence[int]) -> str:
    """Render an upstream-id value as a SLURM ``afterok``-compatible token.

    Accepts a single int (legacy single-tier cascade), a colon-string
    like ``"123:456"`` (already-formatted; passes through), or a
    sequence of ints (joined with ``:``). Empty sequences raise
    ``ValueError`` — callers must not chain on no upstream at all.
    """
    if isinstance(upstream, int):
        return str(upstream)
    if isinstance(upstream, str):
        return upstream
    if isinstance(upstream, Iterable):
        ids = list(upstream)
        if not ids:
            raise ValueError(
                "inject_dependency_flag requires at least one upstream "
                "job id; got an empty sequence."
            )
        if not all(isinstance(i, int) for i in ids):
            raise TypeError(
                "inject_dependency_flag upstream sequence must contain "
                f"only ints; got {[type(i).__name__ for i in ids]}."
            )
        return ":".join(str(i) for i in ids)
    raise TypeError(
        f"inject_dependency_flag upstream must be int, str, or "
        f"Sequence[int]; got {type(upstream).__name__}."
    )


def inject_dependency_flag(command: str, upstream_job_id: int | str | Sequence[int]) -> str:
    """Append ``--depends-on-job-id <id>[:<id>...]`` to a rfm-hpc-submit command.

    Returns the command unchanged when it is not a ``rfm-hpc-submit``
    invocation or when ``--depends-on-job-id`` is already present (so
    re-runs are idempotent and prep / diagnostic commands pass
    through). Accepts a single int (single-tier upstream), a
    pre-formatted colon string, or a sequence of ints (multi-tier
    upstream — every previous-stage tier's reduce id must complete
    before the next stage's array can start, so SLURM ``afterok``
    needs every id, not just the last submitted).
    """
    if "rfm-hpc-submit" not in command:
        return command
    if "--depends-on-job-id" in command:
        return command
    token = _format_upstream(upstream_job_id)
    return f"{command} --depends-on-job-id {token}"


class CascadeChainError(RuntimeError):
    """Cascade stage produced no parseable reduce job id marker.

    The orchestrator must abort rather than continue with a stale
    ``prev_reduce_job_id`` (which would chain the next stage onto an
    earlier stage's reduce and let it race against the current stage's
    work).
    """
