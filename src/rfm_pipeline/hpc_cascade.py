"""Cascade-chaining helpers for HPC orchestrators.

Both ``rfm-pipeline/tools/run_hpc_workflow.py`` and
``bsm-public-rf/scripts/hpc_workflow.py`` consume
``build_remote_submit_command_groups`` and need identical logic to:

1. capture each cascade stage's terminal reduce job id from the
   ``RFM_HPC_SUBMIT_REDUCE_JOB_ID=<id>`` marker emitted on stdout by
   ``rfm-hpc-submit``;
2. rewrite the next stage's commands to append
   ``--depends-on-job-id <id>`` so SLURM enforces the cascade order.

Pulling the logic here keeps both orchestrators in sync and makes the
behavior testable from the rfm-pipeline test suite (bsm has no
package-level tests for this).
"""

from __future__ import annotations

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


def inject_dependency_flag(command: str, upstream_job_id: int) -> str:
    """Append ``--depends-on-job-id <id>`` to a rfm-hpc-submit command.

    Returns the command unchanged when it is not a ``rfm-hpc-submit``
    invocation or when ``--depends-on-job-id`` is already present (so
    re-runs are idempotent and prep / diagnostic commands pass
    through).
    """
    if "rfm-hpc-submit" not in command:
        return command
    if "--depends-on-job-id" in command:
        return command
    return f"{command} --depends-on-job-id {upstream_job_id}"


class CascadeChainError(RuntimeError):
    """Cascade stage produced no parseable reduce job id marker.

    The orchestrator must abort rather than continue with a stale
    ``prev_reduce_job_id`` (which would chain the next stage onto an
    earlier stage's reduce and let it race against the current stage's
    work).
    """
