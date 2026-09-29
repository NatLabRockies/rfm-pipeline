#!/usr/bin/env python3
"""Run the repository's ordered Pixi validation tasks."""

from __future__ import annotations

import argparse
import os
import subprocess
from collections.abc import Sequence

PREP_TASKS = (
    "clean-transients",
    "format-python",
    "format-markdown",
    "fix-notebooks",
)

VALIDATION_TASKS = (
    "build-import-smoke",
    "clean-transients",
    "repo-hygiene",
    "lint",
    "format-check",
    "markdown-check",
    "notebook-check",
    "notebook-workflow-check",
    "compile-check",
    "unit-tests",
    "bsm-manuscript-example-tests",
    "workflow-tests",
    "manuscript-reproduction-smoke",
    "notebook-tests",
    "docs",
    "package-build",
    "package-smoke",
    "clean-transients",
    "repo-hygiene",
    "git-diff-check",
)

FAST_VALIDATION_TASKS = (
    "build-import-smoke",
    "clean-transients",
    "repo-hygiene",
    "lint",
    "format-check",
    "markdown-check",
    "notebook-check",
    "notebook-workflow-check",
    "compile-check",
    "pre-push-tests",
    "bsm-manuscript-example-tests",
    "workflow-tests",
    "docs",
    "package-build",
    "package-smoke",
    "clean-transients",
    "repo-hygiene",
    "git-diff-check",
)


def task_sequence(mode: str) -> tuple[str, ...]:
    """Return the ordered tasks for a validation mode."""
    if mode == "check":
        return VALIDATION_TASKS
    if mode == "fast":
        return FAST_VALIDATION_TASKS
    if mode in {"fix", "clean"}:
        return PREP_TASKS + VALIDATION_TASKS
    raise ValueError(f"unknown gate mode: {mode}")


def run_tasks(tasks: Sequence[str]) -> None:
    """Run each named Pixi task, stopping on the first failure."""
    pixi_bin = os.environ.get("PIXI_BIN", "pixi")
    for task in tasks:
        print(f">>> {pixi_bin} run {task}", flush=True)
        subprocess.run([pixi_bin, "run", task], check=True)


def main() -> None:
    """Parse the mode and execute its task sequence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("check", "fast", "fix", "clean"),
        nargs="?",
        default="check",
    )
    args = parser.parse_args()
    run_tasks(task_sequence(args.mode))


if __name__ == "__main__":
    main()
