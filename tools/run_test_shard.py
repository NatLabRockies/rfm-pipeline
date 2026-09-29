#!/usr/bin/env python3
"""Run one deterministic shard of the repository's pytest files."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from tools.ci_shards import select_weighted_shard
from tools.pytest_nodeids import NODEID_PREFIX

DOUBLE_STAGE_TESTS = frozenset(
    {
        "test_sparse_selection_wrapper_matches_unwrapped_stage_result",
    }
)

FULL_STAGE_TESTS = frozenset(
    {
        "test_ablation_bootstrap_ci_contains_point_estimate",
        "test_ablation_sorted_by_nrmse_and_n_features_positive",
        "test_ablation_table_models_and_nrmse_finite",
        "test_ablation_table_contains_all_five_models",
        "test_ablation_table_final_ols_not_worse_than_null_mean",
        "test_demo_sparse_selection_final_stable_support_nonempty_ci_parity_guard",
        "test_final_artifacts_wrapper_executes_real_stage",
        "test_per_output_n_included_matches_summary",
        "test_per_output_nrmse_mean_matches_macro_point_estimate",
        "test_per_output_nrmse_quantiles_monotonic",
        "test_per_output_nrmse_schema_and_values",
        "test_per_output_nrmse_summary_schema",
        "test_per_output_summary_quantiles_and_worst_outputs",
        "test_reproducibility_example_cli_can_run_manuscript_chain",
        "test_reproducibility_example_runs_manuscript_reproduction_chain",
        "test_run_final_manuscript_artifacts_stage_executes_demo_context",
        "test_run_sparse_selection_stability_stage_executes_demo_context",
        "test_sparse_selection_writes_diagnostics",
    }
)


def estimated_test_weight(nodeid: str) -> int:
    """Estimate relative runtime from the slow integration test's function name."""
    test_name = nodeid.rsplit("::", maxsplit=1)[-1]
    if test_name in DOUBLE_STAGE_TESTS:
        return 240
    if test_name in FULL_STAGE_TESTS:
        return 120
    return 1


def collect_test_ids(test_root: Path) -> list[str]:
    """Collect every pytest node ID below ``test_root`` in stable order."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-qq",
            "-s",
            "-p",
            "tools.pytest_nodeids",
            str(test_root),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    nodeids = [
        line.removeprefix(NODEID_PREFIX)
        for line in result.stdout.splitlines()
        if line.startswith(NODEID_PREFIX)
    ]
    if not nodeids:
        raise RuntimeError(f"pytest collected no tests below {test_root}")
    return nodeids


def build_pytest_command(test_ids: list[str]) -> list[str]:
    """Build the pytest command for a selected group of test items."""
    return [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--durations=20",
        *test_ids,
    ]


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--test-root", type=Path, default=Path("tests"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Collect, select, and execute one test-item shard."""
    args = parse_args(list(sys.argv[1:] if argv is None else argv))
    test_ids = collect_test_ids(args.test_root)
    selected = select_weighted_shard(
        test_ids,
        index=args.shard_index,
        count=args.shard_count,
        weight=estimated_test_weight,
    )
    if not selected:
        raise RuntimeError(f"test shard {args.shard_index} of {args.shard_count} contains no tests")

    print(
        f"Running test shard {args.shard_index + 1}/{args.shard_count} "
        f"({len(selected)} of {len(test_ids)} tests)",
        flush=True,
    )
    subprocess.run(build_pytest_command(selected), check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
