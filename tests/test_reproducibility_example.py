"""Tests for the shipped reproducibility example."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from examples.basic_workflow import run_example


def test_basic_workflow_example_writes_a_reloadable_bundle(tmp_path: Path) -> None:
    run, loaded = run_example(tmp_path / "basic-workflow")

    assert (tmp_path / "basic-workflow" / "manifest.json").is_file()
    assert tuple(loaded) == (
        "all_input_metadata",
        "selected_input_metadata",
        "output_metadata",
        "coef_matrix_standardized",
        "coef_matrix_raw_scale",
        "x_standardization",
        "y_standardization",
        "nrmse_summary",
    )
    assert float(run.holdout_summary.loc[0, "point_estimate"]) > 0.0


def test_reproducibility_example_cli_runs_from_repo_source_tree(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "cli-bundle"
    env = dict(**os.environ)
    env["PYTHONPATH"] = str(repo_root / "src")

    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "examples" / "basic_workflow.py"),
            "--output-dir",
            str(output_dir),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert "Wrote 8 tables to" in completed.stdout
    assert (output_dir / "manifest.json").exists()
