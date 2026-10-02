"""Tests for the shipped end-to-end reproducibility example."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from examples.basic_workflow import run_example
from examples.end_to_end_reproducibility import (
    DATASET_TAG,
    run_reproducibility_example,
)


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


def test_reproducibility_example_function_writes_and_reloads_bundle(tmp_path: Path) -> None:
    result = run_reproducibility_example(tmp_path / DATASET_TAG)

    bundle_root = result["bundle_root"]
    manifest_path = bundle_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest_path.exists()
    assert manifest["dataset_tag"] == DATASET_TAG
    assert manifest["files"]["nrmse_summary"].endswith(".csv")
    assert result["loaded"]["nrmse_summary"].loc[0, "n_boot"] == 25
    holdout_point = float(result["holdout_summary"].loc[0, "point_estimate"])
    assert holdout_point > 0.0
    loaded_point = float(result["loaded"]["nrmse_summary"].loc[0, "point_estimate"])
    assert loaded_point == pytest.approx(holdout_point)


def test_reproducibility_example_cli_runs_from_repo_source_tree(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "cli-bundle"
    env = dict(**os.environ)
    env["PYTHONPATH"] = str(repo_root / "src")

    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "examples" / "end_to_end_reproducibility.py"),
            "--output-dir",
            str(output_dir),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert "Wrote bundle to:" in completed.stdout
    assert (output_dir / "manifest.json").exists()
