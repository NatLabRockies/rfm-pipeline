from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "bsm-manuscript"


def test_bsm_manuscript_example_has_complete_reproduction_surface() -> None:
    required = {
        "README.md",
        "MIGRATION.md",
        "configs/manuscript_case_study.yml",
        "scripts/run_manuscript_pipeline.py",
        "scripts/reproduce_artifacts.py",
        "tests/test_artifact_bundle_consistency.py",
        "artifacts/final_model/coefficient_matrix_raw_scale.csv",
        "artifacts/tables/per_output_nrmse.csv",
        "figures/figure_support_composition.pdf",
    }
    missing = [relative for relative in sorted(required) if not (EXAMPLE / relative).is_file()]
    assert not missing, f"missing BSM manuscript example files: {missing}"


def test_bsm_manuscript_release_figures_validate() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/reproduce_artifacts.py", "--validate-only"],
        cwd=EXAMPLE,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_bsm_manuscript_example_tests_pass() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_artifact_bundle_consistency.py",
            "tests/test_build_metadata_transforms.py",
            "tests/test_configs_loadable.py",
            "tests/test_manuscript_config_reconciliation.py",
            "tests/test_publication_figure_polish.py",
        ],
        cwd=EXAMPLE,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
