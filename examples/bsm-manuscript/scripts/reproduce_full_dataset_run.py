"""Reproduce the full BSM dataset run on HPC.

This script drives the rfm_pipeline orchestration using the BSM publication configs.
It is intended to be run from the bsm-public-rf repo root on a SLURM HPC cluster.

Prerequisites (see README.md for full details):
  - SLURM HPC cluster access (publication run used NREL Kestrel)
  - SCRATCH_DIR environment variable set to your cluster scratch root
    e.g.: export SCRATCH_DIR=/scratch/${USER}
  - SLURM_ACCOUNT environment variable set to your HPC allocation account
    e.g.: export SLURM_ACCOUNT=my_project
  - BSM dataset files at: ${SCRATCH_DIR}/bsm/bsm-public-rf/artifacts/preprocessed_real_data_30k/
  - configs/manuscript_paths.yml configured (copy from configs/manuscript_paths_template.yml)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "hpc" / "kestrel_publication_full_dataset.yml"
RUNNER_PATH = REPO_ROOT / "scripts" / "run_manuscript_pipeline.py"

# Ensure RFM_STUDY_ROOT points to this repo root.
os.environ.setdefault("RFM_STUDY_ROOT", str(REPO_ROOT))

from rfm_pipeline.config import load_config  # noqa: E402

_REQUIRED_ENV = ["SCRATCH_DIR", "SLURM_ACCOUNT"]
_REQUIRED_FILES = [
    REPO_ROOT / "configs" / "manuscript_paths.yml",
    CONFIG_PATH,
]


def _check_prerequisites() -> None:
    """Fail fast with a human-readable message if required env/files are missing."""
    errors: list[str] = []

    for var in _REQUIRED_ENV:
        if not os.environ.get(var):
            errors.append(
                f"  Missing environment variable: {var}\n"
                f"    → export {var}=<your value> before running this script"
            )

    for path in _REQUIRED_FILES:
        if not path.exists():
            if path.name == "manuscript_paths.yml":
                errors.append(
                    f"  Missing required file: {path}\n"
                    f"    → cp configs/manuscript_paths_template.yml configs/manuscript_paths.yml\n"
                    f"    → Edit the copy to set real data paths on your cluster"
                )
            else:
                errors.append(f"  Missing required file: {path}")

    # If manuscript_paths.yml exists, verify it contains real paths rather
    # than the unedited /path/to/... template placeholders. This catches the
    # common reviewer error of copying the template and forgetting to edit
    # it; without this guard the failure would surface deep inside the
    # pipeline with an unhelpful "Dataset root not found" message.
    paths_file = REPO_ROOT / "configs" / "manuscript_paths.yml"
    if paths_file.exists():
        try:
            import yaml

            with open(paths_file, encoding="utf-8") as f:
                paths_data = yaml.safe_load(f) or {}
            unedited = [
                key
                for key, value in paths_data.items()
                if isinstance(value, str)
                and (
                    value.startswith("/path/to/")
                    or value.startswith("/scratch/dhetting/")
                )
            ]
            if unedited:
                errors.append(
                    f"  configs/manuscript_paths.yml still contains template placeholders: "
                    f"{', '.join(unedited)}\n"
                    f"    → Edit each /path/to/... value to point at your local BSM data\n"
                    f"    → See configs/manuscript_paths_template.yml for the expected schema"
                )
        except Exception as exc:  # noqa: BLE001 — surfaced verbatim to user
            errors.append(f"  configs/manuscript_paths.yml failed to parse: {exc}")

    # Validate dataset directory when SCRATCH_DIR is resolvable
    scratch_dir = os.environ.get("SCRATCH_DIR", "")
    if scratch_dir:
        dataset_dir = (
            Path(scratch_dir)
            / "bsm"
            / "bsm-public-rf"
            / "artifacts"
            / "preprocessed_real_data_30k"
        )
        if not dataset_dir.exists():
            errors.append(
                f"  BSM dataset directory not found: {dataset_dir}\n"
                f"    → Obtain the BSM preprocessed dataset and place it at this path.\n"
                f"    → See README.md §'Data Access' for the dataset source."
            )

    if errors:
        print(
            "ERROR: Prerequisites not met for reproduce-full. See README.md for setup.\n"
        )
        for e in errors:
            print(e)
        sys.exit(1)


def main() -> None:
    argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    ).parse_args()
    raise SystemExit(
        "RETIRED: use the content-addressed G11 campaign package; "
        "the legacy full-dataset runner is non-executable."
    )
    _check_prerequisites()

    cfg = load_config(str(CONFIG_PATH))
    artifact_dir = cfg.output.artifact_dir
    if "${" in artifact_dir:
        print(f"WARNING: artifact_dir contains unexpanded variable: {artifact_dir!r}")
        print("  Make sure SCRATCH_DIR and other referenced env vars are exported.")

    print(f"Config loaded: dataset={cfg.dataset.type}")
    print(f"Artifact dir:  {artifact_dir}")
    print("Launching full pipeline runner via subprocess...")

    if not RUNNER_PATH.is_file():
        raise FileNotFoundError(f"Pipeline runner not found: {RUNNER_PATH}")
    subprocess.run([sys.executable, str(RUNNER_PATH), str(CONFIG_PATH)], check=True)


if __name__ == "__main__":
    main()
