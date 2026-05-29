#!/usr/bin/env python3
# Copyright (c) 2026 Dylan Hettinger
"""Generate sensitivity-study job manifests and per-job configs."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bsm_rfm.sensitivity_study import (  # noqa: E402
    generate_study_jobs,
    jobs_to_dataframe,
    sensitivity_study_spec_from_mapping,
)

_SUPPORTED_OVERRIDE_KEYS = {
    "variance_threshold",
    "stages.empirical_null_screening.n_permutations",
    "stages.empirical_null_screening.bh_q_threshold",
    "stages.interaction_discovery.n_permutations",
    "stages.interaction_discovery.p_threshold",
    "stages.sparse_selection.n_stability_subsamples",
    "stages.sparse_selection.lasso_alpha_percentile",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study-dir",
        type=Path,
        default=None,
        help="Override output.study_dir from the study spec.",
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=REPO_ROOT / "configs" / "sensitivity_study" / "study_spec.yml",
        help="Path to study specification YAML.",
    )
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _set_dot_path(mapping: dict[str, Any], dot_path: str, value: Any) -> None:
    cursor = mapping
    parts = dot_path.split(".")
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def main() -> int:
    """Generate per-job config YAMLs and SLURM array manifest."""
    args = _parse_args()
    spec_payload = _load_yaml(args.spec)
    study_spec = sensitivity_study_spec_from_mapping(spec_payload)
    study_dir = args.study_dir or Path(spec_payload["output"]["study_dir"])
    base_config_path = REPO_ROOT / "configs" / "sensitivity_study" / "base_synthetic.yml"
    base_config = _load_yaml(base_config_path)
    n_jobs = int(spec_payload.get("output", {}).get("n_jobs", 8))

    jobs = generate_study_jobs(study_spec)
    jobs_frame = jobs_to_dataframe(jobs)

    study_dir.mkdir(parents=True, exist_ok=True)
    job_config_dir = study_dir / "job_configs"
    artifact_root = study_dir / "artifacts"
    job_config_dir.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)

    jobs_frame.to_csv(study_dir / "jobs.csv", index=False)
    (study_dir / "jobs.json").write_text(
        json.dumps(jobs_frame.to_dict(orient="records"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    slurm_lines: list[str] = []
    for job in jobs:
        artifact_dir = artifact_root / job.job_id
        config_payload = copy.deepcopy(base_config)
        for key, value in job.config_overrides.items():
            if key in _SUPPORTED_OVERRIDE_KEYS:
                applied_key = "algorithm.variance_threshold" if key == "variance_threshold" else key
                _set_dot_path(config_payload, applied_key, value)
        _set_dot_path(config_payload, "runtime.n_jobs", n_jobs)
        _set_dot_path(config_payload, "output.artifact_dir", str(artifact_dir))
        _set_dot_path(config_payload, "output.seed", job.dgp_spec.seed + job.replicate)
        config_payload["synthetic_dgp"] = asdict(job.dgp_spec)
        config_payload["sensitivity_study"] = {
            "job_id": job.job_id,
            "block": job.block,
            "dgp_idx": job.dgp_idx,
            "config_idx": job.config_idx,
            "replicate": job.replicate,
            "n_subsample_levels": job.n_subsample_levels,
            "config_overrides": job.config_overrides,
        }
        config_path = job_config_dir / f"{job.job_id}.yml"
        with config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config_payload, handle, sort_keys=False)
        slurm_lines.append(f"{job.job_id},{config_path},{artifact_dir}")

    (study_dir / "slurm_array.txt").write_text("\n".join(slurm_lines) + "\n", encoding="utf-8")

    pure_dgps = study_spec.pure_synthetic_n_dgps
    bsm_dgps = study_spec.bsm_structure_n_dgps
    print(f"study_dir={study_dir}")
    print(f"total_jobs={len(jobs)}")
    print(f"pure_synthetic_dgps={pure_dgps}")
    print(f"bsm_structure_dgps={bsm_dgps}")
    print(f"pure_configs_per_dgp={study_spec.n_configs_per_dgp_pure}")
    print(f"bsm_configs_per_dgp={study_spec.n_configs_per_dgp_bsm}")
    print(f"replicates={study_spec.n_replicates}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
