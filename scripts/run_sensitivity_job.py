"""run_sensitivity_job.py — Execute one sensitivity study job.

Called by submit_sensitivity_study.sh for each SLURM array task.
Loads the per-job config YAML (DGP spec + config overrides), generates the
synthetic dataset, runs the pipeline, records metrics, and writes a
result.json to the artifact directory.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one sensitivity study job.")
    parser.add_argument("--config", required=True, help="Per-job config YAML path.")
    parser.add_argument("--artifact-dir", required=True, help="Output artifact directory.")
    parser.add_argument("--job-id", required=True, help="Job identifier string.")
    return parser.parse_args()


def main() -> int:
    """Execute one sensitivity study job and write result.json."""
    args = _parse_args()
    config_path = Path(args.config)
    artifact_dir = Path(args.artifact_dir)
    job_id = args.job_id

    artifact_dir.mkdir(parents=True, exist_ok=True)

    result: dict = {
        "job_id": job_id,
        "config_path": str(config_path),
        "artifact_dir": str(artifact_dir),
        "error_message": None,
    }

    try:
        import yaml

        from bsm_rfm.sensitivity_study import run_sensitivity_job

        with open(config_path, encoding="utf-8") as f:
            job_config = yaml.safe_load(f)

        t0 = time.perf_counter()
        job_result = run_sensitivity_job(job_config, artifact_dir=artifact_dir)
        elapsed = time.perf_counter() - t0

        result.update(job_result)
        result["total_wall_seconds"] = elapsed

    except Exception:
        result["error_message"] = traceback.format_exc()
        result_path = artifact_dir / "result.json"
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"ERROR in job {job_id}:\n{result['error_message']}", file=sys.stderr)
        return 1

    result_path = artifact_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Job {job_id} completed. nRMSE_final={result.get('nrmse_final', 'n/a')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
