"""run_sensitivity_job.py — Execute one sensitivity study job.

Called by submit_sensitivity_study.sh for each SLURM array task.
Loads the per-job config YAML (DGP spec + config overrides), generates the
synthetic dataset, writes temporary Parquet files, runs the full pipeline,
extracts nRMSE and support-recovery metrics, and writes result.json.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one sensitivity study job.")
    parser.add_argument("--config", required=True, help="Per-job config YAML path.")
    parser.add_argument("--artifact-dir", required=True, help="Output artifact directory.")
    parser.add_argument("--job-id", required=True, help="Job identifier string.")
    return parser.parse_args()


def _set_dot(mapping: dict[str, Any], dot_path: str, value: Any) -> None:
    """Set a value at a dot-notation path in a nested dict (mutates in place)."""
    parts = dot_path.split(".")
    cursor = mapping
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def _build_pipeline_config(
    job_config: dict[str, Any],
    data_dir: Path,
    artifact_dir: Path,
) -> dict[str, Any]:
    """Build a WorkflowConfig-compatible YAML dict from the job config."""
    base_path = REPO_ROOT / "configs" / "sensitivity_study" / "base_synthetic.yml"
    with base_path.open(encoding="utf-8") as fh:
        pipeline_cfg: dict[str, Any] = yaml.safe_load(fh) or {}

    # Point dataset at the temp synthetic data directory.
    pipeline_cfg["dataset"] = {"type": "synthetic_controlled_dgp", "path": str(data_dir)}

    pipeline_cfg = copy.deepcopy(pipeline_cfg)

    study_meta = job_config.get("sensitivity_study", {})
    overrides: dict[str, Any] = study_meta.get("config_overrides", {})

    # holdout_fraction is applied at DGP generation time; not a pipeline config key.
    _SKIP_OVERRIDE_KEYS = {"holdout_fraction"}

    for key, value in overrides.items():
        if key in _SKIP_OVERRIDE_KEYS:
            continue
        if key == "variance_threshold":
            _set_dot(pipeline_cfg, "algorithm.variance_threshold", value)
        else:
            _set_dot(pipeline_cfg, key, value)

    # Reduce bootstrap count for speed in the sensitivity sweep.
    _set_dot(pipeline_cfg, "stages.final_artifacts.bootstrap_count", 20)

    _set_dot(pipeline_cfg, "output.artifact_dir", str(artifact_dir))
    _set_dot(pipeline_cfg, "output.seed", job_config.get("output", {}).get("seed", 0))
    _set_dot(pipeline_cfg, "runtime.n_jobs", job_config.get("runtime", {}).get("n_jobs", 8))

    return pipeline_cfg


def _extract_metrics(artifact_dir: Path, job_id: str) -> dict[str, Any]:
    """Read nRMSE metrics from the pipeline output artifacts."""
    tables_dir = artifact_dir / "artifacts" / "final_manuscript_artifacts" / "tables"

    # Primary: ablation_table.csv has per-stage nRMSE.
    ablation_path = tables_dir / "ablation_table.csv"
    if ablation_path.exists():
        import pandas as pd

        ablation = pd.read_csv(ablation_path)
        metrics: dict[str, Any] = {"job_id": job_id}
        # Extract null baseline and final OLS nRMSE.
        for col in ablation.columns:
            lower = col.lower()
            if "nrmse" in lower or "stage" in lower or "n_selected" in lower:
                for _, row in ablation.iterrows():
                    stage = str(row.get("stage", "")).replace(" ", "_").lower()
                    if stage and col in row.index:
                        metrics[f"{stage}_{col}"] = row[col]
        # Convenient top-level keys.
        final_row = ablation[ablation["stage"].str.lower().str.contains("ols|final", na=False)]
        if not final_row.empty:
            metrics["nrmse_final"] = float(final_row.iloc[-1].get("nRMSE", float("nan")))
        null_row = ablation[ablation["stage"].str.lower().str.contains("null", na=False)]
        if not null_row.empty:
            metrics["nrmse_null"] = float(null_row.iloc[-1].get("nRMSE", float("nan")))
        if "nrmse_final" in metrics and "nrmse_null" in metrics:
            null = metrics["nrmse_null"]
            final = metrics["nrmse_final"]
            if null and null != 0:
                metrics["nrmse_relative"] = (final - null) / abs(null)
        return metrics

    # Fallback: try final_model_summary.csv.
    fallback = tables_dir / "final_model_summary.csv"
    if fallback.exists():
        import pandas as pd

        row = pd.read_csv(fallback).iloc[0]
        return {
            "job_id": job_id,
            "nrmse_final": float(row.get("final_ols_holdout_nrmse", float("nan"))),
        }

    return {"job_id": job_id, "error_message": f"No metric tables found in {tables_dir}"}


def main() -> int:
    """Generate synthetic data, run pipeline, write result.json."""
    args = _parse_args()
    config_path = Path(args.config)
    artifact_dir = Path(args.artifact_dir)
    job_id = args.job_id

    artifact_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "job_id": job_id,
        "config_path": str(config_path),
        "artifact_dir": str(artifact_dir),
        "error_message": None,
    }

    try:
        with config_path.open(encoding="utf-8") as fh:
            job_config: dict[str, Any] = yaml.safe_load(fh) or {}

        dgp_section: dict[str, Any] = job_config.get("synthetic_dgp", {})
        study_meta: dict[str, Any] = job_config.get("sensitivity_study", {})
        overrides: dict[str, Any] = study_meta.get("config_overrides", {})

        from bsm_rfm.synthetic_dgp import (
            SyntheticDGPSpec,
            generate_bsm_structure_synthetic,
            generate_pure_synthetic,
        )

        holdout_fraction = float(
            overrides.get("holdout_fraction", dgp_section.get("holdout_fraction", 0.05))
        )
        dgp_family = str(dgp_section.get("dgp_family", "pure_synthetic"))

        dgp_spec = SyntheticDGPSpec(
            n_inputs=int(dgp_section["n_inputs"]),
            n_runs=int(dgp_section["n_runs"]),
            n_outputs=int(dgp_section["n_outputs"]),
            sparsity=float(dgp_section["sparsity"]),
            interaction_density=float(dgp_section.get("interaction_density", 0.25)),
            nonlinearity_strength=float(dgp_section.get("nonlinearity_strength", 0.3)),
            noise_snr=float(dgp_section.get("noise_snr", 20.0)),
            holdout_fraction=holdout_fraction,
            dgp_family=dgp_family,
            seed=int(dgp_section.get("seed", 0)) + int(study_meta.get("replicate", 0)),
            factor_model_rank=int(dgp_section.get("factor_model_rank", 20)),
            input_correlation_strength=float(dgp_section.get("input_correlation_strength", 0.3)),
        )

        t0 = time.perf_counter()
        dataset = (
            generate_pure_synthetic(dgp_spec)
            if dgp_family == "pure_synthetic"
            else generate_bsm_structure_synthetic(dgp_spec)
        )
        gen_elapsed = time.perf_counter() - t0

        with tempfile.TemporaryDirectory(prefix=f"bsm_sens_{job_id}_") as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            dataset.input_matrix.to_parquet(data_dir / "X.parquet", index=False)
            dataset.output_matrix.to_parquet(data_dir / "Y.parquet", index=False)
            dataset.holdout_assignments.to_parquet(
                data_dir / "holdout_assignments.parquet", index=False
            )
            dataset.feature_catalog.to_parquet(
                data_dir / "actual_input_feature_catalog.parquet", index=False
            )

            pipeline_cfg = _build_pipeline_config(job_config, data_dir, artifact_dir)
            tmp_cfg_path = Path(tmpdir) / "pipeline_config.yml"
            with tmp_cfg_path.open("w", encoding="utf-8") as fh:
                yaml.safe_dump(pipeline_cfg, fh, sort_keys=False)

            runner = REPO_ROOT / "tools" / "run_manuscript_pipeline.py"
            t1 = time.perf_counter()
            proc = subprocess.run(
                [sys.executable, str(runner), str(tmp_cfg_path)],
                capture_output=True,
                text=True,
                cwd=str(REPO_ROOT),
            )
            pipeline_elapsed = time.perf_counter() - t1

        result["dgp_gen_seconds"] = gen_elapsed
        result["pipeline_seconds"] = pipeline_elapsed
        result["total_wall_seconds"] = gen_elapsed + pipeline_elapsed

        if proc.returncode != 0:
            # Capture last 3 KB of stderr for diagnosis.
            result["error_message"] = proc.stderr[-3000:] if proc.stderr else "no stderr"
            result["stdout_tail"] = proc.stdout[-500:] if proc.stdout else ""
        else:
            metrics = _extract_metrics(artifact_dir, job_id)
            result.update(metrics)

    except Exception:
        result["error_message"] = traceback.format_exc()

    (artifact_dir / "result.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    if result.get("error_message"):
        print(f"ERROR in job {job_id}:\n{result['error_message']}", file=sys.stderr)
        return 1

    print(
        f"Job {job_id} completed. "
        f"nRMSE_final={result.get('nrmse_final', 'n/a')} "
        f"wall={result.get('total_wall_seconds', '?'):.0f}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
