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
    # Honour n_jobs from the per-job config (set by generate_sensitivity_study.py
    # from output.n_jobs in the study spec).  Default to 1 if absent.
    n_jobs = int(job_config.get("runtime", {}).get("n_jobs", 1))
    _set_dot(pipeline_cfg, "runtime.n_jobs", n_jobs)

    return pipeline_cfg


def _null_nrmse_from_dataset(dataset: Any) -> float:
    """Compute holdout macro nRMSE for the null mean predictor.

    Used when the pipeline exits early because empirical null screening removed
    all (or all-but-one) first-order terms, leaving the model null-equivalent.
    """
    import numpy as np

    ha = dataset.holdout_assignments
    Y = dataset.output_matrix

    train_ids = set(ha.loc[ha["split"] == "train", "sample_id"].astype(str))
    holdout_ids = set(ha.loc[ha["split"] != "train", "sample_id"].astype(str))

    Y_train = Y.loc[Y["sample_id"].astype(str).isin(train_ids)].drop(columns=["sample_id"])
    Y_holdout = Y.loc[Y["sample_id"].astype(str).isin(holdout_ids)].drop(columns=["sample_id"])

    if Y_train.empty or Y_holdout.empty:
        return float("nan")

    Y_train_np = Y_train.to_numpy(dtype=float)
    Y_holdout_np = Y_holdout.to_numpy(dtype=float)

    y_mean = Y_train_np.mean(axis=0, keepdims=True)
    residual = Y_holdout_np - y_mean
    rmse_per_output = np.sqrt(np.mean(residual**2, axis=0))

    train_range = Y_train_np.max(axis=0) - Y_train_np.min(axis=0)
    valid = train_range >= 1.0e-6
    if not valid.any():
        return float("nan")

    return float(np.mean(rmse_per_output[valid] / train_range[valid]))


def _extract_metrics(artifact_dir: Path, job_id: str) -> dict[str, Any]:
    """Read nRMSE metrics from the pipeline output artifacts."""
    tables_dir = artifact_dir / "final_manuscript_artifacts" / "tables"

    # Primary: ablation_table.csv has per-stage nRMSE.
    ablation_path = tables_dir / "ablation_table.csv"
    if ablation_path.exists():
        import pandas as pd

        ablation = pd.read_csv(ablation_path)
        metrics: dict[str, Any] = {"job_id": job_id}

        # Column names vary: pipeline uses 'model_name'; older versions used 'stage'.
        name_col = "model_name" if "model_name" in ablation.columns else "stage"
        nrmse_col = "nrmse" if "nrmse" in ablation.columns else "nRMSE"

        # Store per-model metrics keyed by model name.
        for _, row in ablation.iterrows():
            model = str(row.get(name_col, "")).replace(" ", "_").lower()
            if not model:
                continue
            if nrmse_col in row.index:
                metrics[f"{model}_nrmse"] = float(row[nrmse_col])
            for ci_col in ("ci_lower", "ci_upper"):
                if ci_col in row.index:
                    metrics[f"{model}_{ci_col}"] = float(row[ci_col])
            feat_col = "n_features" if "n_features" in row.index else "n_selected"
            if feat_col in row.index:
                metrics[f"{model}_n_features"] = row[feat_col]

        # Convenient top-level keys.
        final_row = ablation[ablation[name_col].str.lower().str.contains("ols|final", na=False)]
        if not final_row.empty:
            last = final_row.iloc[-1]
            metrics["nrmse_final"] = float(last.get(nrmse_col, float("nan")))
            for ci_col in ("ci_lower", "ci_upper"):
                if ci_col in last.index:
                    metrics[f"nrmse_final_{ci_col}"] = float(last[ci_col])
        null_row = ablation[ablation[name_col].str.lower().str.contains("null", na=False)]
        if not null_row.empty:
            metrics["nrmse_null"] = float(null_row.iloc[-1].get(nrmse_col, float("nan")))
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

        from rfm_pipeline.synthetic_dgp import (
            SyntheticDGPSpec,
            generate_calibrated_structure_synthetic,
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
            else generate_calibrated_structure_synthetic(dgp_spec)
        )
        gen_elapsed = time.perf_counter() - t0

        with tempfile.TemporaryDirectory(prefix=f"rfm_sens_{job_id}_") as tmpdir:
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
            stderr = proc.stderr or ""
            # Screens that remove all (or all-but-one) first-order terms are
            # expected for highly sparse or noisy DGPs.  Record as null-equivalent
            # rather than a pipeline failure so SLURM task exits 0.
            _NULL_SCREEN_MSGS = (
                "at least two retained first-order terms",
                "Interaction discovery requires at least two",
            )
            if any(msg in stderr for msg in _NULL_SCREEN_MSGS):
                try:
                    null_nrmse = _null_nrmse_from_dataset(dataset)
                except Exception:
                    null_nrmse = float("nan")
                result["null_screened"] = True
                result["nrmse_null"] = null_nrmse
                result["nrmse_final"] = null_nrmse
                result["nrmse_relative"] = 0.0
                result["null_mean_nrmse"] = null_nrmse
                result["final_ols_nrmse"] = null_nrmse
                result["n_features_retained"] = 0
                result["error_message"] = None
                # Pipeline never ran to completion so it didn't write
                # run_complete.json; write it here so the collector sees
                # this task as finished.
                import datetime

                (artifact_dir / "run_complete.json").write_text(
                    json.dumps(
                        {
                            "status": "complete",
                            "null_screened": True,
                            "completed_at_utc": datetime.datetime.now(
                                datetime.timezone.utc
                            ).isoformat(),
                            "elapsed_seconds": pipeline_elapsed,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            else:
                # Capture last 3 KB of stderr for diagnosis.
                result["error_message"] = stderr[-3000:] if stderr else "no stderr"
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
