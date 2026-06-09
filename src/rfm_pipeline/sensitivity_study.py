# Copyright (c) 2026 Dylan Hettinger
"""Sensitivity-study job generation and result collection utilities."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .synthetic_dgp import SyntheticDGPSpec


@dataclass(frozen=True, slots=True)
class SensitivityStudyJob:
    """One DGP/config/replicate study run."""

    job_id: str
    block: str
    dgp_idx: int
    config_idx: int
    replicate: int
    dgp_spec: SyntheticDGPSpec
    config_overrides: dict[str, Any]
    n_subsample_levels: list[int]


@dataclass(frozen=True, slots=True)
class SensitivityStudySpec:
    """Top-level sensitivity-study design."""

    pure_synthetic_n_dgps: int = 200
    calibrated_n_dgps: int = 50
    n_configs_per_dgp_pure: int = 50
    n_configs_per_dgp_calibrated: int = 30
    n_replicates: int = 5
    pure_synthetic_n_subsample_levels: int = 8
    calibrated_n_subsample_levels: int = 8
    config_lhs_seed: int = 42
    dgp_lhs_seed: int = 0
    fixed_overrides: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Validate that all count fields are positive integers."""
        for field_name in (
            "pure_synthetic_n_dgps",
            "calibrated_n_dgps",
            "n_configs_per_dgp_pure",
            "n_configs_per_dgp_calibrated",
            "n_replicates",
            "pure_synthetic_n_subsample_levels",
            "calibrated_n_subsample_levels",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer; got {value!r}")
        if self.fixed_overrides is not None and not isinstance(self.fixed_overrides, dict):
            raise ValueError(
                f"fixed_overrides must be a dict or None; got {type(self.fixed_overrides).__name__}"
            )


@dataclass(frozen=True, slots=True)
class SensitivityRunResult:
    """Tidied per-stage study output."""

    job_id: str
    subsample_n: int
    stage: str
    nrmse: float
    runtime_seconds: float
    n_features_selected: int
    precision: float
    recall: float
    error_message: str | None


_CONFIG_OPTIONS: tuple[tuple[str, tuple[Any, ...]], ...] = (
    ("holdout_fraction", (0.05, 0.10, 0.15, 0.20)),
    ("variance_threshold", (0.80, 0.85, 0.90, 0.95)),
    ("stages.empirical_null_screening.n_permutations", (51, 101, 201, 401)),
    ("stages.empirical_null_screening.bh_q_threshold", (0.01, 0.05, 0.10, 0.20)),
    ("stages.interaction_discovery.n_permutations", (11, 21, 31, 51, 101)),
    ("stages.interaction_discovery.p_threshold", (0.01, 0.05, 0.10, 0.20)),
    ("stages.sparse_selection.n_stability_subsamples", (10, 25, 50, 100)),
    ("stages.sparse_selection.lasso_alpha_grid_size", (20, 40, 80, 160)),
    ("stages.final_artifacts.delta_threshold_override", (0.001, 0.002, 0.005, 0.010)),
)

_RESULT_COLUMNS = [
    "job_id",
    "subsample_n",
    "stage",
    "nrmse",
    "runtime_seconds",
    "n_features_selected",
    "precision",
    "recall",
    "error_message",
]


def generate_lhs_points(n_points: int, n_dims: int, seed: int) -> np.ndarray:
    """Generate midpoint Latin hypercube samples in [0, 1]^n_dims."""
    if n_points <= 0:
        raise ValueError(f"n_points must be positive; got {n_points!r}")
    if n_dims <= 0:
        raise ValueError(f"n_dims must be positive; got {n_dims!r}")

    rng = np.random.default_rng(seed)
    midpoints = (np.arange(n_points, dtype=np.float64) + 0.5) / n_points
    lhs = np.empty((n_points, n_dims), dtype=np.float64)
    for dim_idx in range(n_dims):
        lhs[:, dim_idx] = midpoints[rng.permutation(n_points)]
    return lhs


def generate_pure_synthetic_dgps(spec: SensitivityStudySpec) -> list[SyntheticDGPSpec]:
    """Generate LHS-sampled pure-synthetic DGPs."""
    lhs = generate_lhs_points(spec.pure_synthetic_n_dgps, 7, spec.dgp_lhs_seed)
    dgps: list[SyntheticDGPSpec] = []
    for idx, row in enumerate(lhs):
        dgps.append(
            SyntheticDGPSpec(
                n_inputs=_scale_int(row[0], 20, 200),
                n_runs=_scale_int(row[1], 5000, 30000),
                n_outputs=_scale_int(row[2], 100, 5000, log_scale=True),
                sparsity=_scale_float(row[3], 0.05, 0.70),
                interaction_density=_scale_float(row[4], 0.0, 0.5),
                nonlinearity_strength=_scale_float(row[5], 0.0, 1.0),
                noise_snr=_scale_float(row[6], 5.0, 100.0, log_scale=True),
                holdout_fraction=0.05,
                dgp_family="pure_synthetic",
                seed=spec.dgp_lhs_seed + idx,
            )
        )
    return dgps


def generate_calibrated_structure_dgps(spec: SensitivityStudySpec) -> list[SyntheticDGPSpec]:
    """Generate LHS-sampled calibrated-structure DGPs (mirroring case-study dataset properties)."""
    lhs = generate_lhs_points(spec.calibrated_n_dgps, 9, spec.dgp_lhs_seed + 10_000)
    dgps: list[SyntheticDGPSpec] = []
    for idx, row in enumerate(lhs):
        dgps.append(
            SyntheticDGPSpec(
                n_inputs=_scale_int(row[0], 100, 200),
                n_runs=_scale_int(row[1], 5000, 30000),
                n_outputs=_scale_int(row[2], 5000, 20000, log_scale=True),
                sparsity=_scale_float(row[3], 0.1, 0.5),
                interaction_density=_scale_float(row[4], 0.1, 0.3),
                nonlinearity_strength=_scale_float(row[5], 0.1, 0.5),
                noise_snr=_scale_float(row[6], 5.0, 50.0, log_scale=True),
                holdout_fraction=0.05,
                dgp_family="calibrated_structure",
                seed=spec.dgp_lhs_seed + 100_000 + idx,
                factor_model_rank=_scale_int(row[7], 10, 40),
                input_correlation_strength=_scale_float(row[8], 0.1, 0.5),
            )
        )
    return dgps


def generate_config_lhs(spec: SensitivityStudySpec, n_configs: int) -> list[dict[str, Any]]:
    """Generate hyperparameter overrides from a Latin hypercube design.

    If ``spec.fixed_overrides`` is set, every config in the returned list is a
    copy of that mapping (used for "production-replica" sweeps where d/n vary
    via the DGP block while the pipeline override set is held constant).
    """
    if spec.fixed_overrides is not None:
        return [dict(spec.fixed_overrides) for _ in range(n_configs)]
    lhs = generate_lhs_points(n_configs, len(_CONFIG_OPTIONS), spec.config_lhs_seed)
    configs: list[dict[str, Any]] = []
    for row in lhs:
        config: dict[str, Any] = {}
        for dim_idx, (key, options) in enumerate(_CONFIG_OPTIONS):
            option_idx = min(len(options) - 1, int(np.floor(row[dim_idx] * len(options))))
            config[key] = options[option_idx]
        configs.append(config)
    return configs


def generate_study_jobs(spec: SensitivityStudySpec) -> list[SensitivityStudyJob]:
    """Enumerate all sensitivity-study jobs."""
    jobs: list[SensitivityStudyJob] = []
    pure_configs = generate_config_lhs(spec, spec.n_configs_per_dgp_pure)
    calibrated_configs = generate_config_lhs(
        replace(spec, config_lhs_seed=spec.config_lhs_seed + 1),
        spec.n_configs_per_dgp_calibrated,
    )

    for block, dgps, configs, n_levels in (
        (
            "pure_synthetic",
            generate_pure_synthetic_dgps(spec),
            pure_configs,
            spec.pure_synthetic_n_subsample_levels,
        ),
        (
            "calibrated_structure",
            generate_calibrated_structure_dgps(spec),
            calibrated_configs,
            spec.calibrated_n_subsample_levels,
        ),
    ):
        for dgp_idx, dgp_spec in enumerate(dgps):
            subsample_levels = _compute_subsample_levels(dgp_spec.n_runs, n_levels)
            for config_idx, config_overrides in enumerate(configs):
                for replicate in range(spec.n_replicates):
                    jobs.append(
                        SensitivityStudyJob(
                            job_id=(f"job_{block}_{dgp_idx:04d}_{config_idx:03d}_{replicate:02d}"),
                            block=block,
                            dgp_idx=dgp_idx,
                            config_idx=config_idx,
                            replicate=replicate,
                            dgp_spec=dgp_spec,
                            config_overrides=dict(config_overrides),
                            n_subsample_levels=subsample_levels,
                        )
                    )
    return jobs


def jobs_to_dataframe(jobs: list[SensitivityStudyJob]) -> pd.DataFrame:
    """Convert jobs to a flat tabular representation."""
    rows: list[dict[str, Any]] = []
    for job in jobs:
        dgp_dict = asdict(job.dgp_spec)
        row: dict[str, Any] = {
            "job_id": job.job_id,
            "block": job.block,
            "dgp_idx": job.dgp_idx,
            "config_idx": job.config_idx,
            "replicate": job.replicate,
            "config_overrides": json.dumps(job.config_overrides, sort_keys=True),
            "n_subsample_levels": ",".join(str(value) for value in job.n_subsample_levels),
            **dgp_dict,
        }
        row.update(job.config_overrides)
        rows.append(row)
    return pd.DataFrame(rows)


def collect_study_results(study_dir: Path) -> pd.DataFrame:
    """Load per-job result JSON files into one tidy frame.

    Supports two directory layouts:
    - Flat: ``study_dir/job_*/result.json``
    - Artifacts subdirectory: ``study_dir/artifacts/job_*/result.json``
      (produced by ``generate_sensitivity_study.py``)
    """
    study_dir = Path(study_dir)
    rows: list[dict[str, Any]] = []
    result_paths = sorted(study_dir.glob("job_*/result.json"))
    if not result_paths:
        result_paths = sorted((study_dir / "artifacts").glob("job_*/result.json"))
    for result_path in result_paths:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        rows.extend(_normalize_result_payload(payload=payload, result_path=result_path))

    if not rows:
        return pd.DataFrame(columns=_RESULT_COLUMNS)

    results = pd.DataFrame(rows)
    jobs_path = study_dir / "jobs.csv"
    if jobs_path.exists():
        jobs = pd.read_csv(jobs_path)
        join_columns = [
            column
            for column in jobs.columns
            if column != "job_id" and column not in results.columns
        ]
        if join_columns:
            results = results.merge(jobs[["job_id", *join_columns]], on="job_id", how="left")
    return results


def sensitivity_study_spec_from_mapping(mapping: dict[str, Any]) -> SensitivityStudySpec:
    """Build a typed study spec from a YAML mapping."""
    study_mapping = dict(mapping.get("study", mapping))
    return SensitivityStudySpec(**study_mapping)


def _normalize_result_payload(*, payload: Any, result_path: Path) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        items = payload
        shared: dict[str, Any] = {}
    elif isinstance(payload, dict) and isinstance(payload.get("results"), list):
        items = payload["results"]
        shared = {key: value for key, value in payload.items() if key != "results"}
    elif isinstance(payload, dict):
        items = [payload]
        shared = {}
    else:
        raise ValueError(f"Unsupported result payload in {result_path}")

    rows: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(f"Unsupported result row in {result_path}: {item!r}")
        row = {**shared, **item}
        if "job_id" not in row:
            row["job_id"] = result_path.parent.name
        rows.append(row)
    return rows


def _compute_subsample_levels(n_runs: int, n_levels: int) -> list[int]:
    lower = max(1000, n_runs // 8)
    if n_levels == 1:
        return [int(n_runs)]
    raw = np.geomspace(lower, n_runs, num=n_levels)
    rounded = np.round(raw).astype(int)
    rounded[0] = lower
    rounded[-1] = n_runs
    rounded = np.maximum.accumulate(rounded)
    for idx in range(1, len(rounded)):
        if rounded[idx] <= rounded[idx - 1]:
            rounded[idx] = min(n_runs, rounded[idx - 1] + 1)
    rounded[-1] = n_runs
    return rounded.tolist()


def _scale_float(value: float, low: float, high: float, *, log_scale: bool = False) -> float:
    if log_scale:
        return float(np.exp(np.log(low) + value * (np.log(high) - np.log(low))))
    return float(low + value * (high - low))


def _scale_int(value: float, low: int, high: int, *, log_scale: bool = False) -> int:
    scaled = _scale_float(value, float(low), float(high), log_scale=log_scale)
    return int(np.clip(round(scaled), low, high))


__all__ = [
    "SensitivityRunResult",
    "SensitivityStudyJob",
    "SensitivityStudySpec",
    "collect_study_results",
    "generate_calibrated_structure_dgps",
    "generate_config_lhs",
    "generate_lhs_points",
    "generate_pure_synthetic_dgps",
    "generate_study_jobs",
    "jobs_to_dataframe",
    "sensitivity_study_spec_from_mapping",
]
