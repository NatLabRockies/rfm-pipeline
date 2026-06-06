# Copyright (c) 2026 Dylan Hettinger
"""Tests for sensitivity-study job generation and result collection."""

from __future__ import annotations

import json

from rfm_pipeline.sensitivity_study import (
    SensitivityStudySpec,
    collect_study_results,
    generate_config_lhs,
    generate_pure_synthetic_dgps,
    generate_study_jobs,
    jobs_to_dataframe,
)


def _small_study_spec() -> SensitivityStudySpec:
    return SensitivityStudySpec(
        pure_synthetic_n_dgps=4,
        calibrated_n_dgps=2,
        n_configs_per_dgp_pure=3,
        n_configs_per_dgp_calibrated=2,
        n_replicates=2,
        pure_synthetic_n_subsample_levels=4,
        calibrated_n_subsample_levels=3,
        config_lhs_seed=17,
        dgp_lhs_seed=5,
    )


def test_generate_pure_synthetic_dgps_count() -> None:
    spec = _small_study_spec()
    dgps = generate_pure_synthetic_dgps(spec)

    assert len(dgps) == spec.pure_synthetic_n_dgps
    assert all(dgp.dgp_family == "pure_synthetic" for dgp in dgps)


def test_generate_config_lhs_count() -> None:
    configs = generate_config_lhs(_small_study_spec(), n_configs=7)

    assert len(configs) == 7
    assert all("variance_threshold" in config for config in configs)
    assert all("stages.empirical_null_screening.n_permutations" in config for config in configs)


def test_study_jobs_job_id_unique() -> None:
    spec = _small_study_spec()
    jobs = generate_study_jobs(spec)

    expected = (
        spec.pure_synthetic_n_dgps * spec.n_configs_per_dgp_pure * spec.n_replicates
        + spec.calibrated_n_dgps * spec.n_configs_per_dgp_calibrated * spec.n_replicates
    )
    assert len(jobs) == expected
    assert len({job.job_id for job in jobs}) == len(jobs)


def test_collect_results_empty_dir(tmp_path) -> None:
    results = collect_study_results(tmp_path)

    assert results.empty


def test_jobs_to_dataframe_roundtrip() -> None:
    jobs = generate_study_jobs(_small_study_spec())
    frame = jobs_to_dataframe(jobs)

    assert len(frame) == len(jobs)
    row = frame.iloc[0]
    job = jobs[0]
    assert row["job_id"] == job.job_id
    assert json.loads(row["config_overrides"]) == job.config_overrides
    assert [
        int(value) for value in row["n_subsample_levels"].split(",") if value
    ] == job.n_subsample_levels  # noqa: E501
    assert row["dgp_family"] == job.dgp_spec.dgp_family
    assert int(row["n_runs"]) == job.dgp_spec.n_runs
