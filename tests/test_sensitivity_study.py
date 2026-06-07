# Copyright (c) 2026 Dylan Hettinger
"""Tests for sensitivity-study job generation and result collection."""

from __future__ import annotations

import json

import pytest

from rfm_pipeline.sensitivity_study import (
    _CONFIG_OPTIONS,
    SensitivityStudySpec,
    _scale_float,
    _scale_int,
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


# ---------------------------------------------------------------------------
# Manuscript Table 3 / Table 4 boundary regression tests
# ---------------------------------------------------------------------------


def test_scale_int_endpoints_recover_manuscript_bounds():
    # Manuscript Table 3 inputs range: 20-200 for pure-synthetic DGPs.
    assert _scale_int(0.0, 20, 200) == 20
    assert _scale_int(1.0, 20, 200) == 200
    # Midpoint sanity.
    assert 100 <= _scale_int(0.5, 20, 200) <= 120


def test_scale_int_log_scale_endpoints():
    # Output count uses log-scale 100-5000.
    assert _scale_int(0.0, 100, 5000, log_scale=True) == 100
    assert _scale_int(1.0, 100, 5000, log_scale=True) == 5000
    # Log-midpoint should be near geometric mean (~707).
    mid = _scale_int(0.5, 100, 5000, log_scale=True)
    assert 600 <= mid <= 800


def test_scale_float_endpoints_recover_bounds():
    assert _scale_float(0.0, 0.05, 0.70) == pytest.approx(0.05)
    assert _scale_float(1.0, 0.05, 0.70) == pytest.approx(0.70)
    assert _scale_float(0.5, 0.05, 0.70) == pytest.approx(0.375)


def test_config_sweep_options_match_manuscript_table4_values():
    """Manuscript Table 4 lists the swept values for each config dimension.

    This test pins the sweep tuples so that an accidental change to
    _CONFIG_OPTIONS will be caught immediately.
    """
    expected = {
        "holdout_fraction": (0.05, 0.10, 0.15, 0.20),
        "variance_threshold": (0.80, 0.85, 0.90, 0.95),
        "stages.empirical_null_screening.n_permutations": (51, 101, 201, 401),
        "stages.empirical_null_screening.bh_q_threshold": (0.01, 0.05, 0.10, 0.20),
        "stages.interaction_discovery.n_permutations": (11, 21, 31, 51, 101),
        "stages.interaction_discovery.p_threshold": (0.01, 0.05, 0.10, 0.20),
        "stages.sparse_selection.n_stability_subsamples": (10, 25, 50, 100),
        "stages.sparse_selection.lasso_alpha_grid_size": (20, 40, 80, 160),
        # Manuscript Table 4: pruning delta sweep is exactly 4 values; no None.
        "stages.final_artifacts.delta_threshold_override": (0.001, 0.002, 0.005, 0.010),
    }
    actual = dict(_CONFIG_OPTIONS)
    for key, vals in expected.items():
        assert actual[key] == vals, f"{key}: expected {vals}, got {actual[key]}"
