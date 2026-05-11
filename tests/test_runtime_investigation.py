"""Tests for runtime investigation workflow tooling."""

from __future__ import annotations

from pathlib import Path

import yaml

from tools.run_runtime_investigation import (
    DEFAULT_PROFILES,
    compute_runtime_projection,
    materialize_ladder_configs,
)


def _base_config() -> dict:
    return {
        "dataset": {"type": "synthetic_full"},
        "algorithm": {"variance_threshold": 0.90},
        "runtime": {"n_jobs": 1, "output_batch_size": None},
        "stages": {
            "empirical_null_screening": {"n_permutations": 1000, "bh_q_threshold": 0.10},
            "interaction_discovery": {"n_permutations": 101, "n_tree_estimators": 100},
            "nonlinear_discovery": {"edf_threshold": 2.5},
            "sparse_selection": {"n_stability_subsamples": 100, "max_candidate_terms": 500},
            "final_artifacts": {"bootstrap_count": 200, "bootstrap_alpha": 0.05},
        },
        "validation": {"fast_mode": False, "fast_mode_overrides": {}},
        "output": {"artifact_dir": "./artifacts/base/", "seed": 123, "verbose": True},
    }


def test_materialize_ladder_configs_creates_profiles_and_dataset_override(tmp_path: Path) -> None:
    output_root = tmp_path / "investigation"
    dataset_path = tmp_path / "my_dataset"
    dataset_path.mkdir()

    configs = materialize_ladder_configs(
        base_config=_base_config(),
        output_root=output_root,
        profiles=DEFAULT_PROFILES,
        dataset_path=dataset_path,
    )

    assert sorted(configs) == ["large", "medium", "small"]
    for profile, cfg_path in configs.items():
        assert cfg_path.exists()
        loaded = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        assert loaded["dataset"]["path"] == str(dataset_path)
        assert loaded["dataset"]["type"] == "custom_dataset"
        assert loaded["output"]["artifact_dir"].endswith(f"/runs/{profile}")

    small_cfg = yaml.safe_load(configs["small"].read_text(encoding="utf-8"))
    large_cfg = yaml.safe_load(configs["large"].read_text(encoding="utf-8"))
    assert (
        small_cfg["stages"]["final_artifacts"]["bootstrap_count"]
        < large_cfg["stages"]["final_artifacts"]["bootstrap_count"]
    )
    assert (
        small_cfg["stages"]["sparse_selection"]["max_candidate_terms"]
        < large_cfg["stages"]["sparse_selection"]["max_candidate_terms"]
    )


def test_compute_runtime_projection_uses_completed_rows_only() -> None:
    rows = [
        {
            "profile": "small",
            "status": "complete",
            "sparse_seconds": 10.0,
            "final_seconds": 20.0,
            "sparse_units": 100.0,
            "final_units": 200.0,
        },
        {
            "profile": "medium",
            "status": "complete",
            "sparse_seconds": 25.0,
            "final_seconds": 40.0,
            "sparse_units": 250.0,
            "final_units": 400.0,
        },
        {
            "profile": "large",
            "status": "failed",
            "sparse_seconds": None,
            "final_seconds": None,
            "sparse_units": 600.0,
            "final_units": 1200.0,
        },
    ]
    target = {
        "target_sparse_units": 500.0,
        "target_final_units": 800.0,
        "target_profile_name": "full",
    }

    projection = compute_runtime_projection(rows, target)

    assert projection["status"] == "ok"
    # median rates: sparse=(0.10,0.10), final=(0.10,0.10) => projected=50+80=130
    assert projection["projected_total_seconds"] == 130.0
    assert projection["projected_total_minutes"] == 2.17
    assert projection["completed_profiles"] == 2


def test_compute_runtime_projection_handles_missing_data() -> None:
    projection = compute_runtime_projection(
        rows=[{"profile": "small", "status": "failed"}],
        target={
            "target_sparse_units": 10.0,
            "target_final_units": 10.0,
            "target_profile_name": "full",
        },
    )
    assert projection["status"] == "insufficient_data"
