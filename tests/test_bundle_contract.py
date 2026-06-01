"""Tests for the canonical export-bundle contract."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from rfm_pipeline import (
    canonical_bundle_loader_keys,
    canonical_manifest_position_map_keys,
    canonical_manifest_top_level_keys,
    canonical_postfit_artifact_names,
    fit_final_ols,
    load_postfit_bundle,
    write_postfit_bundle,
)
from rfm_pipeline.final_ols import build_postfit_artifacts


def _make_demo_fit_result():
    X = pd.DataFrame(
        {
            "x1": [0.0, 1.0, 2.0, 3.0, 4.0],
            "x2": [1.0, 2.0, 0.0, 1.0, 3.0],
        }
    )
    Y = pd.DataFrame(
        {
            "y1": [1.0, 2.0, 2.5, 4.0, 5.5],
            "y2": [0.5, 1.5, 1.0, 2.5, 3.5],
        }
    )
    return fit_final_ols(X, Y)


def _make_demo_nrmse_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "n_outputs": 2,
                "macro_nrmse": 0.12,
                "ci_lower": 0.10,
                "ci_upper": 0.14,
                "n_boot": 25,
                "sample_size": 5,
                "alpha": 0.05,
            }
        ]
    )


def test_manifest_top_level_keys_and_position_map_keys_are_stable() -> None:
    assert canonical_manifest_position_map_keys() == [
        "all_input_position_map",
        "selected_input_position_map",
        "retained_input_position_map",
        "output_position_map",
    ]
    assert canonical_manifest_top_level_keys() == [
        "dataset_tag",
        "n_all_input_features",
        "n_selected_features",
        "n_retained_features",
        "n_outputs",
        "all_input_features",
        "selected_features",
        "retained_features",
        "output_names",
        "files",
        "metrics",
        "evaluation",
        "upstream_provenance",
        *canonical_manifest_position_map_keys(),
    ]


def test_postfit_artifact_names_and_loader_keys_match() -> None:
    assert canonical_bundle_loader_keys() == canonical_postfit_artifact_names()


def test_build_postfit_artifacts_manifest_matches_stable_contract() -> None:
    result = _make_demo_fit_result()
    artifacts = build_postfit_artifacts(
        result,
        dataset_tag="demo-bundle",
        all_input_features=["x1", "x2", "scenario_flag"],
        selected_features=["x1", "x2"],
        evaluation_summary=_make_demo_nrmse_summary(),
        artifact_format="csv",
    )
    manifest = artifacts["manifest"]

    assert list(manifest) == canonical_manifest_top_level_keys()
    assert sorted(manifest["files"]) == sorted(canonical_postfit_artifact_names())


def test_written_bundle_manifest_and_loader_follow_stable_contract(tmp_path: Path) -> None:
    result = _make_demo_fit_result()
    artifacts = build_postfit_artifacts(
        result,
        dataset_tag="demo-bundle",
        all_input_features=["x1", "x2", "scenario_flag"],
        selected_features=["x1", "x2"],
        evaluation_summary=_make_demo_nrmse_summary(),
        artifact_format="csv",
    )

    write_postfit_bundle(artifacts, tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    loaded = load_postfit_bundle(tmp_path)

    assert sorted(manifest["files"]) == sorted(canonical_postfit_artifact_names())
    assert sorted(loaded) == sorted(canonical_bundle_loader_keys())
    assert loaded["nrmse_summary"].loc[0, "n_boot"] == 25


def test_export_bundle_docs_list_stable_contract_helpers() -> None:
    text = Path("docs/export_bundle.md").read_text(encoding="utf-8")
    assert "canonical_manifest_top_level_keys" in text
    assert "canonical_bundle_loader_keys" in text
