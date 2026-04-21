"""Tests for workflow provenance and end-to-end orchestration."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from bsm_rfm.viz_io import load_postfit_bundle
from bsm_rfm.workflow import (
    canonical_case_study_numbers,
    canonical_workflow_stages,
    case_study_number_table,
    run_canonical_workflow,
    workflow_stage_table,
    write_postfit_bundle,
)


def test_canonical_workflow_stage_order_is_stable() -> None:
    stages = canonical_workflow_stages()
    assert [stage.name for stage in stages] == [
        "upstream_null_screening",
        "feature_expansion",
        "modeling_subset_creation",
        "regularized_screening",
        "final_ols",
        "evaluation_export",
        "downstream_visualization",
    ]
    assert [stage.order for stage in stages] == list(range(1, 8))


def test_workflow_stage_table_contains_recovered_provenance_and_current_status() -> None:
    table = workflow_stage_table()
    feature_stage = table.loc[table["name"] == "feature_expansion"].iloc[0]
    screening_stage = table.loc[table["name"] == "regularized_screening"].iloc[0]
    evaluation_stage = table.loc[table["name"] == "evaluation_export"].iloc[0]

    assert feature_stage["provenance"] == "notebook-derived"
    assert feature_stage["source_artifact"] == "make_nonlinear_features.ipynb"
    assert feature_stage["status"] == "spec_recovered_not_fully_ported"
    assert screening_stage["status"] == "implemented_foundation"
    assert evaluation_stage["status"] == "implemented_foundation"


def test_case_study_numbers_include_balanced_subset_and_selected_feature_counts() -> None:
    numbers = {item.key: item.value for item in canonical_case_study_numbers()}
    assert numbers["upstream_sample_size"] == 300000
    assert numbers["modeling_subset_size"] == 20000
    assert numbers["rows_per_boolean_stratum"] == 5000
    assert numbers["selected_feature_count"] == 346


def test_case_study_number_table_preserves_key_metadata() -> None:
    table = case_study_number_table()
    row = table.loc[table["key"] == "null_permutation_count"].iloc[0]
    assert row["value"] == 200
    assert row["provenance"] == "source-derived"


def test_run_canonical_workflow_builds_artifacts_and_manifest_metadata() -> None:
    X_train = pd.DataFrame(
        {
            "x1": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "x2": [1.0, 0.5, 2.0, 1.5, 3.0, 2.5, 3.5, 4.0],
            "x3": [0.2, -1.1, 0.5, -0.7, 1.4, -0.3, 0.8, -1.5],
        }
    )
    Y_train = pd.DataFrame(
        {
            "y1": 1.0 + 2.0 * X_train["x1"] - 1.0 * X_train["x2"],
            "y2": -0.5 + 0.75 * X_train["x1"] + 0.5 * X_train["x2"],
        }
    )
    X_holdout = pd.DataFrame(
        {
            "x1": [8.0, 9.0, 10.0],
            "x2": [4.5, 5.0, 5.5],
            "x3": [1.2, -0.4, 0.1],
        }
    )
    Y_holdout = pd.DataFrame(
        {
            "y1": 1.0 + 2.0 * X_holdout["x1"] - 1.0 * X_holdout["x2"],
            "y2": -0.5 + 0.75 * X_holdout["x1"] + 0.5 * X_holdout["x2"],
        }
    )

    run = run_canonical_workflow(
        X_train,
        Y_train,
        X_holdout,
        Y_holdout,
        dataset_tag="demo-run",
        screening_cv=3,
        screening_l1_ratio=(0.9, 1.0),
        screening_alphas=50,
        screening_random_state=17,
        n_boot=25,
        bootstrap_random_state=19,
        bootstrap_sample_size=len(X_holdout),
        artifact_format="csv",
        upstream_provenance={"source": "unit-test"},
    )

    assert {"x1", "x2"}.issubset(set(run.screening_result.selected_features))
    assert {"x1", "x2"}.issubset(set(run.final_ols_result.feature_names))
    assert run.holdout_summary.loc[0, "point_estimate"] == pytest.approx(0.0)
    assert run.artifact_format == "csv"
    assert run.artifacts["manifest"]["files"]["nrmse_summary"].endswith(".csv")
    assert run.artifacts["manifest"]["metrics"]["holdout_nrmse"]["n_boot"] == 25
    assert run.artifacts["manifest"]["evaluation"]["screening_cv"] == 3
    assert run.artifacts["manifest"]["upstream_provenance"]["source"] == "unit-test"


def test_write_postfit_bundle_round_trips_with_visualization_loader(tmp_path) -> None:
    X_train = pd.DataFrame(
        {
            "x1": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "x2": [1.0, 0.0, 1.5, 0.5, 2.0, 1.0],
            "x3": [0.4, -1.0, 0.7, -0.2, 1.1, -0.6],
        }
    )
    Y_train = pd.DataFrame(
        {
            "y1": 2.0 * X_train["x1"] + X_train["x2"],
            "y2": -1.0 + 0.5 * X_train["x1"] - 2.0 * X_train["x2"],
        }
    )
    X_holdout = pd.DataFrame({"x1": [6.0, 7.0], "x2": [1.5, 2.5], "x3": [0.3, -0.8]})
    Y_holdout = pd.DataFrame(
        {
            "y1": 2.0 * X_holdout["x1"] + X_holdout["x2"],
            "y2": -1.0 + 0.5 * X_holdout["x1"] - 2.0 * X_holdout["x2"],
        }
    )

    run = run_canonical_workflow(
        X_train,
        Y_train,
        X_holdout,
        Y_holdout,
        dataset_tag="roundtrip",
        screening_cv=3,
        screening_l1_ratio=(0.9, 1.0),
        screening_alphas=25,
        screening_random_state=5,
        n_boot=20,
        bootstrap_random_state=7,
        bootstrap_sample_size=len(X_holdout),
        artifact_format="csv",
    )
    written = write_postfit_bundle(run.artifacts, tmp_path)
    loaded = load_postfit_bundle(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))

    assert written["manifest"] == tmp_path / "manifest.json"
    assert written["coef_matrix_raw_scale"].suffix == ".csv"
    assert loaded["coef_matrix_raw_scale"].columns.tolist()[:3] == ["output_name", "x1", "x2"]
    assert loaded["nrmse_summary"].loc[0, "n_boot"] == 20
    assert manifest["files"]["coef_matrix_raw_scale"].endswith(".csv")
