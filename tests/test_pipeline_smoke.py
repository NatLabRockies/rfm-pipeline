"""Tests for test pipeline smoke."""

from __future__ import annotations

import pandas as pd

from rfm_pipeline.artifacts import PipelineManifest
from rfm_pipeline.data import (
    add_scenario_flags,
    fit_standardizers,
    stratified_holdout_split,
    stratified_subset_by_boolean_combination,
)
from rfm_pipeline.features import parse_selected_input_structure
from rfm_pipeline.metrics import bootstrap_macro_nrmse_ci, make_null_mean_prediction

SCENARIOS = [
    "AFSCoff_UAEOROoff",
    "AFSCon_UAEOROoff",
    "AFSCoff_UAEOROon",
    "AFSCon_UAEOROon",
]


def test_balanced_subset_then_holdout_then_null_baseline_metrics():
    rows = []
    outputs = []
    run_id = 0
    for scenario in SCENARIOS:
        for offset in range(7):
            rows.append(
                {
                    "run_id": run_id,
                    "scenario": scenario,
                    "x1": float(offset),
                    "x2": float(offset + 1),
                }
            )
            outputs.append(
                {
                    "run_id": run_id,
                    "scenario": scenario,
                    "y1": float(offset + 2),
                    "y2": float(2 * offset + 1),
                }
            )
            run_id += 1

    X_full = add_scenario_flags(pd.DataFrame(rows))
    subset = stratified_subset_by_boolean_combination(
        X_full,
        n_per_combination=5,
        random_state=17,
    )
    subset_ids = subset[["run_id", "scenario"]].drop_duplicates()
    Y_full = pd.DataFrame(outputs)
    Y = subset_ids.merge(Y_full, on=["run_id", "scenario"], how="left")
    Y = Y.set_index(["run_id", "scenario"])
    X = subset.set_index(["run_id", "scenario"])
    X_train, X_holdout, Y_train, Y_holdout = stratified_holdout_split(
        X,
        Y,
        holdout_fraction=0.25,
        random_state=13,
    )

    feature_columns = ["x1", "x2", "AFSC", "UAEORO"]
    bundle = fit_standardizers(
        X_train[feature_columns],
        X_holdout[feature_columns],
        Y_train,
        Y_holdout,
    )
    null_pred = make_null_mean_prediction(Y_train.to_numpy(), n_rows=len(Y_holdout))
    nrmse = bootstrap_macro_nrmse_ci(
        Y_holdout.to_numpy(),
        null_pred,
        Y_train.to_numpy(),
        n_boot=25,
        sample_size=len(Y_holdout),
        random_state=11,
    )

    structure, type_summary, _ = parse_selected_input_structure(["x1", "x1_quadratic", "x1*x2"])
    manifest = PipelineManifest(
        dataset_tag="demo",
        n_all_input_features=4,
        n_selected_features=3,
        n_retained_features=3,
        n_outputs=2,
        all_input_features=["x1", "x2", "AFSC", "UAEORO"],
        selected_features=["x1", "x1_quadratic", "x1*x2"],
        retained_features=["x1", "x1_quadratic", "x1*x2"],
        output_names=["y1", "y2"],
        files={"nrmse_summary": "postfit_diagnostics/nrmse_summary.parquet"},
        metrics={"null_nrmse": nrmse},
        evaluation={"holdout_fraction": 0.25},
    ).to_dict()

    assert len(subset) == 20
    assert len(X_holdout) == 5
    assert bundle.x_train_scaled.shape[1] == 4
    assert structure.shape[0] == 3
    assert set(type_summary["input_type"]) == {
        "first_order",
        "nonlinear_transformation",
        "second_order",
    }
    assert manifest["selected_input_position_map"]["x1*x2"] == 2
    assert manifest["metrics"]["null_nrmse"]["n_boot"] == 25
