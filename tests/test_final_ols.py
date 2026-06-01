"""Tests for final OLS foundations and post-fit artifact helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rfm_pipeline.final_ols import (
    build_postfit_artifacts,
    canonical_postfit_artifact_names,
    fit_final_ols,
    make_coefficient_matrix_frame,
    make_holdout_nrmse_summary,
    make_standardization_frame,
    notebook_final_ols_contract,
    postfit_artifact_table,
    predict_final_ols,
)


def test_notebook_final_ols_contract_captures_recovered_boundary() -> None:
    contract = notebook_final_ols_contract()
    assert contract.provenance == "notebook-derived"
    assert contract.holdout_fraction == pytest.approx(0.10)
    assert contract.selected_feature_count == 346
    assert contract.coefficient_scales == ("standardized", "raw_scale")
    assert "retained_input_order" in contract.diagnostics


def test_canonical_postfit_artifact_names_match_loader_contract() -> None:
    assert canonical_postfit_artifact_names() == [
        "all_input_metadata",
        "selected_input_metadata",
        "output_metadata",
        "coef_matrix_standardized",
        "coef_matrix_raw_scale",
        "x_standardization",
        "y_standardization",
        "nrmse_summary",
    ]


def test_postfit_artifact_table_preserves_contract_order() -> None:
    table = postfit_artifact_table()
    assert table["artifact_name"].tolist() == canonical_postfit_artifact_names()
    assert table.iloc[3].to_dict()["category"] == "coefficients"
    assert table.iloc[-1].to_dict()["category"] == "evaluation"


def test_make_coefficient_matrix_frame_labels_outputs_and_features() -> None:
    frame = make_coefficient_matrix_frame(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        output_names=["y1", "y2"],
        feature_names=["x1", "x2"],
    )
    assert frame.columns.tolist() == ["output_name", "x1", "x2"]
    assert frame.iloc[1].to_dict() == {"output_name": "y2", "x1": 3.0, "x2": 4.0}


def test_make_coefficient_matrix_frame_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="does not match labels"):
        make_coefficient_matrix_frame(
            np.array([[1.0, 2.0]]),
            output_names=["y1", "y2"],
            feature_names=["x1", "x2"],
        )


def test_make_standardization_frame_preserves_order_and_values() -> None:
    frame = make_standardization_frame(
        names=["x1", "x2"],
        means=np.array([10.0, 20.0]),
        scales=np.array([2.0, 5.0]),
        name_column="feature_name",
    )
    assert frame.to_dict(orient="records") == [
        {
            "feature_name": "x1",
            "original_position": 0,
            "mean": 10.0,
            "scale": 2.0,
        },
        {
            "feature_name": "x2",
            "original_position": 1,
            "mean": 20.0,
            "scale": 5.0,
        },
    ]


def test_make_standardization_frame_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same one-dimensional length"):
        make_standardization_frame(
            names=["x1", "x2"],
            means=np.array([1.0]),
            scales=np.array([1.0, 2.0]),
            name_column="feature_name",
        )


def test_fit_final_ols_recovers_raw_scale_coefficients_and_predictions() -> None:
    X = pd.DataFrame(
        {
            "x1": [0.0, 1.0, 2.0, 3.0, 4.0],
            "x2": [1.0, 0.5, 2.0, 1.5, 3.0],
        },
        index=[10, 11, 12, 13, 14],
    )
    Y = pd.DataFrame(
        {
            "y1": 1.5 + 2.0 * X["x1"] - 1.0 * X["x2"],
            "y2": -0.5 + 0.25 * X["x1"] + 3.0 * X["x2"],
        },
        index=X.index,
    )

    result = fit_final_ols(X, Y)
    pred = predict_final_ols(result, X)

    assert result.feature_names == ("x1", "x2")
    assert result.output_names == ("y1", "y2")
    assert np.allclose(result.coef_raw_scale, np.array([[2.0, -1.0], [0.25, 3.0]]))
    assert np.allclose(result.intercept_raw_scale, np.array([1.5, -0.5]))
    assert pred.index.tolist() == X.index.tolist()
    assert np.allclose(pred.to_numpy(), Y.to_numpy())


def test_make_holdout_nrmse_summary_and_postfit_artifacts_use_canonical_schema() -> None:
    X_train = pd.DataFrame(
        {
            "x1": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "x2": [1.0, 0.0, 1.5, 0.5, 2.0, 1.0],
        }
    )
    Y_train = pd.DataFrame(
        {
            "y1": 2.0 * X_train["x1"] + X_train["x2"],
            "y2": -1.0 + 0.5 * X_train["x1"] - 2.0 * X_train["x2"],
        }
    )
    X_holdout = pd.DataFrame({"x1": [6.0, 7.0], "x2": [1.5, 2.5]})
    Y_holdout = pd.DataFrame(
        {
            "y1": 2.0 * X_holdout["x1"] + X_holdout["x2"],
            "y2": -1.0 + 0.5 * X_holdout["x1"] - 2.0 * X_holdout["x2"],
        }
    )

    result = fit_final_ols(X_train, Y_train)
    summary = make_holdout_nrmse_summary(
        result,
        X_holdout,
        Y_holdout,
        Y_ref=Y_train,
        n_boot=25,
        random_state=17,
        sample_size=len(Y_holdout),
    )
    artifacts = build_postfit_artifacts(
        result,
        dataset_tag="demo",
        all_input_features=["x1", "x2", "scenario_flag"],
        selected_features=["x1", "x2"],
        evaluation_summary=summary,
        upstream_provenance={"stage": "unit-test"},
    )

    assert summary.loc[0, "point_estimate"] == pytest.approx(0.0)
    assert list(summary[["holdout_rows", "n_features", "n_outputs"]].iloc[0]) == [2, 2, 2]
    assert set(canonical_postfit_artifact_names()).issubset(artifacts)
    assert artifacts["coef_matrix_raw_scale"].columns.tolist() == ["output_name", "x1", "x2"]
    assert artifacts["selected_input_metadata"]["input_name"].tolist() == ["x1", "x2"]
    assert artifacts["manifest"]["all_input_position_map"]["scenario_flag"] == 2
    assert artifacts["manifest"]["upstream_provenance"]["stage"] == "unit-test"
