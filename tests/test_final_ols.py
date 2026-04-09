"""Tests for notebook-derived final OLS contract helpers."""

from __future__ import annotations

import numpy as np
import pytest

from bsm_rfm.final_ols import (
    canonical_postfit_artifact_names,
    make_coefficient_matrix_frame,
    make_standardization_frame,
    notebook_final_ols_contract,
    postfit_artifact_table,
)


def test_notebook_final_ols_contract_captures_recovered_boundary():
    contract = notebook_final_ols_contract()
    assert contract.provenance == "notebook_derived"
    assert contract.holdout_fraction == pytest.approx(0.10)
    assert contract.selected_feature_count == 346
    assert contract.coefficient_scales == ("standardized", "raw_scale")
    assert "retained_input_order" in contract.diagnostics


def test_canonical_postfit_artifact_names_match_loader_contract():
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


def test_postfit_artifact_table_preserves_contract_order():
    table = postfit_artifact_table()
    assert table["artifact_name"].tolist() == canonical_postfit_artifact_names()
    assert table.iloc[3].to_dict()["category"] == "coefficients"
    assert table.iloc[-1].to_dict()["category"] == "evaluation"


def test_make_coefficient_matrix_frame_labels_outputs_and_features():
    frame = make_coefficient_matrix_frame(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        output_names=["y1", "y2"],
        feature_names=["x1", "x2"],
    )
    assert frame.columns.tolist() == ["output_name", "x1", "x2"]
    assert frame.iloc[1].to_dict() == {"output_name": "y2", "x1": 3.0, "x2": 4.0}


def test_make_coefficient_matrix_frame_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="does not match labels"):
        make_coefficient_matrix_frame(
            np.array([[1.0, 2.0]]),
            output_names=["y1", "y2"],
            feature_names=["x1", "x2"],
        )


def test_make_standardization_frame_preserves_order_and_values():
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


def test_make_standardization_frame_rejects_length_mismatch():
    with pytest.raises(ValueError, match="same one-dimensional length"):
        make_standardization_frame(
            names=["x1", "x2"],
            means=np.array([1.0]),
            scales=np.array([1.0, 2.0]),
            name_column="feature_name",
        )
