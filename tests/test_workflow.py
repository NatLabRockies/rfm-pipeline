"""Tests for workflow provenance records."""

from __future__ import annotations

from bsm_rfm.workflow import (
    canonical_case_study_numbers,
    canonical_workflow_stages,
    case_study_number_table,
    workflow_stage_table,
)


def test_canonical_workflow_stages_preserve_expected_order_and_labels():
    stages = canonical_workflow_stages()
    assert [stage.stage_key for stage in stages] == [
        "null_screening",
        "feature_expansion",
        "modeling_subset",
        "regularized_screening",
        "final_ols",
        "evaluation_export",
        "downstream_visualization",
    ]
    assert stages[0].implemented_in_package is True
    assert stages[1].source_status == "notebook-derived"


def test_workflow_stage_table_contains_one_row_per_stage():
    stage_table = workflow_stage_table()
    assert stage_table["stage_key"].tolist()[0] == "null_screening"
    assert int(stage_table.shape[0]) == len(canonical_workflow_stages())


def test_case_study_numbers_include_recovered_counts():
    numbers = canonical_case_study_numbers()
    assert numbers["modeling_subset_rows"] == 20000
    assert numbers["rows_per_boolean_stratum"] == 5000
    assert numbers["selected_feature_count"] == 346


def test_case_study_number_table_round_trips_recovered_metrics():
    table = case_study_number_table()
    metric_map = dict(zip(table["metric_name"], table["value"], strict=True))
    assert metric_map["upstream_null_screening_rows"] == 300000
    assert metric_map["full_output_count"] == 23495
