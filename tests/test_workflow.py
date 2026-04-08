"""Tests for workflow provenance helpers."""

from __future__ import annotations

from bsm_rfm.workflow import (
    canonical_case_study_numbers,
    canonical_workflow_stages,
    case_study_number_table,
    workflow_stage_table,
)


def test_canonical_workflow_stages_capture_verified_stage_sequence() -> None:
    stages = canonical_workflow_stages()
    assert [stage.stage_key for stage in stages] == [
        "upstream_null_screening",
        "feature_expansion",
        "modeling_subset_creation",
        "regularized_screening",
        "final_ols",
        "evaluation_export",
        "downstream_visualization",
    ]
    assert stages[0].canonical_source == "null_distribution.py"
    assert stages[2].key_settings["n_per_stratum"] == 5000
    assert stages[3].provenance_status == "notebook-derived"
    assert stages[5].key_settings["bootstrap_replicates"] == 1000


def test_workflow_stage_table_preserves_stage_order() -> None:
    table = workflow_stage_table()
    assert table["stage_key"].tolist()[0] == "upstream_null_screening"
    assert table["stage_key"].tolist()[-1] == "downstream_visualization"
    assert "implementation_status" in table.columns


def test_canonical_case_study_numbers_match_recovered_values() -> None:
    numbers = canonical_case_study_numbers()
    assert numbers["upstream_screening_dataset_size"] == 300000
    assert numbers["modeling_subset_size"] == 20000
    assert numbers["selected_feature_count"] == 346
    assert numbers["bootstrap_replicates"] == 1000


def test_case_study_number_table_contains_key_rows() -> None:
    table = case_study_number_table()
    lookup = dict(zip(table["name"], table["value"], strict=True))
    assert lookup["n_per_boolean_stratum"] == 5000
    assert lookup["full_output_count"] == 23495
