"""Tests for canonical workflow provenance artifacts."""

from __future__ import annotations

from bsm_rfm.workflow import (
    canonical_case_study_numbers,
    canonical_workflow_stages,
    case_study_number_table,
    workflow_stage_table,
)


def test_canonical_workflow_stages_preserve_audited_stage_order():
    stage_keys = [stage.stage_key for stage in canonical_workflow_stages()]
    assert stage_keys == [
        "upstream_null_screening",
        "feature_expansion",
        "modeling_subset_creation",
        "regularized_screening",
        "final_ols",
        "evaluation_export",
        "downstream_visualization",
    ]


def test_workflow_stage_table_exposes_provenance_boundary_labels():
    stage_table = workflow_stage_table().set_index("stage_key")
    assert stage_table.loc["upstream_null_screening", "provenance"] == "source-derived"
    assert stage_table.loc["feature_expansion", "implementation_status"] == (
        "implemented_config_boundary"
    )
    assert stage_table.loc["regularized_screening", "implementation_status"] == ("documented_only")


def test_case_study_number_table_contains_recovered_values():
    number_table = case_study_number_table().set_index("key")
    assert number_table.loc["upstream_rows", "value"] == 300000
    assert number_table.loc["modeling_subset_rows", "value"] == 20000
    assert number_table.loc["selected_second_order_features", "value"] == 244
    assert number_table.loc["bootstrap_replicates", "value"] == 1000


def test_canonical_case_study_numbers_are_unique_by_key():
    keys = [item.key for item in canonical_case_study_numbers()]
    assert len(keys) == len(set(keys))
