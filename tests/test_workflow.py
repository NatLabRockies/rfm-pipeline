"""Tests for the workflow provenance module."""

from __future__ import annotations

from bsm_rfm.workflow import (
    canonical_case_study_numbers,
    canonical_workflow_stages,
    case_study_number_table,
    workflow_stage_table,
)


def test_canonical_workflow_stage_order_is_stable():
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


def test_workflow_stage_table_contains_recovered_provenance_and_status():
    table = workflow_stage_table()
    feature_stage = table.loc[table["name"] == "feature_expansion"].iloc[0]
    assert feature_stage["provenance"] == "notebook-derived"
    assert feature_stage["source_artifact"] == "make_nonlinear_features.ipynb"
    assert feature_stage["status"] == "spec_recovered_not_fully_ported"


def test_case_study_numbers_include_balanced_subset_and_selected_feature_counts():
    numbers = {item.key: item.value for item in canonical_case_study_numbers()}
    assert numbers["upstream_sample_size"] == 300000
    assert numbers["modeling_subset_size"] == 20000
    assert numbers["rows_per_boolean_stratum"] == 5000
    assert numbers["selected_feature_count"] == 346


def test_case_study_number_table_preserves_key_metadata():
    table = case_study_number_table()
    row = table.loc[table["key"] == "null_permutation_count"].iloc[0]
    assert row["value"] == 200
    assert row["provenance"] == "source-derived"
