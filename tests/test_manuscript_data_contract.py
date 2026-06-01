"""Tests for the Phase 1 manuscript data-contract layer."""

from __future__ import annotations

from pathlib import Path

from rfm_pipeline import (
    manuscript_notebook_manifest_table,
    manuscript_notebook_order,
    manuscript_placeholder_path_policy,
    manuscript_required_artifact_table,
    required_manuscript_artifacts,
)


def test_required_manuscript_artifacts_are_frozen() -> None:
    assert required_manuscript_artifacts() == (
        "input_metadata",
        "output_metadata",
        "case_study_input_matrix",
        "case_study_output_matrix",
        "manuscript_feature_catalog",
        "fixed_holdout_assignments",
    )


def test_placeholder_path_policy_is_frozen() -> None:
    assert manuscript_placeholder_path_policy() == {
        "template_file": "configs/manuscript_paths.template.yml",
        "local_override_file": "configs/local/manuscript_paths.local.yml",
        "placeholder_prefix": "REPLACE_WITH_REAL_FILE/",
    }


def test_notebook_order_is_frozen() -> None:
    assert manuscript_notebook_order() == (
        "00_case_study_data_intake.ipynb",
        "01_candidate_library_audit.ipynb",
        "02_output_conditioning.ipynb",
        "03_empirical_null_screen.ipynb",
        "04_interaction_discovery.ipynb",
        "05_nonlinear_discovery.ipynb",
        "06_sparse_selection_and_stability.ipynb",
        "07_final_ols_and_bundle_export.ipynb",
        "08_manuscript_tables_and_figures.ipynb",
    )


def test_contract_files_and_docs_exist() -> None:
    assert Path("configs/manuscript_data_contract.yml").exists()
    assert Path("configs/manuscript_paths.template.yml").exists()
    assert Path("configs/manuscript_runtime.yml").exists()
    assert Path("docs/manuscript_data_contract.md").exists()
    assert Path("notebooks/manuscript/README.md").exists()


def test_manifest_tables_match_contract_lengths() -> None:
    assert len(manuscript_required_artifact_table()) == len(required_manuscript_artifacts())
    assert len(manuscript_notebook_manifest_table()) == len(manuscript_notebook_order())
