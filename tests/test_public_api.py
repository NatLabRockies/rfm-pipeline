"""Tests for the documented public package surface."""

from __future__ import annotations

from pathlib import Path

from bsm_rfm import (
    FeatureExpansionResult,
    FeatureExpansionSpec,
    apply_feature_expansion,
    canonical_bundle_loader_keys,
    canonical_manifest_position_map_keys,
    canonical_manifest_top_level_keys,
    default_feature_expansion_spec,
    load_pipeline_outputs,
    load_postfit_bundle,
    manuscript_notebook_order,
    manuscript_placeholder_path_policy,
    manuscript_required_artifact_table,
    ordered_expanded_feature_names,
    workflow_scope_boundary_table,
)


def test_package_exports_feature_expansion_contract() -> None:
    assert FeatureExpansionSpec.__name__ == "FeatureExpansionSpec"
    assert FeatureExpansionResult.__name__ == "FeatureExpansionResult"
    assert callable(default_feature_expansion_spec)
    assert callable(ordered_expanded_feature_names)
    assert callable(apply_feature_expansion)


def test_package_exports_postfit_bundle_loaders() -> None:
    assert callable(load_postfit_bundle)
    assert callable(load_pipeline_outputs)


def test_api_reference_documents_feature_expansion_and_viz_io_modules() -> None:
    api = Path("docs/api.rst").read_text(encoding="utf-8")
    assert ".. automodule:: bsm_rfm.feature_expansion" in api
    assert ".. automodule:: bsm_rfm.viz_io" in api


def test_docs_include_quickstart_and_export_bundle_guides() -> None:
    quickstart = Path("docs/quickstart.md").read_text(encoding="utf-8")
    export_bundle = Path("docs/export_bundle.md").read_text(encoding="utf-8")
    assert "run_canonical_workflow" in quickstart
    assert "write_postfit_bundle" in quickstart
    assert "load_postfit_bundle" in quickstart
    assert "manifest.json" in export_bundle
    assert "postfit_diagnostics/" in export_bundle


def test_package_exports_workflow_scope_boundary_helper() -> None:
    assert callable(workflow_scope_boundary_table)


def test_docs_include_reproducibility_example_guide() -> None:
    example_doc = Path("docs/reproducibility_example.md").read_text(encoding="utf-8")
    assert "run_reproducibility_example" in example_doc
    assert "examples/end_to_end_reproducibility.py" in example_doc


def test_package_exports_bundle_contract_helpers() -> None:
    assert callable(canonical_bundle_loader_keys)
    assert callable(canonical_manifest_position_map_keys)
    assert callable(canonical_manifest_top_level_keys)


def test_readme_mentions_citation_and_changelog_files() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "CITATION.cff" in readme
    assert "CHANGELOG.md" in readme


def test_package_exports_manuscript_data_contract_helpers() -> None:
    assert callable(manuscript_required_artifact_table)
    assert callable(manuscript_placeholder_path_policy)
    assert isinstance(manuscript_notebook_order(), tuple)


def test_docs_include_manuscript_data_contract_guide() -> None:
    doc = Path("docs/manuscript_data_contract.md").read_text(encoding="utf-8")
    assert "manuscript_feature_catalog" in doc
    assert "configs/manuscript_runtime.yml" in doc
