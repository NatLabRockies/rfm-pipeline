"""Tests for the documented public package surface."""

from __future__ import annotations

from pathlib import Path

from bsm_rfm import (
    DEFAULT_TRANSFORM_LIBRARY,
    EXPONENTIAL,
    INVERSE,
    LOGARITHMIC,
    QUADRATIC,
    SQRT,
    EmpiricalNullScreeningSpec,
    FeatureExpansionResult,
    FeatureExpansionSpec,
    FinalManuscriptArtifactsSpec,
    InteractionDiscoverySpec,
    ManuscriptReproductionAuditResult,
    ManuscriptReproductionAuditStageResult,
    ManuscriptReproductionStageChainResult,
    NonlinearDiscoverySpec,
    OutputConditioningSpec,
    SparseSelectionStabilitySpec,
    TransformDef,
    apply_feature_expansion,
    audit_manuscript_reproduction_outputs,
    build_manuscript_feature_design,
    build_manuscript_notebook_context,
    canonical_bundle_loader_keys,
    canonical_manifest_position_map_keys,
    canonical_manifest_top_level_keys,
    condition_manuscript_outputs,
    default_feature_expansion_spec,
    discover_manuscript_interactions,
    discover_manuscript_nonlinear_transformations,
    empirical_null_screening_spec_from_case_study_config,
    final_manuscript_artifacts_spec_from_case_study_config,
    interaction_discovery_spec_from_case_study_config,
    load_pipeline_outputs,
    load_postfit_bundle,
    manuscript_notebook_order,
    manuscript_placeholder_path_policy,
    manuscript_required_artifact_table,
    manuscript_runtime_summary_table,
    nonlinear_discovery_spec_from_case_study_config,
    ordered_expanded_feature_names,
    output_conditioning_spec_from_case_study_config,
    regenerate_final_manuscript_artifacts,
    resolve_manuscript_runtime,
    run_empirical_null_screening_stage,
    run_final_manuscript_artifacts_stage,
    run_interaction_discovery_stage,
    run_manuscript_reproduction_audit_stage,
    run_manuscript_reproduction_stage_chain,
    run_nonlinear_discovery_stage,
    run_output_conditioning_stage,
    run_sparse_selection_stability_stage,
    screen_manuscript_empirical_null_terms,
    select_manuscript_sparse_support,
    sparse_selection_stability_spec_from_case_study_config,
    warn_nan_transforms,
    workflow_scope_boundary_table,
    write_final_manuscript_artifacts,
    write_interaction_discovery_artifacts,
    write_manuscript_reproduction_audit,
    write_nonlinear_discovery_artifacts,
    write_sparse_selection_stability_artifacts,
)


def test_package_exports_feature_expansion_contract() -> None:
    assert FeatureExpansionSpec.__name__ == "FeatureExpansionSpec"
    assert FeatureExpansionResult.__name__ == "FeatureExpansionResult"
    assert callable(default_feature_expansion_spec)
    assert callable(ordered_expanded_feature_names)
    assert callable(apply_feature_expansion)


def test_package_exports_transform_library() -> None:
    assert TransformDef.__name__ == "TransformDef"
    assert len(DEFAULT_TRANSFORM_LIBRARY) == 5
    assert QUADRATIC.label == "sq"
    assert LOGARITHMIC.label == "log1p"
    assert INVERSE.label == "inv"
    assert SQRT.label == "sqrt"
    assert EXPONENTIAL.label == "exp"
    assert callable(warn_nan_transforms)


def test_package_exports_postfit_bundle_loaders() -> None:
    assert callable(load_postfit_bundle)
    assert callable(load_pipeline_outputs)


def test_api_reference_documents_feature_expansion_viz_io_and_runtime_modules() -> None:
    api = Path("docs/api.rst").read_text(encoding="utf-8")
    assert ".. automodule:: bsm_rfm.feature_expansion" in api
    assert ".. automodule:: bsm_rfm.viz_io" in api
    assert ".. automodule:: bsm_rfm.manuscript_runtime" in api
    assert ".. automodule:: bsm_rfm.manuscript_stages" in api


def test_docs_include_quickstart_export_bundle_and_runtime_guides() -> None:
    quickstart = Path("docs/quickstart.md").read_text(encoding="utf-8")
    export_bundle = Path("docs/export_bundle.md").read_text(encoding="utf-8")
    runtime_doc = Path("docs/manuscript_runtime.md").read_text(encoding="utf-8")
    assert "run_canonical_workflow" in quickstart
    assert "write_postfit_bundle" in quickstart
    assert "load_postfit_bundle" in quickstart
    assert "manifest.json" in export_bundle
    assert "postfit_diagnostics/" in export_bundle
    assert "resolve_manuscript_runtime" in runtime_doc


def test_package_exports_workflow_scope_boundary_helper() -> None:
    assert callable(workflow_scope_boundary_table)


def test_docs_include_reproducibility_example_guide() -> None:
    example_doc = Path("docs/reproducibility_example.md").read_text(encoding="utf-8")
    assert "run_reproducibility_example" in example_doc
    assert "run_manuscript_reproduction_example" in example_doc
    assert "--run-manuscript-chain" in example_doc
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


def test_package_exports_manuscript_runtime_helpers() -> None:
    assert callable(resolve_manuscript_runtime)
    assert callable(build_manuscript_notebook_context)
    assert callable(manuscript_runtime_summary_table)


def test_package_exports_manuscript_output_conditioning_stage() -> None:
    assert OutputConditioningSpec.__name__ == "OutputConditioningSpec"
    assert callable(output_conditioning_spec_from_case_study_config)
    assert callable(condition_manuscript_outputs)
    assert callable(run_output_conditioning_stage)


def test_docs_include_manuscript_data_contract_guide() -> None:
    doc = Path("docs/manuscript_data_contract.md").read_text(encoding="utf-8")
    assert "manuscript_feature_catalog" in doc
    assert "configs/manuscript_runtime.yml" in doc


def test_package_exports_manuscript_empirical_null_screening_stage() -> None:
    assert EmpiricalNullScreeningSpec.__name__ == "EmpiricalNullScreeningSpec"
    assert callable(empirical_null_screening_spec_from_case_study_config)
    assert callable(build_manuscript_feature_design)
    assert callable(screen_manuscript_empirical_null_terms)
    assert callable(run_empirical_null_screening_stage)


def test_package_exports_manuscript_interaction_discovery_stage() -> None:
    assert InteractionDiscoverySpec.__name__ == "InteractionDiscoverySpec"
    assert callable(interaction_discovery_spec_from_case_study_config)
    assert callable(discover_manuscript_interactions)
    assert callable(run_interaction_discovery_stage)
    assert callable(write_interaction_discovery_artifacts)


def test_package_exports_manuscript_nonlinear_discovery_stage() -> None:
    assert NonlinearDiscoverySpec.__name__ == "NonlinearDiscoverySpec"
    assert callable(nonlinear_discovery_spec_from_case_study_config)
    assert callable(discover_manuscript_nonlinear_transformations)
    assert callable(run_nonlinear_discovery_stage)
    assert callable(write_nonlinear_discovery_artifacts)


def test_package_exports_manuscript_sparse_selection_stage() -> None:
    assert SparseSelectionStabilitySpec.__name__ == "SparseSelectionStabilitySpec"
    assert callable(sparse_selection_stability_spec_from_case_study_config)
    assert callable(select_manuscript_sparse_support)
    assert callable(run_sparse_selection_stability_stage)
    assert callable(write_sparse_selection_stability_artifacts)


def test_package_exports_final_manuscript_artifact_stage() -> None:
    assert FinalManuscriptArtifactsSpec.__name__ == "FinalManuscriptArtifactsSpec"
    assert callable(final_manuscript_artifacts_spec_from_case_study_config)
    assert callable(regenerate_final_manuscript_artifacts)
    assert callable(run_final_manuscript_artifacts_stage)
    assert callable(write_final_manuscript_artifacts)


def test_package_exports_manuscript_reproduction_chain() -> None:
    assert ManuscriptReproductionStageChainResult.__name__ == (
        "ManuscriptReproductionStageChainResult"
    )
    assert callable(run_manuscript_reproduction_stage_chain)


def test_package_exports_manuscript_reproduction_audit() -> None:
    assert ManuscriptReproductionAuditResult.__name__ == "ManuscriptReproductionAuditResult"
    assert ManuscriptReproductionAuditStageResult.__name__ == (
        "ManuscriptReproductionAuditStageResult"
    )
    assert callable(audit_manuscript_reproduction_outputs)
    assert callable(run_manuscript_reproduction_audit_stage)
    assert callable(write_manuscript_reproduction_audit)
