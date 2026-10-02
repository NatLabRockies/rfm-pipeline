"""Tests for the documented public package surface."""

from __future__ import annotations

from pathlib import Path

from rfm_pipeline import (
    DEFAULT_TRANSFORM_LIBRARY,
    INVERSE,
    LOGARITHMIC,
    QUADRATIC,
    SQRT,
    CategoricalInputDecl,
    FeatureExpansionResult,
    FeatureExpansionSpec,
    TransformDef,
    apply_feature_expansion,
    build_design_matrix,
    canonical_bundle_loader_keys,
    canonical_manifest_position_map_keys,
    canonical_manifest_top_level_keys,
    default_feature_expansion_spec,
    load_pipeline_outputs,
    load_postfit_bundle,
    ordered_expanded_feature_names,
    run_canonical_workflow,
    warn_nan_transforms,
    write_postfit_bundle,
)


def test_package_exports_modeling_workflow() -> None:
    assert callable(run_canonical_workflow)
    assert callable(write_postfit_bundle)
    assert callable(load_postfit_bundle)
    assert callable(load_pipeline_outputs)


def test_package_exports_feature_expansion_contract() -> None:
    assert FeatureExpansionSpec.__name__ == "FeatureExpansionSpec"
    assert FeatureExpansionResult.__name__ == "FeatureExpansionResult"
    assert CategoricalInputDecl.__name__ == "CategoricalInputDecl"
    assert callable(default_feature_expansion_spec)
    assert callable(ordered_expanded_feature_names)
    assert callable(apply_feature_expansion)
    assert callable(build_design_matrix)


def test_package_exports_transform_library() -> None:
    assert TransformDef.__name__ == "TransformDef"
    assert len(DEFAULT_TRANSFORM_LIBRARY) == 4
    assert [QUADRATIC.label, LOGARITHMIC.label, INVERSE.label, SQRT.label] == [
        "sq",
        "log1p",
        "inv",
        "sqrt",
    ]
    assert callable(warn_nan_transforms)


def test_package_exports_bundle_contract_helpers() -> None:
    assert callable(canonical_bundle_loader_keys)
    assert callable(canonical_manifest_position_map_keys)
    assert callable(canonical_manifest_top_level_keys)


def test_api_reference_documents_every_public_module() -> None:
    api = Path("docs/api.rst").read_text(encoding="utf-8")
    for module in (
        "artifacts",
        "baselines",
        "data",
        "feature_expansion",
        "features",
        "final_ols",
        "metrics",
        "regularized_screening",
        "release_manifest",
        "transforms",
        "viz_io",
        "workflow",
    ):
        assert f".. automodule:: rfm_pipeline.{module}" in api


def test_docs_cover_first_run_fit_export_and_troubleshooting() -> None:
    quickstart = Path("docs/quickstart.md").read_text(encoding="utf-8")
    export_bundle = Path("docs/export_bundle.md").read_text(encoding="utf-8")
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "run_canonical_workflow" in quickstart
    assert "write_postfit_bundle" in quickstart
    assert "load_postfit_bundle" in quickstart
    assert "manifest.json" in export_bundle
    assert "CITATION.cff" in readme
    assert "CHANGELOG.md" in readme


def test_public_tree_has_no_study_output_directories() -> None:
    tracked = Path(".")
    forbidden_paths = (
        tracked / "figures",
        tracked / "notebooks",
        tracked / "paper",
    )
    assert not [path for path in forbidden_paths if path.exists()]
