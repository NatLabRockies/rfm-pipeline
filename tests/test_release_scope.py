"""Tests for release-facing metadata and workflow scope boundaries."""

from __future__ import annotations

from pathlib import Path

import tomllib

from rfm_pipeline.workflow import workflow_scope_boundary_table


def test_workflow_scope_boundary_table_freezes_current_non_foundation_stages() -> None:
    table = workflow_scope_boundary_table()
    assert table["name"].tolist() == ["upstream_null_screening", "feature_expansion"]
    assert table["status"].tolist() == [
        "implemented_adapter",
        "spec_recovered_not_fully_ported",
    ]


def test_scope_boundary_doc_matches_machine_readable_boundary() -> None:
    text = Path("docs/scope_boundary.md").read_text(encoding="utf-8")
    assert "workflow_scope_boundary_table" in text
    assert "upstream_null_screening" in text
    assert "feature_expansion" in text


def test_pyproject_includes_release_facing_repository_metadata() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]
    urls = project["urls"]
    assert urls["Homepage"].endswith("NatLabRockies/bsm-public-rf")
    assert urls["Repository"].endswith("NatLabRockies/bsm-public-rf")
    assert urls["Issues"].endswith("NatLabRockies/bsm-public-rf/issues")
    assert urls["Changelog"].endswith("NatLabRockies/bsm-public-rf/blob/main/CHANGELOG.md")
    assert "keywords" in project and "workflow" in project["keywords"]
    assert "classifiers" in project
    assert any(
        item.startswith("Programming Language :: Python :: 3") for item in project["classifiers"]
    )


def test_repo_includes_mit_license_file_and_metadata() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    license_text = (repo_root / "LICENSE").read_text(encoding="utf-8")
    project = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))["project"]

    assert license_text.startswith("MIT License")
    assert "Permission is hereby granted, free of charge" in license_text
    assert project["license"] == "MIT"
    assert project["license-files"] == ["LICENSE"]
    assert all("License :: OSI Approved ::" not in item for item in project["classifiers"])


def test_repo_includes_changelog_and_citation_metadata() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    changelog_text = (repo_root / "CHANGELOG.md").read_text(encoding="utf-8")
    citation_text = (repo_root / "CITATION.cff").read_text(encoding="utf-8")
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    version = pyproject["project"]["version"]

    assert "# Changelog" in changelog_text
    assert f"## {version}" in changelog_text
    assert "cff-version: 1.2.0" in citation_text
    assert 'title: "rfm-pipeline: Reduced-form modeling workflow package"' in citation_text
    assert f"version: {version}" in citation_text
    assert 'repository-code: "https://github.com/NatLabRockies/rfm-pipeline"' in citation_text
