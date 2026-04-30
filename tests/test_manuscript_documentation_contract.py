"""Tests for public manuscript-reproduction documentation state."""

from __future__ import annotations

from pathlib import Path

PUBLIC_MANUSCRIPT_DOCS = [
    Path("README.md"),
    Path("docs/manuscript_runtime.md"),
    Path("notebooks/manuscript/README.md"),
]


def test_public_manuscript_docs_no_longer_describe_notebooks_as_skeletons() -> None:
    """Phase 3 docs should describe implemented entrypoints, not stale skeletons."""
    for path in PUBLIC_MANUSCRIPT_DOCS:
        text = path.read_text(encoding="utf-8").lower()
        assert "notebook skeleton" not in text


def test_public_manuscript_docs_describe_source_backed_reproduction_chain() -> None:
    """Public docs should expose the implemented source-backed manuscript workflow."""
    runtime_doc = Path("docs/manuscript_runtime.md").read_text(encoding="utf-8")
    notebook_readme = Path("notebooks/manuscript/README.md").read_text(encoding="utf-8")

    required_runtime_snippets = [
        "# Manuscript runtime and source-backed notebooks",
        "## Notebook entrypoints",
        "run_manuscript_reproduction_audit_stage",
        "manuscript-reproduction-smoke",
    ]
    for snippet in required_runtime_snippets:
        assert snippet in runtime_doc

    assert "source-backed stage functions" in notebook_readme
    assert "deterministic handoff artifacts" in notebook_readme


def test_module_plan_records_manuscript_runtime_and_stage_modules() -> None:
    """The live module map should stay synchronized with manuscript-stage implementation."""
    module_plan = Path("docs/module_plan.md").read_text(encoding="utf-8")
    required_snippets = [
        "`bsm_rfm.manuscript_data_contract`",
        "`bsm_rfm.manuscript_runtime`",
        "`bsm_rfm.manuscript_stages`",
        "run_manuscript_reproduction_audit_stage",
        "docs/manuscript_alignment_audit.md",
        "source-backed executable scaffold",
    ]

    for snippet in required_snippets:
        assert snippet in module_plan
