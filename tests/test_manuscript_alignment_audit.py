"""Tests for the manuscript scientific-alignment ledger."""

from __future__ import annotations

from pathlib import Path


def test_docs_index_exposes_manuscript_alignment_audit() -> None:
    """The docs toctree should expose the scientific-alignment status ledger."""
    assert "manuscript_alignment_audit" in Path("docs/index.md").read_text(encoding="utf-8")


def test_alignment_audit_records_non_exact_stage_gaps() -> None:
    """The repo must not silently claim exactness for known approximation stages."""
    audit = Path("docs/manuscript_alignment_audit.md").read_text(encoding="utf-8")
    required_phrases = [
        "should **not** yet claim",
        "tree-based models with SHAP interaction values",
        "residualized product-term contribution",
        "GAM diagnostics",
        "residualized nonlinear contribution",
        "de-biased-LASSO",
        "HC3 interval/drop rule is now implemented",
        "Not acceptable yet: **full exact reproduction of the manuscript workflow**",
    ]
    for phrase in required_phrases:
        assert phrase in audit


def test_runtime_docs_link_alignment_audit_and_do_not_overclaim() -> None:
    """Runtime docs should direct readers to the alignment audit before exactness claims."""
    runtime = Path("docs/manuscript_runtime.md").read_text(encoding="utf-8")
    assert "docs/manuscript_alignment_audit.md" in runtime
    assert "executable does not mean\nmanuscript-exact" in runtime
    assert "final HC3 Wald inferential filter is now implemented locally" in runtime
    assert "still need verification against the private manuscript run" in runtime


def test_readme_distinguishes_audited_scaffold_from_exact_reproduction() -> None:
    """The public README should not imply exact manuscript reproduction is finished."""
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "should not yet be described as a full exact implementation" in readme
    assert "scientific exactness gaps" in readme
    assert "docs/manuscript_alignment_audit.md" in readme


def test_alignment_audit_maps_final_tables_and_figures_to_named_artifacts() -> None:
    """The alignment audit must include a concrete table/figure to artifact mapping."""
    audit = Path("docs/manuscript_alignment_audit.md").read_text(encoding="utf-8")
    required_phrases = [
        "## Manuscript table and figure artifact map",
        "Table 1",
        "Table 2",
        "Figure 1",
        "Figure 2",
        "final_manuscript_artifacts/tables/workflow_stage_summary.csv",
        "final_manuscript_artifacts/tables/model_performance.csv",
        "final_manuscript_artifacts/figures/figure_model_performance_data.csv",
        "final_manuscript_artifacts/figures/figure_support_composition_data.csv",
        "Status: mapping complete, manuscript-value verification pending.",
    ]
    for phrase in required_phrases:
        assert phrase in audit


def test_alignment_audit_externalizes_interaction_and_nonlinear_source_artifacts() -> None:
    """Interaction/nonlinear exactness blockers should name upstream source artifacts."""
    audit = Path("docs/manuscript_alignment_audit.md").read_text(encoding="utf-8")
    required_phrases = [
        "## Externalized manuscript-only upstream artifacts",
        "interaction-discovery stage",
        "nonlinear-discovery stage",
        "private_tree_shap_interaction_workflow",
        "private_gam_nonlinear_discovery_workflow",
        "Source artifact availability in this public repo: no",
    ]
    for phrase in required_phrases:
        assert phrase in audit
