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
        "tree-based SHAP interaction values",
        "GAM EDF > 1 / p < 0.01",
        "de-biased-LASSO",
        "HC3 interval/drop rule is now implemented",
        "Not acceptable yet: **full exact reproduction of the manuscript workflow**",
        "manuscript_aligned_via_shap_gradient_boosting",
        "manuscript_aligned_via_scipy_smoothing_spline",
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
    # README should call out that at least one stage is still a public surrogate
    # and direct the reader to the alignment audit for the full status ledger.
    assert "Public surrogate" in readme
    assert "de-biased LASSO" in readme
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
    """Remaining externalized stages should name upstream source artifacts; implemented stages
    should record their manuscript alignment status."""
    audit = Path("docs/manuscript_alignment_audit.md").read_text(encoding="utf-8")
    required_phrases = [
        "## Externalized manuscript-only upstream artifacts",
        "empirical-null screening stage",
        "null_distribution.py",
        "Delta-null workflow",
        "de-biased-LASSO selection stage",
        "LASSO_to_OLS_v9.ipynb",
        "Source artifact availability in this public repo: no",
        "tree_shap_gradient_boosting",
        "gam_cubic_smoothing_spline",
        "Manuscript aligned",
    ]
    for phrase in required_phrases:
        assert phrase in audit


def test_alignment_audit_tracks_private_run_verification_evidence_requirements() -> None:
    """HC3/table/figure parity gaps should list required private-run evidence artifacts."""
    audit = Path("docs/manuscript_alignment_audit.md").read_text(encoding="utf-8")
    required_phrases = [
        "## Private-run verification evidence ledger",
        "HC3 retained-feature count parity",
        "HC3 dropped-feature count parity",
        "Final coefficient parity",
        "Table 1 value parity",
        "Table 2 value parity",
        "Figure 1 value/layout parity",
        "Figure 2 value/layout parity",
        "Expected private evidence artifact",
        "Verification status",
        "pending private-run evidence",
    ]
    for phrase in required_phrases:
        assert phrase in audit
