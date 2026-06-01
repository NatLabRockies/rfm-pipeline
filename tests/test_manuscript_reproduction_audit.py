"""Tests for manuscript reproduction QA audit artifacts."""

from __future__ import annotations

from pathlib import Path

from rfm_pipeline import (
    build_manuscript_notebook_context,
    run_manuscript_reproduction_audit_stage,
)


def test_run_manuscript_reproduction_audit_stage_writes_manifest_and_checks() -> None:
    context = build_manuscript_notebook_context(
        Path.cwd(),
        "08_manuscript_tables_and_figures.ipynb",
    )

    result = run_manuscript_reproduction_audit_stage(context)

    audit = result.audit
    assert audit.summary.loc[0, "stage"] == "manuscript_reproduction_audit"
    assert audit.summary.loc[0, "qa_status"] == "pass"
    assert audit.summary.loc[0, "n_missing_artifacts"] == 0
    assert audit.summary.loc[0, "n_empty_artifacts"] == 0
    assert audit.summary.loc[0, "n_failed_metric_checks"] == 0
    assert set(result.artifact_paths) == {
        "artifact_manifest",
        "metric_checks",
        "audit_summary",
    }
    assert all(path.exists() for path in result.artifact_paths.values())
    assert set(audit.artifact_manifest["stage"]) == {
        "output_conditioning",
        "empirical_null_screen",
        "interaction_discovery",
        "nonlinear_discovery",
        "sparse_selection",
        "final_manuscript_artifacts",
    }
    positive_check = audit.metric_checks.set_index("check_name").loc[
        "final_ols_holdout_nrmse_positive"
    ]
    assert positive_check["status"] == "pass"
    assert float(positive_check["observed_value"]) > 0.0
