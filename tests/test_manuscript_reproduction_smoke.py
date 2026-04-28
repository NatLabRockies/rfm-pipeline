"""Tests for the manuscript reproduction smoke-check entrypoint."""

from __future__ import annotations

import runpy
from pathlib import Path


def test_manuscript_reproduction_smoke_script_imports_without_installed_package() -> None:
    namespace = runpy.run_path(
        str(Path("tools/check_manuscript_reproduction.py")),
        run_name="not_main",
    )

    assert "check_manuscript_reproduction" in namespace


def test_manuscript_reproduction_smoke_script_declares_current_audit_checks() -> None:
    script = Path("tools/check_manuscript_reproduction.py").read_text(encoding="utf-8")

    required_snippets = [
        "EXPECTED_ARTIFACT_FAMILIES",
        "REQUIRED_METRIC_CHECKS",
        "final_ols_holdout_nrmse_positive",
        "null_mean_holdout_nrmse_finite_positive",
        "final_support_nonempty",
        'repo_root / "src"',
        "sys.path.insert(0, src_root_text)",
    ]
    forbidden_stale_check_names = [
        "null_mean_baseline_nrmse_positive",
        "final_stable_support_nonempty",
    ]

    for snippet in required_snippets:
        assert snippet in script
    for check_name in forbidden_stale_check_names:
        assert check_name not in script


def test_metric_check_validation_reports_missing_final_nrmse_without_traceback() -> None:
    namespace = runpy.run_path(
        str(Path("tools/check_manuscript_reproduction.py")),
        run_name="not_main",
    )

    failures = namespace["validate_metric_check_records"](
        {
            "all_artifacts_exist": {"status": "pass", "observed_value": "1"},
            "all_artifacts_nonempty": {"status": "pass", "observed_value": "1"},
            "final_ols_holdout_nrmse_ci_ordered": {
                "status": "pass",
                "observed_value": "1",
            },
            "null_mean_holdout_nrmse_finite_positive": {
                "status": "pass",
                "observed_value": "1",
            },
            "final_support_nonempty": {"status": "pass", "observed_value": "1"},
            "registered_svg_figures_nonempty": {"status": "pass", "observed_value": "1"},
            "workflow_summary_includes_final_ols": {"status": "pass", "observed_value": "1"},
        }
    )

    assert "final_ols_holdout_nrmse_positive" in failures[0]


def test_metric_check_validation_rejects_nonpositive_or_nonnumeric_final_nrmse() -> None:
    namespace = runpy.run_path(
        str(Path("tools/check_manuscript_reproduction.py")),
        run_name="not_main",
    )
    required = {
        check_name: {"status": "pass", "observed_value": "1"}
        for check_name in namespace["REQUIRED_METRIC_CHECKS"]
    }

    required["final_ols_holdout_nrmse_positive"] = {
        "status": "pass",
        "observed_value": "0",
    }
    zero_failures = namespace["validate_metric_check_records"](required)

    required["final_ols_holdout_nrmse_positive"] = {
        "status": "pass",
        "observed_value": "not-a-number",
    }
    nonnumeric_failures = namespace["validate_metric_check_records"](required)

    assert "not positive" in zero_failures[0]
    assert "not numeric" in nonnumeric_failures[0]


def test_gate_runs_manuscript_reproduction_smoke_before_notebooks() -> None:
    script = Path("test_repo.sh").read_text(encoding="utf-8")
    smoke_position = script.index("manuscript-reproduction-smoke")
    notebook_position = script.index("notebook-tests")

    assert smoke_position < notebook_position
