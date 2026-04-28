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


def test_gate_runs_manuscript_reproduction_smoke_before_notebooks() -> None:
    script = Path("test_repo.sh").read_text(encoding="utf-8")
    smoke_position = script.index("manuscript-reproduction-smoke")
    notebook_position = script.index("notebook-tests")

    assert smoke_position < notebook_position
