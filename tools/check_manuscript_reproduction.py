"""Run the audited manuscript reproduction chain as a repository smoke check."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

EXPECTED_ARTIFACT_FAMILIES = {
    "empirical_null_screen",
    "final_manuscript_artifacts",
    "interaction_discovery",
    "nonlinear_discovery",
    "output_conditioning",
    "sparse_selection",
}

REQUIRED_METRIC_CHECKS = {
    "all_artifacts_exist",
    "all_artifacts_nonempty",
    "final_ols_holdout_nrmse_positive",
    "final_ols_holdout_nrmse_ci_ordered",
    "null_mean_holdout_nrmse_finite_positive",
    "final_support_nonempty",
    "registered_svg_figures_nonempty",
    "workflow_summary_includes_final_ols",
}


def _ensure_local_package_importable(repo_root: Path) -> None:
    """Make the src-layout package importable when the package is not installed."""
    src_root = repo_root / "src"
    src_root_text = str(src_root)
    if src_root_text not in sys.path:
        sys.path.insert(0, src_root_text)


def _make_temp_context(repo_root: Path, output_root: Path):
    """Build a manuscript notebook context rooted in a temporary output directory."""
    _ensure_local_package_importable(repo_root)
    from bsm_rfm import build_manuscript_notebook_context

    context = build_manuscript_notebook_context(
        repo_root,
        "08_manuscript_tables_and_figures.ipynb",
    )
    runtime = replace(context.runtime, output_root=output_root)
    return replace(context, runtime=runtime)


def check_manuscript_reproduction(repo_root: Path) -> list[str]:
    """Return audit failures for a temporary audited manuscript reproduction run."""
    _ensure_local_package_importable(repo_root)
    from bsm_rfm import run_manuscript_reproduction_audit_stage

    failures: list[str] = []
    with TemporaryDirectory(prefix="bsm-rfm-manuscript-reproduction-") as temp_dir:
        context = _make_temp_context(repo_root, Path(temp_dir))
        result = run_manuscript_reproduction_audit_stage(context)
        summary = result.audit.summary.iloc[0]
        metric_checks = result.audit.metric_checks.set_index("check_name")
        artifact_families = set(result.reproduction.artifact_paths)

        if summary["qa_status"] != "pass":
            failures.append(f"audit summary status is {summary['qa_status']!r}, expected 'pass'")
        if int(summary["n_missing_artifacts"]) != 0:
            failures.append(f"missing artifact count is {summary['n_missing_artifacts']}")
        if int(summary["n_empty_artifacts"]) != 0:
            failures.append(f"empty artifact count is {summary['n_empty_artifacts']}")
        if int(summary["n_failed_metric_checks"]) != 0:
            failures.append(f"failed metric-check count is {summary['n_failed_metric_checks']}")
        if artifact_families != EXPECTED_ARTIFACT_FAMILIES:
            missing = sorted(EXPECTED_ARTIFACT_FAMILIES - artifact_families)
            extra = sorted(artifact_families - EXPECTED_ARTIFACT_FAMILIES)
            failures.append(f"artifact families mismatch: missing={missing}, extra={extra}")

        missing_checks = REQUIRED_METRIC_CHECKS - set(metric_checks.index)
        if missing_checks:
            failures.append(f"missing metric checks: {sorted(missing_checks)}")
        for check_name in sorted(REQUIRED_METRIC_CHECKS & set(metric_checks.index)):
            status = metric_checks.loc[check_name, "status"]
            if status != "pass":
                failures.append(f"metric check {check_name!r} status is {status!r}")

        final_nrmse = metric_checks.loc[
            "final_ols_holdout_nrmse_positive",
            "observed_value",
        ]
        if float(final_nrmse) <= 0.0:
            failures.append(f"final holdout nRMSE is not positive: {final_nrmse}")

    return failures


def main() -> int:
    """Run the command-line manuscript reproduction smoke check."""
    repo_root = Path(__file__).resolve().parents[1]
    failures = check_manuscript_reproduction(repo_root)
    if failures:
        print("Manuscript reproduction smoke check failed:")
        for failure in failures:
            print(f" - {failure}")
        return 1
    print("Manuscript reproduction smoke check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
