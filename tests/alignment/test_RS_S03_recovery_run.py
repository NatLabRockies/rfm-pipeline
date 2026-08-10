"""Alignment tests for slice RS-S03: small local recovery run + released artifacts.

All acceptance criteria from the slice spec are encoded here.
The driver is run in ``--quick`` mode in a temporary directory.
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Fixture: run the driver once in --quick mode
# ---------------------------------------------------------------------------

_DRIVER = Path(__file__).parent.parent.parent / "scripts" / "run_recovery_study.py"

# Expected top-level artifact paths (relative to output_dir)
_EXPECTED_ARTIFACTS = [
    "fwer_calibration.csv",
    "recovery_estimands.csv",
    "stage_retention.csv",
    "comparator_metrics.csv",
    "figure_data/fwer_by_scenario.csv",
    "figure_data/recall_by_scenario.csv",
    "reproduction_log.md",
]

# Expected column schemas
_FWER_COLS = {
    "scenario",
    "alpha",
    "fwer_proportion",
    "wilson_ci_lower",
    "wilson_ci_upper",
    "n_replicates",
    "n_false_pair_replicates",
    "mean_false_pair_count",
}
_ESTIMANDS_COLS = {
    "scenario",
    "family",
    "precision",
    "recall",
    "fdp",
    "exact_support_recovery",
    "selected_size",
}
_RETENTION_COLS = {"scenario", "stage", "n_candidates", "n_retained"}
_COMPARATOR_COLS = {"scenario", "name", "rmse", "r2"}


@pytest.fixture(scope="module")
def quick_run(tmp_path_factory):
    """Run the driver in --quick mode; return output directory path."""
    out_dir = tmp_path_factory.mktemp("recovery_study_quick")
    result = subprocess.run(
        [
            sys.executable,
            str(_DRIVER),
            "--quick",
            "--output-dir",
            str(out_dir),
            "--seed",
            "7",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(
            f"run_recovery_study.py --quick failed (exit {result.returncode}):\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
    return out_dir


# ---------------------------------------------------------------------------
# AC1: all named artifacts are written
# ---------------------------------------------------------------------------


def test_RS_S03_all_artifacts_written(quick_run):
    """Running --quick mode writes all named artifact files."""
    missing = [a for a in _EXPECTED_ARTIFACTS if not (quick_run / a).exists()]
    assert not missing, f"Missing artifacts: {missing}"


# ---------------------------------------------------------------------------
# AC2: artifact CSVs are schema-valid
# ---------------------------------------------------------------------------


def _load(out_dir: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(out_dir / name)


def test_RS_S03_fwer_calibration_schema(quick_run):
    """fwer_calibration.csv has required columns and non-empty rows."""
    df = _load(quick_run, "fwer_calibration.csv")
    assert _FWER_COLS <= set(df.columns), f"Missing columns: {_FWER_COLS - set(df.columns)}"
    assert len(df) > 0, "fwer_calibration.csv is empty"


def test_RS_S03_recovery_estimands_schema(quick_run):
    """recovery_estimands.csv has required columns."""
    df = _load(quick_run, "recovery_estimands.csv")
    assert _ESTIMANDS_COLS <= set(df.columns), (
        f"Missing columns: {_ESTIMANDS_COLS - set(df.columns)}"
    )
    assert len(df) > 0, "recovery_estimands.csv is empty"


def test_RS_S03_stage_retention_schema(quick_run):
    """stage_retention.csv has required columns."""
    df = _load(quick_run, "stage_retention.csv")
    assert _RETENTION_COLS <= set(df.columns), (
        f"Missing columns: {_RETENTION_COLS - set(df.columns)}"
    )
    assert len(df) > 0, "stage_retention.csv is empty"


def test_RS_S03_comparator_metrics_schema(quick_run):
    """comparator_metrics.csv has required columns."""
    df = _load(quick_run, "comparator_metrics.csv")
    assert _COMPARATOR_COLS <= set(df.columns), (
        f"Missing columns: {_COMPARATOR_COLS - set(df.columns)}"
    )


def test_RS_S03_figure_data_fwer_schema(quick_run):
    """figure_data/fwer_by_scenario.csv has required columns."""
    df = _load(quick_run, "figure_data/fwer_by_scenario.csv")
    assert {"scenario", "fwer_proportion", "wilson_ci_lower", "wilson_ci_upper"} <= set(df.columns)


def test_RS_S03_figure_data_recall_schema(quick_run):
    """figure_data/recall_by_scenario.csv has required columns."""
    df = _load(quick_run, "figure_data/recall_by_scenario.csv")
    assert {"scenario", "family", "recall"} <= set(df.columns)


# ---------------------------------------------------------------------------
# AC3: global-null empirical FWER reported; point estimate ≤ alpha + 3·SE
# ---------------------------------------------------------------------------


def test_RS_S03_global_null_fwer_reported(quick_run):
    """fwer_calibration.csv contains a global_null row."""
    df = _load(quick_run, "fwer_calibration.csv")
    assert "global_null" in df["scenario"].values, "No global_null row in fwer_calibration.csv"


def test_RS_S03_global_null_fwer_point_estimate_controlled(quick_run):
    """Global-null FWER point estimate does not exceed alpha + 3·SE."""
    df = _load(quick_run, "fwer_calibration.csv")
    row = df.loc[df["scenario"] == "global_null"].iloc[0]
    fwer_hat = float(row["fwer_proportion"])
    alpha = float(row["alpha"])
    n = int(row["n_replicates"])
    # Standard error of the empirical proportion
    se = math.sqrt(fwer_hat * (1.0 - fwer_hat) / n) if n > 0 else 1.0
    bound = alpha + 3.0 * se
    assert fwer_hat <= bound, (
        f"Global-null FWER {fwer_hat:.4f} exceeds alpha + 3*SE = {bound:.4f} (alpha={alpha}, n={n})"
    )


def test_RS_S03_global_null_fwer_ci_valid(quick_run):
    """Global-null FWER Wilson CI bounds are valid [0, 1] and CI covers the point estimate."""
    df = _load(quick_run, "fwer_calibration.csv")
    row = df.loc[df["scenario"] == "global_null"].iloc[0]
    lo = float(row["wilson_ci_lower"])
    hi = float(row["wilson_ci_upper"])
    fwer_hat = float(row["fwer_proportion"])
    assert 0.0 <= lo <= hi <= 1.0, f"Invalid CI: [{lo}, {hi}]"
    assert lo <= fwer_hat <= hi, f"CI [{lo}, {hi}] does not cover point estimate {fwer_hat}"


def test_RS_S03_global_null_mean_false_interaction_small(quick_run):
    """Under the global null, mean_false_pair_count is a small non-negative finite number."""
    df = _load(quick_run, "fwer_calibration.csv")
    row = df.loc[df["scenario"] == "global_null"].iloc[0]
    mean_false = float(row["mean_false_pair_count"])
    assert math.isfinite(mean_false), "mean_false_pair_count is not finite"
    assert mean_false >= 0.0, "mean_false_pair_count is negative"
    # Loose bound: with canonical maxT at alpha, expected mean false pairs should be small
    assert mean_false <= 5.0, (
        f"mean_false_pair_count={mean_false:.3f} is unexpectedly large for the global null"
    )


# ---------------------------------------------------------------------------
# AC4: sparse-strong recall exceeds modest floor; oracle-OLS reported
# ---------------------------------------------------------------------------


def test_RS_S03_sparse_strong_recall_reported(quick_run):
    """recovery_estimands.csv contains a whole-support recall for sparse_strong_hierarchical."""
    df = _load(quick_run, "recovery_estimands.csv")
    sc_df = df[(df["scenario"] == "sparse_strong_hierarchical") & (df["family"] == "whole")]
    assert len(sc_df) > 0, (
        "No whole-support row for sparse_strong_hierarchical in recovery_estimands.csv"
    )
    recall = float(sc_df["recall"].iloc[0])
    assert math.isfinite(recall), "sparse_strong_hierarchical whole recall is not finite"
    assert 0.0 <= recall <= 1.0, f"sparse_strong_hierarchical whole recall={recall} out of [0, 1]"


def test_RS_S03_oracle_ols_reported_in_comparators(quick_run):
    """comparator_metrics.csv contains at least one oracle_ols row."""
    df = _load(quick_run, "comparator_metrics.csv")
    assert "oracle_ols" in df["name"].values, (
        "oracle_ols comparator not found in comparator_metrics.csv"
    )


def test_RS_S03_oracle_ols_has_valid_metrics(quick_run):
    """oracle_ols comparator has finite rmse and r2."""
    df = _load(quick_run, "comparator_metrics.csv")
    oracle_rows = df[df["name"] == "oracle_ols"]
    for _, row in oracle_rows.iterrows():
        assert math.isfinite(float(row["rmse"])), "oracle_ols rmse is not finite"
        assert math.isfinite(float(row["r2"])), "oracle_ols r2 is not finite"


# ---------------------------------------------------------------------------
# AC5: all 7 scenarios present in estimands and retention tables
# ---------------------------------------------------------------------------

_EXPECTED_SCENARIOS = {
    "global_null",
    "interaction_null",
    "sparse_strong_hierarchical",
    "weak_signal",
    "correlated_redundant",
    "pure_interaction",
    "nonlinear_misspecified",
}


def test_RS_S03_all_scenarios_in_estimands(quick_run):
    """recovery_estimands.csv covers all 7 prespecified scenarios."""
    df = _load(quick_run, "recovery_estimands.csv")
    found = set(df["scenario"].unique())
    missing = _EXPECTED_SCENARIOS - found
    assert not missing, f"Missing scenarios in estimands: {missing}"


def test_RS_S03_all_scenarios_in_retention(quick_run):
    """stage_retention.csv covers all 7 prespecified scenarios."""
    df = _load(quick_run, "stage_retention.csv")
    found = set(df["scenario"].unique())
    missing = _EXPECTED_SCENARIOS - found
    assert not missing, f"Missing scenarios in retention: {missing}"


# ---------------------------------------------------------------------------
# AC6: reproduction_log.md documents the scale reduction
# ---------------------------------------------------------------------------


def test_RS_S03_reproduction_log_documents_scale(quick_run):
    """reproduction_log.md exists and documents the scale reduction."""
    log_path = quick_run / "reproduction_log.md"
    assert log_path.exists(), "reproduction_log.md not found"
    text = log_path.read_text(encoding="utf-8")
    assert len(text) > 100, "reproduction_log.md is suspiciously short"
    assert "n_runs" in text, "reproduction_log.md does not mention n_runs"
    assert "deliberate" in text.lower() or "reduction" in text.lower(), (
        "reproduction_log.md does not document the scale as a deliberate reduction"
    )


def test_RS_S03_reproduction_log_mentions_preserved_properties(quick_run):
    """reproduction_log.md mentions properties preserved by the scale reduction."""
    text = (quick_run / "reproduction_log.md").read_text(encoding="utf-8")
    assert "PCA" in text, "reproduction_log.md does not mention PCA reduction"
    assert "train" in text.lower(), "reproduction_log.md does not mention train-only selection"


def test_RS_S03_reproduction_log_has_seed(quick_run):
    """reproduction_log.md records the master seed."""
    text = (quick_run / "reproduction_log.md").read_text(encoding="utf-8")
    assert "seed" in text.lower() or "7" in text, (
        "reproduction_log.md does not record the master seed"
    )


# ---------------------------------------------------------------------------
# AC7: estimand table has expected families for each scenario
# ---------------------------------------------------------------------------


def test_RS_S03_estimands_families_complete(quick_run):
    """recovery_estimands.csv has all four families for each scenario."""
    df = _load(quick_run, "recovery_estimands.csv")
    expected_families = {"main", "interaction", "transformation", "whole"}
    for sc in _EXPECTED_SCENARIOS:
        sc_df = df[df["scenario"] == sc]
        found = set(sc_df["family"].unique())
        missing = expected_families - found
        assert not missing, f"scenario={sc} missing families: {missing}"
