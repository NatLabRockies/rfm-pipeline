"""Tests for stratified and excluded-output error reporting.

Tests:
- Stratified breakdown correctness: per-stratum nRMSE matches manual calculation.
- Multiple strata keys produce independent marginal breakdowns.
- Tail quantiles (p95, p99) are present and ordered correctly.
- Worst-output is identified correctly.
- Excluded-output summary covers all excluded outputs with absolute/domain-scaled RMSE.
- Every headline-metric result carries an explicit metric_population label.
"""

from __future__ import annotations

import numpy as np
import pytest

from rfm_pipeline.metrics import (
    excluded_output_error_summary,
    macro_nrmse_with_ref,
    stratified_nrmse_summary,
)

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

SEED = 7


@pytest.fixture()
def synthetic_strata_data():
    """Outputs with row strata (scenario, year) and an included/excluded mask.

    - 300 rows, 6 outputs.
    - scenario: "ctrl" (rows 0..149), "treat" (rows 150..299).
    - year: "2020" (rows 0..99, 200..249), "2021" (rows 100..199, 250..299).
    - Outputs 0-3: ref range >> min_range → included.
    - Outputs 4-5: ref range ~ 0 → excluded from macro nRMSE.
    """
    rng = np.random.default_rng(SEED)
    n_rows, n_out = 300, 6

    Y_ref = rng.uniform(0, 10, size=(n_rows, n_out))
    # Force outputs 4 and 5 to have near-zero range so they are "excluded".
    Y_ref[:, 4] = 0.0
    Y_ref[:, 5] = 0.0

    Y_true = Y_ref + rng.normal(0, 0.1, size=(n_rows, n_out))
    Y_pred = Y_true + rng.normal(0, 0.5, size=(n_rows, n_out))

    scenario = np.array(["ctrl"] * 150 + ["treat"] * 150)
    year_vals = (
        ["2020"] * 100
        + ["2021"] * 100  # rows 0..199
        + ["2020"] * 50
        + ["2021"] * 50  # rows 200..299
    )
    year = np.array(year_vals)

    output_names = [f"out_{i}" for i in range(n_out)]
    # excluded_mask: True for outputs 4 and 5
    excluded_mask = np.array([False, False, False, False, True, True])

    return {
        "Y_true": Y_true,
        "Y_pred": Y_pred,
        "Y_ref": Y_ref,
        "scenario": scenario,
        "year": year,
        "output_names": output_names,
        "excluded_mask": excluded_mask,
        "n_rows": n_rows,
        "n_out": n_out,
    }


# ---------------------------------------------------------------------------
# Metric population labels are present in overall and stratum results.
# ---------------------------------------------------------------------------


def test_population_label_present(synthetic_strata_data):
    """Every result dict must contain a non-empty metric_population label."""
    d = synthetic_strata_data
    result = stratified_nrmse_summary(
        d["Y_true"],
        d["Y_pred"],
        d["Y_ref"],
        strata={"scenario": d["scenario"], "year": d["year"]},
        output_names=d["output_names"],
    )
    assert "metric_population" in result
    assert isinstance(result["metric_population"], str)
    assert len(result["metric_population"]) > 0

    # Each stratum-level entry also carries metric_population.
    for key, entries in result["by_stratum"].items():
        for entry in entries:
            assert "metric_population" in entry, (
                f"metric_population missing in by_stratum['{key}'] entry {entry['value']}"
            )
            assert len(entry["metric_population"]) > 0


def test_custom_population_label(synthetic_strata_data):
    """Caller-supplied metric_population is forwarded to result."""
    d = synthetic_strata_data
    label = "conditioned_outputs_eval_set"
    result = stratified_nrmse_summary(
        d["Y_true"],
        d["Y_pred"],
        d["Y_ref"],
        strata={"scenario": d["scenario"]},
        output_names=d["output_names"],
        metric_population=label,
    )
    assert result["metric_population"] == label


def test_excluded_output_summary_has_population_label(synthetic_strata_data):
    """excluded_output_error_summary must carry metric_population='excluded_outputs'."""
    d = synthetic_strata_data
    result = excluded_output_error_summary(
        d["Y_true"],
        d["Y_pred"],
        d["excluded_mask"],
        d["output_names"],
    )
    assert result["metric_population"] == "excluded_outputs"


# ---------------------------------------------------------------------------
# Stratified breakdown correctness.
# ---------------------------------------------------------------------------


def test_stratified_breakdown_correctness(synthetic_strata_data):
    """Per-stratum macro_nrmse must match manual macro_nrmse_with_ref."""
    d = synthetic_strata_data
    result = stratified_nrmse_summary(
        d["Y_true"],
        d["Y_pred"],
        d["Y_ref"],
        strata={"scenario": d["scenario"]},
        output_names=d["output_names"],
    )

    strata_entries = {e["value"]: e for e in result["by_stratum"]["scenario"]}
    assert set(strata_entries.keys()) == {"ctrl", "treat"}

    for val, mask_val in [("ctrl", "ctrl"), ("treat", "treat")]:
        mask = d["scenario"] == mask_val
        expected_nrmse, _, _ = macro_nrmse_with_ref(
            d["Y_true"][mask], d["Y_pred"][mask], d["Y_ref"]
        )
        assert strata_entries[val]["macro_nrmse"] == pytest.approx(expected_nrmse, rel=1e-9)


def test_stratified_n_rows_sum(synthetic_strata_data):
    """Sum of per-stratum n_rows must equal total rows."""
    d = synthetic_strata_data
    result = stratified_nrmse_summary(
        d["Y_true"],
        d["Y_pred"],
        d["Y_ref"],
        strata={"scenario": d["scenario"], "year": d["year"]},
        output_names=d["output_names"],
    )
    for key, entries in result["by_stratum"].items():
        total = sum(e["n_rows"] for e in entries)
        assert total == d["n_rows"], f"n_rows sum for '{key}' is {total}, expected {d['n_rows']}"


def test_multiple_strata_keys_independent(synthetic_strata_data):
    """Multiple strata keys produce separate independent marginal breakdowns."""
    d = synthetic_strata_data
    result = stratified_nrmse_summary(
        d["Y_true"],
        d["Y_pred"],
        d["Y_ref"],
        strata={"scenario": d["scenario"], "year": d["year"]},
        output_names=d["output_names"],
    )
    assert "scenario" in result["by_stratum"]
    assert "year" in result["by_stratum"]
    assert len(result["by_stratum"]["scenario"]) == 2  # ctrl, treat
    assert len(result["by_stratum"]["year"]) == 2  # 2020, 2021

    # Verify year breakdown manually.
    year_entries = {e["value"]: e for e in result["by_stratum"]["year"]}
    for yr in ("2020", "2021"):
        mask = d["year"] == yr
        expected, _, _ = macro_nrmse_with_ref(d["Y_true"][mask], d["Y_pred"][mask], d["Y_ref"])
        assert year_entries[yr]["macro_nrmse"] == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# Tail quantiles and worst-output reporting.
# ---------------------------------------------------------------------------


def test_tail_quantiles_present_and_ordered(synthetic_strata_data):
    """tail_nrmse contains requested quantiles and p95 ≤ p99."""
    d = synthetic_strata_data
    result = stratified_nrmse_summary(
        d["Y_true"],
        d["Y_pred"],
        d["Y_ref"],
        strata={"scenario": d["scenario"]},
        output_names=d["output_names"],
        tail_quantiles=(0.95, 0.99),
    )
    tails = result["tail_nrmse"]
    assert 0.95 in tails
    assert 0.99 in tails
    assert np.isfinite(tails[0.95])
    assert np.isfinite(tails[0.99])
    assert tails[0.95] <= tails[0.99]


def test_worst_output_is_maximum_nrmse(synthetic_strata_data):
    """worst_output must be the output with the highest per-output nRMSE."""
    d = synthetic_strata_data
    result = stratified_nrmse_summary(
        d["Y_true"],
        d["Y_pred"],
        d["Y_ref"],
        strata={},  # no strata breakdown needed for this check
        output_names=d["output_names"],
    )
    worst = result["worst_output"]
    assert worst["name"] in d["output_names"]
    assert np.isfinite(worst["nrmse"])

    # Independently compute per-output nRMSE and check worst matches.
    rmse = np.sqrt(np.mean((d["Y_true"] - d["Y_pred"]) ** 2, axis=0))
    ref_range = np.nanmax(d["Y_ref"], axis=0) - np.nanmin(d["Y_ref"], axis=0)
    included = ref_range >= 1e-6
    nrmse = np.where(included, rmse / np.where(included, ref_range, 1.0), np.nan)
    manual_worst_idx = int(np.nanargmax(nrmse))
    assert worst["name"] == d["output_names"][manual_worst_idx]
    assert worst["nrmse"] == pytest.approx(float(nrmse[manual_worst_idx]), rel=1e-9)


def test_median_nrmse_between_min_and_max(synthetic_strata_data):
    """median_nrmse must lie between min and max per-output nRMSE."""
    d = synthetic_strata_data
    result = stratified_nrmse_summary(
        d["Y_true"],
        d["Y_pred"],
        d["Y_ref"],
        strata={},
        output_names=d["output_names"],
    )
    assert np.isfinite(result["median_nrmse"])
    assert result["median_nrmse"] <= result["worst_output"]["nrmse"]
    assert result["median_nrmse"] >= 0.0


# ---------------------------------------------------------------------------
# Excluded-output summary correctness.
# ---------------------------------------------------------------------------


def test_excluded_output_summary_covers_all_excluded(synthetic_strata_data):
    """excluded_output_error_summary reports every excluded output."""
    d = synthetic_strata_data
    result = excluded_output_error_summary(
        d["Y_true"],
        d["Y_pred"],
        d["excluded_mask"],
        d["output_names"],
    )
    assert result["n_excluded"] == int(np.sum(d["excluded_mask"]))
    assert result["n_total"] == d["n_out"]

    # All excluded outputs appear in the outputs list.
    excluded_names = {d["output_names"][i] for i, ex in enumerate(d["excluded_mask"]) if ex}
    reported_excluded = {e["name"] for e in result["outputs"] if e["excluded"]}
    assert reported_excluded == excluded_names


def test_excluded_output_rmse_correct(synthetic_strata_data):
    """RMSE for each excluded output matches manual calculation."""
    d = synthetic_strata_data
    result = excluded_output_error_summary(
        d["Y_true"],
        d["Y_pred"],
        d["excluded_mask"],
        d["output_names"],
    )
    rmse_expected = np.sqrt(np.mean((d["Y_true"] - d["Y_pred"]) ** 2, axis=0))

    for entry in result["outputs"]:
        idx = d["output_names"].index(entry["name"])
        assert entry["rmse"] == pytest.approx(float(rmse_expected[idx]), rel=1e-9)


def test_excluded_domain_scaled_rmse(synthetic_strata_data):
    """domain_scaled_rmse is computed for excluded outputs when domain_scales provided."""
    d = synthetic_strata_data
    rng = np.random.default_rng(SEED + 1)
    domain_scales = rng.uniform(1, 10, size=d["n_out"])
    result = excluded_output_error_summary(
        d["Y_true"],
        d["Y_pred"],
        d["excluded_mask"],
        d["output_names"],
        domain_scales=domain_scales,
    )
    rmse_expected = np.sqrt(np.mean((d["Y_true"] - d["Y_pred"]) ** 2, axis=0))

    for entry in result["outputs"]:
        idx = d["output_names"].index(entry["name"])
        if entry["excluded"]:
            expected_scaled = float(rmse_expected[idx]) / float(domain_scales[idx])
            assert entry["domain_scaled_rmse"] == pytest.approx(expected_scaled, rel=1e-9)
        else:
            # domain_scaled_rmse is only filled for excluded outputs.
            assert entry["domain_scaled_rmse"] is None


def test_excluded_output_summary_median_mean_finite(synthetic_strata_data):
    """median_rmse and mean_rmse are finite when there are excluded outputs."""
    d = synthetic_strata_data
    result = excluded_output_error_summary(
        d["Y_true"],
        d["Y_pred"],
        d["excluded_mask"],
        d["output_names"],
    )
    assert result["n_excluded"] > 0
    assert np.isfinite(result["median_rmse"])
    assert np.isfinite(result["mean_rmse"])


def test_excluded_output_summary_no_excluded():
    """When no outputs are excluded, n_excluded=0 and median/mean are nan."""
    rng = np.random.default_rng(0)
    Y = rng.standard_normal((20, 3))
    result = excluded_output_error_summary(
        Y, Y + 0.1, np.array([False, False, False]), ["a", "b", "c"]
    )
    assert result["n_excluded"] == 0
    assert np.isnan(result["median_rmse"])
    assert np.isnan(result["mean_rmse"])


# ---------------------------------------------------------------------------
# Used and total output counts in the overall summary.
# ---------------------------------------------------------------------------


def test_n_outputs_used_correct(synthetic_strata_data):
    """n_outputs_used reflects only outputs with sufficient ref range."""
    d = synthetic_strata_data
    result = stratified_nrmse_summary(
        d["Y_true"],
        d["Y_pred"],
        d["Y_ref"],
        strata={},
        output_names=d["output_names"],
    )
    # Outputs 4 and 5 have ref range=0, so only 4 out of 6 should be used.
    assert result["n_outputs_total"] == d["n_out"]
    assert result["n_outputs_used"] == d["n_out"] - int(np.sum(d["excluded_mask"]))


# ---------------------------------------------------------------------------
# Error conditions.
# ---------------------------------------------------------------------------


def test_strata_length_mismatch():
    """ValueError when strata array length does not match n_rows."""
    rng = np.random.default_rng(0)
    Y = rng.standard_normal((10, 3))
    with pytest.raises(ValueError, match="n_rows"):
        stratified_nrmse_summary(
            Y,
            Y,
            Y,
            strata={"s": np.array(["A"] * 5)},  # wrong length
            output_names=["a", "b", "c"],
        )


def test_excluded_mask_length_mismatch():
    """ValueError when excluded_mask length does not match n_outputs."""
    rng = np.random.default_rng(0)
    Y = rng.standard_normal((10, 3))
    with pytest.raises(ValueError, match="n_outputs"):
        excluded_output_error_summary(
            Y,
            Y,
            np.array([True, False]),  # wrong length
            ["a", "b", "c"],
        )
