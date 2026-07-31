"""Tests for PA-D: per-output nRMSE eligibility ledger."""

from __future__ import annotations

import numpy as np
import pytest

from rfm_pipeline import (
    build_output_eligibility_ledger,
    macro_nrmse_from_ledger,
    macro_nrmse_with_ref,
)


@pytest.fixture()
def synthetic_data():
    """Small synthetic fixture with a mix of finite-range and zero-range outputs."""
    rng = np.random.default_rng(42)
    n_rows, n_outputs = 50, 8

    Y_ref = rng.normal(size=(n_rows, n_outputs))
    # Make outputs 0 and 3 zero-range (constant) → ineligible
    Y_ref[:, 0] = 5.0
    Y_ref[:, 3] = -2.0

    Y_true = rng.normal(size=(n_rows, n_outputs))
    Y_pred = Y_true + rng.normal(scale=0.1, size=(n_rows, n_outputs))

    return Y_true, Y_pred, Y_ref


def test_ledger_row_count(synthetic_data):
    _, _, Y_ref = synthetic_data
    ledger = build_output_eligibility_ledger(Y_ref)
    assert len(ledger) == Y_ref.shape[1]


def test_ledger_eligible_count_matches_k_used(synthetic_data):
    Y_true, Y_pred, Y_ref = synthetic_data
    ledger = build_output_eligibility_ledger(Y_ref)
    _, info, _ = macro_nrmse_with_ref(Y_true, Y_pred, Y_ref)
    assert ledger["eligible"].sum() == info["k_used"]


def test_ineligible_rows_have_exclusion_reason(synthetic_data):
    _, _, Y_ref = synthetic_data
    ledger = build_output_eligibility_ledger(Y_ref)
    ineligible = ledger[~ledger["eligible"]]
    assert len(ineligible) > 0, "fixture must contain at least one ineligible output"
    assert (ineligible["exclusion_reason"].str.len() > 0).all()


def test_eligible_rows_have_empty_exclusion_reason(synthetic_data):
    _, _, Y_ref = synthetic_data
    ledger = build_output_eligibility_ledger(Y_ref)
    eligible = ledger[ledger["eligible"]]
    assert len(eligible) > 0, "fixture must contain at least one eligible output"
    assert (eligible["exclusion_reason"] == "").all()


def test_macro_nrmse_from_ledger_equals_macro_nrmse_with_ref(synthetic_data):
    Y_true, Y_pred, Y_ref = synthetic_data
    ledger = build_output_eligibility_ledger(Y_ref)
    val_ledger = macro_nrmse_from_ledger(Y_true, Y_pred, ledger)
    val_ref, _, _ = macro_nrmse_with_ref(Y_true, Y_pred, Y_ref)
    assert val_ledger == pytest.approx(val_ref, rel=1e-9)


def test_no_missing_eligible_outputs(synthetic_data):
    Y_true, Y_pred, Y_ref = synthetic_data
    ledger = build_output_eligibility_ledger(Y_ref)
    # Re-compute manually: mean(rmse[eligible] / ref_range[eligible])
    Y_true_arr = np.asarray(Y_true)
    Y_pred_arr = np.asarray(Y_pred)
    rmse = np.sqrt(np.mean((Y_true_arr - Y_pred_arr) ** 2, axis=0))
    eligible_mask = ledger["eligible"].to_numpy(bool)
    ref_range = ledger["ref_range"].to_numpy(float)
    expected = float(np.mean(rmse[eligible_mask] / ref_range[eligible_mask]))
    assert macro_nrmse_from_ledger(Y_true, Y_pred, ledger) == pytest.approx(expected)


def test_ineligible_outputs_do_not_contribute(synthetic_data):
    Y_true, Y_pred, Y_ref = synthetic_data
    ledger = build_output_eligibility_ledger(Y_ref)
    ineligible_ids = ledger.index[~ledger["eligible"]].tolist()
    assert len(ineligible_ids) > 0
    # Perturbing Y_pred on ineligible columns should not change the macro metric
    Y_pred_arr = np.asarray(Y_pred, dtype=np.float64).copy()
    ineligible_col_ids = ledger["output_id"][~ledger["eligible"]].tolist()
    Y_pred_arr[:, ineligible_col_ids] += 1000.0
    val_original = macro_nrmse_from_ledger(Y_true, Y_pred, ledger)
    val_perturbed = macro_nrmse_from_ledger(Y_true, Y_pred_arr, ledger)
    assert val_original == pytest.approx(val_perturbed, rel=1e-9)


def test_counts_not_hardcoded(synthetic_data):
    """Ensure eligible count derives from data, not a fixed number."""
    _, _, Y_ref = synthetic_data
    ledger = build_output_eligibility_ledger(Y_ref)
    n_eligible = int(ledger["eligible"].sum())
    n_ineligible = int((~ledger["eligible"]).sum())
    assert n_eligible + n_ineligible == Y_ref.shape[1]
    # Two constant columns were set → exactly 2 ineligible (range_below_threshold)
    assert n_ineligible == 2


def test_custom_output_ids(synthetic_data):
    _, _, Y_ref = synthetic_data
    ids = [f"out_{i}" for i in range(Y_ref.shape[1])]
    ledger = build_output_eligibility_ledger(Y_ref, output_ids=ids)
    assert list(ledger["output_id"]) == ids


def test_ledger_required_columns(synthetic_data):
    _, _, Y_ref = synthetic_data
    ledger = build_output_eligibility_ledger(Y_ref)
    required = {
        "output_id",
        "ref_min",
        "ref_max",
        "ref_range",
        "ref_variance",
        "range_pass",
        "variance_pass",
        "eligible",
        "exclusion_reason",
        "min_range_threshold",
    }
    assert required.issubset(set(ledger.columns))


def test_min_range_threshold_provenance(synthetic_data):
    _, _, Y_ref = synthetic_data
    threshold = 0.01
    ledger = build_output_eligibility_ledger(Y_ref, min_range=threshold)
    assert (ledger["min_range_threshold"] == threshold).all()
