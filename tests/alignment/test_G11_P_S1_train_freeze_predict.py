"""G11-P-S1: train-freeze-predict and bootstrap paths.

Tests that ``train_fit_and_freeze``, ``holdout_predict``,
``bootstrap_metric_block``, ``reduce_bootstrap_metric_blocks``, and
``build_g11_output_eligibility_ledger`` implement the required stage
separation:

- ``train_fit_and_freeze`` accepts only training bytes (no holdout data).
- ``holdout_predict`` requires a valid, un-tampered freeze manifest; any
  hash mismatch raises ``ModelFreezeAuthorizationError``.
- ``bootstrap_metric_block`` accepts only frozen prediction matrices and
  never refits.
- ``reduce_bootstrap_metric_blocks`` rejects gaps, overlaps, and
  mismatched freeze hashes across shards.
- ``build_g11_output_eligibility_ledger`` uses the exact G11 two-part
  predicate (variance > 1e-12 AND relative_range >= 1e-2) and does NOT
  use ``range >= 1e-6`` as a sufficient condition.

All tests use a minimal synthetic fixture (n_rows=30, n_features=4,
n_outputs=8) to keep runtime under one second.

Covered behaviors
-----------------
1.  ``train_fit_and_freeze`` returns ``TrainFitAndFreezeResult`` with a
    valid ``ModelFreezeManifest``.
2.  Freeze manifest ``freeze_hash`` is consistent with the manifest fields
    (round-trip verification passes).
3.  ``holdout_predict`` succeeds when given the un-tampered result.
4.  ``holdout_predict`` raises ``ModelFreezeAuthorizationError`` on a
    tampered freeze manifest.
5.  ``holdout_predict`` returns ``FrozenPredictionMatrices`` whose
    ``freeze_hash`` matches the manifest.
6.  ``bootstrap_metric_block`` returns a ``BootstrapMetricShard`` with
    draws in ``[draw_start, draw_end)``.
7.  Single full-block ``[0, B)`` reduces to the same per-draw nRMSEs as
    two partitioned blocks ``[0, B//2)`` + ``[B//2, B)`` after reduction.
8.  Shuffled shard order reduces to the same result.
9.  ``reduce_bootstrap_metric_blocks`` rejects a gap (missing draws).
10. ``reduce_bootstrap_metric_blocks`` rejects overlapping draw ranges.
11. ``reduce_bootstrap_metric_blocks`` rejects shards with mismatched
    ``freeze_hash``.
12. ``reduce_bootstrap_metric_blocks`` rejects wrong total draw count.
13. G11 eligibility ledger: output with small variance (< 1e-12) is
    ineligible.
14. G11 eligibility ledger: output with large absolute mean and small
    relative range (range / (|mean| + 1e-12) < 1e-2) is ineligible, even
    when absolute range >= 1e-6 (legacy threshold must not be the gate).
15. G11 eligibility ledger: output passing both criteria is eligible.
16. Eligibility ledger column ``relative_range`` equals
    ``range / (abs(train_mean) + 1e-12)`` to numerical precision.
"""

from __future__ import annotations

import dataclasses
import inspect

import numpy as np
import pandas as pd
import pytest

from rfm_pipeline.manuscript_stages import (
    BootstrapMetricShard,
    FrozenPredictionMatrices,
    ModelFreezeAuthorizationError,
    ModelFreezeManifest,
    TrainFitAndFreezeResult,
    bootstrap_metric_block,
    build_g11_output_eligibility_ledger,
    holdout_predict,
    reduce_bootstrap_metric_blocks,
    train_fit_and_freeze,
    validate_g11_production_eligibility_ledger,
)

# ---------------------------------------------------------------------------
# Synthetic fixture
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(42)
_N_TRAIN = 30
_N_HOLDOUT = 10
_N_FEATURES = 4
_N_OUTPUTS = 8
_CONTRACT_HASH = "aabbccdd" * 8  # 64-char placeholder

_X_TRAIN = _RNG.standard_normal((_N_TRAIN, _N_FEATURES)).astype(np.float64)
_Y_TRAIN = _RNG.standard_normal((_N_TRAIN, _N_OUTPUTS)).astype(np.float64)
_X_HOLDOUT = _RNG.standard_normal((_N_HOLDOUT, _N_FEATURES)).astype(np.float64)
_Y_HOLDOUT = _RNG.standard_normal((_N_HOLDOUT, _N_OUTPUTS)).astype(np.float64)
_FEATURE_NAMES = [f"f{i}" for i in range(_N_FEATURES)]
_OUTPUT_NAMES = [f"out{i}" for i in range(_N_OUTPUTS)]
_HOLDOUT_IDS = tuple(f"holdout-{i}" for i in range(_N_HOLDOUT))


@pytest.fixture(scope="module")
def freeze_result() -> TrainFitAndFreezeResult:
    """Train-only fit, produced once for the whole module."""
    return train_fit_and_freeze(
        _X_TRAIN,
        _Y_TRAIN,
        feature_names=_FEATURE_NAMES,
        output_names=_OUTPUT_NAMES,
        contract_hash=_CONTRACT_HASH,
    )


@pytest.fixture(scope="module")
def frozen_matrices(freeze_result: TrainFitAndFreezeResult) -> FrozenPredictionMatrices:
    """Frozen prediction matrices produced from the training-only result."""
    return holdout_predict(
        freeze_result,
        _X_HOLDOUT,
        _Y_HOLDOUT,
        holdout_ids=_HOLDOUT_IDS,
        strata=np.array(["a"] * 5 + ["b"] * 5),
    )


# ---------------------------------------------------------------------------
# 1. train_fit_and_freeze returns correct types
# ---------------------------------------------------------------------------


def test_train_fit_and_freeze_returns_result(freeze_result: TrainFitAndFreezeResult) -> None:
    assert isinstance(freeze_result, TrainFitAndFreezeResult)
    assert isinstance(freeze_result.freeze_manifest, ModelFreezeManifest)


# ---------------------------------------------------------------------------
# 2. Freeze hash is consistent (round-trip verification)
# ---------------------------------------------------------------------------


def test_freeze_manifest_hash_is_consistent(freeze_result: TrainFitAndFreezeResult) -> None:
    # Round-trip: verify_freeze_hash must not raise on an un-tampered result
    freeze_result.verify_freeze_hash()  # must not raise


# ---------------------------------------------------------------------------
# 3 & 4. holdout_predict: succeeds on valid manifest, raises on tampered
# ---------------------------------------------------------------------------


def test_holdout_predict_succeeds_on_valid_result(
    frozen_matrices: FrozenPredictionMatrices,
) -> None:
    assert isinstance(frozen_matrices, FrozenPredictionMatrices)


def test_holdout_predict_raises_on_tampered_hash(
    freeze_result: TrainFitAndFreezeResult,
) -> None:
    tampered = dataclasses.replace(
        freeze_result,
        freeze_manifest=dataclasses.replace(
            freeze_result.freeze_manifest,
            freeze_hash="0" * 64,
        ),
    )
    with pytest.raises(ModelFreezeAuthorizationError):
        holdout_predict(
            tampered,
            _X_HOLDOUT,
            _Y_HOLDOUT,
            holdout_ids=_HOLDOUT_IDS,
            strata=np.array(["a"] * 5 + ["b"] * 5),
        )


def test_holdout_predict_rejects_a_model_changed_after_freeze(
    freeze_result: TrainFitAndFreezeResult,
) -> None:
    tampered = dataclasses.replace(freeze_result, coef=np.zeros_like(freeze_result.coef))
    with pytest.raises(ModelFreezeAuthorizationError, match="model"):
        holdout_predict(
            tampered,
            _X_HOLDOUT,
            _Y_HOLDOUT,
            holdout_ids=_HOLDOUT_IDS,
            strata=np.array(["a"] * 5 + ["b"] * 5),
        )


# ---------------------------------------------------------------------------
# 5. FrozenPredictionMatrices.freeze_hash matches manifest freeze_hash
# ---------------------------------------------------------------------------


def test_frozen_matrices_freeze_hash_matches_manifest(
    freeze_result: TrainFitAndFreezeResult,
    frozen_matrices: FrozenPredictionMatrices,
) -> None:
    assert frozen_matrices.freeze_hash == freeze_result.freeze_manifest.freeze_hash


def test_holdout_is_structurally_unavailable_to_train_fit() -> None:
    """The only fit entry point has no holdout argument before it emits a freeze."""
    assert not {
        "x_holdout",
        "y_holdout",
        "holdout_ids",
        "truth_ids",
        "prediction_ids",
    }.intersection(inspect.signature(train_fit_and_freeze).parameters)


def test_frozen_predictions_persist_and_validate_row_ids(
    frozen_matrices: FrozenPredictionMatrices,
) -> None:
    assert frozen_matrices.truth_ids == _HOLDOUT_IDS
    assert frozen_matrices.prediction_ids == _HOLDOUT_IDS
    frozen_matrices.validate()


def test_bootstrap_rejects_mismatched_truth_prediction_ids(
    frozen_matrices: FrozenPredictionMatrices,
) -> None:
    mismatched = dataclasses.replace(
        frozen_matrices,
        prediction_ids=tuple(reversed(_HOLDOUT_IDS)),
    )
    with pytest.raises(ValueError, match="IDs"):
        bootstrap_metric_block(mismatched, draw_start=0, draw_end=2, random_seed=7)


def test_bootstrap_rejects_rows_or_ledger_mismatch(
    frozen_matrices: FrozenPredictionMatrices,
) -> None:
    bad_rows = dataclasses.replace(frozen_matrices, y_pred=frozen_matrices.y_pred[:-1])
    with pytest.raises(ValueError, match="rows"):
        bootstrap_metric_block(bad_rows, draw_start=0, draw_end=2, random_seed=7)

    bad_ledger = frozen_matrices.eligibility_ledger.copy()
    bad_ledger.loc[0, "eligible"] = not bool(bad_ledger.loc[0, "eligible"])
    with pytest.raises(ValueError, match="ledger"):
        bootstrap_metric_block(
            dataclasses.replace(frozen_matrices, eligibility_ledger=bad_ledger),
            draw_start=0,
            draw_end=2,
            random_seed=7,
        )


# ---------------------------------------------------------------------------
# 6. bootstrap_metric_block returns correct type and draw range
# ---------------------------------------------------------------------------

_B_TOTAL = 12


def test_bootstrap_metric_block_returns_shard(
    frozen_matrices: FrozenPredictionMatrices,
) -> None:
    shard = bootstrap_metric_block(frozen_matrices, draw_start=0, draw_end=_B_TOTAL, random_seed=7)
    assert isinstance(shard, BootstrapMetricShard)
    assert shard.draw_start == 0
    assert shard.draw_end == _B_TOTAL
    assert len(shard.per_draw_macro_nrmse) == _B_TOTAL


# ---------------------------------------------------------------------------
# 7. Two-partition and full-block reduce to same per-draw values
# ---------------------------------------------------------------------------


def test_two_partition_reduces_to_same_as_full_block(
    frozen_matrices: FrozenPredictionMatrices,
) -> None:
    full = bootstrap_metric_block(frozen_matrices, draw_start=0, draw_end=_B_TOTAL, random_seed=7)
    half = _B_TOTAL // 2
    lo = bootstrap_metric_block(frozen_matrices, draw_start=0, draw_end=half, random_seed=7)
    hi = bootstrap_metric_block(frozen_matrices, draw_start=half, draw_end=_B_TOTAL, random_seed=7)
    reduced = reduce_bootstrap_metric_blocks([lo, hi])
    np.testing.assert_array_equal(reduced.per_draw_macro_nrmse, full.per_draw_macro_nrmse)


# ---------------------------------------------------------------------------
# 8. Shuffled shard order reduces to same result
# ---------------------------------------------------------------------------


def test_shuffled_shard_order_reduces_to_same(
    frozen_matrices: FrozenPredictionMatrices,
) -> None:
    thirds = _B_TOTAL // 3
    s0 = bootstrap_metric_block(frozen_matrices, draw_start=0, draw_end=thirds, random_seed=7)
    s1 = bootstrap_metric_block(
        frozen_matrices, draw_start=thirds, draw_end=2 * thirds, random_seed=7
    )
    s2 = bootstrap_metric_block(
        frozen_matrices, draw_start=2 * thirds, draw_end=_B_TOTAL, random_seed=7
    )
    full = bootstrap_metric_block(frozen_matrices, draw_start=0, draw_end=_B_TOTAL, random_seed=7)
    reduced_shuffled = reduce_bootstrap_metric_blocks([s2, s0, s1])
    np.testing.assert_array_equal(reduced_shuffled.per_draw_macro_nrmse, full.per_draw_macro_nrmse)


# ---------------------------------------------------------------------------
# 9. Reducer rejects gap
# ---------------------------------------------------------------------------


def test_reduce_bootstrap_metric_blocks_rejects_gap(
    frozen_matrices: FrozenPredictionMatrices,
) -> None:
    s0 = bootstrap_metric_block(frozen_matrices, draw_start=0, draw_end=4, random_seed=7)
    # draw 4 missing; next shard starts at 5
    s1 = bootstrap_metric_block(frozen_matrices, draw_start=5, draw_end=_B_TOTAL, random_seed=7)
    with pytest.raises(ValueError, match="gap"):
        reduce_bootstrap_metric_blocks([s0, s1])


# ---------------------------------------------------------------------------
# 10. Reducer rejects overlap
# ---------------------------------------------------------------------------


def test_reduce_bootstrap_metric_blocks_rejects_overlap(
    frozen_matrices: FrozenPredictionMatrices,
) -> None:
    s0 = bootstrap_metric_block(frozen_matrices, draw_start=0, draw_end=6, random_seed=7)
    s1 = bootstrap_metric_block(frozen_matrices, draw_start=5, draw_end=_B_TOTAL, random_seed=7)
    with pytest.raises(ValueError, match="overlap"):
        reduce_bootstrap_metric_blocks([s0, s1])


# ---------------------------------------------------------------------------
# 11. Reducer rejects mismatched freeze_hash
# ---------------------------------------------------------------------------


def test_reduce_bootstrap_metric_blocks_rejects_mismatched_freeze_hash(
    frozen_matrices: FrozenPredictionMatrices,
) -> None:
    half = _B_TOTAL // 2
    s0 = bootstrap_metric_block(frozen_matrices, draw_start=0, draw_end=half, random_seed=7)
    s1 = bootstrap_metric_block(frozen_matrices, draw_start=half, draw_end=_B_TOTAL, random_seed=7)
    bad_s1 = dataclasses.replace(s1, freeze_hash="bad" + "0" * 61)
    with pytest.raises(ValueError, match="freeze_hash"):
        reduce_bootstrap_metric_blocks([s0, bad_s1])


def test_reduce_bootstrap_metric_blocks_rejects_mismatched_schedule(
    frozen_matrices: FrozenPredictionMatrices,
) -> None:
    half = _B_TOTAL // 2
    s0 = bootstrap_metric_block(frozen_matrices, draw_start=0, draw_end=half, random_seed=7)
    s1 = bootstrap_metric_block(
        frozen_matrices,
        draw_start=half,
        draw_end=_B_TOTAL,
        random_seed=8,
    )
    with pytest.raises(ValueError, match="schedule"):
        reduce_bootstrap_metric_blocks([s0, s1])


# ---------------------------------------------------------------------------
# 12. Reducer rejects wrong total draw count
# ---------------------------------------------------------------------------


def test_reduce_bootstrap_metric_blocks_rejects_wrong_total(
    frozen_matrices: FrozenPredictionMatrices,
) -> None:
    # two shards cover only [0, B-1], missing the last draw
    s0 = bootstrap_metric_block(frozen_matrices, draw_start=0, draw_end=_B_TOTAL - 1, random_seed=7)
    with pytest.raises(ValueError, match="total"):
        reduce_bootstrap_metric_blocks([s0], expected_total_draws=_B_TOTAL)


def test_reduce_bootstrap_metric_blocks_rejects_tampered_draw_ids(
    frozen_matrices: FrozenPredictionMatrices,
) -> None:
    shard = bootstrap_metric_block(frozen_matrices, draw_start=0, draw_end=_B_TOTAL, random_seed=7)
    tampered = dataclasses.replace(shard, draw_ids=np.arange(1, _B_TOTAL + 1))
    with pytest.raises(ValueError, match="draw IDs"):
        reduce_bootstrap_metric_blocks([tampered], expected_total_draws=_B_TOTAL)


def test_bootstrap_is_stratified_and_has_no_fit_surface(
    frozen_matrices: FrozenPredictionMatrices,
) -> None:
    shard = bootstrap_metric_block(frozen_matrices, draw_start=0, draw_end=3, random_seed=7)
    assert shard.stratum_counts == (5, 5)
    assert ".fit(" not in inspect.getsource(bootstrap_metric_block)


def test_bootstrap_macro_nrmse_is_mean_of_eligible_outputwise_ratios(
    frozen_matrices: FrozenPredictionMatrices,
) -> None:
    shard = bootstrap_metric_block(frozen_matrices, draw_start=0, draw_end=1, random_seed=7)
    rng = np.random.RandomState(7)
    row_ids = np.concatenate(
        [
            rng.choice(np.arange(5), size=5, replace=True),
            rng.choice(np.arange(5, 10), size=5, replace=True),
        ]
    )
    rmse = np.sqrt(
        np.mean((frozen_matrices.y_holdout[row_ids] - frozen_matrices.y_pred[row_ids]) ** 2, axis=0)
    )
    ledger = frozen_matrices.eligibility_ledger
    eligible = ledger["eligible"].to_numpy(dtype=bool)
    expected = np.mean(rmse[eligible] / ledger["ref_range"].to_numpy()[eligible])
    assert shard.per_draw_macro_nrmse[0] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 13–16. G11 eligibility ledger
# ---------------------------------------------------------------------------


def test_eligibility_ledger_rejects_near_zero_variance() -> None:
    """Output with variance ≤ 1e-12 is ineligible regardless of range."""
    Y = np.zeros((20, 2), dtype=np.float64)
    Y[:, 0] = 1.0  # zero variance — ineligible
    Y[:, 1] = np.arange(20, dtype=np.float64)  # large range — eligible
    ledger = build_g11_output_eligibility_ledger(Y, output_ids=["a", "b"])
    assert not bool(ledger.loc[ledger["output_id"] == "a", "eligible"].iloc[0])
    assert bool(ledger.loc[ledger["output_id"] == "b", "eligible"].iloc[0])


def test_eligibility_ledger_rejects_small_relative_range() -> None:
    """range/(|mean|+1e-12) < 1e-2 is ineligible even when range >= 1e-6 (legacy gate)."""
    Y = np.zeros((20, 1), dtype=np.float64)
    large_mean = 1_000.0
    # range = 1e-4 >= 1e-6 (passes legacy), but relative_range = 1e-4 / 1000 = 1e-7 < 1e-2
    Y[:, 0] = large_mean + np.linspace(0, 1e-4, 20)
    ledger = build_g11_output_eligibility_ledger(Y, output_ids=["x"])
    assert not bool(ledger.iloc[0]["eligible"]), (
        "Output with tiny relative range must be ineligible under G11 predicate"
    )


def test_eligibility_ledger_eligible_output() -> None:
    """Output passing both criteria is eligible."""
    Y = np.zeros((20, 1), dtype=np.float64)
    Y[:, 0] = np.linspace(0, 1, 20)  # variance > 1e-12, range/|mean| large
    ledger = build_g11_output_eligibility_ledger(Y, output_ids=["y"])
    assert bool(ledger.iloc[0]["eligible"])


def test_eligibility_ledger_relative_range_column() -> None:
    """relative_range column equals range / (|mean| + 1e-12)."""
    rng = np.random.default_rng(0)
    Y = rng.uniform(0, 10, size=(25, 5)).astype(np.float64)
    ledger = build_g11_output_eligibility_ledger(Y, output_ids=list(range(5)))
    expected = ledger["ref_range"] / (ledger["ref_mean"].abs() + 1e-12)
    np.testing.assert_allclose(
        ledger["relative_range"].to_numpy(),
        expected.to_numpy(),
        rtol=1e-10,
    )


def test_production_eligibility_contract_requires_exact_accounting() -> None:
    """The frozen production contract reports all 23,495 output decisions."""
    ledger = pd.DataFrame(
        {
            "output_id": np.arange(23_495),
            "ref_min": np.zeros(23_495),
            "ref_max": np.array([1.0] * 9_954 + [0.0] * 13_541),
            "ref_mean": np.array([0.5] * 9_954 + [0.0] * 13_541),
            "ref_variance": np.array([0.1] * 9_954 + [0.0] * 13_541),
            "eligible": np.array([True] * 9_954 + [False] * 13_541),
        }
    )
    expected_accounting = {"total": 23_495, "eligible": 9_954, "excluded": 13_541}
    accounting = validate_g11_production_eligibility_ledger(
        ledger,
        expected_accounting=expected_accounting,
    )
    assert accounting == {"total": 23_495, "eligible": 9_954, "excluded": 13_541}

    with pytest.raises(ValueError, match="eligible"):
        validate_g11_production_eligibility_ledger(
            ledger.assign(
                ref_max=[1.0] * 9_955 + [0.0] * 13_540,
                ref_mean=[0.5] * 9_955 + [0.0] * 13_540,
                ref_variance=[0.1] * 9_955 + [0.0] * 13_540,
                eligible=[True] * 9_955 + [False] * 13_540,
            ),
            expected_accounting=expected_accounting,
        )


def test_production_eligibility_validator_accepts_generic_output_accounting() -> None:
    ledger = build_g11_output_eligibility_ledger(
        np.column_stack(
            [
                np.linspace(0.0, 1.0, 10),
                np.ones(10),
                np.linspace(1.0, 2.0, 10),
                np.full(10, 1_000.0) + np.linspace(0.0, 1e-4, 10),
            ]
        ),
        output_ids=["a", "b", "c", "d"],
    )

    assert validate_g11_production_eligibility_ledger(ledger) == {
        "total": 4,
        "eligible": 2,
        "excluded": 2,
    }
