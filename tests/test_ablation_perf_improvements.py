"""Regression tests for ablation-stage performance improvements.

Verifies:
- parallel ``bootstrap_macro_nrmse_ci`` (n_jobs > 1) produces bit-identical results to serial
- n_jobs=-1 (all CPUs) produces finite, valid output
- superset design caching in ``_compute_ablation_table`` does not change results
"""

from __future__ import annotations

import math
from unittest.mock import patch

import numpy as np
import pandas as pd

from bsm_rfm.metrics import bootstrap_macro_nrmse_ci

# ---------------------------------------------------------------------------
# bootstrap_macro_nrmse_ci parallelism
# ---------------------------------------------------------------------------


def _make_arrays(
    n_rows: int = 50, n_cols: int = 8, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    Y_true = rng.random((n_rows, n_cols)).astype(np.float64)
    Y_pred = Y_true + rng.random((n_rows, n_cols)).astype(np.float64) * 0.1
    Y_ref = rng.random((n_rows * 2, n_cols)).astype(np.float64)
    return Y_true, Y_pred, Y_ref


def test_bootstrap_parallel_matches_serial() -> None:
    """n_jobs=2 must produce bit-identical CI to n_jobs=1 for the same random_state."""
    Y_true, Y_pred, Y_ref = _make_arrays()
    kwargs = dict(n_boot=40, random_state=7)

    serial = bootstrap_macro_nrmse_ci(Y_true, Y_pred, Y_ref, **kwargs, n_jobs=1)
    parallel = bootstrap_macro_nrmse_ci(Y_true, Y_pred, Y_ref, **kwargs, n_jobs=2)

    assert abs(serial["point_estimate"] - parallel["point_estimate"]) < 1e-12
    assert abs(serial["ci_lower"] - parallel["ci_lower"]) < 1e-12
    assert abs(serial["ci_upper"] - parallel["ci_upper"]) < 1e-12
    assert serial["n_boot"] == parallel["n_boot"] == 40


def test_bootstrap_all_cpus_produces_valid_ci() -> None:
    """n_jobs=-1 (all CPUs) must produce a finite, ordered CI containing the point estimate."""
    Y_true, Y_pred, Y_ref = _make_arrays(n_rows=30, n_cols=5, seed=99)

    result = bootstrap_macro_nrmse_ci(Y_true, Y_pred, Y_ref, n_boot=20, n_jobs=-1)

    assert math.isfinite(result["point_estimate"])
    assert math.isfinite(result["ci_lower"])
    assert math.isfinite(result["ci_upper"])
    assert result["ci_lower"] <= result["point_estimate"] <= result["ci_upper"]


def test_bootstrap_parallel_is_deterministic() -> None:
    """Two calls with n_jobs=2 and the same random_state must return identical results."""
    Y_true, Y_pred, Y_ref = _make_arrays(seed=17)
    kwargs = dict(n_boot=30, random_state=55, n_jobs=2)

    first = bootstrap_macro_nrmse_ci(Y_true, Y_pred, Y_ref, **kwargs)
    second = bootstrap_macro_nrmse_ci(Y_true, Y_pred, Y_ref, **kwargs)

    assert first["ci_lower"] == second["ci_lower"]
    assert first["ci_upper"] == second["ci_upper"]
    assert first["point_estimate"] == second["point_estimate"]


# ---------------------------------------------------------------------------
# Superset design caching: _compute_ablation_table calls build once
# ---------------------------------------------------------------------------


def test_ablation_table_builds_design_matrix_once() -> None:
    """_compute_ablation_table must call build_manuscript_feature_design exactly once
    (superset), not once per ablation model."""
    from bsm_rfm.manuscript_stages import FinalManuscriptArtifactsSpec, _compute_ablation_table

    n_train, n_holdout, n_cols = 15, 5, 3
    rng = np.random.default_rng(0)

    train_ids = pd.Series([f"s{i}" for i in range(n_train)])
    holdout_ids = pd.Series([f"s{i}" for i in range(n_train, n_train + n_holdout)])
    all_ids = pd.concat([train_ids, holdout_ids], ignore_index=True)

    feature_catalog = pd.DataFrame(
        {
            "feature_name": ["f0", "f1", "f2"],
            "feature_type": ["first_order", "first_order", "first_order"],
        }
    )
    input_matrix = pd.DataFrame(
        {"sample_id": list(all_ids)} | {f"f{j}": rng.random(n_train + n_holdout) for j in range(3)}
    )
    output_data = rng.random((n_train + n_holdout, n_cols))
    output_df = pd.DataFrame(output_data, columns=[f"y{k}" for k in range(n_cols)])
    output_df.index = list(all_ids)

    y_train = output_df.loc[list(train_ids)]
    y_holdout = output_df.loc[list(holdout_ids)]
    null_predictions = np.tile(y_train.to_numpy().mean(axis=0), (n_holdout, 1))

    screening_retained_terms = pd.DataFrame({"feature_name": ["f2"]})
    prefilter_names = ["f1", "f2"]
    final_names = ["f2"]
    final_metric = {"point_estimate": 0.5, "ci_lower": 0.3, "ci_upper": 0.7}
    spec = FinalManuscriptArtifactsSpec(
        final_predictor_count_reference=1,
        final_first_order_input_count_reference=1,
        intermediate_penalized_holdout_nrmse_reference=0.1,
        final_ols_holdout_nrmse_reference=0.1,
        nrmse_denominator_definition="macro",
        nrmse_min_range=1e-6,
        nrmse_reference_matrix="Y_train",
        bootstrap_count=5,
    )

    import bsm_rfm.manuscript_stages as _stages_mod

    with patch.object(
        _stages_mod,
        "build_manuscript_feature_design",
        wraps=_stages_mod.build_manuscript_feature_design,
    ) as mock_build:
        _compute_ablation_table(
            feature_catalog=feature_catalog,
            input_matrix=input_matrix,
            train_ids=train_ids,
            holdout_ids=holdout_ids,
            y_train=y_train,
            y_holdout=y_holdout,
            screening_retained_terms=screening_retained_terms,
            prefilter_feature_names=prefilter_names,
            final_feature_names=final_names,
            null_predictions=null_predictions,
            final_metric=final_metric,
            spec=spec,
        )

    assert mock_build.call_count == 1, (
        "Expected 1 call to build_manuscript_feature_design (superset), "
        f"got {mock_build.call_count}"
    )
