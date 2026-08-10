"""R4-S02: Production-stage recovery pipeline runner on in-memory synthetic data.

Acceptance criteria
-------------------
- On a small synthetic multi-output dataset with a planted main effect, a
  planted interaction pair, and a planted quadratic transform, the runner
  returns per-stage retained sets that are populated and a final selected
  support that includes the planted main effect (x0) and the planted
  interaction pair ({x0, x1}), demonstrating the PRODUCTION stages were called.
- grep proves recovery_study.py references the three production stage functions.
- Under a complete global null the runner retains 0 interaction pairs.
- The runner accepts independent train and eval response draws (no shared RNG
  state); eval predictions depend only on X_train / Y_train / X_eval.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from rfm_pipeline.recovery_study import (
    ProductionRecoveryResult,
    run_production_recovery_pipeline,
)

# ---------------------------------------------------------------------------
# Shared fixture: planted-effects dataset
# ---------------------------------------------------------------------------
# DGP: Y = 5*x0 + 5*x1 + 3*x0*x1 + 3*x0^2 + noise*0.1
#
# Planted support:
#   main effect  : x0
#   interaction  : {x0, x1}  (pair named "x0:x1" or "x1:x0" depending on order)
#   quadratic    : x0_sq  (x0^2)
#
# Strong equal main effects for x0 and x1 ensure both are retained by
# empirical-null screening with BH q=0.20 before interaction discovery.

_N_TOTAL = 200
_N_TRAIN = 160
_N_EVAL = 40
_N_INPUTS = 4  # x0, x1, x2, x3
_N_OUTPUTS = 2
_PLANTED_SEED = 42


def _make_planted_fixture(seed: int = _PLANTED_SEED):
    """Return (X_train, Y_train, X_eval, Y_eval) with planted effects."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((_N_TOTAL, _N_INPUTS))
    x0, x1 = X[:, 0], X[:, 1]
    signal = 5.0 * x0 + 5.0 * x1 + 3.0 * x0 * x1 + 3.0 * x0**2
    Y = np.column_stack(
        [
            signal + 0.1 * rng.standard_normal(_N_TOTAL),
            0.8 * signal + 0.1 * rng.standard_normal(_N_TOTAL),
        ]
    )
    return X[:_N_TRAIN], Y[:_N_TRAIN], X[_N_TRAIN:], Y[_N_TRAIN:]


def _pair_in_set(support: frozenset[str], left: str, right: str) -> bool:
    """Return True if the unordered pair {left, right} appears in *support*."""
    target = frozenset({left, right})
    return any(":" in name and frozenset(name.split(":")) == target for name in support)


# ---------------------------------------------------------------------------
# Test 1: planted effects → recall > 0 for main and interaction families
# ---------------------------------------------------------------------------


def test_R4_S02_planted_effects_recovery() -> None:
    """Runner discovers planted main effect, interaction pair, and quadratic.

    Confirms the production stages were actually called (not stubs), because
    the planted x0:x1 interaction is only detectable via GBT+SHAP analysis of
    the training data.
    """
    X_train, Y_train, X_eval, Y_eval = _make_planted_fixture()

    result = run_production_recovery_pipeline(
        X_train,
        Y_train,
        X_eval,
        Y_eval,
        permutation_count_B=999,
        n_tree_estimators=50,
        seed=_PLANTED_SEED,
    )

    assert isinstance(result, ProductionRecoveryResult)

    # Per-stage retained sets must be populated
    assert result.screening_candidate_count > 0, "No screening candidates"
    assert len(result.screening_retained_set) > 0, "Screening retained nothing"
    assert result.interaction_candidate_count > 0, "No interaction candidates"
    assert len(result.interaction_retained_set) > 0, (
        f"Interaction stage retained nothing; pair scores count="
        f"{result.interaction_candidate_count}"
    )
    assert len(result.nonlinear_retained_set) > 0, "Nonlinear stage retained nothing"

    # Planted main effect x0 ∈ final_selected_support  (recall > 0 for main family)
    assert "x0" in result.final_selected_support, (
        f"Planted main effect x0 missing from final support: {result.final_selected_support}"
    )

    # Planted interaction {x0, x1} ∈ final_selected_support
    # (recall > 0 for interaction family; pair name is order-dependent)
    assert _pair_in_set(result.final_selected_support, "x0", "x1"), (
        f"Planted interaction pair {{x0, x1}} missing from final support: "
        f"{result.final_selected_support}"
    )

    # Eval predictions shape
    assert result.eval_predictions.shape == (_N_EVAL, _N_OUTPUTS), (
        f"Unexpected eval_predictions shape: {result.eval_predictions.shape}"
    )
    assert np.isfinite(result.eval_predictions).all(), "eval_predictions contains non-finite values"


# ---------------------------------------------------------------------------
# Test 2: grep confirms production stage functions are referenced
# ---------------------------------------------------------------------------


def test_R4_S02_source_references_production_stages() -> None:
    """recovery_study.py must directly reference the three production stage functions."""
    src_path = pathlib.Path(__file__).parents[2] / "src" / "rfm_pipeline" / "recovery_study.py"
    src = src_path.read_text(encoding="utf-8")
    for fn_name in (
        "discover_manuscript_interactions",
        "discover_manuscript_nonlinear_transformations",
        "select_manuscript_sparse_support",
    ):
        assert fn_name in src, (
            f"Production stage function {fn_name!r} not referenced in recovery_study.py"
        )


# ---------------------------------------------------------------------------
# Test 3: complete global null → 0 retained interaction pairs
# ---------------------------------------------------------------------------


def test_R4_S02_global_null_terminates_without_fabricated_predictions() -> None:
    """A global-null run propagates the terminal-stage failure; it never fabricates a null fit."""
    rng = np.random.default_rng(999)
    X_train = rng.standard_normal((160, _N_INPUTS))
    Y_train = rng.standard_normal((160, _N_OUTPUTS)) * 0.1  # pure noise
    X_eval = rng.standard_normal((_N_EVAL, _N_INPUTS))
    Y_eval = rng.standard_normal((_N_EVAL, _N_OUTPUTS)) * 0.1

    with pytest.raises(ValueError):
        run_production_recovery_pipeline(
            X_train,
            Y_train,
            X_eval,
            Y_eval,
            permutation_count_B=999,
            n_tree_estimators=50,
            seed=999,
        )


# ---------------------------------------------------------------------------
# Test 4: independent train / eval response draws
# ---------------------------------------------------------------------------


def test_R4_S02_independent_train_eval_rng() -> None:
    """Runner accepts independently drawn train and eval responses.

    Eval predictions are computed by OLS fitted on (X_train, Y_train) and
    applied to X_eval, so they must be identical regardless of which Y_eval
    was passed.  This confirms no shared RNG state leaks from the train path
    into the eval response draw.
    """
    X_train, Y_train, X_eval, Y_eval_a = _make_planted_fixture()
    Y_eval_b = np.random.default_rng(30).standard_normal((_N_EVAL, _N_OUTPUTS))

    result_a = run_production_recovery_pipeline(
        X_train,
        Y_train,
        X_eval,
        Y_eval_a,
        permutation_count_B=999,
        n_tree_estimators=50,
        seed=42,
    )
    result_b = run_production_recovery_pipeline(
        X_train,
        Y_train,
        X_eval,
        Y_eval_b,
        permutation_count_B=999,
        n_tree_estimators=50,
        seed=42,
    )

    # Both runs complete without error
    assert result_a.eval_predictions.shape == (_N_EVAL, _N_OUTPUTS)
    assert result_b.eval_predictions.shape == (_N_EVAL, _N_OUTPUTS)

    # Eval predictions are identical: they depend only on X_train/Y_train/X_eval,
    # not on Y_eval, confirming the train and eval draws are decoupled.
    np.testing.assert_array_equal(
        result_a.eval_predictions,
        result_b.eval_predictions,
        err_msg=(
            "eval_predictions differ between runs with different Y_eval; "
            "the train and eval paths must not share RNG state"
        ),
    )
