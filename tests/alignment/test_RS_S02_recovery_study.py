"""Alignment tests for slice RS-S02: prespecified recovery-study scenarios,
recovery estimands, empirical interaction FWER, and new baselines.

All acceptance criteria from the slice spec are encoded here.
Fixed seeds ensure determinism.
"""

from __future__ import annotations

import numpy as np
import pytest

from rfm_pipeline.baselines import (
    ElasticNetBaseline,
    GBTBaseline,
    OracleOLSBaseline,
    compare_baselines,
)
from rfm_pipeline.recovery_study import (
    RecoveryScenario,
    empirical_interaction_fwer,
    prespecified_recovery_scenarios,
    recovery_estimands,
)
from rfm_pipeline.synthetic_dgp import DGPTrueSupport

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_CATALOG_TRANSFORMS = frozenset({"sin", "sq", "log1p", "exp"})


@pytest.fixture(scope="module")
def all_scenarios() -> list[RecoveryScenario]:
    """Pre-generate once per test session (seeds are fixed → deterministic)."""
    return prespecified_recovery_scenarios()


@pytest.fixture(scope="module")
def scenario_map(all_scenarios) -> dict[str, RecoveryScenario]:
    return {sc.name: sc for sc in all_scenarios}


# ---------------------------------------------------------------------------
# Scenario manifest structure
# ---------------------------------------------------------------------------


def test_RS_S02_exactly_7_scenarios(all_scenarios):
    assert len(all_scenarios) == 7, f"Expected 7 scenarios, got {len(all_scenarios)}"


def test_RS_S02_all_scenario_names_unique(all_scenarios):
    names = [sc.name for sc in all_scenarios]
    assert len(names) == len(set(names)), "Duplicate scenario names"


def test_RS_S02_each_scenario_has_dgptrue_support(all_scenarios):
    for sc in all_scenarios:
        assert isinstance(sc.true_support, DGPTrueSupport), (
            f"{sc.name}: true_support is not a DGPTrueSupport"
        )


def test_RS_S02_seeds_match_spec(all_scenarios):
    for sc in all_scenarios:
        assert sc.seed == sc.spec.seed, (
            f"{sc.name}: RecoveryScenario.seed ({sc.seed}) != spec.seed ({sc.spec.seed})"
        )


# ---------------------------------------------------------------------------
# Scenario-specific planted-support assertions
# ---------------------------------------------------------------------------


def test_RS_S02_global_null_no_interactions(scenario_map):
    sc = scenario_map["global_null"]
    assert len(sc.true_support.true_active_interactions) == 0, (
        "global_null must plant no interactions"
    )
    assert len(sc.true_support.true_active_nonlinear) == 0, (
        "global_null must plant no nonlinear transforms"
    )


def test_RS_S02_interaction_null_no_interactions(scenario_map):
    sc = scenario_map["interaction_null"]
    assert len(sc.true_support.true_active_interactions) == 0, (
        "interaction_null must have no planted interactions"
    )
    assert len(sc.true_support.true_active_nonlinear) > 0, (
        "interaction_null must plant nonlinear transforms"
    )


def test_RS_S02_pure_interaction_negligible_main_effects(scenario_map):
    sc = scenario_map["pure_interaction"]
    assert len(sc.true_support.true_active_inputs) == 0, (
        "pure_interaction: planted main-effect support must be empty (~zero main effects)"
    )
    assert len(sc.true_support.true_active_interactions) > 0, (
        "pure_interaction: must plant at least one interaction"
    )


def test_RS_S02_nonlinear_scenario_has_misspecified_transform_in_spec(scenario_map):
    sc = scenario_map["nonlinear_misspecified"]
    assert len(sc.misspecified_transforms) >= 1, (
        "nonlinear_misspecified: misspecified_transforms must be non-empty"
    )


def test_RS_S02_misspecified_transforms_outside_catalog(scenario_map):
    sc = scenario_map["nonlinear_misspecified"]
    for t in sc.misspecified_transforms:
        base = t.split("_")[0]  # "cubic_x0" -> "cubic"
        assert base not in _CATALOG_TRANSFORMS, (
            f"misspecified transform '{t}' base '{base}' is inside the catalog"
        )


def test_RS_S02_misspecified_transform_in_true_support(scenario_map):
    sc = scenario_map["nonlinear_misspecified"]
    for t in sc.misspecified_transforms:
        assert t in sc.true_support.true_active_nonlinear, (
            f"misspecified transform '{t}' is flagged but not in true_active_nonlinear"
        )


def test_RS_S02_sparse_strong_hierarchical_has_interactions(scenario_map):
    sc = scenario_map["sparse_strong_hierarchical"]
    assert len(sc.true_support.true_active_interactions) > 0, (
        "sparse_strong_hierarchical must plant at least one interaction"
    )


# ---------------------------------------------------------------------------
# recovery_estimands: correct metrics on hand-built pairs
# ---------------------------------------------------------------------------


def test_RS_S02_estimands_perfect_recovery():
    true = DGPTrueSupport(
        true_active_inputs=frozenset({"x0", "x1"}),
        true_active_interactions=frozenset({("x0", "x1")}),
        true_active_nonlinear=frozenset({"sin_x0"}),
    )
    selected = DGPTrueSupport(
        true_active_inputs=frozenset({"x0", "x1"}),
        true_active_interactions=frozenset({("x0", "x1")}),
        true_active_nonlinear=frozenset({"sin_x0"}),
    )
    est = recovery_estimands(true, selected)

    for family in ("main", "interaction", "transformation", "whole"):
        m = est[family]
        assert m["precision"] == 1.0, f"{family}: precision should be 1.0"
        assert m["recall"] == 1.0, f"{family}: recall should be 1.0"
        assert m["fdp"] == 0.0, f"{family}: fdp should be 0.0"
        assert m["exact_support_recovery"] is True, (
            f"{family}: exact_support_recovery should be True"
        )

    assert est["main"]["selected_size"] == 2
    assert est["interaction"]["selected_size"] == 1
    assert est["transformation"]["selected_size"] == 1


def test_RS_S02_estimands_partial_selection():
    true = DGPTrueSupport(
        true_active_inputs=frozenset({"x0", "x1", "x2"}),
        true_active_interactions=frozenset({("x0", "x1"), ("x1", "x2")}),
        true_active_nonlinear=frozenset({"sin_x0", "sq_x1"}),
    )
    # Main: TP={x0,x1}, FP={x3}, FN={x2}  → precision=2/3, recall=2/3, fdp=1/3
    # Interaction: TP={(x0,x1)}, FP={(x0,x3)}, FN={(x1,x2)}  → p=r=fdp=0.5
    # Transform: TP={sin_x0}, FP={log1p_x0}, FN={sq_x1}  → p=r=fdp=0.5
    selected = DGPTrueSupport(
        true_active_inputs=frozenset({"x0", "x1", "x3"}),
        true_active_interactions=frozenset({("x0", "x1"), ("x0", "x3")}),
        true_active_nonlinear=frozenset({"sin_x0", "log1p_x0"}),
    )
    est = recovery_estimands(true, selected)

    m = est["main"]
    assert m["selected_size"] == 3
    assert abs(m["precision"] - 2 / 3) < 1e-10
    assert abs(m["recall"] - 2 / 3) < 1e-10
    assert abs(m["fdp"] - 1 / 3) < 1e-10
    assert m["exact_support_recovery"] is False

    ix = est["interaction"]
    assert ix["selected_size"] == 2
    assert abs(ix["precision"] - 0.5) < 1e-10
    assert abs(ix["recall"] - 0.5) < 1e-10
    assert abs(ix["fdp"] - 0.5) < 1e-10
    assert ix["exact_support_recovery"] is False

    tf = est["transformation"]
    assert tf["selected_size"] == 2
    assert abs(tf["precision"] - 0.5) < 1e-10
    assert abs(tf["recall"] - 0.5) < 1e-10
    assert abs(tf["fdp"] - 0.5) < 1e-10
    assert tf["exact_support_recovery"] is False


def test_RS_S02_estimands_empty_selected():
    """When nothing is selected: precision=1.0 (convention), recall=0.0, fdp=0.0."""
    true = DGPTrueSupport(
        true_active_inputs=frozenset({"x0"}),
        true_active_interactions=frozenset(),
        true_active_nonlinear=frozenset({"sin_x0"}),
    )
    selected = DGPTrueSupport(
        true_active_inputs=frozenset(),
        true_active_interactions=frozenset(),
        true_active_nonlinear=frozenset(),
    )
    est = recovery_estimands(true, selected)
    assert est["main"]["precision"] == 1.0
    assert est["main"]["recall"] == 0.0
    assert est["main"]["fdp"] == 0.0
    assert est["main"]["exact_support_recovery"] is False
    assert est["main"]["selected_size"] == 0


def test_RS_S02_estimands_all_null():
    """When true and selected are both empty: all metrics are 1.0/True/0."""
    null_support = DGPTrueSupport(
        true_active_inputs=frozenset(),
        true_active_interactions=frozenset(),
        true_active_nonlinear=frozenset(),
    )
    est = recovery_estimands(null_support, null_support)
    for family in ("main", "interaction", "transformation", "whole"):
        m = est[family]
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["fdp"] == 0.0
        assert m["exact_support_recovery"] is True
        assert m["selected_size"] == 0


def test_RS_S02_estimands_unordered_interaction_pairs():
    """Interaction pairs (a,b) and (b,a) must be treated as identical."""
    true = DGPTrueSupport(
        true_active_inputs=frozenset(),
        true_active_interactions=frozenset({("x0", "x1")}),
        true_active_nonlinear=frozenset(),
    )
    # Selected uses reversed tuple order — should still match.
    selected = DGPTrueSupport(
        true_active_inputs=frozenset(),
        true_active_interactions=frozenset({("x1", "x0")}),
        true_active_nonlinear=frozenset(),
    )
    est = recovery_estimands(true, selected)
    assert est["interaction"]["precision"] == 1.0
    assert est["interaction"]["recall"] == 1.0
    assert est["interaction"]["exact_support_recovery"] is True


def test_RS_S02_estimands_keys_present():
    """Return dict must contain all required keys for every family."""
    null = DGPTrueSupport(
        true_active_inputs=frozenset(),
        true_active_interactions=frozenset(),
        true_active_nonlinear=frozenset(),
    )
    est = recovery_estimands(null, null)
    required_keys = {"precision", "recall", "fdp", "exact_support_recovery", "selected_size"}
    for family in ("main", "interaction", "transformation", "whole"):
        missing = required_keys - set(est[family].keys())
        assert not missing, f"{family}: missing keys {missing}"


# ---------------------------------------------------------------------------
# empirical_interaction_fwer
# ---------------------------------------------------------------------------


def test_RS_S02_fwer_fixed_flag_vector():
    # 3 out of 5 replicates have >=1 false pair → proportion = 0.6
    flags = [1, 0, 2, 1, 0]
    result = empirical_interaction_fwer(flags)
    assert result["fwer_proportion"] == pytest.approx(0.6)
    assert result["n_replicates"] == 5
    assert result["n_false_pair_replicates"] == 3
    assert result["mean_false_pair_count"] == pytest.approx(0.8)


def test_RS_S02_fwer_wilson_interval_covers_proportion():
    flags = [1, 0, 2, 1, 0]
    result = empirical_interaction_fwer(flags)
    lo = result["wilson_ci_lower"]
    hi = result["wilson_ci_upper"]
    p = result["fwer_proportion"]
    assert lo <= p <= hi, f"Wilson CI [{lo:.4f}, {hi:.4f}] does not cover proportion {p}"


def test_RS_S02_fwer_wilson_interval_in_unit_interval():
    for flags in ([0, 0, 0], [1, 1, 1], [1, 0, 1, 0, 1]):
        result = empirical_interaction_fwer(flags)
        assert 0.0 <= result["wilson_ci_lower"] <= result["wilson_ci_upper"] <= 1.0, (
            f"Wilson CI outside [0,1]: {result['wilson_ci_lower']}, {result['wilson_ci_upper']}"
        )


def test_RS_S02_fwer_all_false():
    result = empirical_interaction_fwer([0, 0, 0, 0])
    assert result["fwer_proportion"] == 0.0
    assert result["wilson_ci_lower"] == pytest.approx(0.0, abs=1e-10)
    assert result["n_false_pair_replicates"] == 0


def test_RS_S02_fwer_all_true():
    result = empirical_interaction_fwer([1, 1, 1, 1])
    assert result["fwer_proportion"] == 1.0
    assert result["wilson_ci_upper"] == pytest.approx(1.0, abs=1e-10)
    assert result["n_false_pair_replicates"] == 4


def test_RS_S02_fwer_bool_flags_accepted():
    """Boolean flags must be accepted as a valid input type."""
    result = empirical_interaction_fwer([True, False, True])
    assert result["fwer_proportion"] == pytest.approx(2 / 3)
    assert result["n_false_pair_replicates"] == 2


def test_RS_S02_fwer_empty_input():
    result = empirical_interaction_fwer([])
    assert result["n_replicates"] == 0
    assert result["fwer_proportion"] == 0.0


def test_RS_S02_fwer_required_keys():
    result = empirical_interaction_fwer([0, 1])
    required = {
        "fwer_proportion",
        "wilson_ci_lower",
        "wilson_ci_upper",
        "n_replicates",
        "n_false_pair_replicates",
        "mean_false_pair_count",
    }
    missing = required - set(result.keys())
    assert not missing, f"Missing keys: {missing}"


# ---------------------------------------------------------------------------
# OracleOLSBaseline
# ---------------------------------------------------------------------------


@pytest.fixture()
def regression_data():
    rng = np.random.default_rng(42)
    X = rng.random((100, 5))
    W = np.array([[1.0, 0.5], [0.0, 0.0], [0.3, 1.0], [0.0, 0.0], [0.0, 0.0]])
    Y = X @ W + 0.05 * rng.normal(size=(100, 2))
    return X, Y


def test_RS_S02_oracle_ols_fit_predict_shape(regression_data):
    X, Y = regression_data
    bl = OracleOLSBaseline(
        true_active_inputs=frozenset({"x0", "x2"}),
        feature_names=["x0", "x1", "x2", "x3", "x4"],
    )
    bl.fit(X, Y)
    Y_pred = bl.predict(X)
    assert Y_pred.shape == Y.shape


def test_RS_S02_oracle_ols_name():
    bl = OracleOLSBaseline(true_active_inputs=frozenset({"x0"}))
    assert bl.name == "oracle_ols"


def test_RS_S02_oracle_ols_model_size_after_fit(regression_data):
    X, Y = regression_data
    bl = OracleOLSBaseline(
        true_active_inputs=frozenset({"x0", "x2"}),
        feature_names=["x0", "x1", "x2", "x3", "x4"],
    )
    bl.fit(X, Y)
    assert bl.model_size_bytes() > 0


def test_RS_S02_oracle_ols_predict_before_fit_raises():
    bl = OracleOLSBaseline(true_active_inputs=frozenset({"x0"}))
    with pytest.raises(RuntimeError):
        bl.predict(np.zeros((5, 5)))


def test_RS_S02_oracle_ols_no_feature_names_uses_all_cols(regression_data):
    X, Y = regression_data
    bl = OracleOLSBaseline(true_active_inputs=frozenset({"x0", "x2"}))
    bl.fit(X, Y)
    Y_pred = bl.predict(X)
    assert Y_pred.shape == Y.shape


# ---------------------------------------------------------------------------
# GBTBaseline (nonlinear surrogate)
# ---------------------------------------------------------------------------


def test_RS_S02_gbt_fit_predict_multi_output(regression_data):
    X, Y = regression_data
    bl = GBTBaseline(n_estimators=10, max_depth=2)
    bl.fit(X, Y)
    Y_pred = bl.predict(X)
    assert Y_pred.shape == Y.shape


def test_RS_S02_gbt_fit_predict_single_output():
    rng = np.random.default_rng(0)
    X = rng.random((50, 4))
    y = X[:, 0] + rng.normal(scale=0.1, size=50)
    bl = GBTBaseline(n_estimators=5)
    bl.fit(X, y)
    y_pred = bl.predict(X)
    assert y_pred.shape == (50,)


def test_RS_S02_gbt_name():
    bl = GBTBaseline(n_estimators=50, max_depth=4)
    assert "gbt" in bl.name
    assert "50" in bl.name
    assert "4" in bl.name


def test_RS_S02_gbt_predict_before_fit_raises():
    bl = GBTBaseline()
    with pytest.raises(RuntimeError):
        bl.predict(np.zeros((5, 3)))


# ---------------------------------------------------------------------------
# compare_baselines harness integration
# ---------------------------------------------------------------------------


def test_RS_S02_baselines_in_comparison_harness(regression_data):
    X, Y = regression_data
    X_train, Y_train = X[:70], Y[:70]
    X_eval, Y_eval = X[70:], Y[70:]

    true_active = frozenset({"x0", "x2"})
    feature_names = ["x0", "x1", "x2", "x3", "x4"]

    baselines = [
        OracleOLSBaseline(true_active_inputs=true_active, feature_names=feature_names),
        GBTBaseline(n_estimators=10, max_depth=2),
        ElasticNetBaseline(alpha=0.01),
    ]
    df = compare_baselines(baselines, X_train, Y_train, X_eval, Y_eval)

    assert len(df) == 3
    required_cols = {"name", "rmse", "r2", "fit_time_s", "eval_time_s", "peak_memory_mb"}
    missing = required_cols - set(df.columns)
    assert not missing, f"Missing harness columns: {missing}"

    # All three baseline types appear in the harness output.
    names = set(df["name"])
    assert "oracle_ols" in names
    assert any("gbt" in n for n in names)
    assert any("elasticnet" in n for n in names)

    assert (df["rmse"] >= 0).all()
    assert df["rmse"].apply(np.isfinite).all()


def test_RS_S02_elasticnet_baseline_present_in_harness(regression_data):
    """Confirm the sparse-linear (ElasticNet) comparator is available."""
    X, Y = regression_data
    X_train, Y_train = X[:70], Y[:70]
    X_eval, Y_eval = X[70:], Y[70:]
    bl = ElasticNetBaseline(alpha=0.01)
    df = compare_baselines([bl], X_train, Y_train, X_eval, Y_eval)
    assert len(df) == 1
    assert "elasticnet" in df["name"].iloc[0]
