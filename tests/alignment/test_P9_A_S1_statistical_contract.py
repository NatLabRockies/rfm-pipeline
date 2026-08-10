"""P9-A-S1: Freeze the statistical contract — production-function tests.

Verifies the finite-B global maxT equation, >= tie convention, and boundary
conditions that define the prespecified statistical contract. Do NOT weaken
these assertions; they are the authoritative evidence for gate G0.

Imported directly from the production module (manuscript_stages).
"""

from __future__ import annotations

import numpy as np
import pytest

from rfm_pipeline.manuscript_stages import (
    max_t_adjusted_pvalues,
    min_permutations_required,
    multiplicity_controlled_interaction_selection,
)

# ---------------------------------------------------------------------------
# Finite-B equation: p_adj_j = (1 + #{b : max_b >= obs_j}) / (B + 1)
# ---------------------------------------------------------------------------


def test_P9_A_S1_equation_formula_from_spec():
    """Verify the prespecified equation for each pair at a worked example.

    Equation: p_adj_j = (1 + #{b : rowmax_b >= obs_j}) / (B + 1)
    where rowmax_b = max over all pairs of null_statistics[b, :].
    """
    # B=3 permutations, 2 pairs.
    obs = np.array([4.0, 2.0])
    null = np.array(
        [
            [3.0, 1.5],  # rowmax = 3.0
            [5.0, 2.0],  # rowmax = 5.0
            [1.0, 0.5],  # rowmax = 1.0
        ]
    )
    # For obs[0]=4.0: rowmaxes [3,5,1] -> #{>= 4.0} = {5.0} -> count=1 -> (1+1)/4 = 0.5
    # For obs[1]=2.0: rowmaxes [3,5,1] -> #{>= 2.0} = {3.0,5.0} -> count=2 -> (1+2)/4 = 0.75
    padj = max_t_adjusted_pvalues(obs, null)
    np.testing.assert_allclose(padj, [0.5, 0.75], rtol=1e-12)


def test_P9_A_S1_ge_tie_convention_exact_boundary():
    """Verify >= tie convention: a null row-max that EQUALS the observed score
    must be counted as an exceedance (not ignored).

    This is the critical tie-convention test: with > the count would be 0;
    with >= the count is 1. The contract specifies >=.
    """
    # B=3, 1 pair. obs=5.0. null rowmaxes = [5.0, 3.0, 2.0].
    # With >=: #{b : rowmax_b >= 5.0} = 1 -> p = (1+1)/(3+1) = 0.5
    # With >:  #{b : rowmax_b >  5.0} = 0 -> p = (1+0)/(3+1) = 0.25
    obs1 = np.array([5.0])
    null1 = np.array([[5.0], [3.0], [2.0]])
    padj = max_t_adjusted_pvalues(obs1, null1)
    # >= tie: (1+1)/(3+1) = 0.5
    assert padj[0] == pytest.approx(0.5, abs=1e-12), (
        f"Expected 0.5 with >= tie convention; got {padj[0]:.6f}. "
        "Check that ties (null max == obs) are counted as exceedances."
    )


def test_P9_A_S1_ge_tie_strict_greater_would_differ():
    """Confirm the >= tie result differs from what strict > would give.

    This test documents the intentional contract choice: strict > is wrong.
    """
    obs = np.array([5.0])
    null = np.array([[5.0], [3.0], [2.0]])  # B=3, only 1 row-max equals obs
    padj = max_t_adjusted_pvalues(obs, null)
    # >= gives 0.5; strict > would give 0.25.
    assert padj[0] != pytest.approx(0.25, abs=1e-12), (
        "Contract requires >=; strict > would give 0.25 but is wrong."
    )
    assert padj[0] == pytest.approx(0.5, abs=1e-12)


def test_P9_A_S1_finite_B_minimum_pvalue_is_one_over_Bplus1():
    """Minimum achievable p-value is 1/(B+1), achieved when obs >> all null.

    With B permutations and no exceedances (obs > every row-max), the formula
    gives (1+0)/(B+1) = 1/(B+1). This is the finite-B resolution floor.
    """
    for B in [1, 4, 9, 19, 99, 499, 999]:
        obs = np.array([1e9])  # dominates all null
        null = np.zeros((B, 1))
        padj = max_t_adjusted_pvalues(obs, null)
        expected = 1.0 / (B + 1)
        assert padj[0] == pytest.approx(expected, rel=1e-12), (
            f"B={B}: minimum p-value should be 1/(B+1)={expected:.6g}; got {padj[0]:.6g}"
        )


def test_P9_A_S1_finite_B_maximum_pvalue_is_one():
    """Maximum p-value is 1.0, achieved when all row-maxima >= obs."""
    for B in [1, 4, 9, 19, 99]:
        obs = np.array([0.0])  # all null maxima >= 0
        null = np.ones((B, 1))
        padj = max_t_adjusted_pvalues(obs, null)
        assert padj[0] == pytest.approx(1.0, abs=1e-12), (
            f"B={B}: maximum p-value should be 1.0; got {padj[0]:.6g}"
        )


def test_P9_A_S1_B1_boundary_cases():
    """With B=1, the only two achievable p-values are 0.5 and 1.0."""
    # obs > null row-max -> p = 1/(1+1) = 0.5
    obs_high = np.array([10.0])
    null_low = np.array([[1.0]])
    padj_high = max_t_adjusted_pvalues(obs_high, null_low)
    assert padj_high[0] == pytest.approx(0.5, abs=1e-12)

    # obs <= null row-max -> p = (1+1)/(1+1) = 1.0
    obs_low = np.array([1.0])
    null_high = np.array([[10.0]])
    padj_low = max_t_adjusted_pvalues(obs_low, null_high)
    assert padj_low[0] == pytest.approx(1.0, abs=1e-12)

    # obs == null row-max (tie) -> p = 1.0 (counted via >=)
    obs_eq = np.array([5.0])
    null_eq = np.array([[5.0]])
    padj_eq = max_t_adjusted_pvalues(obs_eq, null_eq)
    assert padj_eq[0] == pytest.approx(1.0, abs=1e-12)


def test_P9_A_S1_pvalue_range_is_open_0_closed_1():
    """Adjusted p-values lie in (0, 1]: strictly positive, at most 1.0."""
    rng = np.random.default_rng(42)
    for B in [1, 9, 49, 199]:
        null = rng.normal(size=(B, 20))
        obs = rng.normal(size=(20,))
        padj = max_t_adjusted_pvalues(obs, null)
        assert np.all(padj > 0.0), f"B={B}: p-values must be strictly positive"
        assert np.all(padj <= 1.0), f"B={B}: p-values must be at most 1.0"


def test_P9_A_S1_min_permutations_for_alpha005():
    """min_permutations_required returns a value >= 1/(alpha) - 1 for family_size=1."""
    # For alpha=0.05, family_size=1, min B is ceil(1/0.05) - 1 = 19.
    min_B = min_permutations_required(quantile=0.95, family_size=1)
    assert min_B >= 19, f"min_B for alpha=0.05, family=1 should be >= 19; got {min_B}"
    # The minimum achievable p-value 1/(min_B+1) must be <= alpha.
    assert 1.0 / (min_B + 1) <= 0.05 + 1e-12


def test_P9_A_S1_method_max_t_uses_padj_le_alpha():
    """Confirm max_t selects pairs where p_adj <= alpha."""
    obs = np.array([10.0, 5.0, 1.0])
    # B=9 null. Row-maxima all < 10 but some >= 5.
    rng = np.random.default_rng(0)
    null = rng.uniform(0, 8, size=(9, 3))  # all row-maxima <= 8
    # Force obs[0]=10 to have no exceedances: min p = 1/10 = 0.1
    # Force obs[2]=1 to have many exceedances: p close to 1.
    selected, p_values, _ = multiplicity_controlled_interaction_selection(
        obs, null, alpha=0.15, method="max_t"
    )
    # All selected pairs must have p_adj <= 0.15.
    assert np.all(p_values[selected] <= 0.15 + 1e-12)
    # All rejected pairs must have p_adj > 0.15.
    assert np.all(p_values[~selected] > 0.15 - 1e-12)


def test_P9_A_S1_pvalue_sum_formula_consistency():
    """Verify sum of exceedances + 1 in numerator, B+1 in denominator."""
    obs = np.array([2.0, 6.0, 4.0])
    null = np.array(
        [
            [1.0, 5.0, 3.0],  # rowmax=5
            [3.0, 7.0, 2.0],  # rowmax=7
            [2.5, 4.0, 1.0],  # rowmax=4
            [0.5, 8.0, 6.0],  # rowmax=8
            [1.5, 3.0, 2.0],  # rowmax=3
        ]
    )
    # rowmaxes = [5, 7, 4, 8, 3]
    # obs[0]=2: #{>= 2} = {5,7,4,8,3} = 5  -> (1+5)/6 = 1.0
    # obs[1]=6: #{>= 6} = {7,8} = 2         -> (1+2)/6 = 0.5
    # obs[2]=4: #{>= 4} = {5,7,4,8} = 4    -> (1+4)/6 = 5/6
    padj = max_t_adjusted_pvalues(obs, null)
    np.testing.assert_allclose(padj[0], 1.0, atol=1e-12)
    np.testing.assert_allclose(padj[1], 3.0 / 6.0, atol=1e-12)
    np.testing.assert_allclose(padj[2], 5.0 / 6.0, atol=1e-12)
