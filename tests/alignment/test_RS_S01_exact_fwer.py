"""Gate for slice RS-S01: exact/conservative finite-permutation maxT
interaction-FWER rule.

Authored test-first. The exact Westfall-Young single-step maxT rule controls the
family-wise error rate at any finite number of permutations B under the complete
null, unlike the quantile-threshold path. Do not weaken these assertions.
"""

from __future__ import annotations

import numpy as np
import pytest

from rfm_pipeline.manuscript_stages import (
    max_t_adjusted_pvalues,
    multiplicity_controlled_interaction_selection,
)


def test_RS_S01_max_t_adjusted_pvalue_formula():
    # 3 pairs, B=4 null draws. Row maxima -> [2, 4, 1, 6].
    obs = np.array([5.0, 1.0, 3.0])
    null = np.array(
        [
            [2.0, 0.5, 1.0],
            [4.0, 0.8, 2.0],
            [1.0, 0.2, 0.9],
            [6.0, 0.3, 2.5],
        ]
    )
    # p_adj_j = (1 + #{b : rowmax_b >= obs_j}) / (B + 1), B = 4
    #   obs=5 -> {6}       -> (1+1)/5 = 0.4
    #   obs=1 -> {2,4,1,6} -> (1+4)/5 = 1.0
    #   obs=3 -> {4,6}     -> (1+2)/5 = 0.6
    padj = max_t_adjusted_pvalues(obs, null)
    np.testing.assert_allclose(padj, [0.4, 1.0, 0.6])


def test_RS_S01_adjusted_pvalues_monotone_in_observed():
    rng = np.random.default_rng(0)
    null = rng.normal(size=(200, 10))
    obs = np.linspace(-1.0, 4.0, 10)
    padj = max_t_adjusted_pvalues(obs, null)
    order = np.argsort(obs)
    # Larger observed score => smaller-or-equal adjusted p-value.
    assert np.all(np.diff(padj[order]) <= 1e-12)


def test_RS_S01_exact_method_selects_padj_le_alpha():
    obs = np.array([5.0, 1.0, 3.0])
    null = np.array([[2.0, 0.5, 1.0], [4.0, 0.8, 2.0], [1.0, 0.2, 0.9], [6.0, 0.3, 2.5]])
    selected, p_values, _ = multiplicity_controlled_interaction_selection(
        obs, null, alpha=0.5, method="max_t"
    )
    # padj = [0.4, 1.0, 0.6]; alpha=0.5 -> only pair 0.
    np.testing.assert_allclose(p_values, [0.4, 1.0, 0.6])
    assert selected.tolist() == [True, False, False]


def test_RS_S01_global_null_fwer_is_controlled():
    rng = np.random.default_rng(12345)
    alpha = 0.2
    n_pairs = 15
    B = 199
    n_rep = 600
    false_reject = 0
    for _ in range(n_rep):
        # Complete null: obs and the B null rows are one exchangeable draw.
        data = rng.normal(size=(B + 1, n_pairs))
        obs = data[0]
        null = data[1:]
        selected, _, _ = multiplicity_controlled_interaction_selection(
            obs, null, alpha=alpha, method="max_t"
        )
        if selected.any():
            false_reject += 1
    emp_fwer = false_reject / n_rep
    se = (alpha * (1.0 - alpha) / n_rep) ** 0.5
    assert emp_fwer <= alpha + 3.0 * se, (
        f"empirical global-null FWER {emp_fwer:.3f} exceeds {alpha} + 3*SE ({alpha + 3 * se:.3f})"
    )


def test_RS_S01_input_validation():
    obs = np.array([1.0, 2.0])
    null = np.zeros((5, 2))
    with pytest.raises(ValueError):
        max_t_adjusted_pvalues(obs.reshape(2, 1), null)  # obs not 1-D
    with pytest.raises(ValueError):
        max_t_adjusted_pvalues(obs, null[:, :1])  # column mismatch
    with pytest.raises(ValueError):
        multiplicity_controlled_interaction_selection(obs, null, alpha=1.5, method="max_t")
