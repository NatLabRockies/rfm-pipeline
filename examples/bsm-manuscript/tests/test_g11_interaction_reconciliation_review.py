"""Independent review calculations for reconciled G11 interactions."""

from __future__ import annotations

import numpy as np

from scripts.review_g11_interaction_reconciliation import independent_max_t


def test_independent_max_t_uses_finite_permutation_correction() -> None:
    observed = np.asarray([0.9, 0.25])
    null = np.asarray([[0.1, 0.2], [0.3, 0.1], [0.4, 0.2]])

    selected, adjusted, threshold = independent_max_t(
        observed, null, alpha=0.25
    )

    assert selected.tolist() == [True, False]
    assert adjusted.tolist() == [0.25, 0.75]
    assert threshold == 0.4
