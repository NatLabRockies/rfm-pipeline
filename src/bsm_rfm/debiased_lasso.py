"""De-biased LASSO stub module.

This stub provides the placeholder entrypoint `compute_debiased_lasso_artifacts`
so test-first development can proceed. The full implementation will be added
once the artifact schema and deterministic behaviors are specified by tests.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def compute_debiased_lasso_artifacts(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    *,
    config: dict | None = None,
) -> dict[str, Any]:
    """Provide placeholder API for the de-biased LASSO artifact generator.

    This raises NotImplementedError until the full implementation is provided.
    """
    raise NotImplementedError(
        "compute_debiased_lasso_artifacts placeholder (see docs/debiased_lasso_contract.md)"
    )
