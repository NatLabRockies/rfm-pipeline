"""Test that sparse-selection stage writes a consolidated diagnostics artifact.

This is a test-first addition to ensure CI/local parity issues expose diagnostics
early in the pipeline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from bsm_rfm import manuscript_stages
from bsm_rfm.manuscript_runtime import build_manuscript_notebook_context
from bsm_rfm.manuscript_stages import (
    SparseSelectionStabilitySpec,
    _select_component_lasso_by_ebic,
    run_sparse_selection_stability_stage,
    write_sparse_selection_stability_artifacts,
)


def test_sparse_selection_writes_diagnostics(tmp_path: Path) -> None:
    context = build_manuscript_notebook_context(
        Path.cwd(), "06_sparse_selection_and_stability.ipynb"
    )

    stage_result = run_sparse_selection_stability_stage(context)
    sparse_result = stage_result.sparse_selection

    paths = write_sparse_selection_stability_artifacts(sparse_result, tmp_path)
    assert "sparse_selection_diagnostics" in paths

    df = pd.read_csv(paths["sparse_selection_diagnostics"])
    assert df.shape[0] == 1

    expected_cols = {
        "n_candidate_terms",
        "n_full_support_terms",
        "n_final_stable_support_terms",
        "mean_resample_jaccard",
        "mean_resample_spearman",
        "passes_jaccard_threshold",
        "passes_spearman_threshold",
        "final_stable_support_nonempty",
    }
    assert expected_cols.issubset(set(df.columns))

    assert int(df.loc[0, "n_candidate_terms"]) > 0
    assert int(df.loc[0, "n_full_support_terms"]) > 0
    assert int(df.loc[0, "n_final_stable_support_terms"]) > 0
    assert bool(df.loc[0, "final_stable_support_nonempty"]) is True


def test_sparse_selection_uses_warm_start_on_lasso_path(monkeypatch) -> None:
    calls: list[dict] = []

    class SpyLasso:
        def __init__(self, **kwargs) -> None:
            calls.append(dict(kwargs))
            self._n_features = 0
            self.coef_ = np.array([], dtype=float)

        def set_params(self, **kwargs) -> SpyLasso:
            calls.append(dict(kwargs))
            return self

        def fit(self, x: np.ndarray, y: np.ndarray) -> SpyLasso:
            self._n_features = x.shape[1]
            self.coef_ = np.zeros(self._n_features, dtype=float)
            return self

    monkeypatch.setattr(manuscript_stages, "Lasso", SpyLasso)

    x = np.array(
        [
            [1.0, 0.0, 2.0],
            [0.5, 1.5, 1.0],
            [1.5, 0.5, 0.0],
            [2.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    y = np.array([1.0, 0.5, 1.5, 2.0], dtype=float)
    active = np.array([True, True, True], dtype=bool)
    spec = SparseSelectionStabilitySpec(
        model_class="l1_penalized_linear_model_per_retained_component",
        ebic_gamma=0.5,
        support_aggregation_rule="union_nonzero_support_across_retained_components",
        resampling_scheme="2_subsamples_of_80_percent_rows_without_replacement_seed_123",
        subsample_count=2,
        subsample_fraction=0.80,
        jaccard_threshold=0.50,
        spearman_threshold=0.50,
        random_seed=123,
    )

    _select_component_lasso_by_ebic(
        x,
        y,
        active_features=active,
        spec=spec,
    )

    constructor_calls = [call for call in calls if "selection" in call]
    assert constructor_calls, "Expected at least one Lasso constructor call."
    assert all(call.get("warm_start") is True for call in constructor_calls)


def test_sparse_selection_suppresses_lasso_convergence_warning(monkeypatch) -> None:
    class WarningLasso:
        def __init__(self, **kwargs) -> None:
            self.coef_ = np.array([], dtype=float)

        def set_params(self, **kwargs) -> WarningLasso:
            return self

        def fit(self, x: np.ndarray, y: np.ndarray) -> WarningLasso:
            import warnings

            self.coef_ = np.zeros(x.shape[1], dtype=float)
            warnings.warn("did not converge", ConvergenceWarning, stacklevel=2)
            return self

    monkeypatch.setattr(manuscript_stages, "Lasso", WarningLasso)

    x = np.array(
        [
            [1.0, 0.0, 2.0],
            [0.5, 1.5, 1.0],
            [1.5, 0.5, 0.0],
            [2.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    y = np.array([1.0, 0.5, 1.5, 2.0], dtype=float)
    active = np.array([True, True, True], dtype=bool)
    spec = SparseSelectionStabilitySpec(
        model_class="l1_penalized_linear_model_per_retained_component",
        ebic_gamma=0.5,
        support_aggregation_rule="union_nonzero_support_across_retained_components",
        resampling_scheme="2_subsamples_of_80_percent_rows_without_replacement_seed_123",
        subsample_count=2,
        subsample_fraction=0.80,
        jaccard_threshold=0.50,
        spearman_threshold=0.50,
        random_seed=123,
    )

    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _select_component_lasso_by_ebic(
            x,
            y,
            active_features=active,
            spec=spec,
        )
    assert not any(isinstance(w.message, ConvergenceWarning) for w in caught)
