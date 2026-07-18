"""Tests for manuscript interaction-discovery stage."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import rfm_pipeline.manuscript_stages as manuscript_stages
from rfm_pipeline.manuscript_runtime import (
    build_manuscript_notebook_context,
)
from rfm_pipeline.manuscript_stages import (
    InteractionDiscoverySpec,
    discover_manuscript_interactions,
    interaction_discovery_spec_from_case_study_config,
    run_interaction_discovery_stage,
    write_interaction_discovery_artifacts,
)


def test_interaction_discovery_guard_on_uses_corrected_family_size_fwer(
    monkeypatch,
) -> None:
    """Guard-ON production path: FWER adequacy is family_size=1 (not n_pairs)."""
    sample_ids = list(range(1, 41))
    x1 = [-1.0, -1.0, 1.0, 1.0] * 10
    x2 = [-1.0, 1.0, -1.0, 1.0] * 10
    x3 = [1.0 if v % 2 == 0 else -1.0 for v in sample_ids]
    pca_signal = [a * b for a, b in zip(x1, x2, strict=True)]
    inputs = pd.DataFrame({"sample_id": sample_ids, "x1": x1, "x2": x2, "x3": x3})
    catalog = pd.DataFrame(
        {
            "feature_name": ["x1", "x2", "x3"],
            "feature_type": ["first_order"] * 3,
        }
    )
    holdout = pd.DataFrame({"sample_id": sample_ids, "split": ["train"] * 32 + ["holdout"] * 8})
    pca_scores = pd.DataFrame({"sample_id": sample_ids, "PC1": pca_signal})
    retained_terms = pd.DataFrame(
        {
            "feature_name": ["x1", "x2", "x3"],
            "feature_type": ["first_order"] * 3,
        }
    )
    # With family_size=n_pairs=3 the OLD guard would require B >= 599 for
    # alpha=0.05. Under the corrected FWER family_size=1 rule, B=19 is enough.
    spec = InteractionDiscoverySpec(
        method="tree_shap_interaction_values",
        aggregation_rule="max_over_components_of_mean_absolute_shap_interaction",
        null_threshold_quantile=0.995,
        retained_pairs_reference=1,
        permutation_count_B=19,
        random_seed=123,
        n_jobs=1,
        family_error_method="fwer_max_stat",
        family_error_alpha=0.05,
        # enforce_permutation_adequacy=True (default) -- guard is ON
    )

    def _fake_scorer(
        y_base: np.ndarray,
        permute_response: bool,
        *,
        n_pairs: int,
        n_comp: int,
        **_: object,
    ) -> tuple[np.ndarray, np.ndarray]:
        _ = (y_base, permute_response)
        scores = np.ones(n_pairs, dtype=float)
        component_scores = np.ones((n_pairs, n_comp), dtype=float)
        return scores, component_scores

    monkeypatch.setattr(
        manuscript_stages,
        "_score_interaction_permutation",
        _fake_scorer,
    )

    result = discover_manuscript_interactions(
        inputs, catalog, holdout, pca_scores, retained_terms, spec
    )
    assert set(result.pair_scores["pair_name"]) == {"x1:x2", "x1:x3", "x2:x3"}
    assert result.summary.loc[0, "n_candidate_pairs"] == 3


def test_interaction_discovery_guard_on_bh_fdr_requires_n_pairs_budget() -> None:
    """Guard-ON production path: BH-FDR adequacy scales with n_pairs."""
    sample_ids = list(range(1, 41))
    x1 = [-1.0, -1.0, 1.0, 1.0] * 10
    x2 = [-1.0, 1.0, -1.0, 1.0] * 10
    x3 = [1.0 if v % 2 == 0 else -1.0 for v in sample_ids]
    pca_signal = [a * b for a, b in zip(x1, x2, strict=True)]
    inputs = pd.DataFrame({"sample_id": sample_ids, "x1": x1, "x2": x2, "x3": x3})
    catalog = pd.DataFrame(
        {
            "feature_name": ["x1", "x2", "x3"],
            "feature_type": ["first_order"] * 3,
        }
    )
    holdout = pd.DataFrame({"sample_id": sample_ids, "split": ["train"] * 32 + ["holdout"] * 8})
    pca_scores = pd.DataFrame({"sample_id": sample_ids, "PC1": pca_signal})
    retained_terms = pd.DataFrame(
        {
            "feature_name": ["x1", "x2", "x3"],
            "feature_type": ["first_order"] * 3,
        }
    )
    # BH-FDR with 3 pairs and alpha=0.05 requires B >= ceil(3/0.05)-1 = 59.
    spec = InteractionDiscoverySpec(
        method="tree_shap_interaction_values",
        aggregation_rule="max_over_components_of_mean_absolute_shap_interaction",
        null_threshold_quantile=0.995,
        retained_pairs_reference=1,
        permutation_count_B=19,
        random_seed=123,
        n_jobs=1,
        family_error_method="bh_fdr",
        family_error_alpha=0.05,
    )
    with pytest.raises(manuscript_stages.PermutationAdequacyError):
        discover_manuscript_interactions(inputs, catalog, holdout, pca_scores, retained_terms, spec)

    config = {
        "case_study": {
            "interface": {"holdout_random_seed": 456},
            "empirical_null_screen": {"permutation_count_B": 11},
            "interaction_discovery": {
                "method": "tree_shap_interaction_values",
                "aggregation_rule": "max_over_components_of_mean_absolute_shap_interaction",
                "null_threshold_quantile": 0.9,
                "retained_pairs": 12,
                "permutation_count_B": 13,
                "n_tree_estimators": 17,
                "max_tree_depth": 2,
                "max_shap_samples": 41,
                "parallel_batch_timeout_seconds": 17,
                "parallel_backend": "loky",
            },
        }
    }

    spec = interaction_discovery_spec_from_case_study_config(config)

    assert spec.permutation_count_B == 13
    assert spec.random_seed == 456
    assert spec.n_tree_estimators == 17
    assert spec.max_tree_depth == 2
    assert spec.max_shap_samples == 41
    assert spec.parallel_batch_timeout_seconds == 17
    assert spec.parallel_backend == "loky"


def test_interaction_discovery_retains_residual_pair_signal_and_writes_artifacts(
    tmp_path: Path,
) -> None:
    sample_ids = list(range(1, 41))
    x1 = [-1.0, -1.0, 1.0, 1.0] * 10
    x2 = [-1.0, 1.0, -1.0, 1.0] * 10
    pca_signal = [left * right for left, right in zip(x1, x2, strict=True)]
    inputs = pd.DataFrame({"sample_id": sample_ids, "x1": x1, "x2": x2})
    catalog = pd.DataFrame(
        {
            "feature_name": ["x1", "x2"],
            "feature_type": ["first_order", "first_order"],
            "origin": ["test"] * 2,
        }
    )
    holdout = pd.DataFrame({"sample_id": sample_ids, "split": ["train"] * 32 + ["holdout"] * 8})
    pca_scores = pd.DataFrame({"sample_id": sample_ids, "PC1": pca_signal})
    retained_terms = pd.DataFrame(
        {
            "feature_name": ["x1", "x2"],
            "feature_type": ["first_order", "first_order"],
        }
    )
    spec = InteractionDiscoverySpec(
        method="tree_shap_interaction_values",
        aggregation_rule="max_over_components_of_mean_absolute_shap_interaction",
        null_threshold_quantile=0.95,
        retained_pairs_reference=367,
        permutation_count_B=49,
        random_seed=123,
    )

    result = discover_manuscript_interactions(
        inputs,
        catalog,
        holdout,
        pca_scores,
        retained_terms,
        spec,
    )

    assert result.summary.loc[0, "stage"] == "interaction_discovery"
    assert result.summary.loc[0, "n_candidate_pairs"] == 1
    assert result.summary.loc[0, "n_retained_pairs"] == 1
    assert result.pair_scores.loc[0, "pair_name"] == "x1:x2"
    assert result.pair_scores.loc[0, "retained"]
    assert result.pair_scores.loc[0, "empirical_null_retained"]
    assert result.summary.loc[0, "public_implementation_method"] == "tree_shap_gradient_boosting"
    assert result.summary.loc[0, "source_workflow_equivalence_status"] == (
        "manuscript_aligned_via_shap_gradient_boosting"
    )
    assert result.provenance.loc[0, "manuscript_method"] == "tree_shap_interaction_values"
    assert result.provenance.loc[0, "public_implementation_method"] == "tree_shap_gradient_boosting"

    paths = write_interaction_discovery_artifacts(result, tmp_path)
    assert sorted(paths) == [
        "component_interaction_scores",
        "interaction_discovery_provenance",
        "interaction_discovery_summary",
        "interaction_null_summary",
        "interaction_pair_scores",
        "retained_interaction_pairs",
    ]
    assert paths["retained_interaction_pairs"].read_text(encoding="utf-8").startswith("pair_name")
    provenance_text = paths["interaction_discovery_provenance"].read_text(encoding="utf-8")
    assert "tree_shap_gradient_boosting" in provenance_text
    assert "manuscript_aligned_via_shap_gradient_boosting" in provenance_text


def test_interaction_discovery_generates_all_pairs_from_retained_first_order_terms() -> None:
    sample_ids = list(range(1, 41))
    x1 = [-1.0, -1.0, 1.0, 1.0] * 10
    x2 = [-1.0, 1.0, -1.0, 1.0] * 10
    x3 = [1.0 if value % 2 == 0 else -1.0 for value in sample_ids]
    pca_signal = [left * right for left, right in zip(x1, x2, strict=True)]
    inputs = pd.DataFrame({"sample_id": sample_ids, "x1": x1, "x2": x2, "x3": x3})
    catalog = pd.DataFrame(
        {
            "feature_name": ["x1", "x2", "x3"],
            "feature_type": ["first_order", "first_order", "first_order"],
        }
    )
    holdout = pd.DataFrame({"sample_id": sample_ids, "split": ["train"] * 32 + ["holdout"] * 8})
    pca_scores = pd.DataFrame({"sample_id": sample_ids, "PC1": pca_signal})
    retained_terms = pd.DataFrame(
        {
            "feature_name": ["x1", "x2", "x3"],
            "feature_type": ["first_order", "first_order", "first_order"],
        }
    )
    spec = InteractionDiscoverySpec(
        method="tree_shap_interaction_values",
        aggregation_rule="max_over_components_of_mean_absolute_shap_interaction",
        null_threshold_quantile=0.95,
        retained_pairs_reference=367,
        permutation_count_B=19,
        random_seed=123,
        enforce_permutation_adequacy=False,
    )

    result = discover_manuscript_interactions(
        inputs,
        catalog,
        holdout,
        pca_scores,
        retained_terms,
        spec,
    )

    assert set(result.pair_scores["pair_name"]) == {"x1:x2", "x1:x3", "x2:x3"}
    assert result.summary.loc[0, "n_candidate_pairs"] == 3


def test_interaction_discovery_respects_candidate_pair_range(monkeypatch) -> None:
    sample_ids = list(range(1, 21))
    x1 = [-1.0, -1.0, 1.0, 1.0] * 5
    x2 = [-1.0, 1.0, -1.0, 1.0] * 5
    x3 = [1.0 if value % 2 == 0 else -1.0 for value in sample_ids]
    pca_signal = [left * right for left, right in zip(x1, x2, strict=True)]
    inputs = pd.DataFrame({"sample_id": sample_ids, "x1": x1, "x2": x2, "x3": x3})
    catalog = pd.DataFrame(
        {
            "feature_name": ["x1", "x2", "x3"],
            "feature_type": ["first_order", "first_order", "first_order"],
        }
    )
    holdout = pd.DataFrame({"sample_id": sample_ids, "split": ["train"] * 16 + ["holdout"] * 4})
    pca_scores = pd.DataFrame({"sample_id": sample_ids, "PC1": pca_signal})
    retained_terms = pd.DataFrame(
        {
            "feature_name": ["x1", "x2", "x3"],
            "feature_type": ["first_order", "first_order", "first_order"],
        }
    )
    spec = InteractionDiscoverySpec(
        method="tree_shap_interaction_values",
        aggregation_rule="max_over_components_of_mean_absolute_shap_interaction",
        null_threshold_quantile=0.9,
        retained_pairs_reference=1,
        permutation_count_B=2,
        random_seed=123,
        n_jobs=1,
        enforce_permutation_adequacy=False,
    )

    def _fake_score_interaction_permutation(
        y_base: np.ndarray,
        permute_response: bool,
        *,
        n_pairs: int,
        n_comp: int,
        **_: object,
    ) -> tuple[np.ndarray, np.ndarray]:
        _ = (y_base, permute_response)
        scores = np.ones(n_pairs, dtype=float)
        component_scores = np.ones((n_pairs, n_comp), dtype=float)
        return scores, component_scores

    monkeypatch.setattr(
        manuscript_stages,
        "_score_interaction_permutation",
        _fake_score_interaction_permutation,
    )

    # Candidate ordering for x1,x2,x3 is: [x1:x2, x1:x3, x2:x3].
    result = discover_manuscript_interactions(
        inputs,
        catalog,
        holdout,
        pca_scores,
        retained_terms,
        spec,
        pair_start_idx=1,
        pair_end_idx=2,
    )
    assert result.summary.loc[0, "n_candidate_pairs"] == 1
    assert set(result.pair_scores["pair_name"]) == {"x1:x3"}


def test_interaction_discovery_rejects_empty_candidate_pair_range() -> None:
    sample_ids = list(range(1, 21))
    x1 = [-1.0, -1.0, 1.0, 1.0] * 5
    x2 = [-1.0, 1.0, -1.0, 1.0] * 5
    pca_signal = [left * right for left, right in zip(x1, x2, strict=True)]
    inputs = pd.DataFrame({"sample_id": sample_ids, "x1": x1, "x2": x2})
    catalog = pd.DataFrame(
        {
            "feature_name": ["x1", "x2"],
            "feature_type": ["first_order", "first_order"],
        }
    )
    holdout = pd.DataFrame({"sample_id": sample_ids, "split": ["train"] * 16 + ["holdout"] * 4})
    pca_scores = pd.DataFrame({"sample_id": sample_ids, "PC1": pca_signal})
    retained_terms = pd.DataFrame(
        {
            "feature_name": ["x1", "x2"],
            "feature_type": ["first_order", "first_order"],
        }
    )
    spec = InteractionDiscoverySpec(
        method="tree_shap_interaction_values",
        aggregation_rule="max_over_components_of_mean_absolute_shap_interaction",
        null_threshold_quantile=0.9,
        retained_pairs_reference=1,
        permutation_count_B=2,
        random_seed=123,
        n_jobs=1,
        enforce_permutation_adequacy=False,
    )

    with pytest.raises(ValueError, match="candidate range is empty"):
        discover_manuscript_interactions(
            inputs,
            catalog,
            holdout,
            pca_scores,
            retained_terms,
            spec,
            pair_start_idx=5,
            pair_end_idx=6,
        )


def test_run_interaction_discovery_stage_executes_demo_context() -> None:
    context = build_manuscript_notebook_context(
        Path.cwd(),
        "04_interaction_discovery.ipynb",
    )

    result = run_interaction_discovery_stage(context)

    assert result.interactions.summary.loc[0, "stage"] == "interaction_discovery"
    assert result.interactions.summary.loc[0, "n_candidate_pairs"] == 1
    assert result.artifact_paths["interaction_pair_scores"].exists()


def test_interaction_discovery_rejects_invalid_parallel_backend() -> None:
    sample_ids = list(range(1, 21))
    x1 = [-1.0, -1.0, 1.0, 1.0] * 5
    x2 = [-1.0, 1.0, -1.0, 1.0] * 5
    pca_signal = [left * right for left, right in zip(x1, x2, strict=True)]
    inputs = pd.DataFrame({"sample_id": sample_ids, "x1": x1, "x2": x2})
    catalog = pd.DataFrame(
        {
            "feature_name": ["x1", "x2"],
            "feature_type": ["first_order", "first_order"],
        }
    )
    holdout = pd.DataFrame({"sample_id": sample_ids, "split": ["train"] * 16 + ["holdout"] * 4})
    pca_scores = pd.DataFrame({"sample_id": sample_ids, "PC1": pca_signal})
    retained_terms = pd.DataFrame(
        {
            "feature_name": ["x1", "x2"],
            "feature_type": ["first_order", "first_order"],
        }
    )
    spec = InteractionDiscoverySpec(
        method="tree_shap_interaction_values",
        aggregation_rule="max_over_components_of_mean_absolute_shap_interaction",
        null_threshold_quantile=0.9,
        retained_pairs_reference=1,
        permutation_count_B=2,
        random_seed=123,
        n_jobs=2,
        parallel_backend="invalid_backend",
        enforce_permutation_adequacy=False,
    )

    with pytest.raises(ValueError, match="parallel_backend"):
        discover_manuscript_interactions(inputs, catalog, holdout, pca_scores, retained_terms, spec)


def test_interaction_discovery_uses_configured_parallel_backend_without_fallback(
    monkeypatch,
) -> None:
    sample_ids = list(range(1, 21))
    x1 = [-1.0, -1.0, 1.0, 1.0] * 5
    x2 = [-1.0, 1.0, -1.0, 1.0] * 5
    pca_signal = [left * right for left, right in zip(x1, x2, strict=True)]
    inputs = pd.DataFrame({"sample_id": sample_ids, "x1": x1, "x2": x2})
    catalog = pd.DataFrame(
        {
            "feature_name": ["x1", "x2"],
            "feature_type": ["first_order", "first_order"],
        }
    )
    holdout = pd.DataFrame({"sample_id": sample_ids, "split": ["train"] * 16 + ["holdout"] * 4})
    pca_scores = pd.DataFrame({"sample_id": sample_ids, "PC1": pca_signal})
    retained_terms = pd.DataFrame(
        {
            "feature_name": ["x1", "x2"],
            "feature_type": ["first_order", "first_order"],
        }
    )
    spec = InteractionDiscoverySpec(
        method="tree_shap_interaction_values",
        aggregation_rule="max_over_components_of_mean_absolute_shap_interaction",
        null_threshold_quantile=0.9,
        retained_pairs_reference=1,
        permutation_count_B=3,
        random_seed=123,
        n_jobs=2,
        parallel_batch_timeout_seconds=1,
        parallel_backend="threading",
        enforce_permutation_adequacy=False,
    )

    calls: list[tuple[int, str]] = []

    class FakeParallel:
        def __init__(self, n_jobs: int, **kwargs: object) -> None:
            self.n_jobs = n_jobs
            self.backend = str(kwargs.get("backend", "loky"))

        def __enter__(self) -> FakeParallel:
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def __call__(
            self,
            jobs: list[tuple[object, tuple[object, ...], dict[str, object]]],
        ) -> list[tuple[np.ndarray, np.ndarray]]:
            calls.append((self.n_jobs, self.backend))
            results: list[tuple[np.ndarray, np.ndarray]] = []
            for func, args, kwargs in jobs:
                results.append(func(*args, **kwargs))
            return results

    def _fake_score_interaction_permutation(
        y_base: np.ndarray,
        permute_response: bool,
        *,
        n_pairs: int,
        n_comp: int,
        **_: object,
    ) -> tuple[np.ndarray, np.ndarray]:
        _ = (y_base, permute_response)
        scores = np.ones(n_pairs, dtype=float)
        component_scores = np.ones((n_pairs, n_comp), dtype=float)
        return scores, component_scores

    monkeypatch.setattr(manuscript_stages, "Parallel", FakeParallel)
    monkeypatch.setattr(
        manuscript_stages,
        "_score_interaction_permutation",
        _fake_score_interaction_permutation,
    )

    result = discover_manuscript_interactions(
        inputs,
        catalog,
        holdout,
        pca_scores,
        retained_terms,
        spec,
    )

    assert calls[0] == (2, "threading")
    assert all(n_jobs == 2 and backend == "threading" for n_jobs, backend in calls)
    assert len(result.pair_scores) == 1


def test_interaction_discovery_accepts_dask_backend_with_executor(
    monkeypatch,
) -> None:
    sample_ids = list(range(1, 21))
    x1 = [-1.0, -1.0, 1.0, 1.0] * 5
    x2 = [-1.0, 1.0, -1.0, 1.0] * 5
    pca_signal = [left * right for left, right in zip(x1, x2, strict=True)]
    inputs = pd.DataFrame({"sample_id": sample_ids, "x1": x1, "x2": x2})
    catalog = pd.DataFrame(
        {
            "feature_name": ["x1", "x2"],
            "feature_type": ["first_order", "first_order"],
        }
    )
    holdout = pd.DataFrame({"sample_id": sample_ids, "split": ["train"] * 16 + ["holdout"] * 4})
    pca_scores = pd.DataFrame({"sample_id": sample_ids, "PC1": pca_signal})
    retained_terms = pd.DataFrame(
        {
            "feature_name": ["x1", "x2"],
            "feature_type": ["first_order", "first_order"],
        }
    )
    spec = InteractionDiscoverySpec(
        method="tree_shap_interaction_values",
        aggregation_rule="max_over_components_of_mean_absolute_shap_interaction",
        null_threshold_quantile=0.9,
        retained_pairs_reference=1,
        permutation_count_B=3,
        random_seed=123,
        n_jobs=2,
        parallel_backend="dask",
        enforce_permutation_adequacy=False,
    )

    class FakeExecutor:
        def map(self, fn, items, **kwargs):  # noqa: ANN001, ANN003
            _ = kwargs
            return [fn(item) for item in items]

        def close(self) -> None:
            return None

    def _fake_get_executor(backend: str, **kwargs: object):  # noqa: ANN003
        assert backend == "dask"
        assert kwargs["n_workers"] == 2
        return FakeExecutor()

    def _fake_score_interaction_permutation(
        y_base: np.ndarray,
        permute_response: bool,
        *,
        n_pairs: int,
        n_comp: int,
        **_: object,
    ) -> tuple[np.ndarray, np.ndarray]:
        _ = (y_base, permute_response)
        scores = np.ones(n_pairs, dtype=float)
        component_scores = np.ones((n_pairs, n_comp), dtype=float)
        return scores, component_scores

    monkeypatch.setattr(manuscript_stages, "get_executor", _fake_get_executor)
    monkeypatch.setattr(
        manuscript_stages,
        "_score_interaction_permutation",
        _fake_score_interaction_permutation,
    )

    result = discover_manuscript_interactions(
        inputs,
        catalog,
        holdout,
        pca_scores,
        retained_terms,
        spec,
    )
    assert len(result.pair_scores) == 1


def test_interaction_discovery_falls_back_to_joblib_when_dask_executor_fails(
    monkeypatch,
) -> None:
    sample_ids = list(range(1, 21))
    x1 = [-1.0, -1.0, 1.0, 1.0] * 5
    x2 = [-1.0, 1.0, -1.0, 1.0] * 5
    pca_signal = [left * right for left, right in zip(x1, x2, strict=True)]
    inputs = pd.DataFrame({"sample_id": sample_ids, "x1": x1, "x2": x2})
    catalog = pd.DataFrame(
        {
            "feature_name": ["x1", "x2"],
            "feature_type": ["first_order", "first_order"],
        }
    )
    holdout = pd.DataFrame({"sample_id": sample_ids, "split": ["train"] * 16 + ["holdout"] * 4})
    pca_scores = pd.DataFrame({"sample_id": sample_ids, "PC1": pca_signal})
    retained_terms = pd.DataFrame(
        {
            "feature_name": ["x1", "x2"],
            "feature_type": ["first_order", "first_order"],
        }
    )
    spec = InteractionDiscoverySpec(
        method="tree_shap_interaction_values",
        aggregation_rule="max_over_components_of_mean_absolute_shap_interaction",
        null_threshold_quantile=0.9,
        retained_pairs_reference=1,
        permutation_count_B=3,
        random_seed=123,
        n_jobs=2,
        parallel_backend="dask",
        enforce_permutation_adequacy=False,
    )

    class FakeParallel:
        def __init__(self, n_jobs: int, **kwargs: object) -> None:
            self.n_jobs = n_jobs
            self.backend = str(kwargs.get("backend", "loky"))

        def __enter__(self) -> FakeParallel:
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001
            return False

        def __call__(
            self,
            jobs: list[tuple[object, tuple[object, ...], dict[str, object]]],
        ) -> list[tuple[np.ndarray, np.ndarray]]:
            _ = (self.n_jobs, self.backend)
            return [func(*args, **kwargs) for func, args, kwargs in jobs]

    def _fake_score_interaction_permutation(
        y_base: np.ndarray,
        permute_response: bool,
        *,
        n_pairs: int,
        n_comp: int,
        **_: object,
    ) -> tuple[np.ndarray, np.ndarray]:
        _ = (y_base, permute_response)
        scores = np.ones(n_pairs, dtype=float)
        component_scores = np.ones((n_pairs, n_comp), dtype=float)
        return scores, component_scores

    def _raising_get_executor(backend: str, **kwargs: object):  # noqa: ANN003
        _ = (backend, kwargs)
        raise RuntimeError("dask unavailable")

    monkeypatch.setattr(manuscript_stages, "get_executor", _raising_get_executor)
    monkeypatch.setattr(manuscript_stages, "Parallel", FakeParallel)
    monkeypatch.setattr(
        manuscript_stages,
        "_score_interaction_permutation",
        _fake_score_interaction_permutation,
    )

    result = discover_manuscript_interactions(
        inputs,
        catalog,
        holdout,
        pca_scores,
        retained_terms,
        spec,
    )
    assert len(result.pair_scores) == 1


def test_interaction_discovery_resumes_from_checkpointed_permutation_scores(
    monkeypatch,
    tmp_path: Path,
) -> None:
    sample_ids = list(range(1, 21))
    x1 = [-1.0, -1.0, 1.0, 1.0] * 5
    x2 = [-1.0, 1.0, -1.0, 1.0] * 5
    pca_signal = [left * right for left, right in zip(x1, x2, strict=True)]
    inputs = pd.DataFrame({"sample_id": sample_ids, "x1": x1, "x2": x2})
    catalog = pd.DataFrame(
        {
            "feature_name": ["x1", "x2"],
            "feature_type": ["first_order", "first_order"],
        }
    )
    holdout = pd.DataFrame({"sample_id": sample_ids, "split": ["train"] * 16 + ["holdout"] * 4})
    pca_scores = pd.DataFrame({"sample_id": sample_ids, "PC1": pca_signal})
    retained_terms = pd.DataFrame(
        {
            "feature_name": ["x1", "x2"],
            "feature_type": ["first_order", "first_order"],
        }
    )
    spec = InteractionDiscoverySpec(
        method="tree_shap_interaction_values",
        aggregation_rule="max_over_components_of_mean_absolute_shap_interaction",
        null_threshold_quantile=0.9,
        retained_pairs_reference=1,
        permutation_count_B=5,
        random_seed=123,
        n_jobs=1,
        parallel_backend="threading",
        enforce_permutation_adequacy=False,
    )
    checkpoint_root = tmp_path / "interaction_checkpoints"
    monkeypatch.setenv("RFM_PROGRESS_BATCH_SIZE", "2")

    interrupted_call_counter = {"count": 0}

    def _fail_midway_score_interaction_permutation(
        y_base: np.ndarray,
        permute_response: bool,
        *,
        n_pairs: int,
        n_comp: int,
        **_: object,
    ) -> tuple[np.ndarray, np.ndarray]:
        _ = (y_base, permute_response)
        interrupted_call_counter["count"] += 1
        if interrupted_call_counter["count"] >= 3:
            raise RuntimeError("simulated worker interruption")
        score = float(interrupted_call_counter["count"])
        return np.full(n_pairs, score), np.full((n_pairs, n_comp), score)

    monkeypatch.setattr(
        manuscript_stages,
        "_score_interaction_permutation",
        _fail_midway_score_interaction_permutation,
    )

    with pytest.raises(RuntimeError, match="parallel batch failed"):
        discover_manuscript_interactions(
            inputs,
            catalog,
            holdout,
            pca_scores,
            retained_terms,
            spec,
            checkpoint_dir=checkpoint_root,
        )

    score_files_after_interrupt = sorted(checkpoint_root.rglob("score_*.npz"))
    assert len(score_files_after_interrupt) == 2

    resumed_call_counter = {"count": 0}

    def _resume_score_interaction_permutation(
        y_base: np.ndarray,
        permute_response: bool,
        *,
        n_pairs: int,
        n_comp: int,
        **_: object,
    ) -> tuple[np.ndarray, np.ndarray]:
        _ = (y_base, permute_response)
        resumed_call_counter["count"] += 1
        score = float(resumed_call_counter["count"] + 10)
        return np.full(n_pairs, score), np.full((n_pairs, n_comp), score)

    monkeypatch.setattr(
        manuscript_stages,
        "_score_interaction_permutation",
        _resume_score_interaction_permutation,
    )

    resumed_result = discover_manuscript_interactions(
        inputs,
        catalog,
        holdout,
        pca_scores,
        retained_terms,
        spec,
        checkpoint_dir=checkpoint_root,
    )
    assert resumed_call_counter["count"] == 4
    assert len(resumed_result.pair_scores) == 1

    loaded_call_counter = {"count": 0}

    def _should_not_run_score_interaction_permutation(
        y_base: np.ndarray,
        permute_response: bool,
        *,
        n_pairs: int,
        n_comp: int,
        **_: object,
    ) -> tuple[np.ndarray, np.ndarray]:
        _ = (y_base, permute_response)
        loaded_call_counter["count"] += 1
        raise AssertionError("checkpoint resume should skip recomputation")

    monkeypatch.setattr(
        manuscript_stages,
        "_score_interaction_permutation",
        _should_not_run_score_interaction_permutation,
    )

    loaded_result = discover_manuscript_interactions(
        inputs,
        catalog,
        holdout,
        pca_scores,
        retained_terms,
        spec,
        checkpoint_dir=checkpoint_root,
    )
    assert loaded_call_counter["count"] == 0
    assert len(loaded_result.pair_scores) == 1
