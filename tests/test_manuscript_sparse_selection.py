"""Tests for manuscript sparse-selection and stability stage."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import rfm_pipeline.manuscript_stages as manuscript_stages
from rfm_pipeline.manuscript_runtime import (
    build_manuscript_notebook_context,
)
from rfm_pipeline.manuscript_stages import (
    SparseSelectionStabilitySpec,
    run_sparse_selection_stability_stage,
    select_manuscript_sparse_support,
    write_sparse_selection_stability_artifacts,
)


def test_sparse_selection_retains_stable_signal_feature_and_writes_artifacts(
    tmp_path: Path,
) -> None:
    sample_ids = list(range(1, 61))
    x1 = [float(index) for index in range(60)]
    x2 = [1.0 if index % 2 == 0 else -1.0 for index in range(60)]
    x3 = [float((index % 5) - 2) for index in range(60)]
    inputs = pd.DataFrame({"sample_id": sample_ids, "x1": x1, "x2": x2, "x3": x3})
    catalog = pd.DataFrame(
        {
            "feature_name": ["x1", "x2", "x3"],
            "feature_type": ["first_order", "first_order", "first_order"],
            "origin": ["test", "test", "test"],
        }
    )
    holdout = pd.DataFrame({"sample_id": sample_ids, "split": ["train"] * 48 + ["holdout"] * 12})
    pca_scores = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "PC1": [3.0 * value for value in x1],
            "PC2": [-1.5 * value for value in x1],
        }
    )
    retained_terms = pd.DataFrame({"feature_name": ["x1", "x2", "x3"]})
    retained_pairs = pd.DataFrame({"pair_name": []})
    retained_transformations = pd.DataFrame({"feature_name": []})
    spec = SparseSelectionStabilitySpec(
        model_class="l1_penalized_linear_model_per_retained_component",
        ebic_gamma=0.5,
        support_aggregation_rule="union_nonzero_support_across_retained_components",
        resampling_scheme="8_subsamples_of_80_percent_rows_without_replacement_seed_123",
        subsample_count=8,
        subsample_fraction=0.80,
        jaccard_threshold=0.50,
        spearman_threshold=0.50,
        random_seed=123,
    )

    result = select_manuscript_sparse_support(
        inputs,
        catalog,
        holdout,
        pca_scores,
        retained_terms,
        retained_pairs,
        retained_transformations,
        spec,
    )

    assert result.summary.loc[0, "stage"] == "sparse_selection_and_stability"
    assert result.summary.loc[0, "n_candidate_terms"] == 3
    assert "x1" in result.final_stable_support["feature_name"].tolist()
    assert result.component_model_selection["selected_support_size"].max() >= 1
    x1_row = result.stability_feature_summary.loc[
        result.stability_feature_summary["feature_name"] == "x1"
    ].iloc[0]
    assert bool(x1_row["full_support_selected"])
    assert x1_row["stability_selection_frequency"] >= 0.5

    paths = write_sparse_selection_stability_artifacts(result, tmp_path)
    assert sorted(paths) == [
        "component_coefficients",
        "component_model_selection",
        "final_stable_support",
        "sparse_selection_diagnostics",
        "sparse_selection_provenance",
        "sparse_selection_summary",
        "stability_feature_summary",
        "stability_resample_summary",
        "support_candidates",
    ]
    assert paths["final_stable_support"].read_text(encoding="utf-8").startswith("feature_name")
    provenance = pd.read_csv(paths["sparse_selection_provenance"])
    assert provenance.loc[0, "public_implementation_status"] == ("source_backed_public_surrogate")
    assert provenance.loc[0, "source_workflow_reference"] == "notebook_pca_debiased_lasso"
    assert provenance.loc[0, "source_workflow_equivalence_status"] == "not_yet_validated"


def test_sparse_selection_materializes_dynamic_terms_absent_from_static_catalog() -> None:
    sample_ids = list(range(1, 61))
    x1 = [float(index) for index in range(60)]
    x2 = [float((index % 5) + 1) for index in range(60)]
    inputs = pd.DataFrame({"sample_id": sample_ids, "x1": x1, "x2": x2})
    catalog = pd.DataFrame(
        {
            "feature_name": ["x1", "x2"],
            "feature_type": ["first_order", "first_order"],
            "origin": ["test", "test"],
        }
    )
    holdout = pd.DataFrame({"sample_id": sample_ids, "split": ["train"] * 48 + ["holdout"] * 12})
    pca_scores = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "PC1": [
                2.0 * x1_value + 1.5 * x2_value for x1_value, x2_value in zip(x1, x2, strict=True)
            ],
        }
    )
    retained_terms = pd.DataFrame({"feature_name": ["x1", "x2"]})
    retained_pairs = pd.DataFrame({"pair_name": ["x1:x2"]})
    retained_transformations = pd.DataFrame({"feature_name": ["x2_inv"]})
    spec = SparseSelectionStabilitySpec(
        model_class="l1_penalized_linear_model_per_retained_component",
        ebic_gamma=0.5,
        support_aggregation_rule="union_nonzero_support_across_retained_components",
        resampling_scheme="8_subsamples_of_80_percent_rows_without_replacement_seed_123",
        subsample_count=8,
        subsample_fraction=0.80,
        jaccard_threshold=0.50,
        spearman_threshold=0.50,
        random_seed=123,
    )

    result = select_manuscript_sparse_support(
        inputs,
        catalog,
        holdout,
        pca_scores,
        retained_terms,
        retained_pairs,
        retained_transformations,
        spec,
    )

    assert result.summary.loc[0, "n_candidate_terms"] == 4
    support_candidates = result.support_candidates.set_index("feature_name")
    assert support_candidates.loc["x1:x2", "feature_type"] == "interaction"
    assert support_candidates.loc["x2_inv", "feature_type"] == "transformation"


def test_sparse_selection_applies_deterministic_top_k_candidate_cap() -> None:
    sample_ids = list(range(1, 61))
    x1 = [float(index) for index in range(60)]
    x2 = [float((index % 5) + 1) for index in range(60)]
    inputs = pd.DataFrame({"sample_id": sample_ids, "x1": x1, "x2": x2})
    catalog = pd.DataFrame(
        {
            "feature_name": ["x1", "x2"],
            "feature_type": ["first_order", "first_order"],
            "origin": ["test", "test"],
        }
    )
    holdout = pd.DataFrame({"sample_id": sample_ids, "split": ["train"] * 48 + ["holdout"] * 12})
    pca_scores = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "PC1": [
                2.0 * x1_value + 1.5 * x2_value for x1_value, x2_value in zip(x1, x2, strict=True)
            ],
        }
    )
    retained_terms = pd.DataFrame(
        {
            "feature_name": ["x1", "x2"],
            "empirical_p_value": [1.0e-6, 0.2],
            "observed_statistic": [100.0, 1.0],
        }
    )
    retained_pairs = pd.DataFrame(
        {
            "pair_name": ["x1:x2"],
            "empirical_p_value": [0.05],
            "observed_score": [2.0],
        }
    )
    retained_transformations = pd.DataFrame(
        {
            "feature_name": ["x2_inv"],
            "empirical_p_value": [0.9],
            "curvature_score": [0.1],
        }
    )
    spec = SparseSelectionStabilitySpec(
        model_class="l1_penalized_linear_model_per_retained_component",
        ebic_gamma=0.5,
        support_aggregation_rule="union_nonzero_support_across_retained_components",
        resampling_scheme="8_subsamples_of_80_percent_rows_without_replacement_seed_123",
        subsample_count=8,
        subsample_fraction=0.80,
        jaccard_threshold=0.50,
        spearman_threshold=0.50,
        random_seed=123,
        max_candidate_terms=2,
    )

    result = select_manuscript_sparse_support(
        inputs,
        catalog,
        holdout,
        pca_scores,
        retained_terms,
        retained_pairs,
        retained_transformations,
        spec,
    )

    assert int(result.summary.loc[0, "n_candidate_terms"]) == 2
    assert set(result.support_candidates["feature_name"]) == {"x1", "x1:x2"}


def test_run_sparse_selection_stability_stage_executes_demo_context() -> None:
    context = build_manuscript_notebook_context(
        Path.cwd(),
        "06_sparse_selection_and_stability.ipynb",
    )

    result = run_sparse_selection_stability_stage(context)

    assert result.sparse_selection.summary.loc[0, "stage"] == "sparse_selection_and_stability"
    assert result.sparse_selection.summary.loc[0, "n_candidate_terms"] >= 1
    assert result.sparse_selection.summary.loc[0, "n_final_stable_support_terms"] >= 1
    assert not result.sparse_selection.final_stable_support.empty
    assert result.artifact_paths["support_candidates"].exists()


def test_demo_sparse_selection_final_stable_support_nonempty_ci_parity_guard() -> None:
    """Guardrail: demo fixture must produce non-empty final_stable_support on all platforms.

    This test exists to catch the CI/local parity failure mode where GitHub Linux produced
    empty final_stable_support while macOS tests passed.  It fires before downstream
    final-artifact and reproduction-chain tests can obscure the root cause.
    """
    context = build_manuscript_notebook_context(
        Path.cwd(),
        "06_sparse_selection_and_stability.ipynb",
    )

    result = run_sparse_selection_stability_stage(context)
    ss = result.sparse_selection

    summary = ss.summary.loc[0]
    n_candidates = int(summary["n_candidate_terms"])
    n_full = int(summary["n_full_support_terms"])
    n_final = int(summary["n_final_stable_support_terms"])

    feat = ss.stability_feature_summary
    mean_jaccard = float(feat["mean_resample_jaccard"].mean()) if len(feat) else float("nan")
    mean_spearman = float(feat["mean_resample_spearman"].mean()) if len(feat) else float("nan")
    passes_jaccard = bool(feat["passes_jaccard_threshold"].all()) if len(feat) else False
    passes_spearman = bool(feat["passes_spearman_threshold"].all()) if len(feat) else False

    diagnostic = (
        f"n_candidate_terms={n_candidates}, "
        f"n_full_support_terms={n_full}, "
        f"n_final_stable_support_terms={n_final}, "
        f"mean_resample_jaccard={mean_jaccard:.3f}, "
        f"mean_resample_spearman={mean_spearman:.3f}, "
        f"all_pass_jaccard={passes_jaccard}, "
        f"all_pass_spearman={passes_spearman}"
    )

    assert n_candidates > 0, f"No candidate terms reached sparse selection. {diagnostic}"
    assert n_full > 0, (
        f"EBIC/L1 full support is empty — no feature survived L1 selection on any component. "
        f"{diagnostic}"
    )
    assert n_final > 0, (
        f"final_stable_support is empty after stability filtering — "
        f"check Jaccard/Spearman thresholds or fixture signal strength. {diagnostic}"
    )
    assert not ss.final_stable_support.empty, (
        f"final_stable_support DataFrame is empty. {diagnostic}"
    )


def test_sparse_selection_reuses_checkpointed_resamples(
    monkeypatch,
    tmp_path: Path,
) -> None:
    sample_ids = list(range(1, 61))
    x1 = [float(index) for index in range(60)]
    x2 = [1.0 if index % 2 == 0 else -1.0 for index in range(60)]
    x3 = [float((index % 5) - 2) for index in range(60)]
    inputs = pd.DataFrame({"sample_id": sample_ids, "x1": x1, "x2": x2, "x3": x3})
    catalog = pd.DataFrame(
        {
            "feature_name": ["x1", "x2", "x3"],
            "feature_type": ["first_order", "first_order", "first_order"],
            "origin": ["test", "test", "test"],
        }
    )
    holdout = pd.DataFrame({"sample_id": sample_ids, "split": ["train"] * 48 + ["holdout"] * 12})
    pca_scores = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "PC1": [3.0 * value for value in x1],
            "PC2": [-1.5 * value for value in x1],
        }
    )
    retained_terms = pd.DataFrame({"feature_name": ["x1", "x2", "x3"]})
    retained_pairs = pd.DataFrame({"pair_name": []})
    retained_transformations = pd.DataFrame({"feature_name": []})
    spec = SparseSelectionStabilitySpec(
        model_class="l1_penalized_linear_model_per_retained_component",
        ebic_gamma=0.5,
        support_aggregation_rule="union_nonzero_support_across_retained_components",
        resampling_scheme="6_subsamples_of_80_percent_rows_without_replacement_seed_123",
        subsample_count=6,
        subsample_fraction=0.80,
        jaccard_threshold=0.50,
        spearman_threshold=0.50,
        random_seed=123,
        n_jobs=1,
    )

    call_counter = {"count": 0}

    def _fake_run_one_stability_resample(
        resample_id: int,
        row_indices: np.ndarray,
        x_scaled: np.ndarray,
        y_scaled: np.ndarray,
        feature_names: list[str],
        component_names: list[str],
        spec: SparseSelectionStabilitySpec,
        active_features: np.ndarray,
        full_support_mask: np.ndarray,
        full_importance: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        _ = (row_indices, y_scaled, component_names, spec, active_features, full_support_mask)
        call_counter["count"] += 1
        support = np.zeros(len(feature_names), dtype=bool)
        support[0] = True
        importance = np.zeros(len(feature_names), dtype=float)
        importance[0] = float(full_importance.max() if len(full_importance) else 1.0)
        summary_row = {
            "resample_id": int(resample_id),
            "subsample_size": int(len(x_scaled)),
            "selected_support_size": int(support.sum()),
            "jaccard_with_full_support": 1.0,
            "spearman_with_full_importance": 1.0,
        }
        return support, importance, summary_row

    monkeypatch.setattr(
        manuscript_stages,
        "_run_one_stability_resample",
        _fake_run_one_stability_resample,
    )
    checkpoint_root = tmp_path / "sparse_selection"
    first = select_manuscript_sparse_support(
        inputs,
        catalog,
        holdout,
        pca_scores,
        retained_terms,
        retained_pairs,
        retained_transformations,
        spec,
        checkpoint_dir=checkpoint_root,
    )
    assert call_counter["count"] > 0

    def _should_not_recompute(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("checkpointed stability resamples should be reused")

    monkeypatch.setattr(
        manuscript_stages,
        "_run_one_stability_resample",
        _should_not_recompute,
    )
    second = select_manuscript_sparse_support(
        inputs,
        catalog,
        holdout,
        pca_scores,
        retained_terms,
        retained_pairs,
        retained_transformations,
        spec,
        checkpoint_dir=checkpoint_root,
    )
    assert first.summary.equals(second.summary)


def test_sparse_selection_supports_active_resample_subsets(
    monkeypatch,
    tmp_path: Path,
) -> None:
    sample_ids = list(range(1, 61))
    x1 = [float(index) for index in range(60)]
    x2 = [1.0 if index % 2 == 0 else -1.0 for index in range(60)]
    x3 = [float((index % 5) - 2) for index in range(60)]
    inputs = pd.DataFrame({"sample_id": sample_ids, "x1": x1, "x2": x2, "x3": x3})
    catalog = pd.DataFrame(
        {
            "feature_name": ["x1", "x2", "x3"],
            "feature_type": ["first_order", "first_order", "first_order"],
            "origin": ["test", "test", "test"],
        }
    )
    holdout = pd.DataFrame({"sample_id": sample_ids, "split": ["train"] * 48 + ["holdout"] * 12})
    pca_scores = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "PC1": [3.0 * value for value in x1],
            "PC2": [-1.5 * value for value in x1],
        }
    )
    retained_terms = pd.DataFrame({"feature_name": ["x1", "x2", "x3"]})
    retained_pairs = pd.DataFrame({"pair_name": []})
    retained_transformations = pd.DataFrame({"feature_name": []})
    spec = SparseSelectionStabilitySpec(
        model_class="l1_penalized_linear_model_per_retained_component",
        ebic_gamma=0.5,
        support_aggregation_rule="union_nonzero_support_across_retained_components",
        resampling_scheme="6_subsamples_of_80_percent_rows_without_replacement_seed_123",
        subsample_count=6,
        subsample_fraction=0.80,
        jaccard_threshold=0.50,
        spearman_threshold=0.50,
        random_seed=123,
        n_jobs=1,
    )

    call_counter = {"count": 0}

    def _fake_run_one_stability_resample(
        resample_id: int,
        row_indices: np.ndarray,
        x_scaled: np.ndarray,
        y_scaled: np.ndarray,
        feature_names: list[str],
        component_names: list[str],
        spec: SparseSelectionStabilitySpec,
        active_features: np.ndarray,
        full_support_mask: np.ndarray,
        full_importance: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        _ = (row_indices, y_scaled, component_names, spec, active_features, full_support_mask)
        call_counter["count"] += 1
        support = np.zeros(len(feature_names), dtype=bool)
        support[0] = True
        importance = np.zeros(len(feature_names), dtype=float)
        importance[0] = float(full_importance.max() if len(full_importance) else 1.0)
        summary_row = {
            "resample_id": int(resample_id),
            "subsample_size": int(len(x_scaled)),
            "selected_support_size": int(support.sum()),
            "jaccard_with_full_support": 1.0,
            "spearman_with_full_importance": 1.0,
        }
        return support, importance, summary_row

    monkeypatch.setattr(
        manuscript_stages,
        "_run_one_stability_resample",
        _fake_run_one_stability_resample,
    )
    checkpoint_root = tmp_path / "sparse_selection_subset"
    active_subset = {1, 4}

    select_manuscript_sparse_support(
        inputs,
        catalog,
        holdout,
        pca_scores,
        retained_terms,
        retained_pairs,
        retained_transformations,
        spec,
        checkpoint_dir=checkpoint_root,
        active_resample_indices=active_subset,
    )
    assert call_counter["count"] == len(active_subset)

    def _should_not_recompute(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("checkpointed subset resamples should be reused")

    monkeypatch.setattr(
        manuscript_stages,
        "_run_one_stability_resample",
        _should_not_recompute,
    )
    select_manuscript_sparse_support(
        inputs,
        catalog,
        holdout,
        pca_scores,
        retained_terms,
        retained_pairs,
        retained_transformations,
        spec,
        checkpoint_dir=checkpoint_root,
        active_resample_indices=active_subset,
    )
