"""Tests for canonical manuscript interaction discovery."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import rfm_pipeline.manuscript_stages as manuscript_stages
from rfm_pipeline.interaction_contract import canonical_execution_contract_from_specs
from rfm_pipeline.manuscript_runtime import build_manuscript_notebook_context
from rfm_pipeline.manuscript_stages import (
    InteractionDiscoverySpec,
    discover_interaction_scores_only,
    discover_manuscript_interactions,
    reduce_interaction_draw_blocks,
    run_interaction_discovery_stage,
    score_interaction_draw_block,
    write_interaction_discovery_artifacts,
)

DRAW_COUNT = 199


def _spec(
    *,
    draws: int = DRAW_COUNT,
    **overrides: object,
) -> InteractionDiscoverySpec:
    values: dict[str, object] = {
        "method": "tree_shap_interaction_values",
        "aggregation_rule": "max_over_components_of_mean_absolute_shap_interaction",
        "null_threshold_quantile": 0.95,
        "retained_pairs_reference": 0,
        "permutation_count_B": draws,
        "random_seed": 123,
        "n_jobs": 1,
        "selection_method": "max_t",
        "selection_alpha": 0.05,
        "minimum_selection_draws": DRAW_COUNT,
    }
    values.update(overrides)
    return InteractionDiscoverySpec(**values)


def _fixture(
    *,
    n_features: int = 2,
    n_rows: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sample_ids = np.arange(1, n_rows + 1)
    values = {
        "x1": np.resize(np.array([-1.0, -1.0, 1.0, 1.0]), n_rows),
        "x2": np.resize(np.array([-1.0, 1.0, -1.0, 1.0]), n_rows),
        "x3": np.resize(np.array([1.0, -1.0]), n_rows),
    }
    feature_names = [f"x{index}" for index in range(1, n_features + 1)]
    inputs = pd.DataFrame(
        {"sample_id": sample_ids, **{name: values[name] for name in feature_names}}
    )
    catalog = pd.DataFrame(
        {
            "feature_name": feature_names,
            "feature_type": ["first_order"] * len(feature_names),
        }
    )
    holdout = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "split": ["train"] * (n_rows - 4) + ["holdout"] * 4,
        }
    )
    pca_scores = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "PC1": inputs["x1"] * inputs["x2"],
        }
    )
    return inputs, catalog, holdout, pca_scores, catalog.copy()


def _fake_score(
    y_base: np.ndarray,
    permute_response: bool,
    *,
    n_pairs: int,
    n_comp: int,
    **_: object,
) -> tuple[np.ndarray, np.ndarray]:
    _ = y_base
    score = 0.1 if permute_response else 0.8
    scores = np.full(n_pairs, score, dtype=float)
    return scores, np.full((n_pairs, n_comp), score, dtype=float)


def test_interaction_discovery_requires_canonical_draw_adequacy() -> None:
    """The in-memory route refuses to start below the canonical draw floor."""
    inputs, catalog, holdout, pca_scores, retained_terms = _fixture()
    spec = _spec(draws=DRAW_COUNT - 1)

    with pytest.raises(ValueError, match="draw adequacy"):
        discover_manuscript_interactions(
            inputs,
            catalog,
            holdout,
            pca_scores,
            retained_terms,
            spec,
        )


def test_interaction_discovery_rejects_legacy_or_noncanonical_selectors() -> None:
    """The neutral maxT selector has no legacy or FDR alias."""
    with pytest.raises(ValueError, match="selection_method"):
        _spec(selection_method="fwer_max_stat_exact")
    with pytest.raises(ValueError, match="selection_method"):
        _spec(selection_method="bh_fdr")


def test_interaction_discovery_retains_signal_and_writes_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    inputs, catalog, holdout, pca_scores, retained_terms = _fixture()
    monkeypatch.setattr(manuscript_stages, "_score_interaction_permutation", _fake_score)

    result = discover_manuscript_interactions(
        inputs,
        catalog,
        holdout,
        pca_scores,
        retained_terms,
        _spec(),
    )

    assert result.summary.loc[0, "stage"] == "interaction_discovery"
    assert result.summary.loc[0, "status"] == "completed"
    assert result.summary.loc[0, "n_candidate_pairs"] == 1
    assert result.summary.loc[0, "n_retained_pairs"] == 1
    assert result.pair_scores.loc[0, "pair_name"] == "x1:x2"
    assert result.pair_scores.loc[0, "retained"]
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


def test_interaction_discovery_generates_full_candidate_family(monkeypatch) -> None:
    inputs, catalog, holdout, pca_scores, retained_terms = _fixture(n_features=3)
    monkeypatch.setattr(manuscript_stages, "_score_interaction_permutation", _fake_score)

    result = discover_manuscript_interactions(
        inputs,
        catalog,
        holdout,
        pca_scores,
        retained_terms,
        _spec(),
    )

    assert result.pair_scores["pair_name"].tolist() == ["x1:x2", "x1:x3", "x2:x3"]
    assert result.summary.loc[0, "n_candidate_pairs"] == 3


def test_draw_blocks_keep_the_complete_candidate_family(monkeypatch) -> None:
    inputs, catalog, holdout, pca_scores, retained_terms = _fixture(n_features=3)
    spec = _spec()
    contract = canonical_execution_contract_from_specs(spec)
    monkeypatch.setattr(manuscript_stages, "_score_interaction_permutation", _fake_score)

    first = score_interaction_draw_block(
        inputs,
        catalog,
        holdout,
        pca_scores,
        retained_terms,
        spec,
        draw_start=0,
        draw_end=100,
        contract=contract,
    )
    second = score_interaction_draw_block(
        inputs,
        catalog,
        holdout,
        pca_scores,
        retained_terms,
        spec,
        draw_start=100,
        draw_end=DRAW_COUNT,
        contract=contract,
    )
    assert first.pair_names == ("x1:x2", "x1:x3", "x2:x3")
    assert second.pair_names == first.pair_names
    assert reduce_interaction_draw_blocks(
        [first, second], spec=spec, contract=contract
    ).null_scores.shape == (DRAW_COUNT, 3)


def test_score_only_discovery_rejects_empty_draw_block() -> None:
    inputs, catalog, holdout, pca_scores, retained_terms = _fixture()
    spec = _spec()

    with pytest.raises(ValueError, match="draw block"):
        score_interaction_draw_block(
            inputs,
            catalog,
            holdout,
            pca_scores,
            retained_terms,
            spec,
            draw_start=5,
            draw_end=5,
            contract=canonical_execution_contract_from_specs(spec),
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


def test_score_only_discovery_rejects_invalid_parallel_backend() -> None:
    inputs, catalog, holdout, pca_scores, retained_terms = _fixture()
    spec = _spec(n_jobs=2, parallel_backend="invalid_backend")

    with pytest.raises(ValueError, match="parallel_backend"):
        discover_interaction_scores_only(
            inputs,
            catalog,
            holdout,
            pca_scores,
            retained_terms,
            spec,
            contract=canonical_execution_contract_from_specs(spec),
        )


def test_score_only_discovery_uses_configured_threading_backend(monkeypatch) -> None:
    inputs, catalog, holdout, pca_scores, retained_terms = _fixture()
    spec = _spec(n_jobs=2, parallel_backend="threading")
    calls: list[tuple[int, str]] = []

    class FakeParallel:
        def __init__(self, n_jobs: int, **kwargs: object) -> None:
            self.n_jobs = n_jobs
            self.backend = str(kwargs.get("backend", "loky"))

        def __enter__(self) -> FakeParallel:
            return self

        def __exit__(self, exc_type, exc, traceback) -> bool:  # noqa: ANN001
            return False

        def __call__(self, jobs):
            calls.append((self.n_jobs, self.backend))
            return [function(*args, **kwargs) for function, args, kwargs in jobs]

    monkeypatch.setattr(manuscript_stages, "Parallel", FakeParallel)
    monkeypatch.setattr(manuscript_stages, "_score_interaction_permutation", _fake_score)

    result = discover_manuscript_interactions(
        inputs,
        catalog,
        holdout,
        pca_scores,
        retained_terms,
        spec,
    )

    assert calls
    assert all(call == (2, "threading") for call in calls)
    assert len(result.pair_scores) == 1


def test_score_only_discovery_uses_dask_without_fallback(monkeypatch) -> None:
    """Dask dispatch is used directly and an executor error is not recovered."""
    inputs, catalog, holdout, pca_scores, retained_terms = _fixture()
    spec = _spec(n_jobs=2, parallel_backend="dask")

    class FakeExecutor:
        def __init__(self) -> None:
            self.closed = False

        def map(self, function, items):
            return [function(item) for item in items]

        def close(self) -> None:
            self.closed = True

    executor = FakeExecutor()
    monkeypatch.setattr(
        manuscript_stages,
        "get_executor",
        lambda backend, **kwargs: executor,
    )
    monkeypatch.setattr(manuscript_stages, "_score_interaction_permutation", _fake_score)
    artifact = discover_interaction_scores_only(
        inputs,
        catalog,
        holdout,
        pca_scores,
        retained_terms,
        spec,
        contract=canonical_execution_contract_from_specs(spec),
    )
    assert artifact.status == "score_only_completed"
    assert executor.closed

    monkeypatch.setattr(
        manuscript_stages,
        "get_executor",
        lambda backend, **kwargs: (_ for _ in ()).throw(RuntimeError("dask unavailable")),
    )
    with pytest.raises(RuntimeError, match="dask unavailable"):
        discover_interaction_scores_only(
            inputs,
            catalog,
            holdout,
            pca_scores,
            retained_terms,
            spec,
            contract=canonical_execution_contract_from_specs(spec),
        )


def test_score_only_discovery_resumes_valid_checkpoints_and_propagates_failures(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """A failed score raises; only valid completed checkpoints are reused."""
    inputs, catalog, holdout, pca_scores, retained_terms = _fixture()
    spec = _spec(parallel_backend="threading")
    checkpoint_root = tmp_path / "interaction_checkpoints"
    monkeypatch.setenv("RFM_PROGRESS_BATCH_SIZE", "2")
    interrupted_calls = {"count": 0}

    def _fail_midway(
        y_base: np.ndarray,
        permute_response: bool,
        *,
        n_pairs: int,
        n_comp: int,
        **_: object,
    ) -> tuple[np.ndarray, np.ndarray]:
        _ = y_base, permute_response
        interrupted_calls["count"] += 1
        if interrupted_calls["count"] >= 3:
            raise RuntimeError("simulated worker interruption")
        score = float(interrupted_calls["count"])
        return np.full(n_pairs, score), np.full((n_pairs, n_comp), score)

    monkeypatch.setattr(manuscript_stages, "_score_interaction_permutation", _fail_midway)
    contract = canonical_execution_contract_from_specs(spec)
    with pytest.raises(RuntimeError, match="simulated worker interruption"):
        discover_interaction_scores_only(
            inputs,
            catalog,
            holdout,
            pca_scores,
            retained_terms,
            spec,
            checkpoint_dir=checkpoint_root,
            contract=contract,
        )

    assert len(list(checkpoint_root.rglob("score_*.npz"))) == 2
    resumed_calls = {"count": 0}

    def _resume(
        y_base: np.ndarray,
        permute_response: bool,
        *,
        n_pairs: int,
        n_comp: int,
        **_: object,
    ) -> tuple[np.ndarray, np.ndarray]:
        _ = y_base, permute_response
        resumed_calls["count"] += 1
        score = float(resumed_calls["count"] + 10)
        return np.full(n_pairs, score), np.full((n_pairs, n_comp), score)

    monkeypatch.setattr(manuscript_stages, "_score_interaction_permutation", _resume)
    artifact = discover_interaction_scores_only(
        inputs,
        catalog,
        holdout,
        pca_scores,
        retained_terms,
        spec,
        checkpoint_dir=checkpoint_root,
        contract=contract,
    )
    assert resumed_calls["count"] == DRAW_COUNT - 1
    assert artifact.status == "score_only_completed"

    def _should_not_run(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("completed score checkpoints must be reused")

    monkeypatch.setattr(manuscript_stages, "_score_interaction_permutation", _should_not_run)
    loaded = discover_interaction_scores_only(
        inputs,
        catalog,
        holdout,
        pca_scores,
        retained_terms,
        spec,
        checkpoint_dir=checkpoint_root,
        contract=contract,
    )
    assert loaded.status == "score_only_completed"


def _uniform_inputs(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=n), rng.uniform(-1.0, 1.0, size=n)


def test_condition_pca_scores_removes_additive_main_effects_preserves_interactions() -> None:
    """Additive linear+quadratic main effects are residualized out; interactions survive."""
    n = 400
    sample_ids = list(range(1, n + 1))
    x1, x2 = _uniform_inputs(n, seed=7)
    inputs = pd.DataFrame({"sample_id": sample_ids, "x1": x1, "x2": x2})
    holdout = pd.DataFrame({"sample_id": sample_ids, "split": ["train"] * n})
    retained_terms = pd.DataFrame(
        {"feature_name": ["x1", "x2"], "feature_type": ["first_order"] * 2}
    )
    main_effect = 3.0 * x1 + 2.0 * (x1**2)
    interaction = 5.0 * x1 * x2
    pca_scores = pd.DataFrame(
        {"sample_id": sample_ids, "PC_main": main_effect, "PC_int": interaction}
    )

    conditioned = manuscript_stages._condition_pca_scores_on_main_effects(
        inputs,
        holdout,
        pca_scores,
        retained_terms,
        degree=2,
    )

    assert float(np.var(conditioned["PC_main"].to_numpy())) < 1e-6 * float(np.var(main_effect))
    assert float(np.var(conditioned["PC_int"].to_numpy())) > 0.9 * float(np.var(interaction))


def test_condition_pca_scores_degree_one_leaves_quadratic_curvature() -> None:
    """Degree=1 removes only linear structure; quadratic curvature remains."""
    n = 400
    sample_ids = list(range(1, n + 1))
    x1, _ = _uniform_inputs(n, seed=11)
    inputs = pd.DataFrame({"sample_id": sample_ids, "x1": x1})
    holdout = pd.DataFrame({"sample_id": sample_ids, "split": ["train"] * n})
    retained_terms = pd.DataFrame({"feature_name": ["x1"], "feature_type": ["first_order"]})
    curvature = 2.0 * (x1**2)
    pca_scores = pd.DataFrame({"sample_id": sample_ids, "PC1": curvature})

    conditioned = manuscript_stages._condition_pca_scores_on_main_effects(
        inputs,
        holdout,
        pca_scores,
        retained_terms,
        degree=1,
    )
    assert float(np.var(conditioned["PC1"].to_numpy())) > 0.5 * float(np.var(curvature))


def test_condition_pca_scores_degree_zero_returns_unchanged() -> None:
    """Degree < 1 disables conditioning and returns the input unchanged."""
    n = 20
    sample_ids = list(range(1, n + 1))
    x1, _ = _uniform_inputs(n, seed=3)
    inputs = pd.DataFrame({"sample_id": sample_ids, "x1": x1})
    holdout = pd.DataFrame({"sample_id": sample_ids, "split": ["train"] * n})
    retained_terms = pd.DataFrame({"feature_name": ["x1"], "feature_type": ["first_order"]})
    pca_scores = pd.DataFrame({"sample_id": sample_ids, "PC1": x1})

    out = manuscript_stages._condition_pca_scores_on_main_effects(
        inputs,
        holdout,
        pca_scores,
        retained_terms,
        degree=0,
    )
    pd.testing.assert_frame_equal(out, pca_scores)


def test_interaction_spec_conditions_main_effects_by_default() -> None:
    """The corrected main-effect conditioning is enabled by default."""
    spec = _spec()
    assert spec.condition_main_effects is True
    assert spec.main_effect_conditioning_degree == 2
