"""Tests for canonical score-only HPC interaction reduction."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from rfm_pipeline.distributed.checkpoint import CheckpointManager
from rfm_pipeline.distributed.manifest import ShardManifest
from rfm_pipeline.interaction_contract import (
    ScoreOnlyInteractionArtifact,
    build_control_snapshot,
    canonical_execution_contract_from_specs,
)
from rfm_pipeline.manuscript_stages import (
    InteractionDiscoverySpec,
    discover_interaction_scores_only,
    reduce_score_only_interaction_artifacts,
    score_interaction_draw_block,
)

DRAW_COUNT = 199


def _spec(
    *,
    draws: int = DRAW_COUNT,
    random_seed: int = 123,
    **overrides: object,
) -> InteractionDiscoverySpec:
    return InteractionDiscoverySpec(
        method="tree_shap_interaction_values",
        aggregation_rule="max_over_components_of_mean_absolute_shap_interaction",
        null_threshold_quantile=0.9,
        retained_pairs_reference=0,
        permutation_count_B=draws,
        random_seed=random_seed,
        selection_method="max_t",
        selection_alpha=0.05,
        minimum_selection_draws=DRAW_COUNT,
        **overrides,
    )


def _contract_and_snapshot(
    pair_names: tuple[str, ...],
    spec: InteractionDiscoverySpec,
):
    contract = canonical_execution_contract_from_specs(spec)
    snapshot = build_control_snapshot(
        contract,
        candidate_pair_names=pair_names,
        training_sample_ids=np.arange(1, 9, dtype=np.int64),
        feature_matrix=np.arange(24, dtype=float).reshape(8, 3),
        response_matrix=np.arange(16, dtype=float).reshape(8, 2),
        component_names=("PC1", "PC2"),
    )
    return contract, snapshot


def _artifact(
    snapshot,
    *,
    start: int,
    end: int,
    observed: np.ndarray | None = None,
    null: np.ndarray | None = None,
) -> ScoreOnlyInteractionArtifact:
    width = len(snapshot.candidate_pair_names)
    return ScoreOnlyInteractionArtifact(
        status="score_only_completed",
        draw_range_start=start,
        draw_range_end=end,
        pair_names=snapshot.candidate_pair_names,
        observed_scores=(
            np.asarray(observed, dtype=float)
            if observed is not None
            else (
                np.linspace(0.6, 0.9, width, dtype=float)
                if start == 0
                else np.empty(0, dtype=float)
            )
        ),
        null_scores=(
            np.zeros((end - start, width), dtype=float)
            if null is None
            else np.asarray(null, dtype=float)
        ),
        draw_ids=np.arange(start, end, dtype=np.int64),
        control_snapshot=snapshot,
    )


def _shard_metadata(shard_id: str, artifact: ScoreOnlyInteractionArtifact) -> dict[str, object]:
    return {
        "shard_id": shard_id,
        "stage": "interaction_discovery",
        "shard_mode": "score_only",
        "status": artifact.status,
        "draw_range_start": artifact.draw_range_start,
        "draw_range_end": artifact.draw_range_end,
        "control_snapshot_sha256": artifact.control_snapshot.checksum,
        "candidate_family_sha256": artifact.control_snapshot.candidate_family_sha256,
        "candidate_family_count": len(artifact.control_snapshot.candidate_pair_names),
    }


def _write_score_only_shard(
    output_root,
    shard_id: str,
    artifact: ScoreOnlyInteractionArtifact,
) -> dict[str, object]:
    artifact.write_to(output_root / shard_id)
    return _shard_metadata(shard_id, artifact)


def _patch_hpc_reducer(
    monkeypatch: pytest.MonkeyPatch,
    *,
    contract,
    spec: InteractionDiscoverySpec,
) -> None:
    from rfm_pipeline import hpc_reduce

    monkeypatch.setattr(hpc_reduce, "load_canonical_execution_contract", lambda _: contract)
    monkeypatch.setattr(hpc_reduce, "load_config", lambda _: object())
    monkeypatch.setattr(hpc_reduce, "apply_fast_mode_overrides", lambda value: value)
    monkeypatch.setattr(hpc_reduce, "config_to_legacy_case_study", lambda _: object())
    monkeypatch.setattr(
        hpc_reduce,
        "interaction_discovery_spec_from_case_study_config",
        lambda _: spec,
    )


def test_hpc_shard_worker_writes_self_verifying_score_only_artifacts(
    tmp_path,
    monkeypatch,
) -> None:
    """Interaction workers emit only the canonical persisted score artifact."""
    from rfm_pipeline import hpc_shard_worker

    x_path = tmp_path / "X.parquet"
    holdout_path = tmp_path / "holdout_assignments.parquet"
    catalog_path = tmp_path / "actual_input_feature_catalog.parquet"
    pca_scores_path = tmp_path / "pca_scores.csv"
    retained_terms_path = tmp_path / "retained_terms.csv"
    x_df = pd.DataFrame(
        {
            "sample_id": list(range(1, 11)),
            "x1": [0.0, 1.0] * 5,
            "x2": [1.0, 0.0] * 5,
        }
    )
    x_df.to_parquet(x_path)
    pd.DataFrame(
        {"sample_id": list(range(1, 11)), "split": ["train"] * 8 + ["holdout"] * 2}
    ).to_parquet(holdout_path)
    catalog = pd.DataFrame(
        {
            "feature_name": ["x1", "x2"],
            "feature_type": ["first_order", "first_order"],
        }
    )
    catalog.to_parquet(catalog_path)
    pd.DataFrame({"sample_id": list(range(1, 11)), "PC1": [0.0, 1.0] * 5}).to_csv(
        pca_scores_path,
        index=False,
    )
    catalog.to_csv(retained_terms_path, index=False)

    shard = ShardManifest(
        shard_id="task-0000",
        stage="interaction_discovery",
        input_paths=[
            str(x_path),
            str(holdout_path),
            str(catalog_path),
            str(pca_scores_path),
            str(retained_terms_path),
        ],
        output_path=str(tmp_path / "out" / "task-0000"),
        expected_rows=10,
        expected_columns=2,
        feature_start_idx=0,
        feature_end_idx=DRAW_COUNT,
    )
    spec = _spec()
    contract, snapshot = _contract_and_snapshot(("x1:x2",), spec)
    artifact = _artifact(snapshot, start=0, end=DRAW_COUNT)
    monkeypatch.setattr(hpc_shard_worker, "_load_interaction_spec", lambda _: spec)
    monkeypatch.setattr(hpc_shard_worker, "load_canonical_execution_contract", lambda _: contract)
    discover_call: dict[str, object] = {}

    def _fake_discover(*args, **kwargs):  # noqa: ANN002, ANN003
        discover_call.update(kwargs)
        return artifact

    monkeypatch.setattr(hpc_shard_worker, "score_interaction_draw_block", _fake_discover)
    cm = CheckpointManager(str(tmp_path / "out"), shard.shard_id)
    cm.mark_running()
    hpc_shard_worker._run_shard_stage(
        shard,
        cm,
        SimpleNamespace(config="canonical.yml"),
    )

    result = json.loads((cm.final_dir / "shard_result.json").read_text(encoding="utf-8"))
    assert result["shard_mode"] == "score_only"
    assert result["control_snapshot_sha256"] == snapshot.checksum
    assert result["candidate_family_count"] == 1
    assert (cm.final_dir / "score_only_interaction.npz").is_file()
    assert (cm.final_dir / "score_only_interaction.json").is_file()
    assert not (cm.final_dir / "retained_interaction_pairs.csv").exists()
    assert discover_call["draw_start"] == 0
    assert discover_call["draw_end"] == DRAW_COUNT
    assert discover_call["contract"] is contract


def test_hpc_reduce_merges_verified_artifacts(tmp_path, monkeypatch) -> None:
    """The reducer makes one global decision from persisted canonical artifacts."""
    from rfm_pipeline import hpc_reduce

    output_root = tmp_path / "outputs"
    merged_dir = tmp_path / "merged"
    merged_dir.mkdir()
    spec = _spec()
    contract, snapshot = _contract_and_snapshot(("x1:x2", "x1:x3"), spec)
    artifacts = [
        _artifact(snapshot, start=0, end=100),
        _artifact(snapshot, start=100, end=DRAW_COUNT),
    ]
    shard_results = [
        _write_score_only_shard(output_root, f"task-{index:04d}", artifact)
        for index, artifact in enumerate(artifacts)
    ]
    _patch_hpc_reducer(monkeypatch, contract=contract, spec=spec)

    hpc_reduce._reduce_interaction_discovery(
        shard_results,
        merged_dir,
        output_root,
        config_path="canonical.yml",
    )

    scores = pd.read_csv(merged_dir / "interaction_pair_scores_merged.csv")
    summary = json.loads(
        (merged_dir / "interaction_discovery_merged.json").read_text(encoding="utf-8")
    )
    assert set(scores["pair_name"]) == {"x1:x2", "x1:x3"}
    assert summary["reducer"] == "canonical_score_only_family_decision"
    assert summary["status"] == "completed"
    assert summary["canonical_hashes"]["control_snapshot_sha256"] == snapshot.checksum


def test_empty_interaction_family_builds_one_terminal_hpc_shard(tmp_path, monkeypatch) -> None:
    """The HPC path represents a globally empty family with one terminal artifact."""
    from rfm_pipeline import hpc_submit
    from rfm_pipeline.distributed import manifest

    artifact_dir = tmp_path / "artifacts"
    retained_terms_dir = artifact_dir / "empirical_null_screen"
    retained_terms_dir.mkdir(parents=True)
    pd.DataFrame(columns=["feature_name", "feature_type"]).to_csv(
        retained_terms_dir / "retained_terms.csv",
        index=False,
    )

    monkeypatch.setattr(
        manifest,
        "resolve_interaction_discovery_shard_inputs",
        lambda *_args, **_kwargs: {"retained_terms": "retained_terms.csv"},
    )
    args = SimpleNamespace(
        stage="interaction_discovery",
        config=tmp_path / "canonical.yml",
    )

    shards = hpc_submit._build_fresh_manifest(
        args,
        artifact_dir,
        n_shards=5,
        workflow=SimpleNamespace(dataset=SimpleNamespace(path=None)),
    )

    assert len(shards) == 1
    assert shards[0].feature_start_idx is None
    assert shards[0].feature_end_idx is None


def test_global_reducer_uses_nulls_from_every_shard() -> None:
    """A pair is rejected when a later draw block raises the global max null."""
    spec = _spec()
    contract, snapshot = _contract_and_snapshot(("x1:x2", "x1:x3"), spec)
    first = _artifact(
        snapshot,
        start=0,
        end=100,
        observed=np.array([0.5, 0.9]),
        null=np.full((100, 2), 0.1),
    )
    second = _artifact(
        snapshot,
        start=100,
        end=DRAW_COUNT,
        null=np.full((DRAW_COUNT - 100, 2), 0.8),
    )

    result = reduce_score_only_interaction_artifacts(
        [first, second],
        spec=spec,
        contract=contract,
    )

    assert set(result.retained_pairs["pair_name"]) == {"x1:x3"}


def test_artifact_rejects_noncanonical_draw_order() -> None:
    """A shard cannot relabel or reorder the shared response-permutation schedule."""
    spec = _spec()
    _, snapshot = _contract_and_snapshot(("x1:x2",), spec)
    draw_ids = np.arange(DRAW_COUNT, dtype=np.int64)
    draw_ids[-1] = DRAW_COUNT

    with pytest.raises(ValueError, match="canonical contiguous draw range"):
        ScoreOnlyInteractionArtifact(
            status="score_only_completed",
            draw_range_start=0,
            draw_range_end=DRAW_COUNT,
            pair_names=("x1:x2",),
            observed_scores=np.array([0.5]),
            null_scores=np.ones((DRAW_COUNT, 1)),
            draw_ids=draw_ids,
            control_snapshot=snapshot,
        )


def _make_interaction_fixture(seed: int = 42):
    rng = np.random.default_rng(seed)
    samples = np.arange(1, 13)
    x_df = pd.DataFrame(
        {
            "sample_id": samples,
            "x1": rng.standard_normal(len(samples)),
            "x2": rng.standard_normal(len(samples)),
            "x3": rng.standard_normal(len(samples)),
        }
    )
    holdout_df = pd.DataFrame({"sample_id": samples, "split": "train"})
    pca_df = pd.DataFrame(
        {
            "sample_id": samples,
            "PC1": rng.standard_normal(len(samples)),
            "PC2": rng.standard_normal(len(samples)),
        }
    )
    catalog = pd.DataFrame(
        {
            "feature_name": ["x1", "x2", "x3"],
            "feature_type": ["first_order"] * 3,
        }
    )
    return x_df, holdout_df, pca_df, catalog


def test_monolithic_and_multishard_scoring_share_full_feature_identity(monkeypatch) -> None:
    """Draw blocks score the full-family feature matrix and reduce identically."""
    import rfm_pipeline.manuscript_stages as stages

    x_df, holdout_df, pca_df, catalog = _make_interaction_fixture()
    spec = _spec(
        random_seed=19,
        n_tree_estimators=5,
        max_tree_depth=2,
        max_shap_samples=12,
        min_component_variance_fraction=0.0,
        condition_main_effects=False,
        parallel_backend="threading",
    )
    contract = canonical_execution_contract_from_specs(spec)
    feature_widths: set[int] = set()
    weights = {"x1:x2": 0.8, "x1:x3": 0.7, "x2:x3": 0.6}

    def _fake_score(**kwargs):
        feature_widths.add(kwargs["x_feat"].shape[1])
        values = np.array([weights[item[0]] for item in kwargs["candidates"]], dtype=float)
        if kwargs["permute_response"]:
            values *= 0.25
        return values, np.repeat(values[:, None], kwargs["n_comp"], axis=1)

    monkeypatch.setattr(stages, "_score_interaction_permutation", _fake_score)
    kwargs = {
        "input_matrix": x_df,
        "feature_catalog": catalog,
        "holdout_assignments": holdout_df,
        "pca_scores": pca_df,
        "retained_terms": catalog,
        "spec": spec,
        "contract": contract,
    }
    monolithic = discover_interaction_scores_only(**kwargs)
    first = score_interaction_draw_block(**kwargs, draw_start=0, draw_end=100)
    second = score_interaction_draw_block(**kwargs, draw_start=100, draw_end=DRAW_COUNT)
    one_shard = reduce_score_only_interaction_artifacts(
        [monolithic],
        spec=spec,
        contract=contract,
    )
    many_shards = reduce_score_only_interaction_artifacts(
        [first, second],
        spec=spec,
        contract=contract,
    )

    assert feature_widths == {3}
    pd.testing.assert_frame_equal(
        one_shard.pair_scores.reset_index(drop=True),
        many_shards.pair_scores.reset_index(drop=True),
    )


def test_reducer_rejects_shuffled_artifact_arrival() -> None:
    """Reducer refuses reordering rather than silently repairing shard order."""
    spec = _spec()
    contract, snapshot = _contract_and_snapshot(("x1:x2", "x1:x3"), spec)
    first = _artifact(snapshot, start=0, end=100)
    second = _artifact(snapshot, start=100, end=DRAW_COUNT)

    with pytest.raises(ValueError, match="reordered|contiguous"):
        reduce_score_only_interaction_artifacts(
            [second, first],
            spec=spec,
            contract=contract,
        )


def test_production_permutation_uses_one_joint_row_order(monkeypatch) -> None:
    import rfm_pipeline.manuscript_stages as stages

    y_base = np.arange(24, dtype=float).reshape(8, 3)
    captured: list[np.ndarray] = []

    def _capture_fit(x_feat, y, **kwargs):  # noqa: ANN001
        _ = x_feat, kwargs
        captured.append(np.asarray(y))
        return object()

    monkeypatch.setattr(stages, "_fit_tree_for_shap", _capture_fit)
    monkeypatch.setattr(
        stages,
        "_shap_mean_abs_interaction_matrix",
        lambda model, x_feat, max_samples, rng: np.zeros((2, 2)),
    )
    stages._score_interaction_permutation(
        y_base=y_base,
        permute_response=True,
        x_feat=np.ones((8, 2)),
        n_pairs=1,
        n_comp=3,
        active_comp_indices=[0, 1, 2],
        pair_to_indices={"x1:x2": (0, 1)},
        candidates=[("x1:x2", "x1", "x2")],
        n_estimators=1,
        max_depth=1,
        max_shap_samples=8,
        seed=17,
    )

    assert len(captured) == 3
    first_order = np.argsort(captured[0])
    for component in captured[1:]:
        np.testing.assert_array_equal(np.argsort(component), first_order)


def test_load_interaction_result_falls_back_to_merged_distributed_outputs(tmp_path) -> None:
    from rfm_pipeline.distributed.stage_loaders import _load_interaction_discovery_result

    merged_root = tmp_path / "hpc_shards_interaction_discovery" / "_merged"
    merged_root.mkdir(parents=True)
    pd.DataFrame(
        [
            {"pair_name": "x1:x2", "interaction_score": 0.8, "retained": True},
            {"pair_name": "x1:x3", "interaction_score": 0.7, "retained": False},
        ]
    ).to_csv(merged_root / "interaction_pair_scores_merged.csv", index=False)
    pd.DataFrame([{"pair_name": "x1:x2", "interaction_score": 0.8, "retained": True}]).to_csv(
        merged_root / "retained_interaction_pairs_merged.csv", index=False
    )
    (merged_root / "interaction_discovery_merged.json").write_text(
        json.dumps({"n_shards": 2, "n_merged_pair_scores": 2, "n_merged_retained_pairs": 1}),
        encoding="utf-8",
    )

    loaded = _load_interaction_discovery_result(tmp_path)

    assert set(loaded.pair_scores["pair_name"]) == {"x1:x2", "x1:x3"}
    assert set(loaded.retained_pairs["pair_name"]) == {"x1:x2"}
    assert loaded.summary.loc[0, "artifact_source"] == "distributed_merged_fallback"
    assert int(loaded.summary.loc[0, "n_shards"]) == 2


def test_hpc_shard_worker_dispatches_noninteraction_stage(tmp_path, monkeypatch) -> None:
    from rfm_pipeline import hpc_shard_worker

    config_path = tmp_path / "case_study.yaml"
    config_path.write_text("pipeline:\n  stage_order: [output_conditioning]\n", encoding="utf-8")
    shard = ShardManifest(
        shard_id="task-0000",
        stage="output_conditioning",
        input_paths=[str(config_path)],
        output_path=str(tmp_path / "out" / "task-0000"),
        expected_rows=10,
        expected_columns=5,
        feature_start_idx=0,
        feature_end_idx=2,
    )
    cm = CheckpointManager(str(tmp_path / "out"), shard.shard_id)
    cm.mark_running()
    call_record: dict[str, str | None] = {}

    def _fake_run_output_conditioning_shard(
        shard_manifest: ShardManifest,
        checkpoint_manager: CheckpointManager,
        *,
        config_path: str,
    ) -> None:
        _ = shard_manifest
        call_record["config_path"] = config_path
        (checkpoint_manager.staging_dir / "shard_result.json").write_text(
            json.dumps({"shard_id": "task-0000", "stage": "output_conditioning"}),
            encoding="utf-8",
        )

    monkeypatch.setattr(
        hpc_shard_worker,
        "_run_output_conditioning_shard",
        _fake_run_output_conditioning_shard,
    )
    hpc_shard_worker._run_shard_stage(shard, cm, SimpleNamespace(config=str(config_path)))

    assert call_record["config_path"] == str(config_path)
    assert (cm.final_dir / "_SUCCESS.json").exists()


def test_hpc_reduce_materializes_noninteraction_stage(tmp_path, monkeypatch) -> None:
    from rfm_pipeline import hpc_reduce

    output_root = tmp_path / "outputs"
    output_dir = tmp_path / "merged"
    output_root.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    for shard_id in ("task-0000", "task-0001"):
        shard_dir = output_root / shard_id
        shard_dir.mkdir(parents=True)
        (shard_dir / "shard_result.json").write_text(
            json.dumps(
                {
                    "shard_id": shard_id,
                    "stage": "output_conditioning",
                    "status": "checkpoint_warmup_complete",
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(
        hpc_reduce,
        "_load_workflow_tables_and_case_config",
        lambda config_path: (
            {
                "case_study_output_matrix": pd.DataFrame(),
                "fixed_holdout_assignments": pd.DataFrame(),
            },
            object(),
        ),
    )
    monkeypatch.setattr(
        hpc_reduce,
        "output_conditioning_spec_from_case_study_config",
        lambda case_config: object(),
    )
    monkeypatch.setattr(
        hpc_reduce,
        "condition_manuscript_outputs",
        lambda outputs, holdout, spec: SimpleNamespace(
            summary=pd.DataFrame([{"stage": "output_conditioning"}])
        ),
    )
    monkeypatch.setattr(
        hpc_reduce,
        "write_output_conditioning_artifacts",
        lambda conditioning, artifact_root: {
            "summary": artifact_root / "output_conditioning/summary.csv"
        },
    )

    hpc_reduce._reduce_checkpoint_warmed_stage(
        stage="output_conditioning",
        shard_results=[
            {"shard_id": "task-0000", "stage": "output_conditioning"},
            {"shard_id": "task-0001", "stage": "output_conditioning"},
        ],
        output_dir=output_dir,
        output_root=output_root,
        config_path=str(tmp_path / "case_study.yaml"),
    )

    merged = json.loads((output_dir / "output_conditioning_merged.json").read_text())
    assert merged["stage"] == "output_conditioning"
    assert merged["status"] == "materialized_from_checkpoints"


def test_reducer_requires_config_path(tmp_path) -> None:
    """Interaction reduction has no default-spec fallback."""
    from rfm_pipeline import hpc_reduce

    output_root = tmp_path / "outputs"
    output_dir = tmp_path / "merged"
    output_root.mkdir()
    output_dir.mkdir()

    with pytest.raises(ValueError, match="config_path"):
        hpc_reduce._reduce_interaction_discovery(
            shard_results=[],
            output_dir=output_dir,
            output_root=output_root,
            config_path=None,
        )


def test_artifact_rejects_nonfinite_scores() -> None:
    spec = _spec()
    _, snapshot = _contract_and_snapshot(("x1:x2",), spec)

    with pytest.raises(ValueError, match="non-finite"):
        _artifact(
            snapshot,
            start=0,
            end=DRAW_COUNT,
            observed=np.array([np.nan]),
        )
    with pytest.raises(ValueError, match="non-finite"):
        _artifact(
            snapshot,
            start=0,
            end=DRAW_COUNT,
            null=np.full((DRAW_COUNT, 1), np.inf),
        )


def test_reducer_rejects_duplicate_or_incomplete_ranges() -> None:
    spec = _spec()
    contract, snapshot = _contract_and_snapshot(("x1:x2", "x1:x3"), spec)
    first = _artifact(snapshot, start=0, end=100)

    with pytest.raises(ValueError, match="reordered|contiguous"):
        reduce_score_only_interaction_artifacts(
            [first, first],
            spec=spec,
            contract=contract,
        )
    with pytest.raises(ValueError, match="coverage"):
        reduce_score_only_interaction_artifacts(
            [first],
            spec=spec,
            contract=contract,
        )


def test_hpc_reducer_rejects_non_score_only_shard(tmp_path, monkeypatch) -> None:
    from rfm_pipeline import hpc_reduce

    output_root = tmp_path / "outputs"
    output_dir = tmp_path / "merged"
    output_root.mkdir()
    output_dir.mkdir()
    spec = _spec()
    contract, _ = _contract_and_snapshot(("x1:x2",), spec)
    _patch_hpc_reducer(monkeypatch, contract=contract, spec=spec)

    with pytest.raises(ValueError, match="score-only"):
        hpc_reduce._reduce_interaction_discovery(
            shard_results=[
                {
                    "shard_id": "task-0000",
                    "stage": "interaction_discovery",
                    "shard_mode": "full_decision",
                }
            ],
            output_dir=output_dir,
            output_root=output_root,
            config_path="canonical.yml",
        )


def test_hpc_reducer_rejects_manifest_identity_drift(tmp_path, monkeypatch) -> None:
    from rfm_pipeline import hpc_reduce

    output_root = tmp_path / "outputs"
    output_dir = tmp_path / "merged"
    output_dir.mkdir()
    spec = _spec()
    contract, snapshot = _contract_and_snapshot(("x1:x2",), spec)
    artifact = _artifact(snapshot, start=0, end=DRAW_COUNT)
    metadata = _write_score_only_shard(output_root, "task-0000", artifact)
    metadata["candidate_family_count"] = 2
    _patch_hpc_reducer(monkeypatch, contract=contract, spec=spec)

    with pytest.raises(ValueError, match="candidate-family count"):
        hpc_reduce._reduce_interaction_discovery(
            [metadata],
            output_dir,
            output_root,
            config_path="canonical.yml",
        )


def test_hpc_reducer_emits_canonical_hashes(tmp_path, monkeypatch) -> None:
    from rfm_pipeline import hpc_reduce

    output_root = tmp_path / "outputs"
    output_dir = tmp_path / "merged"
    output_dir.mkdir()
    spec = _spec()
    contract, snapshot = _contract_and_snapshot(("x1:x2", "x1:x3"), spec)
    shard_results = [
        _write_score_only_shard(
            output_root,
            "task-0000",
            _artifact(snapshot, start=0, end=100),
        ),
        _write_score_only_shard(
            output_root,
            "task-0001",
            _artifact(snapshot, start=100, end=DRAW_COUNT),
        ),
    ]
    _patch_hpc_reducer(monkeypatch, contract=contract, spec=spec)

    hpc_reduce._reduce_interaction_discovery(
        shard_results,
        output_dir,
        output_root,
        config_path="canonical.yml",
    )
    hashes = json.loads(
        (output_dir / "interaction_discovery_merged.json").read_text(encoding="utf-8")
    )["canonical_hashes"]
    required = {
        "contract_sha256",
        "control_snapshot_sha256",
        "family_sha256",
        "null_matrix_sha256",
        "p_values_sha256",
        "retained_set_sha256",
    }
    assert required <= set(hashes)
    assert all(isinstance(value, str) and len(value) == 64 for value in hashes.values())


def test_reducer_rejects_full_snapshot_identity_drift() -> None:
    """Matching draw ranges do not permit different response or feature identities."""
    spec = _spec()
    contract, snapshot = _contract_and_snapshot(("x1:x2", "x1:x3"), spec)
    mismatched_snapshot = replace(
        snapshot,
        feature_matrix_sha256="0" * 64,
    )
    first = _artifact(snapshot, start=0, end=100)
    second = _artifact(mismatched_snapshot, start=100, end=DRAW_COUNT)

    with pytest.raises(ValueError, match="full control identity"):
        reduce_score_only_interaction_artifacts(
            [first, second],
            spec=spec,
            contract=contract,
        )
