"""Tests for HPC shard worker + reduce merge integration."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd

from rfm_pipeline.distributed.checkpoint import CheckpointManager
from rfm_pipeline.distributed.manifest import ShardManifest
from rfm_pipeline.manuscript_stages import InteractionDiscoveryResult, InteractionDiscoverySpec


def _minimal_interaction_result() -> InteractionDiscoveryResult:
    pair_scores = pd.DataFrame(
        [
            {
                "pair_name": "x1:x2",
                "interaction_score": 0.5,
                "empirical_null_threshold": 0.2,
                "empirical_p_value": 0.01,
                "empirical_null_retained": True,
                "retained": True,
            }
        ]
    )
    component_scores = pd.DataFrame(
        [{"pair_name": "x1:x2", "component_name": "PC1", "interaction_score": 0.5}]
    )
    null_summary = pd.DataFrame(
        [{"pair_name": "x1:x2", "null_mean_score": 0.1, "null_std_score": 0.01}]
    )
    retained_pairs = pair_scores.loc[pair_scores["retained"]].copy()
    provenance = pd.DataFrame(
        [
            {
                "manuscript_method": "tree_shap_interaction_values",
                "public_implementation_method": "tree_shap_gradient_boosting",
                "public_implementation_status": "manuscript_aligned",
                "source_workflow_reference": "private_tree_shap_interaction_workflow",
                "source_workflow_equivalence_status": (
                    "manuscript_aligned_via_shap_gradient_boosting"
                ),
                "null_threshold_quantile": 0.9,
                "permutation_count_B": 3,
            }
        ]
    )
    summary = pd.DataFrame(
        [
            {
                "stage": "interaction_discovery",
                "n_candidate_pairs": 1,
                "n_retained_pairs": 1,
                "n_training_rows": 10,
                "n_empirical_null_retained_pairs": 1,
                "null_threshold_quantile": 0.9,
                "retained_pairs_reference": 1,
                "public_implementation_method": "tree_shap_gradient_boosting",
                "public_implementation_status": "manuscript_aligned",
                "source_workflow_reference": "private_tree_shap_interaction_workflow",
                "source_workflow_equivalence_status": (
                    "manuscript_aligned_via_shap_gradient_boosting"
                ),
            }
        ]
    )
    return InteractionDiscoveryResult(
        pair_scores=pair_scores,
        component_interaction_scores=component_scores,
        interaction_null_summary=null_summary,
        retained_pairs=retained_pairs,
        provenance=provenance,
        summary=summary,
    )


def test_hpc_shard_worker_writes_interaction_artifacts(tmp_path, monkeypatch):
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
            "x3": [0.5] * 10,
        }
    )
    x_df.to_parquet(x_path)
    pd.DataFrame(
        {"sample_id": list(range(1, 11)), "split": ["train"] * 8 + ["holdout"] * 2}
    ).to_parquet(holdout_path)
    pd.DataFrame(
        {
            "feature_name": ["x1", "x2", "x3"],
            "feature_type": ["first_order", "first_order", "first_order"],
        }
    ).to_parquet(catalog_path)
    pd.DataFrame({"sample_id": list(range(1, 11)), "PC1": [0.0, 1.0] * 5}).to_csv(
        pca_scores_path, index=False
    )
    pd.DataFrame({"feature_name": ["x1", "x2", "x3"], "feature_type": ["first_order"] * 3}).to_csv(
        retained_terms_path, index=False
    )

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
        expected_columns=3,
        feature_start_idx=0,
        feature_end_idx=2,
    )

    cm = CheckpointManager(str(tmp_path / "out"), shard.shard_id)
    cm.mark_running()

    def _fake_spec(_: str | None) -> InteractionDiscoverySpec:
        return InteractionDiscoverySpec(
            method="tree_shap_interaction_values",
            aggregation_rule="max_over_components_of_mean_absolute_shap_interaction",
            null_threshold_quantile=0.9,
            retained_pairs_reference=1,
            permutation_count_B=3,
            random_seed=123,
            n_jobs=1,
        )

    monkeypatch.setattr(hpc_shard_worker, "_load_interaction_spec", _fake_spec)
    discover_call: dict[str, object] = {}

    def _fake_discover(*args, **kwargs):  # noqa: ANN002, ANN003
        _ = args
        discover_call.update(kwargs)
        return _minimal_interaction_result()

    monkeypatch.setattr(
        hpc_shard_worker,
        "discover_manuscript_interactions",
        _fake_discover,
    )

    hpc_shard_worker._run_interaction_shard(shard, cm, config_path=None)

    shard_result = json.loads((cm.staging_dir / "shard_result.json").read_text())
    assert shard_result["stage"] == "interaction_discovery"
    assert shard_result["n_retained_pairs"] == 1
    assert discover_call["pair_start_idx"] == 0
    assert discover_call["pair_end_idx"] == 2
    assert (cm.staging_dir / "retained_interaction_pairs.csv").exists()
    assert (cm.staging_dir / "interaction_pair_scores.csv").exists()


def test_hpc_reduce_merges_interaction_pair_outputs(tmp_path):
    from tools import hpc_reduce

    output_root = tmp_path / "outputs"
    merged_dir = tmp_path / "merged"
    output_root.mkdir(parents=True)
    merged_dir.mkdir(parents=True)

    shard_results = []
    for shard_id, pair_name, score in [
        ("task-0000", "x1:x2", 0.4),
        ("task-0001", "x1:x3", 0.7),
    ]:
        shard_dir = output_root / shard_id
        shard_dir.mkdir(parents=True)
        pd.DataFrame(
            [{"pair_name": pair_name, "interaction_score": score, "retained": True}]
        ).to_csv(shard_dir / "retained_interaction_pairs.csv", index=False)
        pd.DataFrame(
            [
                {
                    "pair_name": pair_name,
                    "interaction_score": score,
                    "empirical_null_threshold": 0.1,
                    "empirical_p_value": 0.01,
                    "retained": True,
                    "empirical_null_retained": True,
                }
            ]
        ).to_csv(shard_dir / "interaction_pair_scores.csv", index=False)
        shard_results.append(
            {
                "shard_id": shard_id,
                "stage": "interaction_discovery",
                "retained_pairs_file": "retained_interaction_pairs.csv",
                "pair_scores_file": "interaction_pair_scores.csv",
            }
        )

    hpc_reduce._reduce_interaction_discovery(
        shard_results=shard_results,
        output_dir=merged_dir,
        output_root=output_root,
    )

    merged_retained = pd.read_csv(merged_dir / "retained_interaction_pairs_merged.csv")
    merged_scores = pd.read_csv(merged_dir / "interaction_pair_scores_merged.csv")
    assert set(merged_retained["pair_name"]) == {"x1:x2", "x1:x3"}
    assert set(merged_scores["pair_name"]) == {"x1:x2", "x1:x3"}


def test_load_interaction_result_falls_back_to_merged_distributed_outputs(tmp_path):
    from rfm_pipeline.distributed.stage_loaders import _load_interaction_discovery_result

    merged_root = tmp_path / "hpc_shards_interaction_discovery" / "_merged"
    merged_root.mkdir(parents=True)
    pd.DataFrame(
        [
            {"pair_name": "x1:x2", "interaction_score": 0.8, "retained": True},
            {"pair_name": "x1:x3", "interaction_score": 0.7, "retained": False},
        ]
    ).to_csv(merged_root / "interaction_pair_scores_merged.csv", index=False)
    pd.DataFrame(
        [
            {"pair_name": "x1:x2", "interaction_score": 0.8, "retained": True},
        ]
    ).to_csv(merged_root / "retained_interaction_pairs_merged.csv", index=False)
    (merged_root / "interaction_discovery_merged.json").write_text(
        json.dumps({"n_shards": 2, "n_merged_pair_scores": 2, "n_merged_retained_pairs": 1})
    )

    loaded = _load_interaction_discovery_result(tmp_path)

    assert set(loaded.pair_scores["pair_name"]) == {"x1:x2", "x1:x3"}
    assert set(loaded.retained_pairs["pair_name"]) == {"x1:x2"}
    assert loaded.summary.loc[0, "artifact_source"] == "distributed_merged_fallback"
    assert int(loaded.summary.loc[0, "n_shards"]) == 2


def test_hpc_shard_worker_dispatches_noninteraction_stage(tmp_path, monkeypatch):
    from rfm_pipeline import hpc_shard_worker

    config_path = tmp_path / "case_study.yaml"
    config_path.write_text("pipeline:\n  stage_order: [output_conditioning]\n")
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
            json.dumps({"shard_id": "task-0000", "stage": "output_conditioning"})
        )

    monkeypatch.setattr(
        hpc_shard_worker,
        "_run_output_conditioning_shard",
        _fake_run_output_conditioning_shard,
    )

    hpc_shard_worker._run_shard_stage(
        shard,
        cm,
        SimpleNamespace(config=str(config_path)),
    )

    assert call_record["config_path"] == str(config_path)
    assert (cm.final_dir / "_SUCCESS.json").exists()


def test_hpc_reduce_materializes_noninteraction_stage(tmp_path, monkeypatch):
    from tools import hpc_reduce

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
            )
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
