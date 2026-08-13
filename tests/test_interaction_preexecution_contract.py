"""Focused pre-execution contract tests for global interaction selection."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from rfm_pipeline.interaction_contract import (
    ScoreOnlyInteractionArtifact,
    build_control_snapshot,
    canonical_execution_contract_from_specs,
    load_canonical_execution_contract,
    verify_control_snapshot,
)
from rfm_pipeline.manuscript_stages import (
    FinalManuscriptArtifactsSpec,
    InteractionDiscoverySpec,
    discover_interaction_scores_only,
    reduce_score_only_interaction_artifacts,
)
from rfm_pipeline.recovery_study import ProductionRecoveryResult


def _interaction_spec(*, draws: int = 999) -> InteractionDiscoverySpec:
    return InteractionDiscoverySpec(
        method="tree_shap_interaction_values",
        aggregation_rule="max_over_components_of_mean_absolute_shap_interaction",
        null_threshold_quantile=0.9,
        retained_pairs_reference=0,
        permutation_count_B=draws,
        random_seed=17,
        selection_method="max_stat_adjusted_p_mc",
        selection_alpha=0.5,
        minimum_selection_draws=999,
        enforce_permutation_adequacy=False,
    )


def _final_spec() -> FinalManuscriptArtifactsSpec:
    return FinalManuscriptArtifactsSpec(
        final_predictor_count_reference=0,
        final_first_order_input_count_reference=0,
        intermediate_penalized_holdout_nrmse_reference=0.0,
        final_ols_holdout_nrmse_reference=0.0,
        nrmse_denominator_definition="macro",
        nrmse_min_range=1.0e-6,
        nrmse_reference_matrix="Y_train",
    )


def _contract_and_snapshot(
    pair_names: tuple[str, ...],
    *,
    draws: int = 999,
):
    contract = canonical_execution_contract_from_specs(
        _interaction_spec(draws=draws),
        replace(
            _final_spec(),
            hc3_output_subset_mode="all",
            pruning_error_scale_quantile=0.9,
            pruning_remove_count_override=1,
        ),
    )
    snapshot = build_control_snapshot(
        contract,
        candidate_pair_names=pair_names,
        training_sample_ids=np.array([10, 11, 12, 13], dtype=np.int64),
        feature_matrix=np.arange(12, dtype=float).reshape(4, 3),
        response_matrix=np.arange(8, dtype=float).reshape(4, 2),
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
            else (np.full(width, 0.9, dtype=float) if start == 0 else np.empty(0, dtype=float))
        ),
        null_scores=(
            np.zeros((end - start, width), dtype=float)
            if null is None
            else np.asarray(null, dtype=float)
        ),
        draw_ids=np.arange(start, end, dtype=np.int64),
        control_snapshot=snapshot,
    )


def test_canonical_loader_captures_terminal_controls_and_snapshot_identity(tmp_path) -> None:
    """The canonical loader freezes interaction and terminal controls together."""
    config_path = tmp_path / "contract.yml"
    config_path.write_text(
        """
dataset:
  type: synthetic_300_sample
stages:
  interaction_discovery:
    n_permutations: 1000
    selection_method: max_stat_adjusted_p_mc
    selection_alpha: 0.10
    minimum_selection_draws: 999
  final_artifacts:
    hc3_output_subset_mode: target_list
    hc3_output_names: [y1]
    pruning_error_scale_quantile: 0.90
    pruning_remove_count_override: 2
output:
  seed: 17
""".strip()
        + "\n",
        encoding="utf-8",
    )

    contract = load_canonical_execution_contract(config_path)
    assert contract.interaction_controls["selection_method"] == "max_stat_adjusted_p_mc"
    assert contract.interaction_controls["selection_alpha"] == pytest.approx(0.10)
    assert contract.terminal_controls["hc3_output_subset_mode"] == "target_list"
    assert contract.terminal_controls["pruning_remove_count_override"] == 2
    assert contract.terminal_controls["refit_method"] == "ordinary_least_squares_after_pruning"

    snapshot = build_control_snapshot(
        contract,
        candidate_pair_names=("x1:x2",),
        training_sample_ids=np.array([1, 2, 3, 4]),
        feature_matrix=np.arange(8, dtype=float).reshape(4, 2),
        response_matrix=np.arange(4, dtype=float).reshape(4, 1),
        component_names=("PC1",),
    )
    verify_control_snapshot(snapshot, contract)

    with pytest.raises(ValueError, match="candidate-family"):
        verify_control_snapshot(
            replace(snapshot, candidate_pair_names=("x2:x1",)),
            contract,
        )


def test_contract_preflights_terminal_controls_and_snapshot_dimensions() -> None:
    """Terminal controls and score identities fail before scoring can begin."""
    default_contract = canonical_execution_contract_from_specs(_interaction_spec())
    assert (
        default_contract.terminal_controls["inferential_filter_interval_method"]
        == "hc3_wald_95_percent_drop_if_zero_compatible_for_all_outputs"
    )

    with pytest.raises(ValueError, match="HC3 Wald"):
        canonical_execution_contract_from_specs(
            _interaction_spec(),
            replace(
                _final_spec(),
                inferential_filter_interval_method="unsupported_interval",
            ),
        )

    with pytest.raises(ValueError, match="at most one pruning override"):
        canonical_execution_contract_from_specs(
            _interaction_spec(),
            replace(
                _final_spec(),
                pruning_delta_threshold_override=0.01,
                pruning_remove_count_override=1,
            ),
        )

    contract = canonical_execution_contract_from_specs(_interaction_spec(), _final_spec())
    with pytest.raises(ValueError, match="same row count"):
        build_control_snapshot(
            contract,
            candidate_pair_names=("x1:x2",),
            training_sample_ids=np.array([1, 2, 3, 4]),
            feature_matrix=np.arange(9, dtype=float).reshape(3, 3),
            response_matrix=np.arange(8, dtype=float).reshape(4, 2),
            component_names=("PC1", "PC2"),
        )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("stage_seed", 18, "stage seed"),
        ("score_seed_schedule_sha256", "0" * 64, "score-seed schedule"),
        (
            "permutation_index_schedule_sha256",
            "0" * 64,
            "permutation-index schedule",
        ),
    ],
)
def test_control_snapshot_recomputes_seed_and_permutation_schedules(
    field: str,
    replacement: object,
    message: str,
) -> None:
    """A syntactically valid but drifted schedule identity cannot be reused."""
    contract, snapshot = _contract_and_snapshot(("x1:x2", "x1:x3"))

    with pytest.raises(ValueError, match=message):
        verify_control_snapshot(replace(snapshot, **{field: replacement}), contract)


def test_generic_config_rejects_conflicting_terminal_pruning_overrides(tmp_path) -> None:
    """Explicit generic controls cannot silently choose between two pruning overrides."""
    config_path = tmp_path / "conflicting-pruning.yml"
    config_path.write_text(
        """
dataset:
  type: synthetic_300_sample
stages:
  final_artifacts:
    delta_threshold_override: 0.01
    pruning_remove_count_override: 1
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="at most one pruning override"):
        load_canonical_execution_contract(config_path)


def test_legacy_exact_fwer_method_name_is_rejected_without_alias() -> None:
    """Only the neutral Monte Carlo maximum-statistic identifier is accepted."""
    with pytest.raises(ValueError, match="selection_method"):
        replace(_interaction_spec(), selection_method="fwer_max_stat_exact")


def test_persisted_score_only_artifact_and_memory_reducer_share_strict_contract(tmp_path) -> None:
    """Persisted and in-memory artifacts use one reducer and reject tampering."""
    contract, snapshot = _contract_and_snapshot(("x1:x2", "x1:x3"))
    first = _artifact(snapshot, start=0, end=500)
    second = _artifact(snapshot, start=500, end=999)

    first.write_to(tmp_path / "first")
    second.write_to(tmp_path / "second")
    restored = [
        ScoreOnlyInteractionArtifact.read_from(tmp_path / "first"),
        ScoreOnlyInteractionArtifact.read_from(tmp_path / "second"),
    ]

    result = reduce_score_only_interaction_artifacts(
        restored,
        spec=_interaction_spec(),
        contract=contract,
    )
    assert result.summary.loc[0, "status"] == "completed"
    assert result.pair_scores["pair_name"].tolist() == ["x1:x2", "x1:x3"]

    with pytest.raises(ValueError, match="order|reorder|range"):
        reduce_score_only_interaction_artifacts(
            list(reversed(restored)),
            spec=_interaction_spec(),
            contract=contract,
        )
    with pytest.raises(ValueError, match="coverage|range"):
        reduce_score_only_interaction_artifacts(
            [restored[0]],
            spec=_interaction_spec(),
            contract=contract,
        )

    np.savez_compressed(
        tmp_path / "first" / "score_only_interaction.npz",
        observed_scores=np.array([999.0]),
        null_scores=np.zeros((3, 1)),
        draw_ids=np.arange(3),
    )
    with pytest.raises(ValueError, match="checksum"):
        ScoreOnlyInteractionArtifact.read_from(tmp_path / "first")


def test_hpc_reducer_uses_the_same_persisted_artifact_reducer(tmp_path, monkeypatch) -> None:
    """The HPC path loads the persisted artifact then calls the canonical reducer."""
    from rfm_pipeline import hpc_reduce

    contract, snapshot = _contract_and_snapshot(("x1:x2", "x1:x3"))
    output_root = tmp_path / "shards"
    merged = tmp_path / "merged"
    merged.mkdir()
    artifacts = [_artifact(snapshot, start=0, end=500), _artifact(snapshot, start=500, end=999)]
    shard_results = []
    for index, artifact in enumerate(artifacts):
        shard_id = f"task-{index:04d}"
        artifact.write_to(output_root / shard_id)
        shard_results.append(
            {
                "shard_id": shard_id,
                "shard_mode": "score_only",
                "status": artifact.status,
                "draw_range_start": artifact.draw_range_start,
                "draw_range_end": artifact.draw_range_end,
                "control_snapshot_sha256": artifact.control_snapshot.checksum,
                "candidate_family_sha256": artifact.control_snapshot.candidate_family_sha256,
                "candidate_family_count": len(artifact.control_snapshot.candidate_pair_names),
            }
        )

    monkeypatch.setattr(hpc_reduce, "load_canonical_execution_contract", lambda _: contract)
    monkeypatch.setattr(hpc_reduce, "load_config", lambda _: object())
    monkeypatch.setattr(hpc_reduce, "apply_fast_mode_overrides", lambda value: value)
    monkeypatch.setattr(hpc_reduce, "config_to_legacy_case_study", lambda _: object())
    monkeypatch.setattr(
        hpc_reduce,
        "interaction_discovery_spec_from_case_study_config",
        lambda _: _interaction_spec(),
    )

    hpc_reduce._reduce_interaction_discovery(
        shard_results,
        merged,
        output_root,
        config_path="canonical.yml",
    )
    summary = __import__("json").loads(
        (merged / "interaction_discovery_merged.json").read_text(encoding="utf-8")
    )
    assert summary["reducer"] == "canonical_score_only_family_decision"
    assert summary["status"] == "completed"
    assert summary["canonical_hashes"]["control_snapshot_sha256"] == snapshot.checksum

    tampered_results = [dict(result) for result in shard_results]
    tampered_results[0]["candidate_family_count"] = 99
    with pytest.raises(ValueError, match="candidate-family count"):
        hpc_reduce._reduce_interaction_discovery(
            tampered_results,
            merged,
            output_root,
            config_path="canonical.yml",
        )


def test_one_pair_is_valid_and_empty_family_is_terminal() -> None:
    """A one-pair family reduces normally; a zero-pair family ends explicitly."""
    one_contract, one_snapshot = _contract_and_snapshot(("x1:x2",))
    one_result = reduce_score_only_interaction_artifacts(
        [_artifact(one_snapshot, start=0, end=999)],
        spec=_interaction_spec(),
        contract=one_contract,
    )
    assert one_result.summary.loc[0, "status"] == "completed"
    assert one_result.pair_scores["pair_name"].tolist() == ["x1:x2"]

    empty_contract, empty_snapshot = _contract_and_snapshot(())
    empty = ScoreOnlyInteractionArtifact.empty_terminal(empty_snapshot)
    empty_result = reduce_score_only_interaction_artifacts(
        [empty],
        spec=_interaction_spec(),
        contract=empty_contract,
    )
    assert empty_result.summary.loc[0, "status"] == "empty_candidate_family"
    assert empty_result.pair_scores.empty
    assert empty_result.retained_pairs.empty


def test_score_only_execution_handles_one_pair_and_empty_family(monkeypatch) -> None:
    """The execution path preserves one pair and emits a terminal empty artifact."""
    import rfm_pipeline.manuscript_stages as stages

    inputs = pd.DataFrame(
        {
            "sample_id": [1, 2, 3, 4],
            "x1": [0.0, 1.0, 0.0, 1.0],
            "x2": [1.0, 0.0, 1.0, 0.0],
        }
    )
    holdout = pd.DataFrame({"sample_id": [1, 2, 3, 4], "split": ["train"] * 4})
    pca = pd.DataFrame({"sample_id": [1, 2, 3, 4], "PC1": [0.0, 1.0, 0.0, 1.0]})
    catalog = pd.DataFrame(
        {"feature_name": ["x1", "x2"], "feature_type": ["first_order", "first_order"]}
    )
    spec = _interaction_spec()
    spec = replace(spec, selection_alpha=0.5)
    contract = canonical_execution_contract_from_specs(spec)

    monkeypatch.setattr(
        stages,
        "_score_interaction_permutation",
        lambda **kwargs: (
            np.full(kwargs["n_pairs"], 0.8),
            np.full((kwargs["n_pairs"], kwargs["n_comp"]), 0.8),
        ),
    )
    one_pair = discover_interaction_scores_only(
        inputs,
        catalog,
        holdout,
        pca,
        catalog,
        spec,
        contract=contract,
    )
    assert one_pair.status == "score_only_completed"
    assert one_pair.pair_names == ("x1:x2",)

    with monkeypatch.context() as mismatch_patch:

        def _unexpected_conditioning(*args, **kwargs):
            raise AssertionError("control mismatch must reject before score conditioning")

        mismatch_patch.setattr(
            stages,
            "_condition_pca_scores_on_main_effects",
            _unexpected_conditioning,
        )
        with pytest.raises(ValueError, match="Canonical interaction controls"):
            discover_interaction_scores_only(
                inputs,
                catalog,
                holdout,
                pca,
                catalog,
                replace(spec, n_tree_estimators=spec.n_tree_estimators + 1),
                contract=contract,
            )

    one_feature = catalog.iloc[:1].copy()
    empty = discover_interaction_scores_only(
        inputs.loc[:, ["sample_id", "x1"]],
        one_feature,
        holdout,
        pca,
        one_feature,
        spec,
        contract=contract,
    )
    assert empty.status == "empty_candidate_family"


def test_empty_support_never_silently_returns_null_predictions() -> None:
    """An empty family is explicit and carries no fabricated prediction matrix."""
    result = ProductionRecoveryResult(
        screening_candidate_count=160,
        screening_retained_set=frozenset(),
        interaction_candidate_count=0,
        interaction_retained_set=frozenset(),
        nonlinear_candidate_count=0,
        nonlinear_retained_set=frozenset(),
        final_selected_support=frozenset(),
        eval_predictions=np.empty((0, 0)),
        terminal_status="empty_candidate_family",
        contract_hash="0" * 64,
        interaction_artifact_checksums=("1" * 64,),
        model_freeze_hash=None,
    )
    assert result.terminal_status == "empty_candidate_family"
    assert result.eval_predictions.size == 0
    assert result.model_freeze_hash is None
