"""G11-A-S1: draw-block interaction scoring keeps one complete pair family."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import rfm_pipeline.manuscript_stages as stages
from rfm_pipeline.interaction_contract import (
    ScoreOnlyInteractionArtifact,
    canonical_execution_contract_from_specs,
)
from rfm_pipeline.manuscript_stages import (
    InteractionDiscoverySpec,
    discover_interaction_scores_only,
    reduce_interaction_draw_blocks,
    reduce_score_only_interaction_artifacts,
    score_interaction_draw_block,
)


def _spec() -> InteractionDiscoverySpec:
    return InteractionDiscoverySpec(
        method="tree_shap_interaction_values",
        aggregation_rule="max_over_components_of_mean_absolute_shap_interaction",
        null_threshold_quantile=0.95,
        retained_pairs_reference=0,
        permutation_count_B=199,
        random_seed=20260811,
        n_jobs=1,
        selection_method="max_t",
        selection_alpha=0.05,
        minimum_selection_draws=199,
    )


def _tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    inputs = pd.DataFrame(
        {
            "sample_id": range(1, 9),
            "x1": [-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0],
            "x2": [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
            "x3": [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
        }
    )
    catalog = pd.DataFrame(
        {"feature_name": ["x1", "x2", "x3"], "feature_type": ["first_order"] * 3}
    )
    holdout = pd.DataFrame({"sample_id": range(1, 9), "split": ["train"] * 8})
    pca = pd.DataFrame({"sample_id": range(1, 9), "PC1": inputs["x1"] * inputs["x2"]})
    return inputs, catalog, holdout, pca, catalog.copy()


def _fake_score(
    y_base: np.ndarray,
    permute_response: bool,
    *,
    n_pairs: int,
    n_comp: int,
    seed: int,
    **_: object,
) -> tuple[np.ndarray, np.ndarray]:
    _ = y_base
    value = float(seed % 101) if permute_response else 999.0
    return np.full(n_pairs, value), np.full((n_pairs, n_comp), value)


def _block(
    tables: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
    spec: InteractionDiscoverySpec,
    contract: object,
    start: int,
    end: int,
) -> ScoreOnlyInteractionArtifact:
    return score_interaction_draw_block(
        *tables,
        spec,
        draw_start=start,
        draw_end=end,
        contract=contract,
    )


def test_serial_and_complete_family_draw_blocks_are_identical(monkeypatch) -> None:
    """Two contiguous draw blocks exactly reproduce a serial family decision."""
    monkeypatch.setattr(stages, "_score_interaction_permutation", _fake_score)
    tables = _tables()
    spec = _spec()
    contract = canonical_execution_contract_from_specs(spec)

    serial = discover_interaction_scores_only(*tables, spec, contract=contract)
    first = _block(tables, spec, contract, 0, 73)
    second = _block(tables, spec, contract, 73, 199)
    combined = reduce_interaction_draw_blocks([first, second], spec=spec, contract=contract)

    assert first.pair_names == ("x1:x2", "x1:x3", "x2:x3")
    assert second.pair_names == first.pair_names
    np.testing.assert_array_equal(combined.observed_scores, serial.observed_scores)
    np.testing.assert_array_equal(combined.null_scores, serial.null_scores)
    serial_result = reduce_score_only_interaction_artifacts([serial], spec=spec, contract=contract)
    block_result = reduce_score_only_interaction_artifacts([combined], spec=spec, contract=contract)
    pd.testing.assert_frame_equal(block_result.pair_scores, serial_result.pair_scores)


@pytest.mark.parametrize(
    ("ranges", "message"),
    [
        ([(0, 73), (74, 199)], "coverage"),
        ([(0, 73), (72, 199)], "reordered|duplicate|contiguous"),
    ],
)
def test_draw_block_reducer_rejects_gaps_and_duplicate_draws(monkeypatch, ranges, message) -> None:
    monkeypatch.setattr(stages, "_score_interaction_permutation", _fake_score)
    tables = _tables()
    spec = _spec()
    contract = canonical_execution_contract_from_specs(spec)
    artifacts = [_block(tables, spec, contract, start, end) for start, end in ranges]

    with pytest.raises(ValueError, match=message):
        reduce_interaction_draw_blocks(artifacts, spec=spec, contract=contract)


def test_draw_block_reducer_rejects_reordered_mixed_and_corrupt_artifacts(
    monkeypatch, tmp_path
) -> None:
    """Persisted blocks are content-addressed and may not be reordered or mixed."""
    monkeypatch.setattr(stages, "_score_interaction_permutation", _fake_score)
    tables = _tables()
    spec = _spec()
    contract = canonical_execution_contract_from_specs(spec)
    first = _block(tables, spec, contract, 0, 100)
    second = _block(tables, spec, contract, 100, 199)

    with pytest.raises(ValueError, match="reordered"):
        reduce_interaction_draw_blocks([second, first], spec=spec, contract=contract)
    mixed = replace(
        second,
        control_snapshot=replace(second.control_snapshot, response_matrix_sha256="0" * 64),
    )
    with pytest.raises(ValueError, match="identity"):
        reduce_interaction_draw_blocks([first, mixed], spec=spec, contract=contract)

    first.write_to(tmp_path / "first")
    scores_path = tmp_path / "first" / "score_only_interaction.npz"
    scores_path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        ScoreOnlyInteractionArtifact.read_from(tmp_path / "first")

    second.write_to(tmp_path / "incomplete")
    metadata_path = tmp_path / "incomplete" / "score_only_interaction.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    del metadata["payload_sha256"]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete"):
        ScoreOnlyInteractionArtifact.read_from(tmp_path / "incomplete")


def test_hpc_manifest_partitions_interaction_draws_not_pairs(tmp_path) -> None:
    """The scheduler-facing manifest spans B draws even when C(P, 2) differs."""
    from rfm_pipeline.distributed.manifest import build_manifest
    from rfm_pipeline.hpc_submit import _estimate_stage_partition_span

    retained = tmp_path / "empirical_null_screen"
    retained.mkdir()
    pd.DataFrame({"feature_name": ["x1", "x2", "x3"], "feature_type": ["first_order"] * 3}).to_csv(
        retained / "retained_terms.csv", index=False
    )
    workflow = SimpleNamespace(
        stages=SimpleNamespace(interaction_discovery=SimpleNamespace(n_permutations=1000))
    )

    draws = _estimate_stage_partition_span("interaction_discovery", workflow, tmp_path)
    shards = build_manifest(
        "interaction_discovery", [], str(tmp_path / "shards"), n_shards=3, expected_columns=draws
    )

    assert draws == 999
    assert [(shard.feature_start_idx, shard.feature_end_idx) for shard in shards] == [
        (0, 333),
        (333, 666),
        (666, 999),
    ]
