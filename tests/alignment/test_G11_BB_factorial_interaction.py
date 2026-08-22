"""Prospective Gate-B correction for binary--binary parity interactions."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import rfm_pipeline.manuscript_stages as stages
from rfm_pipeline.campaign_contract import G11_CONTRACT
from rfm_pipeline.interaction_contract import (
    ScoreOnlyInteractionArtifact,
    build_control_snapshot,
    canonical_execution_contract_from_specs,
)
from rfm_pipeline.manuscript_stages import InteractionDiscoverySpec


def _type_aware_spec(*, draws: int = 999) -> InteractionDiscoverySpec:
    return InteractionDiscoverySpec(
        method="type_aware_tree_shap_binary_factorial",
        aggregation_rule="max_over_components_by_detector",
        null_threshold_quantile=0.995,
        retained_pairs_reference=0,
        permutation_count_B=draws,
        random_seed=20260818,
        n_jobs=1,
        selection_method="max_stat_adjusted_p_mc",
        selection_alpha=0.05,
        tree_family_alpha=0.025,
        binary_binary_family_alpha=0.025,
        binary_binary_method="studentized_factorial_contrast_hc3",
        binary_binary_minimum_cell_count=2,
        minimum_selection_draws=draws,
    )


def _balanced_binary_design(repeats: int = 12) -> tuple[np.ndarray, np.ndarray]:
    left = np.tile(np.array([-1.0, -1.0, 1.0, 1.0]), repeats)
    right = np.tile(np.array([-1.0, 1.0, -1.0, 1.0]), repeats)
    return left, right


def test_studentized_factorial_detects_parity_and_rejects_additive_signal() -> None:
    """The BB statistic targets the only nonadditive degree of freedom in a 2x2 table."""
    left, right = _balanced_binary_design()
    rng = np.random.default_rng(41)
    parity = 2.0 * left * right + rng.normal(scale=0.15, size=left.size)
    additive = 1.5 * left - 0.75 * right + rng.normal(scale=0.15, size=left.size)
    response = np.column_stack([parity, additive])

    scores = stages._studentized_binary_factorial_component_scores(
        left,
        right,
        response,
        active_component_indices=[0, 1],
        minimum_cell_count=2,
    )

    assert scores[0] > 20.0
    assert scores[1] < 2.0


def test_studentized_factorial_is_invariant_to_binary_coding_and_pair_order() -> None:
    left, right = _balanced_binary_design()
    rng = np.random.default_rng(77)
    response = (left * right + rng.normal(scale=0.2, size=left.size))[:, None]

    baseline = stages._studentized_binary_factorial_component_scores(
        left,
        right,
        response,
        active_component_indices=[0],
        minimum_cell_count=2,
    )
    recoded = stages._studentized_binary_factorial_component_scores(
        np.where(right > 0, 17.0, 3.0),
        np.where(left > 0, 0.0, 1.0),
        response,
        active_component_indices=[0],
        minimum_cell_count=2,
    )

    np.testing.assert_allclose(recoded, baseline, rtol=1.0e-12, atol=1.0e-12)


@pytest.mark.parametrize(
    "right",
    [
        np.array([-1.0, 1.0, -1.0, 1.0]),
        np.array([-1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0]),
    ],
)
def test_studentized_factorial_requires_four_adequately_populated_cells(
    right: np.ndarray,
) -> None:
    left = np.resize(np.array([-1.0, -1.0, 1.0, 1.0]), right.size)
    response = np.arange(right.size, dtype=float)[:, None]

    with pytest.raises(ValueError, match="four.*cells|cell count"):
        stages._studentized_binary_factorial_component_scores(
            left,
            right,
            response,
            active_component_indices=[0],
            minimum_cell_count=2,
        )


def test_type_aware_detector_assignment_separates_bb_from_tree_family() -> None:
    left, right = _balanced_binary_design(repeats=3)
    continuous = np.linspace(-2.0, 2.0, left.size)
    features = np.column_stack([left, right, continuous])
    candidates = [
        ("binary_a:binary_b", "binary_a", "binary_b"),
        ("binary_a:continuous", "binary_a", "continuous"),
        ("binary_b:continuous", "binary_b", "continuous"),
    ]
    pair_indices = {
        "binary_a:binary_b": (0, 1),
        "binary_a:continuous": (0, 2),
        "binary_b:continuous": (1, 2),
    }

    detectors = stages._interaction_pair_detectors(
        features,
        candidates,
        pair_indices,
        method="type_aware_tree_shap_binary_factorial",
    )

    assert detectors == (
        "studentized_binary_factorial",
        "tree_shap",
        "tree_shap",
    )


def test_type_aware_score_only_path_recovers_bb_without_tree_fit(monkeypatch) -> None:
    """A BB-only candidate family is scored and reduced without fitting a tree."""
    left, right = _balanced_binary_design(repeats=20)
    rng = np.random.default_rng(909)
    sample_ids = np.arange(left.size)
    inputs = pd.DataFrame({"sample_id": sample_ids, "binary_a": left, "binary_b": right})
    catalog = pd.DataFrame(
        {
            "feature_name": ["binary_a", "binary_b"],
            "feature_type": ["first_order", "first_order"],
        }
    )
    assignments = pd.DataFrame({"sample_id": sample_ids, "split": "train"})
    pca_scores = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "PC1": 2.0 * left * right + rng.normal(scale=0.2, size=left.size),
        }
    )
    spec = _type_aware_spec(draws=199)
    monkeypatch.setattr(
        stages,
        "_fit_tree_for_shap",
        lambda *args, **kwargs: pytest.fail("BB-only family must not fit a tree"),
    )

    result = stages.discover_manuscript_interactions(
        inputs,
        catalog,
        assignments,
        pca_scores,
        catalog.copy(),
        spec,
    )

    row = result.pair_scores.iloc[0]
    assert row["pair_name"] == "binary_a:binary_b"
    assert row["detector_method"] == "studentized_binary_factorial"
    assert row["detector_family_alpha"] == pytest.approx(0.025)
    assert bool(row["retained"])
    assert row["empirical_p_value"] <= 0.025


def test_partitioned_max_t_never_compares_incommensurate_raw_scores() -> None:
    """Each detector family receives its frozen alpha and its own row maximum."""
    observed = np.array([5.0, 0.90, 0.80])
    null = np.array(
        [
            [10.0, 0.10, 0.10],
            [9.0, 0.20, 0.20],
            [8.0, 0.30, 0.30],
            [7.0, 0.40, 0.40],
            [6.0, 0.50, 0.50],
            [5.5, 0.60, 0.60],
            [5.4, 0.70, 0.70],
            [5.3, 0.75, 0.75],
            [5.2, 0.79, 0.79],
            [5.1, 0.81, 0.81],
            [5.0, 0.82, 0.82],
            [4.9, 0.83, 0.83],
            [4.8, 0.84, 0.84],
            [4.7, 0.85, 0.85],
            [4.6, 0.86, 0.86],
            [4.5, 0.87, 0.87],
            [4.4, 0.88, 0.88],
            [4.3, 0.89, 0.89],
            [4.2, 0.895, 0.895],
            [4.1, 0.899, 0.899],
            [4.0, 0.70, 0.70],
            [3.9, 0.60, 0.60],
            [3.8, 0.50, 0.50],
            [3.7, 0.40, 0.40],
            [3.6, 0.30, 0.30],
            [3.5, 0.20, 0.20],
            [3.4, 0.10, 0.10],
            [3.3, 0.05, 0.05],
            [3.2, 0.04, 0.04],
            [3.1, 0.03, 0.03],
            [3.0, 0.02, 0.02],
            [2.9, 0.01, 0.01],
            [2.8, 0.00, 0.00],
            [2.7, 0.00, 0.00],
            [2.6, 0.00, 0.00],
            [2.5, 0.00, 0.00],
            [2.4, 0.00, 0.00],
            [2.3, 0.00, 0.00],
            [2.2, 0.00, 0.00],
            [2.1, 0.00, 0.00],
        ]
    )
    detectors = ("studentized_binary_factorial", "tree_shap", "tree_shap")

    selected, p_values, thresholds, alphas = (
        stages._partitioned_multiplicity_controlled_interaction_selection(
            observed,
            null,
            detectors,
            tree_family_alpha=0.025,
            binary_binary_family_alpha=0.025,
            method="max_stat_adjusted_p_mc",
        )
    )

    assert selected.tolist() == [False, True, False]
    assert p_values[1] <= 0.025
    assert thresholds[0] > 1.0
    assert thresholds[1] < 1.0
    assert alphas.tolist() == [0.025, 0.025, 0.025]


def test_detector_identity_is_persisted_and_tamper_evident(tmp_path) -> None:
    spec = _type_aware_spec(draws=999)
    contract = canonical_execution_contract_from_specs(spec)
    snapshot = build_control_snapshot(
        contract,
        candidate_pair_names=("binary_a:binary_b", "binary_a:continuous"),
        candidate_pair_detectors=("studentized_binary_factorial", "tree_shap"),
        training_sample_ids=np.arange(8),
        feature_matrix=np.arange(24, dtype=float).reshape(8, 3),
        response_matrix=np.arange(8, dtype=float).reshape(8, 1),
        component_names=("PC1",),
    )
    artifact = ScoreOnlyInteractionArtifact(
        status="score_only_completed",
        draw_range_start=0,
        draw_range_end=999,
        pair_names=snapshot.candidate_pair_names,
        observed_scores=np.array([3.0, 0.4]),
        null_scores=np.zeros((999, 2)),
        draw_ids=np.arange(999),
        control_snapshot=snapshot,
    )
    artifact.write_to(tmp_path)
    loaded = ScoreOnlyInteractionArtifact.read_from(tmp_path)
    assert loaded.control_snapshot.candidate_pair_detectors == (
        "studentized_binary_factorial",
        "tree_shap",
    )

    metadata_path = tmp_path / "score_only_interaction.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["control_snapshot"]["candidate_pair_detectors"][0] = "tree_shap"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="detector|payload checksum"):
        ScoreOnlyInteractionArtifact.read_from(tmp_path)


def test_type_aware_alpha_partition_must_preserve_overall_fwer() -> None:
    with pytest.raises(ValueError, match="sum.*selection_alpha"):
        _type_aware_spec().__class__(
            **{
                **_type_aware_spec().__dict__,
                "tree_family_alpha": 0.04,
                "binary_binary_family_alpha": 0.04,
            }
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tree_family_alpha", 0.015),
        ("binary_binary_family_alpha", 0.015),
        ("binary_binary_minimum_cell_count", 3),
    ],
)
def test_g11_contract_rejects_drift_from_frozen_family_controls(
    field: str,
    value: float | int,
) -> None:
    with pytest.raises(ValueError, match="frozen|0.020|exactly two"):
        replace(G11_CONTRACT, **{field: value})


def test_g11_amended_detector_alphas_target_reliable_fixed_family_calibration() -> None:
    """The amended contract spends 0.04 FWER while preserving strong-signal power."""
    assert G11_CONTRACT.schema_version == "g11_campaign_contract_v11"
    assert G11_CONTRACT.generation == 12
    assert G11_CONTRACT.tree_family_alpha == pytest.approx(0.020)
    assert G11_CONTRACT.binary_binary_family_alpha == pytest.approx(0.020)
    assert (
        G11_CONTRACT.tree_family_alpha + G11_CONTRACT.binary_binary_family_alpha
    ) == pytest.approx(0.04)
