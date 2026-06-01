# Copyright (c) 2026 Dylan Hettinger
"""Tests for standalone synthetic DGP generation."""

from __future__ import annotations

import pandas as pd

from rfm_pipeline.synthetic_dgp import (
    SyntheticDGPSpec,
    generate_bsm_structure_synthetic,
    generate_pure_synthetic,
)


def _small_spec(*, seed: int = 7, family: str = "pure_synthetic") -> SyntheticDGPSpec:
    return SyntheticDGPSpec(
        n_inputs=10,
        n_runs=100,
        n_outputs=5,
        sparsity=0.3,
        interaction_density=0.25,
        nonlinearity_strength=0.4,
        noise_snr=10.0,
        holdout_fraction=0.10,
        dgp_family=family,
        seed=seed,
        factor_model_rank=4,
        input_correlation_strength=0.2,
    )


def test_pure_synthetic_output_shapes() -> None:
    dataset = generate_pure_synthetic(_small_spec())

    assert dataset.input_matrix.shape == (100, 11)
    assert dataset.output_matrix.shape == (100, 6)
    assert dataset.holdout_assignments.shape == (100, 2)
    assert set(dataset.feature_catalog.columns) == {"feature_name", "feature_type", "origin"}


def test_pure_synthetic_sample_ids() -> None:
    dataset = generate_pure_synthetic(_small_spec())

    sample_ids = dataset.input_matrix["sample_id"]
    assert sample_ids.is_unique
    assert sample_ids.tolist()[0] == "s000000"
    assert sample_ids.tolist()[-1] == "s000099"

    holdout = dataset.holdout_assignments
    assert (holdout["split"] == "holdout").sum() == 10
    assert holdout.tail(10)["split"].eq("holdout").all()
    assert holdout.head(90)["split"].eq("train").all()


def test_pure_synthetic_feature_catalog_columns() -> None:
    dataset = generate_pure_synthetic(_small_spec())

    assert list(dataset.feature_catalog.columns) == ["feature_name", "feature_type", "origin"]
    assert (
        dataset.feature_catalog["feature_type"]
        .isin({"first_order", "interaction", "transformation"})
        .all()
    )


def test_pure_synthetic_no_nan() -> None:
    dataset = generate_pure_synthetic(_small_spec())

    assert not dataset.input_matrix.isna().any().any()
    assert not dataset.output_matrix.isna().any().any()
    assert not dataset.holdout_assignments.isna().any().any()
    assert not dataset.feature_catalog.isna().any().any()


def test_bsm_structure_output_shapes() -> None:
    dataset = generate_bsm_structure_synthetic(_small_spec(family="bsm_structure"))

    assert dataset.input_matrix.shape == (100, 11)
    assert dataset.output_matrix.shape == (100, 6)
    assert dataset.holdout_assignments.shape == (100, 2)
    assert dataset.spec.dgp_family == "bsm_structure"


def test_support_recovery_precision_recall() -> None:
    spec = _small_spec(seed=11)
    dataset = generate_pure_synthetic(spec)

    expected_active = max(1, round(spec.sparsity * spec.n_inputs))
    assert len(dataset.true_support.true_active_inputs) == expected_active
    assert dataset.true_support.true_active_inputs <= frozenset(
        f"x{i}" for i in range(spec.n_inputs)
    )
    for left, right in dataset.true_support.true_active_interactions:
        assert left in dataset.true_support.true_active_inputs
        assert right in dataset.true_support.true_active_inputs
        assert left < right


def test_reproducible_with_same_seed() -> None:
    spec = _small_spec(seed=19)
    left = generate_pure_synthetic(spec)
    right = generate_pure_synthetic(spec)

    pd.testing.assert_frame_equal(left.input_matrix, right.input_matrix)
    pd.testing.assert_frame_equal(left.output_matrix, right.output_matrix)
    pd.testing.assert_frame_equal(left.holdout_assignments, right.holdout_assignments)
    pd.testing.assert_frame_equal(left.feature_catalog, right.feature_catalog)
    assert left.true_support == right.true_support
