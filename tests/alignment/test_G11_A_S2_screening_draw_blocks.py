"""G11 screening permutation blocks use one schedule and one artifact per block."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rfm_pipeline.manuscript_stages import (
    EmpiricalNullScreeningSpec,
    OutputConditioningSpec,
    ScreeningNullBlock,
    condition_manuscript_outputs,
    reduce_screening_draw_blocks,
    score_screening_draw_block,
)


def _fixture():
    rng = np.random.default_rng(14)
    n = 48
    x = rng.normal(size=(n, 5))
    y = np.column_stack([1.5 * x[:, 0] + rng.normal(scale=0.3, size=n), rng.normal(size=n)])
    sample_ids = [f"row-{index}" for index in range(n)]
    input_matrix = pd.DataFrame(
        {"sample_id": sample_ids, **{f"x{i}": x[:, i] for i in range(x.shape[1])}}
    )
    output_matrix = pd.DataFrame({"sample_id": sample_ids, "y0": y[:, 0], "y1": y[:, 1]})
    assignments = pd.DataFrame({"sample_id": sample_ids, "split": ["train"] * 40 + ["holdout"] * 8})
    catalog = pd.DataFrame(
        {"feature_name": [f"x{i}" for i in range(5)], "feature_type": "first_order"}
    )
    conditioning = condition_manuscript_outputs(
        output_matrix,
        assignments,
        OutputConditioningSpec(
            epsilon_var=0.0,
            epsilon_snr=0.0,
            snr_delta=1.0,
            method="pca",
            retained_components=2,
            retained_variance_fraction=1.0,
        ),
    )
    spec = EmpiricalNullScreeningSpec(
        statistic="coefficient_row_l2_norm",
        permutation_count_B=19,
        bh_q_screen=0.20,
        retained_terms_reference=0,
        random_seed=17,
    )
    return input_matrix, catalog, assignments, conditioning.pca_scores, spec


def test_screening_two_draw_blocks_equal_one_full_block():
    args = _fixture()
    full = score_screening_draw_block(*args, draw_start=0, draw_end=19)
    left = score_screening_draw_block(*args, draw_start=0, draw_end=7)
    right = score_screening_draw_block(*args, draw_start=7, draw_end=19)

    full_result = reduce_screening_draw_blocks([full], spec=args[-1])
    split_result = reduce_screening_draw_blocks([left, right], spec=args[-1])
    pd.testing.assert_frame_equal(
        full_result.feature_screening_statistics,
        split_result.feature_screening_statistics,
    )
    assert full.schedule_hash == left.schedule_hash == right.schedule_hash
    assert int(split_result.summary.loc[0, "n_training_rows"]) == 40


def test_screening_block_round_trip_is_one_bounded_npz(tmp_path):
    block = score_screening_draw_block(*_fixture(), draw_start=0, draw_end=7)
    block.write(tmp_path)
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "screening_block.json",
        "screening_block.npz",
    ]
    restored = ScreeningNullBlock.read(tmp_path)
    np.testing.assert_array_equal(restored.null_statistics, block.null_statistics)
    assert restored.artifact_hash == block.artifact_hash


def test_screening_reducer_rejects_gap_and_reordering():
    args = _fixture()
    left = score_screening_draw_block(*args, draw_start=0, draw_end=7)
    right = score_screening_draw_block(*args, draw_start=8, draw_end=19)
    with pytest.raises(ValueError, match="contiguous"):
        reduce_screening_draw_blocks([left, right], spec=args[-1])
    with pytest.raises(ValueError, match="contiguous"):
        reduce_screening_draw_blocks([right, left], spec=args[-1])
