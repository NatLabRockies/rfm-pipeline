"""Recovery runner contract after the Generation-11 production-path repair."""

from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import rfm_pipeline.recovery_study as recovery
from rfm_pipeline.campaign_contract import G11_CONTRACT
from rfm_pipeline.recovery_study import run_production_recovery_pipeline


def _fixture(*, n_inputs: int = 160) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(20260812)
    x_train = rng.normal(size=(24, n_inputs))
    x_eval = rng.normal(size=(8, n_inputs))
    if n_inputs == 160:
        x_train[:, 158:] = rng.integers(0, 2, size=(24, 2))
        x_eval[:, 158:] = rng.integers(0, 2, size=(8, 2))
    y_train = rng.normal(size=(24, 2))
    y_eval = rng.normal(size=(8, 2))
    return x_train, y_train, x_eval, y_eval


def test_recovery_runner_accepts_only_one_typed_scientific_contract() -> None:
    """Callers cannot tune B, alpha, trees, or stability outside the contract."""
    parameters = inspect.signature(run_production_recovery_pipeline).parameters
    assert {
        "execution_contract",
        "artifact_dir",
        "seed",
    } <= set(parameters)
    assert {
        "permutation_count_B",
        "B_screen",
        "B_interaction",
        "alpha",
        "selection_method",
        "n_tree_estimators",
        "n_stability_subsamples",
    }.isdisjoint(parameters)


def test_recovery_runner_requires_all_158_continuous_and_two_binary_inputs(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="158 continuous plus 2 binary"):
        run_production_recovery_pipeline(
            *_fixture(n_inputs=159),
            artifact_dir=tmp_path / "wrong-interface",
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("B_screen", 3198, "permutation floors"),
        ("B_interaction", 998, "permutation floors"),
        ("n_tree_estimators", 249, "250 tree estimators"),
        ("n_stability_subsamples", 49, "50 stability subsamples"),
    ],
)
def test_recovery_runner_rejects_contract_drift_before_scoring(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: int,
    message: str,
) -> None:
    monkeypatch.setattr(
        recovery,
        "condition_manuscript_outputs",
        lambda *args, **kwargs: pytest.fail("contract drift reached scientific scoring"),
    )
    with pytest.raises(ValueError, match=message):
        run_production_recovery_pipeline(
            *_fixture(),
            execution_contract=replace(G11_CONTRACT, **{field: value}),
            artifact_dir=tmp_path / field,
        )


def test_recovery_runner_source_uses_persisted_global_reducer_and_terminal_freeze() -> None:
    """The submitted path contains the required persistence and terminal stages."""
    source = inspect.getsource(run_production_recovery_pipeline)
    required = (
        "score_interaction_draw_block",
        "artifact.write_to",
        "ScoreOnlyInteractionArtifact.read_from",
        "reduce_score_only_interaction_artifacts",
        "select_manuscript_sparse_support",
        "terminal_train_fit_and_freeze",
        "write_terminal_train_fit_and_freeze",
        "holdout_predict",
    )
    for symbol in required:
        assert symbol in source
    assert "discover_manuscript_interactions" not in source
    assert "except" not in source


def test_evaluation_truth_is_not_materialized_before_terminal_freeze() -> None:
    """Both terminal branches freeze a model before converting evaluation truth."""
    source = inspect.getsource(run_production_recovery_pipeline)
    truth_reads = [
        index
        for index in range(len(source))
        if source.startswith("y_eval_arr = _to_arr(Y_eval)", index)
    ]
    assert len(truth_reads) == 2
    intercept_freeze = source.index("write_train_fit_and_freeze(")
    terminal_freeze = source.index("write_terminal_train_fit_and_freeze(")
    assert intercept_freeze < truth_reads[0]
    assert terminal_freeze < truth_reads[1]
