"""Recovery runner contract after the Generation-11 production-path repair."""

from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import rfm_pipeline.recovery_study as recovery
from rfm_pipeline.campaign_contract import G11_CONTRACT
from rfm_pipeline.manuscript_stages import (
    FinalManuscriptArtifactsSpec,
    _terminal_train_fit_and_freeze,
    terminal_train_fit_and_freeze,
    write_terminal_train_fit_and_freeze,
)
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
    assert "completed_empty_hc3_support" in source


def _terminal_spec() -> FinalManuscriptArtifactsSpec:
    return FinalManuscriptArtifactsSpec(
        final_predictor_count_reference=0,
        final_first_order_input_count_reference=0,
        intermediate_penalized_holdout_nrmse_reference=0.0,
        final_ols_holdout_nrmse_reference=0.0,
        nrmse_denominator_definition="training_response_range",
        nrmse_min_range=1.0e-12,
        nrmse_reference_matrix="Y_train",
        bootstrap_count=2,
        inferential_filter_alpha=0.05,
        n_jobs=1,
    )


def test_recovery_terminal_freezes_training_mean_for_empty_hc3_support(
    tmp_path: Path,
) -> None:
    """An empty recovery support is a measured outcome, not a failed replicate."""
    index = pd.Index(range(12), name="sample_id")
    x_train = pd.DataFrame({"x0": np.zeros(12)}, index=index)
    y_train = pd.DataFrame(
        {
            "y0": np.linspace(-1.0, 1.0, 12),
            "y1": np.linspace(2.0, 3.0, 12),
        },
        index=index,
    )

    terminal = _terminal_train_fit_and_freeze(
        x_train,
        y_train,
        spec=_terminal_spec(),
        contract_hash="a" * 64,
        allow_empty_hc3_support=True,
    )

    assert terminal.prefilter_feature_names == ("x0",)
    assert terminal.hc3_feature_names == ()
    assert terminal.final_feature_names == ()
    assert terminal.freeze_result.freeze_manifest.feature_names == ()
    np.testing.assert_allclose(
        terminal.freeze_result.intercept,
        y_train.to_numpy(dtype=float).mean(axis=0),
    )
    assert not terminal.hc3_inferential_filter_summary["hc3_retained_after_filter"].any()
    assert terminal.feature_pruning_impact.empty
    assert terminal.feature_pruning_summary.loc[0, "status"] == "not_run_empty_hc3_support"

    output_dir = tmp_path / "empty-hc3-terminal"
    write_terminal_train_fit_and_freeze(terminal=terminal, output_dir=output_dir)
    assert (output_dir / "terminal_manifest.json").exists()
    assert (output_dir / "model" / "freeze_manifest.json").exists()


def test_manuscript_terminal_still_rejects_empty_hc3_support() -> None:
    """Recovery handling must not weaken the publication terminal gate."""
    index = pd.Index(range(12), name="sample_id")
    x_train = pd.DataFrame({"x0": np.zeros(12)}, index=index)
    y_train = pd.DataFrame({"y0": np.linspace(-1.0, 1.0, 12)}, index=index)

    with pytest.raises(ValueError, match="retained no final features"):
        terminal_train_fit_and_freeze(
            x_train,
            y_train,
            spec=_terminal_spec(),
            contract_hash="b" * 64,
        )


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


def test_recovery_runner_records_persisted_interaction_payload_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The production path must use the score artifact's canonical identity."""
    import pandas as pd

    payload_sha256 = "b" * 64

    class PayloadOnlyArtifact:
        def write_to(self, directory: Path) -> None:
            directory.mkdir(parents=True)

        @property
        def payload_sha256(self) -> str:
            return payload_sha256

    monkeypatch.setattr(
        recovery,
        "condition_manuscript_outputs",
        lambda *args, **kwargs: SimpleNamespace(
            pca_scores=pd.DataFrame({"sample_id": range(24), "PC1": np.zeros(24)})
        ),
    )
    monkeypatch.setattr(
        recovery,
        "screen_manuscript_empirical_null_terms",
        lambda *args, **kwargs: SimpleNamespace(
            retained_terms=pd.DataFrame(columns=["feature_name", "feature_type"]),
            feature_screening_statistics=pd.DataFrame({"feature_name": ["x0"]}),
        ),
    )
    monkeypatch.setattr(
        recovery,
        "score_interaction_draw_block",
        lambda *args, **kwargs: PayloadOnlyArtifact(),
    )
    monkeypatch.setattr(
        recovery.ScoreOnlyInteractionArtifact,
        "read_from",
        classmethod(lambda cls, directory: PayloadOnlyArtifact()),
    )
    monkeypatch.setattr(
        recovery,
        "reduce_score_only_interaction_artifacts",
        lambda *args, **kwargs: SimpleNamespace(
            retained_pairs=pd.DataFrame(columns=["pair_name"]),
            pair_scores=pd.DataFrame(columns=["pair_name"]),
        ),
    )
    freeze = SimpleNamespace(freeze_manifest=SimpleNamespace(freeze_hash="c" * 64))
    monkeypatch.setattr(recovery, "train_fit_and_freeze", lambda *args, **kwargs: freeze)
    monkeypatch.setattr(recovery, "write_train_fit_and_freeze", lambda *args, **kwargs: None)
    frozen = SimpleNamespace(y_pred=np.zeros((8, 2)))
    monkeypatch.setattr(recovery, "holdout_predict", lambda *args, **kwargs: frozen)
    monkeypatch.setattr(
        recovery,
        "write_frozen_prediction_matrices",
        lambda *args, **kwargs: None,
    )

    result = run_production_recovery_pipeline(
        *_fixture(),
        artifact_dir=tmp_path / "payload-hashes",
    )

    assert result.interaction_artifact_checksums == (payload_sha256,)
