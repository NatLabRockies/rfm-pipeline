"""Regression coverage for the full-scale G11 pilot recovery worker."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import rfm_pipeline.hpc_campaign_package as campaign
import rfm_pipeline.recovery_study as recovery_study


def test_pilot_recovery_preserves_feature_names_for_comparator_designs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The recovery runner and algebraic materializer must share one name contract."""
    observed: dict[str, object] = {}
    algebraic_names = ("x000", "x003", "x000:x001", "x002_sq", "binary_0")

    def fake_pipeline(
        x_train: pd.DataFrame,
        y_train: np.ndarray,
        x_eval: pd.DataFrame,
        y_eval: np.ndarray,
        **_: object,
    ) -> SimpleNamespace:
        expected_names = [f"x{index:03d}" for index in range(158)] + [
            "binary_0",
            "binary_1",
        ]
        assert isinstance(x_train, pd.DataFrame)
        assert isinstance(x_eval, pd.DataFrame)
        assert x_train.columns.tolist() == expected_names
        assert x_eval.columns.tolist() == expected_names
        observed["x_train"] = x_train.copy()
        observed["x_eval"] = x_eval.copy()
        return SimpleNamespace(
            algebraic_candidate_names=algebraic_names,
            eval_predictions=np.zeros((len(x_eval), y_train.shape[1]), dtype=float),
            screening_retained_set=frozenset({"x000", "x001"}),
            interaction_retained_set=frozenset({"x000:x001"}),
            final_selected_support=frozenset({"x000", "x000:x001"}),
        )

    def fake_comparators(**kwargs: object) -> SimpleNamespace:
        x_train = observed["x_train"]
        x_eval = observed["x_eval"]
        assert isinstance(x_train, pd.DataFrame)
        assert isinstance(x_eval, pd.DataFrame)
        expected_train = np.column_stack(
            [
                x_train["x000"],
                x_train["x003"],
                x_train["x000"] * x_train["x001"],
                x_train["x002"] ** 2,
                x_train["binary_0"],
            ]
        )
        expected_eval = np.column_stack(
            [
                x_eval["x000"],
                x_eval["x003"],
                x_eval["x000"] * x_eval["x001"],
                x_eval["x002"] ** 2,
                x_eval["binary_0"],
            ]
        )
        np.testing.assert_allclose(kwargs["algebraic_train_design"], expected_train)
        np.testing.assert_allclose(kwargs["algebraic_eval_design"], expected_eval)
        return SimpleNamespace(
            schedule_sha256="a" * 64,
            predictions={"proposed_terminal_workflow": kwargs["proposed_predictions"]},
        )

    monkeypatch.setattr(recovery_study, "run_production_recovery_pipeline", fake_pipeline)
    monkeypatch.setattr(recovery_study, "run_recovery_comparators", fake_comparators)
    monkeypatch.setattr(
        recovery_study,
        "write_recovery_comparator_result",
        lambda *_args, **_kwargs: {},
    )

    result = campaign._run_pilot_recovery(
        {"schedule_hash": "1" * 64, "config_hash": "2" * 64},
        tmp_path,
    )

    assert result["status"] == "completed"
    assert result["comparator_names"] == ["proposed_terminal_workflow"]
