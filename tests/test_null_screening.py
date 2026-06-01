"""Tests for test null screening."""

from __future__ import annotations

import pandas as pd

from rfm_pipeline.null_screening import NullScreeningConfig, run_null_screening_with_source


class FakeSourceModule:
    def __init__(self):
        self.calls: dict[str, object] = {}

    def run_delta_for_screening(self, items, **kwargs):
        self.calls["items"] = list(items)
        self.calls["kwargs"] = kwargs
        observed = pd.DataFrame(
            {"dataset": ["d1", "d1"], "variable": ["x1", "x2"], "delta": [0.3, 0.1]}
        )
        significant = pd.DataFrame(
            {
                "dataset": ["d1", "d1"],
                "output": ["y1", "y1"],
                "variable": ["x1", "x2"],
                "is_significant": [True, False],
            }
        )
        return observed, significant, {("d1", "y1"): {"cutoff": 0.2}}

    def determine_influential_inputs(self, sig_all: pd.DataFrame, *, min_outputs: int = 1):
        self.calls["min_outputs_determine"] = min_outputs
        assert sig_all["variable"].tolist() == ["x1", "x2"]
        return {("m1", "s1"): ["x1"]}

    def ensure_influentials_or_fallback(
        self,
        infl_map,
        observed_all: pd.DataFrame,
        *,
        min_outputs: int,
    ):
        self.calls["min_outputs_fallback"] = min_outputs
        assert observed_all["variable"].tolist() == ["x1", "x2"]
        return infl_map


def test_run_null_screening_with_source_delegates_to_recovered_script_interface():
    source = FakeSourceModule()
    cfg = NullScreeningConfig(B=50, alpha=0.01, method="fwer-max", min_outputs=2)
    result = run_null_screening_with_source(source, items=[("d1", "payload")], config=cfg)
    assert source.calls["items"] == [("d1", "payload")]
    kwargs = source.calls["kwargs"]
    assert kwargs["B"] == 50
    assert kwargs["alpha"] == 0.01
    assert kwargs["method"] == "fwer-max"
    assert source.calls["min_outputs_determine"] == 2
    assert source.calls["min_outputs_fallback"] == 2
    assert result.influential_inputs == {("m1", "s1"): ["x1"]}
    assert result.provenance["screening_method"] == (
        "SALib delta sensitivity with permutation null"
    )
