"""Tests for test viz io."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bsm_rfm.viz_io import load_pipeline_outputs


def test_load_pipeline_outputs_reads_canonical_postfit_tables(tmp_path: Path):
    postfit = tmp_path / "postfit_diagnostics"
    postfit.mkdir()
    names = [
        "all_input_metadata",
        "selected_input_metadata",
        "output_metadata",
        "coef_matrix_standardized",
        "coef_matrix_raw_scale",
        "x_standardization",
        "y_standardization",
        "nrmse_summary",
    ]
    for name in names:
        pd.DataFrame({"a": [1]}).to_csv(postfit / f"{name}.csv", index=False)
    loaded = load_pipeline_outputs(tmp_path)
    assert set(loaded) == set(names)
    assert loaded["nrmse_summary"].iloc[0, 0] == 1
