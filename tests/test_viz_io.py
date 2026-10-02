"""Tests for test viz io."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from rfm_pipeline.viz_io import load_postfit_bundle


def test_load_postfit_bundle_follows_manifest_file_map(tmp_path: Path):
    postfit = tmp_path / "custom_tables"
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
    (tmp_path / "manifest.json").write_text(
        json.dumps({"files": {name: f"custom_tables/{name}.csv" for name in names}}),
        encoding="utf-8",
    )

    loaded = load_postfit_bundle(tmp_path)

    assert set(loaded) == set(names)
    assert loaded["nrmse_summary"].iloc[0, 0] == 1


def test_load_postfit_bundle_requires_manifest(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="manifest.json"):
        load_postfit_bundle(tmp_path)


def test_load_postfit_bundle_rejects_empty_required_table(tmp_path: Path):
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
    postfit = tmp_path / "postfit_diagnostics"
    postfit.mkdir()
    for name in names:
        path = postfit / f"{name}.csv"
        if name == "coef_matrix_raw_scale":
            path.write_text("", encoding="utf-8")
        else:
            pd.DataFrame({"a": [1]}).to_csv(path, index=False)
    (tmp_path / "manifest.json").write_text(
        json.dumps({"files": {name: f"postfit_diagnostics/{name}.csv" for name in names}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="coef_matrix_raw_scale.*empty"):
        load_postfit_bundle(tmp_path)
