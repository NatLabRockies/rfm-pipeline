"""PA-C0: feature-catalog generator must be case-study-agnostic.

The generic catalog generator previously hardcoded BSM binary predictor names
(``AFSC``, ``UAEORO``) in its structural-column exclusion sets, which both
(a) leaked case-study identifiers into the generic ``scripts/`` surface and
(b) silently dropped those binary predictors from the catalog — the exact Gate A
defect that excluded 2 of the 160 inputs from the run of record.

These tests pin the generalized contract:
- structural id columns are configurable, defaulting to generic names only,
- non-structural columns (including binary 0/1 predictors) flow through as
  first-order features regardless of their name,
- no removed case-study token appears in the generator source.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

_SCRIPT = Path(__file__).parents[2] / "scripts" / "generate_feature_catalog.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("_gen_feature_catalog", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_matrix(tmp_path: Path) -> Path:
    df = pd.DataFrame(
        {
            "sample_id": [0, 1, 2, 3],
            "scenario": ["a", "b", "a", "b"],
            "run_id": [10, 11, 12, 13],
            "num_x": [0.1, 0.2, 0.3, 0.4],
            "num_y": [1.0, 2.0, 3.0, 4.0],
            # Binary predictors with case-study-style names must NOT be dropped.
            "AFSC": [0, 1, 0, 1],
            "UAEORO": [1, 1, 0, 0],
        }
    )
    path = tmp_path / "X.parquet"
    df.to_parquet(path)
    return path


def test_binary_predictors_are_retained_as_features(tmp_path: Path) -> None:
    module = _load_module()
    _X, feature_cols = module.load_input_matrix(_write_matrix(tmp_path))
    assert "AFSC" in feature_cols
    assert "UAEORO" in feature_cols
    # Structural columns still excluded by default.
    for structural in ("sample_id", "scenario", "run_id"):
        assert structural not in feature_cols


def test_id_columns_are_configurable(tmp_path: Path) -> None:
    module = _load_module()
    _X, feature_cols = module.load_input_matrix(
        _write_matrix(tmp_path), id_columns={"sample_id", "AFSC"}
    )
    assert "AFSC" not in feature_cols
    assert "UAEORO" in feature_cols
    # Columns not named in id_columns are retained even if conventionally structural.
    assert "scenario" in feature_cols


def test_default_id_columns_contain_no_casestudy_names() -> None:
    module = _load_module()
    for token in ("AFSC", "UAEORO"):
        assert token not in set(module.DEFAULT_ID_COLUMNS)


def test_no_casestudy_tokens_in_generator_source() -> None:
    source = _SCRIPT.read_text(encoding="utf-8")
    hits = [token for token in ("AFSC", "UAEORO") if token in source]
    assert not hits, f"Case-study tokens found in {_SCRIPT.name}: {hits}"


def test_first_order_catalog_includes_binary_features(tmp_path: Path) -> None:
    module = _load_module()
    _X, feature_cols = module.load_input_matrix(_write_matrix(tmp_path))
    catalog = module.generate_first_order_features(feature_cols)
    names = set(catalog["feature_name"])
    assert {"AFSC", "UAEORO", "num_x", "num_y"} <= names


@pytest.mark.parametrize("strategy", ["safe", "all"])
def test_nonlinear_transforms_do_not_error_on_binary(tmp_path: Path, strategy: str) -> None:
    module = _load_module()
    X, feature_cols = module.load_input_matrix(_write_matrix(tmp_path))
    # Must not raise (e.g. inverse of a column containing zeros) and must not
    # emit degenerate transforms of two-point-support binary predictors.
    catalog = module.generate_nonlinear_transforms(X, feature_cols, strategy)
    for name in catalog["feature_name"]:
        assert not name.startswith("AFSC_")
        assert not name.startswith("UAEORO_")
