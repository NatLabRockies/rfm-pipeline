"""G1-S04: Generic demo manuscript fixtures (rename AFSC/UAEORO -> cat_a/cat_b).

Acceptance criteria exercised here:

- ``write_demo_manuscript_artifacts`` writes an input matrix whose two boolean
  categorical columns are named ``cat_a`` / ``cat_b`` (not ``AFSC`` / ``UAEORO``).
- ``input_metadata`` and ``manuscript_feature_catalog`` reference the generic
  names and remain internally consistent with the input matrix column set.
- No ``AFSC`` / ``UAEORO`` token remains in ``manuscript_runtime.py``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from rfm_pipeline import manuscript_runtime
from rfm_pipeline.manuscript_runtime import write_demo_manuscript_artifacts

_RUNTIME_PY = Path(manuscript_runtime.__file__)


def test_demo_artifacts_use_generic_categorical_names(tmp_path: Path) -> None:
    written = write_demo_manuscript_artifacts(tmp_path)

    input_matrix = pd.read_csv(written["case_study_input_matrix"])
    input_metadata = pd.read_csv(written["input_metadata"])

    assert "cat_a" in input_matrix.columns
    assert "cat_b" in input_matrix.columns
    assert "AFSC" not in input_matrix.columns
    assert "UAEORO" not in input_matrix.columns

    assert set(input_metadata["input_name"]) == {"x1", "x2", "cat_a", "cat_b"}
    assert "AFSC" not in set(input_metadata["input_name"])
    assert "UAEORO" not in set(input_metadata["input_name"])


def test_feature_catalog_first_order_names_align_with_input_matrix(tmp_path: Path) -> None:
    written = write_demo_manuscript_artifacts(tmp_path)
    input_matrix = pd.read_csv(written["case_study_input_matrix"])
    feature_catalog = pd.read_csv(written["manuscript_feature_catalog"])

    first_order = feature_catalog[feature_catalog["feature_type"] == "first_order"]
    first_order_names = set(first_order["feature_name"])
    input_cols = set(input_matrix.columns) - {"sample_id"}

    assert first_order_names == input_cols
    for token in ("AFSC", "UAEORO"):
        assert token not in set(feature_catalog["feature_name"])


def test_manuscript_runtime_source_has_no_afsc_uaeoro_tokens() -> None:
    text = _RUNTIME_PY.read_text(encoding="utf-8")
    assert "AFSC" not in text, "AFSC token must not remain in manuscript_runtime.py"
    assert "UAEORO" not in text, "UAEORO token must not remain in manuscript_runtime.py"
