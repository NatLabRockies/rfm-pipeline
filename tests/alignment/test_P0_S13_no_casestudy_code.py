"""P0-S13: case-study-specific modules removed from generic src/.

Covers:
- Importing removed modules raises ImportError.
- `import rfm_pipeline` still succeeds.
- Removed symbols are absent from rfm_pipeline.__all__.
"""

from __future__ import annotations

import importlib

import pytest


def test_track_a_v3_import_raises() -> None:
    with pytest.raises(ImportError):
        importlib.import_module("rfm_pipeline.track_a_v3")


def test_wave5_track_a_import_raises() -> None:
    with pytest.raises(ImportError):
        importlib.import_module("rfm_pipeline.wave5_track_a")


def test_rfm_pipeline_top_level_import_succeeds() -> None:
    import rfm_pipeline  # noqa: F401


def test_removed_symbols_absent_from_all() -> None:
    import rfm_pipeline

    removed = {
        "build_runtime_model_frame",
        "compute_near_bsm_sample_weights",
        "estimate_bsm_measurements_knn",
        "aggregate_wave5_replicates",
        "compute_efficiency_targets",
        "select_model_feature_columns",
    }
    present = removed & set(rfm_pipeline.__all__)
    assert not present, f"Removed symbols still in __all__: {present}"
