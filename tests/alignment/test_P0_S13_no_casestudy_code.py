"""P0-S13: case-study-specific modules removed from generic src/.

Covers:
- Importing removed modules raises ImportError.
- `import rfm_pipeline` still succeeds.
- Removed symbols are absent from rfm_pipeline.__all__.
- No denylisted removed-case-study tokens reappear in src/rfm_pipeline/.

Known pre-existing exceptions (NOT checked by the denylist scan):
  - ``AFSC`` / ``UAEORO`` default column-name strings and ``add_scenario_flags``
    in ``src/rfm_pipeline/data.py`` are scenario-helper residuals that require a
    separate generalisation milestone.  Tracked in ``docs/scope_backlog.md``.
    They are intentionally excluded from the denylist below so the test remains
    green until that work is done, while still blocking re-introduction of the
    *removed* planning-add-on code.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Tokens that MUST NOT appear anywhere in src/rfm_pipeline/**/*.py.
# These correspond to removed case-study-specific modules / symbols.
# ---------------------------------------------------------------------------
_DENYLIST: frozenset[str] = frozenset(
    [
        "track_a_v3",
        "wave5_track_a",
        "wave5",
        "wave1234",
        "build_runtime_model_frame",
        "compute_near_bsm_sample_weights",
        "estimate_bsm_measurements_knn",
        "aggregate_wave5_replicates",
        "compute_efficiency_targets",
        "select_model_feature_columns",
    ]
)

_SRC_ROOT = Path(__file__).parents[2] / "src" / "rfm_pipeline"


# ---------------------------------------------------------------------------
# Original import / __all__ checks (must be retained)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Filesystem denylist scan
# ---------------------------------------------------------------------------


def test_no_removed_casestudy_tokens_in_src() -> None:
    """Fail if any denylisted removed-case-study token reappears in src/."""
    hits: list[str] = []
    for py_file in sorted(_SRC_ROOT.rglob("*.py")):
        source = py_file.read_text(encoding="utf-8")
        for token in sorted(_DENYLIST):
            if token in source:
                hits.append(f"{py_file.relative_to(_SRC_ROOT.parents[1])}:{token!r}")

    assert not hits, "Removed case-study tokens found in src/rfm_pipeline/:\n" + "\n".join(
        f"  {h}" for h in hits
    )
