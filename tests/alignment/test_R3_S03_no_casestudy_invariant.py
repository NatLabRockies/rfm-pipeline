"""R3-S03: Make the no-case-study-code invariant test honest (F4 test integrity).

This slice upgrades ``test_P0_S13_no_casestudy_code.py`` by adding a real
filesystem scan that ensures removed case-study tokens cannot silently
re-enter ``src/rfm_pipeline/``.

Previously the test only checked that two removed modules failed to import.
It did NOT scan ``src/`` for case-study literals, leaving the door open for
case-study-specific code to re-enter under different names.

This test:
  - scans ``src/rfm_pipeline/**/*.py`` for a denylist of removed-case-study
    tokens that MUST be absent.
  - documents the known pre-existing scenario-helper exceptions
    (``AFSC`` / ``UAEORO`` / ``add_scenario_flags``) as a tracked
    generalisation-backlog item rather than silently ignoring them.

See also: ``test_P0_S13_no_casestudy_code.py`` (original import/``__all__``
checks are retained there and still pass).

Known pre-existing exceptions (NOT in the denylist):
  ``AFSC``, ``UAEORO``, ``add_scenario_flags`` in ``src/rfm_pipeline/data.py``
  are scenario-helper residuals requiring a separate generalisation milestone.
  Tracked in ``docs/scope_backlog.md``.
"""

from __future__ import annotations

from pathlib import Path

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


def test_R3_S03_no_removed_casestudy_tokens_in_src() -> None:
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
