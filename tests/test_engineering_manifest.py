"""Tests for the repository engineering manifest."""

from pathlib import Path

MANIFEST_PATH = Path("docs/ENGINEERING_MANIFEST.md")


def test_engineering_manifest_exists_and_is_indexed() -> None:
    """The authoritative engineering manifest should be part of the public docs."""
    assert MANIFEST_PATH.exists()
    assert "ENGINEERING_MANIFEST" in Path("docs/index.md").read_text(encoding="utf-8")


def test_engineering_manifest_records_current_completed_work() -> None:
    """The manifest should reflect implemented live-repo reproduction infrastructure."""
    text = MANIFEST_PATH.read_text(encoding="utf-8")
    required_phrases = [
        "Phase 3 source-backed stage chain",
        "QA audit layer",
        "manuscript-reproduction-smoke",
        "docs/manuscript_alignment_audit.md",
    ]

    for phrase in required_phrases:
        assert phrase in text


def test_engineering_manifest_records_remaining_exactness_gaps() -> None:
    """The manifest should preserve the known manuscript-exactness boundaries."""
    text = MANIFEST_PATH.read_text(encoding="utf-8")
    required_phrases = [
        "tree-SHAP interaction",
        "GAM EDF/p-value",
        "de-biased-LASSO",
        "real-data HC3",
        "table and figure verification",
    ]

    for phrase in required_phrases:
        assert phrase in text


def test_engineering_manifest_does_not_reopen_completed_phase_three_slices() -> None:
    """Completed Phase 3 infrastructure should not be described as future next work."""
    text = MANIFEST_PATH.read_text(encoding="utf-8")
    stale_phrases = [
        "next Phase 3 slice should implement final manuscript table",
        "Only after that should Phase 3 implementation begin",
    ]

    for phrase in stale_phrases:
        assert phrase not in text
