"""Tests for P2-S01: one-command release/reproduction manifest with checksums (F7)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rfm_pipeline.release_manifest import (
    ReleaseManifest,
    build_manifest,
    verify_manifest,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def synthetic_artifact_tree(tmp_path: Path) -> Path:
    """Create a small synthetic artifact directory tree."""
    (tmp_path / "tables").mkdir()
    (tmp_path / "figures").mkdir()
    (tmp_path / "tables" / "table1.csv").write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    (tmp_path / "tables" / "table2.csv").write_text("x,y\n5,6\n", encoding="utf-8")
    (tmp_path / "figures" / "fig1.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    return tmp_path


# ---------------------------------------------------------------------------
# P2-S01-A: manifest build over a synthetic artifact tree
# ---------------------------------------------------------------------------


def test_P2_S01_build_manifest_records_all_artifacts(
    synthetic_artifact_tree: Path,
) -> None:
    artifact_files = list(synthetic_artifact_tree.rglob("*"))
    artifact_files = [p for p in artifact_files if p.is_file()]

    manifest = build_manifest(
        artifact_paths=artifact_files,
        environment_ref="pixi.lock",
        base_dir=synthetic_artifact_tree,
    )

    assert len(manifest.artifacts) == len(artifact_files)
    recorded_paths = {a.path for a in manifest.artifacts}
    for p in artifact_files:
        rel = str(p.relative_to(synthetic_artifact_tree))
        assert rel in recorded_paths, f"Missing entry for {rel}"


def test_P2_S01_build_manifest_checksums_are_sha256(
    synthetic_artifact_tree: Path,
) -> None:
    import hashlib

    csv_path = synthetic_artifact_tree / "tables" / "table1.csv"
    manifest = build_manifest(
        artifact_paths=[csv_path],
        environment_ref="pixi.lock",
        base_dir=synthetic_artifact_tree,
    )
    entry = manifest.artifacts[0]
    expected = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    assert entry.sha256 == expected
    assert len(entry.sha256) == 64


def test_P2_S01_build_manifest_size_bytes_correct(
    synthetic_artifact_tree: Path,
) -> None:
    csv_path = synthetic_artifact_tree / "tables" / "table1.csv"
    manifest = build_manifest(
        artifact_paths=[csv_path],
        base_dir=synthetic_artifact_tree,
    )
    entry = manifest.artifacts[0]
    assert entry.size_bytes == csv_path.stat().st_size


# ---------------------------------------------------------------------------
# P2-S01-B: environment reference present
# ---------------------------------------------------------------------------


def test_P2_S01_environment_ref_present(synthetic_artifact_tree: Path) -> None:
    csv_path = synthetic_artifact_tree / "tables" / "table1.csv"
    manifest = build_manifest(
        artifact_paths=[csv_path],
        environment_ref="pixi.lock",
        base_dir=synthetic_artifact_tree,
    )
    assert manifest.environment_ref == "pixi.lock"
    assert manifest.environment_ref  # non-empty


def test_P2_S01_manifest_created_at_present(synthetic_artifact_tree: Path) -> None:
    csv_path = synthetic_artifact_tree / "tables" / "table1.csv"
    manifest = build_manifest(artifact_paths=[csv_path])
    assert manifest.created_at
    # ISO-8601 UTC format check
    assert "T" in manifest.created_at


# ---------------------------------------------------------------------------
# P2-S01-C: checksum verification detects a tampered file
# ---------------------------------------------------------------------------


def test_P2_S01_verify_clean_manifest_passes(synthetic_artifact_tree: Path) -> None:
    artifact_files = [p for p in synthetic_artifact_tree.rglob("*") if p.is_file()]
    manifest = build_manifest(
        artifact_paths=artifact_files,
        environment_ref="pixi.lock",
        base_dir=synthetic_artifact_tree,
    )
    report = verify_manifest(manifest, base_dir=synthetic_artifact_tree)
    assert report.ok
    assert report.missing == []
    assert report.tampered == []


def test_P2_S01_verify_detects_tampered_file(synthetic_artifact_tree: Path) -> None:
    csv_path = synthetic_artifact_tree / "tables" / "table1.csv"
    manifest = build_manifest(
        artifact_paths=[csv_path],
        environment_ref="pixi.lock",
        base_dir=synthetic_artifact_tree,
    )

    # Tamper the file after building the manifest.
    csv_path.write_text("TAMPERED CONTENT\n", encoding="utf-8")

    report = verify_manifest(manifest, base_dir=synthetic_artifact_tree)
    assert not report.ok
    assert len(report.tampered) == 1
    assert report.missing == []


def test_P2_S01_verify_detects_missing_file(synthetic_artifact_tree: Path) -> None:
    csv_path = synthetic_artifact_tree / "tables" / "table1.csv"
    manifest = build_manifest(
        artifact_paths=[csv_path],
        environment_ref="pixi.lock",
        base_dir=synthetic_artifact_tree,
    )

    # Remove the file.
    csv_path.unlink()

    report = verify_manifest(manifest, base_dir=synthetic_artifact_tree)
    assert not report.ok
    assert len(report.missing) == 1
    assert report.tampered == []


# ---------------------------------------------------------------------------
# P2-S01-D: manifest round-trips through JSON
# ---------------------------------------------------------------------------


def test_P2_S01_manifest_json_round_trip(synthetic_artifact_tree: Path) -> None:
    artifact_files = [p for p in synthetic_artifact_tree.rglob("*") if p.is_file()]
    manifest = build_manifest(
        artifact_paths=artifact_files,
        environment_ref="pixi.lock",
        base_dir=synthetic_artifact_tree,
    )
    serialised = manifest.to_json()
    recovered = ReleaseManifest.from_json(serialised)
    assert recovered.manifest_version == manifest.manifest_version
    assert recovered.environment_ref == manifest.environment_ref
    assert len(recovered.artifacts) == len(manifest.artifacts)
    for orig, rec in zip(manifest.artifacts, recovered.artifacts, strict=True):
        assert orig.path == rec.path
        assert orig.sha256 == rec.sha256
        assert orig.size_bytes == rec.size_bytes


def test_P2_S01_manifest_to_dict_is_serialisable(synthetic_artifact_tree: Path) -> None:
    csv_path = synthetic_artifact_tree / "tables" / "table1.csv"
    manifest = build_manifest(artifact_paths=[csv_path], environment_ref="pixi.lock")
    d = manifest.to_dict()
    # Must be JSON-serialisable with standard library.
    json.dumps(d)
    assert "artifacts" in d
    assert "environment_ref" in d
    assert "manifest_version" in d
    assert "created_at" in d


# ---------------------------------------------------------------------------
# P2-S01-E: missing artifact raises on build
# ---------------------------------------------------------------------------


def test_P2_S01_build_raises_on_nonexistent_artifact(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_manifest(artifact_paths=[tmp_path / "does_not_exist.csv"])
