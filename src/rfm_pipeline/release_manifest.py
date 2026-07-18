"""Release/reproduction manifest builder for the rfm-pipeline.

Provides one-command enumeration of declared artifacts, SHA-256 checksums,
sizes, and an environment/lockfile reference.  A verify mode recomputes
checksums and reports drift without modifying the manifest.

F7 closure: reproducibility resources must be enumerable and verifiable.
"""

from __future__ import annotations

import datetime
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ArtifactEntry:
    """Single artifact record within a release manifest."""

    path: str
    sha256: str
    size_bytes: int


@dataclass
class ReleaseManifest:
    """Machine-readable release/reproduction manifest.

    Parameters
    ----------
    manifest_version:
        Schema version string for forward-compatibility.
    created_at:
        ISO-8601 UTC timestamp at build time.
    environment_ref:
        Path or identifier of the lockfile / environment descriptor used.
    artifacts:
        Ordered list of artifact entries with checksums and sizes.
    """

    manifest_version: str = "1.0"
    created_at: str = field(default_factory=lambda: _utcnow_iso())
    environment_ref: str = ""
    artifacts: list[ArtifactEntry] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return the manifest as a JSON-serializable dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize the manifest to JSON text."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReleaseManifest:
        """Build a manifest from a dictionary representation."""
        artifacts = [ArtifactEntry(**a) for a in data.get("artifacts", [])]
        return cls(
            manifest_version=data.get("manifest_version", "1.0"),
            created_at=data.get("created_at", ""),
            environment_ref=data.get("environment_ref", ""),
            artifacts=artifacts,
        )

    @classmethod
    def from_json(cls, text: str) -> ReleaseManifest:
        """Deserialize a manifest from JSON text."""
        return cls.from_dict(json.loads(text))


# ---------------------------------------------------------------------------
# Checksum helpers
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of *path*."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _utcnow_iso() -> str:
    return datetime.datetime.now(tz=datetime.timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_manifest(
    artifact_paths: list[Path | str],
    environment_ref: str = "",
    base_dir: Path | str | None = None,
) -> ReleaseManifest:
    """Build a :class:`ReleaseManifest` from a list of artifact paths.

    Parameters
    ----------
    artifact_paths:
        Iterable of file paths to include.  Each must exist.
    environment_ref:
        Path or label for the lockfile/environment descriptor (e.g.
        ``"pixi.lock"``).  May be empty.
    base_dir:
        When provided, artifact paths in the manifest are recorded relative
        to this directory.  Defaults to the absolute path.

    Returns
    -------
    ReleaseManifest
    """
    base = Path(base_dir).resolve() if base_dir is not None else None
    entries: list[ArtifactEntry] = []
    for raw in artifact_paths:
        p = Path(raw).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Artifact not found: {p}")
        recorded_path = str(p.relative_to(base)) if base is not None else str(p)
        entries.append(
            ArtifactEntry(
                path=recorded_path,
                sha256=_sha256_file(p),
                size_bytes=p.stat().st_size,
            )
        )
    return ReleaseManifest(
        environment_ref=environment_ref,
        artifacts=entries,
    )


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


@dataclass
class DriftReport:
    """Result of a manifest verification pass."""

    ok: bool
    missing: list[str] = field(default_factory=list)
    tampered: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)


def verify_manifest(
    manifest: ReleaseManifest,
    base_dir: Path | str | None = None,
) -> DriftReport:
    """Recompute checksums for every entry and report drift.

    Parameters
    ----------
    manifest:
        Previously built manifest.
    base_dir:
        Root directory to resolve relative paths against.  If *None*, paths
        are interpreted as-is (absolute or relative to cwd).

    Returns
    -------
    DriftReport
        ``ok`` is ``True`` only when every artifact is present and unchanged.
    """
    base = Path(base_dir).resolve() if base_dir is not None else Path(".")
    missing: list[str] = []
    tampered: list[str] = []
    messages: list[str] = []

    for entry in manifest.artifacts:
        p = (base / entry.path) if not Path(entry.path).is_absolute() else Path(entry.path)
        if not p.exists():
            missing.append(entry.path)
            messages.append(f"MISSING  {entry.path}")
            continue
        actual = _sha256_file(p)
        if actual != entry.sha256:
            tampered.append(entry.path)
            messages.append(
                f"TAMPERED {entry.path}  expected={entry.sha256[:16]}…  actual={actual[:16]}…"
            )

    return DriftReport(
        ok=not missing and not tampered,
        missing=missing,
        tampered=tampered,
        messages=messages,
    )
