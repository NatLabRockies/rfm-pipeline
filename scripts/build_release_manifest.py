#!/usr/bin/env python
r"""One-command release/reproduction manifest builder.

Usage (via Pixi):
    pixi run python scripts/build_release_manifest.py build \\
        --artifacts outputs/table1.csv outputs/fig1.pdf \\
        --env-ref pixi.lock \\
        --out release_manifest.json

    pixi run python scripts/build_release_manifest.py verify \\
        --manifest release_manifest.json

Exits non-zero on verification drift or missing artifacts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from the repo root without installing.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rfm_pipeline.release_manifest import (
    ReleaseManifest,
    build_manifest,
    verify_manifest,
)


def cmd_build(args: argparse.Namespace) -> int:
    """Build a manifest from parsed command-line arguments."""
    artifact_paths = [Path(p) for p in args.artifacts]
    manifest = build_manifest(
        artifact_paths=artifact_paths,
        environment_ref=args.env_ref or "",
        base_dir=args.base_dir or None,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(manifest.to_json(), encoding="utf-8")
    print(f"Manifest written to {out_path}  ({len(manifest.artifacts)} artifact(s))")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Verify a manifest from parsed command-line arguments."""
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 2
    manifest = ReleaseManifest.from_json(manifest_path.read_text(encoding="utf-8"))
    report = verify_manifest(manifest, base_dir=args.base_dir or None)
    for msg in report.messages:
        print(msg)
    if report.ok:
        print(f"OK  {len(manifest.artifacts)} artifact(s) verified, no drift detected.")
        return 0
    else:
        print(
            f"DRIFT DETECTED: {len(report.missing)} missing, {len(report.tampered)} tampered.",
            file=sys.stderr,
        )
        return 1


def main(argv: list[str] | None = None) -> int:
    """Parse command-line arguments and run the selected subcommand."""
    parser = argparse.ArgumentParser(description="Build or verify a release/reproduction manifest.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="Build a new manifest.")
    p_build.add_argument(
        "--artifacts", nargs="+", required=True, metavar="PATH", help="Artifact files."
    )
    p_build.add_argument("--env-ref", default="pixi.lock", help="Lockfile/env reference.")
    p_build.add_argument("--base-dir", default=None, help="Base directory for relative paths.")
    p_build.add_argument("--out", default="release_manifest.json", help="Output manifest path.")

    p_verify = sub.add_parser("verify", help="Verify an existing manifest.")
    p_verify.add_argument("--manifest", default="release_manifest.json", help="Manifest path.")
    p_verify.add_argument("--base-dir", default=None, help="Base directory for relative paths.")

    args = parser.parse_args(argv)
    if args.command == "build":
        return cmd_build(args)
    return cmd_verify(args)


if __name__ == "__main__":
    sys.exit(main())
