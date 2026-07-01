#!/usr/bin/env python
"""Regenerate manuscript SVG figures from pre-collected run artifacts.

Reads the CSV data files produced by the final_manuscript_artifacts stage and
re-renders all SVG figures using the current (updated) rendering functions in
manuscript_stages.py.  Use this after collecting a run locally to apply any
styling or rendering improvements without re-running the full pipeline.

Usage:
    pixi run python scripts/regenerate_manuscript_figures.py \
        --artifacts-dir artifacts/publication_full_dataset_distributed_results/\
publication_full_dataset_distributed_20260526_short_hp1

The script overwrites the SVG files in
<artifacts-dir>/artifacts/final_manuscript_artifacts/figures/
and prints a summary of what was written.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repo src is on the path when run directly
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from rfm_pipeline._svg_pdf import save_svg_as_pdf  # noqa: E402
from rfm_pipeline.manuscript_stages import regenerate_figures_from_committed_data  # noqa: E402


def _save_pdf(svg_path: Path) -> None:
    """Convert SVG to PDF using the shared Chrome-based helper."""
    save_svg_as_pdf(svg_path)


def regenerate(artifacts_root: Path) -> dict[str, Path]:
    """Regenerate all manuscript SVG figures from collected artifact CSVs."""
    fig_dir = artifacts_root / "artifacts" / "final_manuscript_artifacts" / "figures"
    table_dir = artifacts_root / "artifacts" / "final_manuscript_artifacts" / "tables"

    if not fig_dir.exists():
        raise FileNotFoundError(f"Figures directory not found: {fig_dir}")

    written = regenerate_figures_from_committed_data(
        figure_data_dir=fig_dir,
        tables_dir=table_dir,
        output_dir=fig_dir,
    )
    return written


def main() -> None:
    """Parse arguments and regenerate manuscript figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts-dir",
        required=True,
        help="Local root of the collected run (e.g. artifacts/publication_full_dataset_distributed_results/...)",  # noqa: E501
    )
    args = parser.parse_args()

    artifacts_root = Path(args.artifacts_dir).expanduser().resolve()
    if not artifacts_root.exists():
        print(f"error: artifacts directory not found: {artifacts_root}", file=sys.stderr)
        sys.exit(1)

    print(f"Regenerating figures from: {artifacts_root}")
    written = regenerate(artifacts_root)
    for _name, path in sorted(written.items()):
        print(f"  wrote {path.name}  ({path.stat().st_size:,} bytes)")
        _save_pdf(path)
        print(f"  wrote {path.stem}.pdf")
    print(f"\nDone — {len(written)} figures written.")


if __name__ == "__main__":
    main()
