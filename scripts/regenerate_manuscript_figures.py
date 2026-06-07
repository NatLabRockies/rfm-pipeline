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
import re
import subprocess
import sys
import tempfile
from pathlib import Path

# Ensure repo src is on the path when run directly
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from rfm_pipeline._chrome import find_chrome as _find_chrome  # noqa: E402
from rfm_pipeline.manuscript_stages import regenerate_figures_from_committed_data  # noqa: E402


def _save_pdf(svg_path: Path) -> None:
    """Convert SVG to PDF using Chrome headless."""
    svg_text = svg_path.read_text(encoding="utf-8")
    m = re.search(r'<svg[^>]+width="(\d+(?:\.\d+)?)"[^>]+height="(\d+(?:\.\d+)?)"', svg_text)
    w, h = (int(float(m.group(1))), int(float(m.group(2)))) if m else (1200, 600)
    html_content = (
        f"<!DOCTYPE html><html><head><style>"
        f"@page{{size:{w}px {h}px;margin:0}}"
        f"html,body{{margin:0;padding:0;width:{w}px;height:{h}px;overflow:hidden}}"
        f"</style></head><body><img src='file://{svg_path.resolve()}' "
        f"width='{w}' height='{h}'/></body></html>"
    )
    chrome = _find_chrome()
    pdf_path = svg_path.with_suffix(".pdf")
    with tempfile.NamedTemporaryFile(suffix=".html", mode="w", delete=False) as f:
        f.write(html_content)
        tmp_html = f.name
    try:
        subprocess.run(
            [
                chrome,
                "--headless=new",
                "--no-sandbox",
                "--disable-gpu",
                f"--print-to-pdf={pdf_path}",
                "--print-to-pdf-no-header",
                f"file://{tmp_html}",
            ],
            check=True,
            capture_output=True,
        )
    finally:
        Path(tmp_html).unlink(missing_ok=True)


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
