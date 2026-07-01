"""Shared SVG→PDF conversion via headless Chrome."""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

from rfm_pipeline._chrome import find_chrome

_SVG_SIZE_RE = re.compile(r'<svg[^>]+width="(\d+(?:\.\d+)?)"[^>]+height="(\d+(?:\.\d+)?)"')


def _svg_size(svg_text: str) -> tuple[int, int]:
    match = _SVG_SIZE_RE.search(svg_text)
    if match:
        return int(float(match.group(1))), int(float(match.group(2)))
    return 1200, 600


def save_svg_as_pdf(svg_path: Path) -> Path:
    """Render `svg_path` to a same-stem `.pdf` file using headless Chrome."""
    svg_text = svg_path.read_text(encoding="utf-8")
    width, height = _svg_size(svg_text)
    html_content = (
        "<!DOCTYPE html><html><head><style>"
        f"@page{{size:{width}px {height}px;margin:0}}"
        f"html,body{{margin:0;padding:0;width:{width}px;height:{height}px;overflow:hidden}}"
        "</style></head><body>"
        f"<img src='file://{svg_path.resolve()}' width='{width}' height='{height}'/>"
        "</body></html>"
    )

    pdf_path = svg_path.with_suffix(".pdf")
    chrome = find_chrome()
    with tempfile.TemporaryDirectory(prefix="rfm-svg-pdf-", dir=svg_path.parent) as tmp_dir:
        tmp_html = Path(tmp_dir) / f"{svg_path.stem}.print.html"
        tmp_html.write_text(html_content, encoding="utf-8")
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
    return pdf_path
