from __future__ import annotations

import subprocess
from pathlib import Path

from rfm_pipeline._svg_pdf import save_svg_as_pdf


def test_save_svg_as_pdf_invokes_headless_chrome(tmp_path: Path, monkeypatch) -> None:
    svg_path = tmp_path / "figure.svg"
    svg_path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="320"></svg>\n',
        encoding="utf-8",
    )

    call: dict[str, list[str]] = {}

    monkeypatch.setattr("rfm_pipeline._svg_pdf.find_chrome", lambda: "/opt/chrome")

    def _fake_run(
        cmd: list[str], *, check: bool, capture_output: bool
    ) -> subprocess.CompletedProcess[bytes]:
        assert check is True
        assert capture_output is True
        call["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr("rfm_pipeline._svg_pdf.subprocess.run", _fake_run)

    pdf_path = save_svg_as_pdf(svg_path)
    assert pdf_path == svg_path.with_suffix(".pdf")
    cmd = call["cmd"]
    assert cmd[0] == "/opt/chrome"
    assert f"--print-to-pdf={pdf_path}" in cmd
    assert cmd[-1].startswith("file://")


def test_target_scripts_use_shared_svg_pdf_helper() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    targets = [
        repo_root / "scripts" / "plot_sensitivity_results.py",
        repo_root / "scripts" / "plot_sensitivity_rf_figures.py",
        repo_root / "scripts" / "regenerate_manuscript_figures.py",
    ]

    for path in targets:
        text = path.read_text(encoding="utf-8")
        assert "from rfm_pipeline._svg_pdf import save_svg_as_pdf" in text
        assert "NamedTemporaryFile" not in text
