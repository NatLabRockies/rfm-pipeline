#!/usr/bin/env python
"""Build compact Track A v3 analytics/figures for manuscript add-on framing."""
# ruff: noqa: E501

from __future__ import annotations

import json
import math
import subprocess
import tempfile
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

from rfm_pipeline._chrome import find_chrome as _find_chrome

ROOT = Path(__file__).resolve().parents[1]
IN_DIR = ROOT / "artifacts" / "sensitivity" / "wave5_measurement_models_v3"
OUT_DIR = IN_DIR
DOCS_OUT = ROOT / "docs" / "manuscripts"


def _save_pdf(svg_path: Path) -> Path:
    svg_text = svg_path.read_text(encoding="utf-8")
    width = 1100
    height = 650
    if 'width="' in svg_text and 'height="' in svg_text:
        try:
            width = int(float(svg_text.split('width="', 1)[1].split('"', 1)[0]))
            height = int(float(svg_text.split('height="', 1)[1].split('"', 1)[0]))
        except (IndexError, ValueError):
            pass

    html_content = (
        "<!DOCTYPE html><html><head><style>"
        f"@page{{size:{width}px {height}px;margin:0}}"
        f"html,body{{margin:0;padding:0;width:{width}px;height:{height}px;overflow:hidden}}"
        "</style></head><body>"
        f"<img src='file://{svg_path.resolve()}' width='{width}' height='{height}'/>"
        "</body></html>"
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
    return pdf_path


def _band(v: float, q1: float, q2: float) -> str:
    if v <= q1:
        return "likely acceptable"
    if v <= q2:
        return "borderline"
    return "likely unacceptable"


def _write_svg(path: Path, width: int, height: int, body: str) -> None:
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        f"{body}</svg>"
    )
    path.write_text(svg, encoding="utf-8")


def _build_bsm_interval_figure(
    bsm: dict[str, float],
    q1: float,
    q2: float,
    out_svg: Path,
) -> None:
    width, height = 1100, 500
    left, right, top, bottom = 90, 40, 80, 90
    x_min = 0.0
    x_max = max(0.14, float(bsm["nrmse_pred_high95"]) * 1.05)
    plot_w = width - left - right
    axis_y = 260

    def xpix(v: float) -> float:
        return left + (v - x_min) / max(x_max - x_min, 1e-9) * plot_w

    bg = [
        f'<rect x="{xpix(0)}" y="{top}" width="{xpix(q1) - xpix(0)}" height="{height - top - bottom}" '
        'fill="#2ca02c" fill-opacity="0.14"/>',
        f'<rect x="{xpix(q1)}" y="{top}" width="{xpix(q2) - xpix(q1)}" height="{height - top - bottom}" '
        'fill="#ffbf00" fill-opacity="0.16"/>',
        f'<rect x="{xpix(q2)}" y="{top}" width="{xpix(x_max) - xpix(q2)}" height="{height - top - bottom}" '
        'fill="#d62728" fill-opacity="0.14"/>',
    ]
    grid = []
    for t in np.linspace(0, x_max, 8):
        x = xpix(float(t))
        grid.append(
            f'<line x1="{x}" y1="{top}" x2="{x}" y2="{height - bottom}" stroke="#bbbbbb" '
            'stroke-width="1" stroke-dasharray="2,4"/>'
        )
        grid.append(
            f'<line x1="{x}" y1="{height - bottom}" x2="{x}" y2="{height - bottom + 8}" '
            'stroke="#444444" stroke-width="1.5"/>'
        )
        grid.append(
            f'<text x="{x}" y="{height - bottom + 26}" font-size="14" text-anchor="middle">{t:.3f}</text>'
        )

    low = xpix(float(bsm["nrmse_pred_low95"]))
    mid = xpix(float(bsm["nrmse_pred"]))
    high = xpix(float(bsm["nrmse_pred_high95"]))
    obs = xpix(float(bsm["nrmse_obs"]))
    main = [
        f'<line x1="{low}" y1="{axis_y}" x2="{high}" y2="{axis_y}" stroke="#1f77b4" stroke-width="4"/>',
        f'<line x1="{low}" y1="{axis_y - 10}" x2="{low}" y2="{axis_y + 10}" stroke="#1f77b4" stroke-width="3"/>',
        f'<line x1="{high}" y1="{axis_y - 10}" x2="{high}" y2="{axis_y + 10}" stroke="#1f77b4" stroke-width="3"/>',
        f'<circle cx="{mid}" cy="{axis_y}" r="7" fill="#1f77b4"/>',
        f'<line x1="{obs}" y1="{top}" x2="{obs}" y2="{height - bottom}" stroke="#000000" stroke-width="3" stroke-dasharray="9,7"/>',
    ]
    title = '<text x="550" y="34" font-size="24" text-anchor="middle">Planning estimate for BSM (uncertainty-aware)</text>'
    axis = (
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" '
        'stroke="#333333" stroke-width="2"/>'
        f'<text x="{(left + width - right) / 2}" y="{height - 20}" font-size="18" text-anchor="middle">nRMSE</text>'
    )
    txt = escape(
        f"Pred={bsm['nrmse_pred']:.4f}, Obs={bsm['nrmse_obs']:.4f}, Error={bsm['pct_err']:+.1f}%, "
        f"OOD nearest-distance={bsm['ood_diagnostics']['nearest_knob_distance']:.2f}"
    )
    note = (
        f'<rect x="{width - 520}" y="{height - 118}" width="500" height="20" fill="#ffffff" fill-opacity="0.92"/>'
        f'<text x="{width - 16}" y="{height - 103}" font-size="13" text-anchor="end">{txt}</text>'
    )
    legend = [
        '<line x1="96" y1="58" x2="124" y2="58" stroke="#1f77b4" stroke-width="4"/>'
        '<line x1="96" y1="49" x2="96" y2="67" stroke="#1f77b4" stroke-width="3"/>'
        '<line x1="124" y1="49" x2="124" y2="67" stroke="#1f77b4" stroke-width="3"/>'
        '<circle cx="110" cy="58" r="5" fill="#1f77b4"/>'
        '<text x="136" y="63" font-size="13">Estimate with 95% interval</text>',
        '<line x1="390" y1="58" x2="430" y2="58" stroke="#000" stroke-width="3" stroke-dasharray="8,6"/>'
        '<text x="440" y="63" font-size="13">Observed</text>',
        '<rect x="540" y="50" width="14" height="10" fill="#2ca02c" fill-opacity="0.14"/>'
        '<text x="560" y="59" font-size="12">\u2264 0.0421</text>',
        '<rect x="640" y="50" width="14" height="10" fill="#ffbf00" fill-opacity="0.16"/>'
        '<text x="660" y="59" font-size="12">(0.0421, 0.0532]</text>',
        '<rect x="808" y="50" width="14" height="10" fill="#d62728" fill-opacity="0.14"/>'
        '<text x="828" y="59" font-size="12">&gt; 0.0532</text>',
    ]

    _write_svg(out_svg, width, height, "".join(bg + grid + main + [title, axis, note] + legend))


def _build_cv_scatter_figure(
    preds: pd.DataFrame,
    cv_r2: float,
    out_svg: Path,
) -> None:
    obs = preds["nrmse_obs"].to_numpy(dtype=float)
    pred = preds["nrmse_pred_hybrid_ridge"].to_numpy(dtype=float)
    low = min(obs.min(), pred.min())
    high = max(obs.max(), pred.max())
    width, height = 700, 700
    left, right, top, bottom = 85, 30, 60, 85
    plot_w = width - left - right
    plot_h = height - top - bottom

    def xpix(v: float) -> float:
        return left + (v - low) / max(high - low, 1e-9) * plot_w

    def ypix(v: float) -> float:
        return top + (1.0 - (v - low) / max(high - low, 1e-9)) * plot_h

    pts = []
    for xo, yp in zip(obs, pred, strict=True):
        pts.append(
            f'<circle cx="{xpix(float(xo)):.2f}" cy="{ypix(float(yp)):.2f}" r="2.8" fill="#1f77b4" fill-opacity="0.65"/>'
        )
    line = (
        f'<line x1="{xpix(low)}" y1="{ypix(low)}" x2="{xpix(high)}" y2="{ypix(high)}" '
        'stroke="#000" stroke-width="2" stroke-dasharray="7,5"/>'
    )
    grid = []
    for t in np.linspace(low, high, 7):
        x = xpix(float(t))
        y = ypix(float(t))
        grid.append(
            f'<line x1="{x}" y1="{top}" x2="{x}" y2="{height - bottom}" stroke="#d5d5d5" stroke-width="1"/>'
        )
        grid.append(
            f'<line x1="{left}" y1="{y}" x2="{width - right}" y2="{y}" stroke="#d5d5d5" stroke-width="1"/>'
        )
    axes = (
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#333" stroke-width="2"/>'
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#333" stroke-width="2"/>'
        f'<text x="{(left + width - right) / 2}" y="{height - 24}" font-size="17" text-anchor="middle">Observed nRMSE</text>'
        f'<text x="22" y="{(top + height - bottom) / 2}" font-size="17" text-anchor="middle" transform="rotate(-90 22 {(top + height - bottom) / 2})">Predicted nRMSE (Track A v3)</text>'
    )
    title = '<text x="350" y="32" font-size="24" text-anchor="middle">Wave5 CV fit quality (hybrid model)</text>'
    annot = (
        f'<rect x="{left + 10}" y="{top + 10}" width="180" height="34" fill="#fff" fill-opacity="0.9"/>'
        f'<text x="{left + 20}" y="{top + 33}" font-size="16">Group-CV R² = {cv_r2:.3f}</text>'
    )
    _write_svg(out_svg, width, height, "".join(grid + [line] + pts + [axes, title, annot]))


def main() -> None:
    """Generate planning artifacts and copy manuscript-ready figure files."""
    summary = json.loads((IN_DIR / "wave5_track_a_v3_summary.json").read_text(encoding="utf-8"))
    preds = pd.read_csv(IN_DIR / "wave5_track_a_v3_predictions.csv")
    bsm = summary["metrics"]["bsm_prediction"]

    q1 = float(preds["nrmse_obs"].quantile(1 / 3))
    q2 = float(preds["nrmse_obs"].quantile(2 / 3))
    bsm_band_pred = _band(float(bsm["nrmse_pred"]), q1, q2)
    bsm_band_obs = _band(float(bsm["nrmse_obs"]), q1, q2)

    planning = {
        "model": "track_a_v3_hybrid",
        "claim_scope": "workflow planning add-on (uncertainty-aware), not precision prediction",
        "in_domain_cv": {
            "hybrid_ridge_nrmse_cv_r2": summary["metrics"]["hybrid_ridge_nrmse_cv"]["r2"],
            "hybrid_ridge_nrmse_cv_rmse": summary["metrics"]["hybrid_ridge_nrmse_cv"]["rmse"],
        },
        "bsm": {
            "nrmse_pred": bsm["nrmse_pred"],
            "nrmse_obs": bsm["nrmse_obs"],
            "pct_err": bsm["pct_err"],
            "nrmse_pred_low95": bsm["nrmse_pred_low95"],
            "nrmse_pred_high95": bsm["nrmse_pred_high95"],
            "ood_nearest_knob_distance": bsm["ood_diagnostics"]["nearest_knob_distance"],
            "predicted_band": bsm_band_pred,
            "observed_band": bsm_band_obs,
        },
        "runtime": {
            "runtime_cv_r2": summary["metrics"]["runtime_hybrid_ridge_cv"]["r2"],
            "runtime_guidance": "report as coarse runtime regime only (fast/moderate/slow), not point estimate",
        },
        "decision_bands": {
            "likely_acceptable_max": q1,
            "borderline_max": q2,
            "likely_unacceptable_min": q2,
            "basis": "tertiles of wave5 observed nRMSE",
        },
    }

    (OUT_DIR / "track_a_v3_planning_summary.json").write_text(
        json.dumps(planning, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame(
        [
            {"band": "likely acceptable", "nrmse_min": 0.0, "nrmse_max": q1},
            {"band": "borderline", "nrmse_min": q1, "nrmse_max": q2},
            {"band": "likely unacceptable", "nrmse_min": q2, "nrmse_max": math.inf},
        ]
    ).to_csv(OUT_DIR / "track_a_v3_decision_bands.csv", index=False)

    fig1_svg = OUT_DIR / "fig_track_a_v3_bsm_planning_interval.svg"
    fig2_svg = OUT_DIR / "fig_track_a_v3_cv_scatter.svg"
    _build_bsm_interval_figure(bsm=bsm, q1=q1, q2=q2, out_svg=fig1_svg)
    _build_cv_scatter_figure(
        preds=preds,
        cv_r2=float(summary["metrics"]["hybrid_ridge_nrmse_cv"]["r2"]),
        out_svg=fig2_svg,
    )
    fig1_pdf = _save_pdf(fig1_svg)
    fig2_pdf = _save_pdf(fig2_svg)

    DOCS_OUT.mkdir(parents=True, exist_ok=True)
    for path in [fig1_svg, fig1_pdf, fig2_svg, fig2_pdf]:
        target = DOCS_OUT / path.name
        target.write_bytes(path.read_bytes())

    print("Wrote planning analytics:")
    print(f"  {OUT_DIR / 'track_a_v3_planning_summary.json'}")
    print(f"  {OUT_DIR / 'track_a_v3_decision_bands.csv'}")
    print("Wrote figures:")
    print(f"  {fig1_svg}")
    print(f"  {fig1_pdf}")
    print(f"  {fig2_svg}")
    print(f"  {fig2_pdf}")


if __name__ == "__main__":
    main()
