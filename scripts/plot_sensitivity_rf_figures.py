#!/usr/bin/env python3
# Copyright (c) 2026 Dylan Hettinger
"""Generate RF meta-regression figures for sensitivity section.

Figures produced:
  fig_sensitivity_rf_importance.svg   -- two-panel RF feature importance
  fig_sensitivity_bsm_validation.svg  -- \u03b3 histogram + BSM validation dot plot
"""

from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rfm_pipeline._svg_pdf import save_svg_as_pdf  # noqa: E402
from rfm_pipeline.manuscript_stages import (  # noqa: E402
    _SVG_COLOR_ACCENT_GREEN,
    _SVG_COLOR_ACCENT_ORANGE,
    _SVG_COLOR_BACKGROUND,
    _SVG_COLOR_DANGER,
    _SVG_COLOR_EDGE,
    _SVG_COLOR_LIGHT,
    _SVG_COLOR_PRIMARY,
    _SVG_COLOR_TEXT,
    _SVG_COLOR_TEXT_MUTED,
    _SVG_COLOR_TITLE,
    _SVG_FONT_FAMILY,
    _SVG_FS_AXIS,
    _SVG_FS_LABEL,
    _SVG_FS_LEGEND,
    _SVG_FS_SMALL,
    _SVG_FS_TICK,
    _SVG_FS_TITLE,
)

# ── Hardcoded sensitivity results ───────────────────────────────────────────

QUALITY_IMPORTANCE = [
    ("BH threshold (q)", 0.452),
    ("Sparsity (s)", 0.141),
    ("Screening permutations", 0.102),
    ("Input count (d)", 0.077),
    ("Interaction density (\u03c1)", 0.072),
    ("Nonlinearity strength (\u03ba)", 0.054),
    ("Signal-to-noise ratio (\u03c3)", 0.047),
    ("Interaction p-threshold", 0.021),
]

RUNTIME_IMPORTANCE = [
    ("Sparsity (s)", 0.400),
    ("Input count (d)", 0.365),
    ("Run count (n)", 0.077),
    ("Interaction permutations", 0.053),
    ("Stability subsamples", 0.049),
    ("Interaction density (\u03c1)", 0.016),
    ("Signal-to-noise ratio (\u03c3)", 0.013),
    ("Nonlinearity strength (\u03ba)", 0.012),
]

FORMULA_TOP10: list[tuple[str, float]] = []  # retained for reference only — not plotted

# BSM production r_BSM: (0.0721 - 0.1653) / 0.1653 = -0.564
# (uses current pipeline nRMSE with production-dataset null nRMSE)
BSM_R = -0.564
# NOTE: median of γ across successful sensitivity runs is now computed
# dynamically inside ``render_bsm_validation`` from the supplied results CSV
# (avoids drift when wave count changes; was hardcoded -0.462 against wave1).

BSM_NRMSE_ACTUAL = 0.0721
# Wave1234 RF quality model applied to BSM production feature vector
# (d=135, n=28750, sparsity=0.28, ρ=0.15, κ=0.20, σ=22.3, perms_scr=201,
#  q=0.05, perms_int=31, p=0.05, n_stab=50, lasso_grid=40, var_thresh=0.9).
# Wave4 added 30 calibrated rows at the production override corner, which
# tightened the RF PI and shifted γ slightly toward 0 (−0.371 vs −0.388).
# The PI now fully EXCLUDES the actual production value 0.0721 — confirming
# the gap is sensitivity-harness vs production-pipeline, not training
# coverage. γ point = -0.3710 → nRMSE = 0.1653·(1+γ); 80% per-tree PI on γ
# [-0.3815, -0.3458] → nRMSE [0.1022, 0.1081].
BSM_RF_PRED = 0.1040
BSM_RF_P10 = 0.1022
BSM_RF_P90 = 0.1081
BSM_ANALOG_NRMSE = [
    0.07655,
    0.07662,
    0.07665,
    0.07665,
    0.07681,
    0.07681,
    0.07688,
    0.07695,
    0.07697,
    0.07697,
    0.07699,
    0.07702,
    0.07702,
    0.07702,
    0.07710,
    0.07710,
    0.07718,
    0.07718,
    0.07724,
    0.07724,
    0.07729,
    0.07732,
    0.07737,
    0.07747,
    0.07748,
]

# ── SVG helpers ──────────────────────────────────────────────────────────────

_FONT = _SVG_FONT_FAMILY
_FS_TITLE = _SVG_FS_TITLE
_FS_AXIS = _SVG_FS_AXIS
_FS_TICK = _SVG_FS_TICK
_FS_LABEL = _SVG_FS_LABEL
_FS_VAL = _SVG_FS_LEGEND


def _canvas(w: int, h: int, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}">'
        f'<rect width="100%" height="100%" fill="{_SVG_COLOR_BACKGROUND}"/>'
        f"{body}</svg>"
    )


def _panel_title(text: str, x: int, y: int, rule_x1: int, rule_x2: int) -> str:
    return (
        f'<text x="{x}" y="{y}" font-family="{_FONT}" font-size="{_FS_TITLE}" '
        f'font-weight="700" fill="{_SVG_COLOR_TITLE}">{html.escape(text)}</text>'
        f'<line x1="{rule_x1}" y1="{y + 6}" x2="{rule_x2}" y2="{y + 6}" '
        f'stroke="{_SVG_COLOR_EDGE}" stroke-width="0.8"/>'
    )


def _ax_label(text: str, x: int, y: int, anchor: str = "middle") -> str:
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="{_FONT}" '
        f'font-size="{_FS_AXIS}" fill="{_SVG_COLOR_TEXT}">{html.escape(text)}</text>'
    )


def _tick_label(text: str, x: float, y: float, anchor: str = "middle") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-family="{_FONT}" '
        f'font-size="{_FS_TICK}" fill="{_SVG_COLOR_TEXT_MUTED}">{html.escape(text)}</text>'
    )


def _save_svg(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content + "\n", encoding="utf-8")


def _save_pdf(svg_path: Path) -> None:
    """Convert SVG to PDF using the shared Chrome-based helper."""
    save_svg_as_pdf(svg_path)


# ── Figure 1: RF feature importance ─────────────────────────────────────────


def _render_importance_panel(
    body: list[str],
    data: list[tuple[str, float]],
    color: str,
    title: str,
    x_offset: int,
    bar_x_start: int,
    bar_max_w: int,
    y_bar_top: int,
    bar_h: int,
    bar_gap: int,
    y_axis_bottom: int,
    canvas_w_for_axis_label: int,
) -> None:
    max_val = max(v for _, v in data)
    stride = bar_h + bar_gap
    rule_x2 = bar_x_start + bar_max_w + 5

    # Title + rule
    body.append(_panel_title(title, x_offset + 10, y_bar_top - 28, x_offset + 5, rule_x2))

    # Bottom spine (x-axis)
    ax_y = y_axis_bottom
    body.append(
        f'<line x1="{bar_x_start}" y1="{ax_y}" x2="{bar_x_start + bar_max_w}" y2="{ax_y}" '
        f'stroke="{_SVG_COLOR_EDGE}" stroke-width="1.2"/>'
    )
    # Left spine
    body.append(
        f'<line x1="{bar_x_start}" y1="{y_bar_top}" x2="{bar_x_start}" y2="{ax_y}" '
        f'stroke="{_SVG_COLOR_EDGE}" stroke-width="1.2"/>'
    )

    # Ticks at 0.0, 0.1, 0.2, 0.3
    tick_vals = [0.0, 0.1, 0.2, 0.3]
    for tv in tick_vals:
        tx = bar_x_start + (tv / max_val) * bar_max_w
        body.append(
            f'<line x1="{tx:.1f}" y1="{ax_y}" x2="{tx:.1f}" y2="{ax_y + 4}" '
            f'stroke="{_SVG_COLOR_EDGE}" stroke-width="1"/>'
        )
        # Faint grid line
        body.append(
            f'<line x1="{tx:.1f}" y1="{y_bar_top}" x2="{tx:.1f}" y2="{ax_y}" '
            f'stroke="{_SVG_COLOR_LIGHT}" stroke-width="0.5" stroke-dasharray="2 3"/>'
        )
        body.append(_tick_label(f"{tv:.1f}", tx, ax_y + 14))

    # Axis label
    ax_center = bar_x_start + bar_max_w / 2
    body.append(
        _ax_label(
            "RF feature importance (mean decrease in impurity)",
            int(ax_center),
            ax_y + 28,
        )
    )

    # Bars + labels
    label_x_end = bar_x_start - 8
    for i, (label, val) in enumerate(data):
        y = y_bar_top + i * stride
        bw = (val / max_val) * bar_max_w
        # background track
        body.append(
            f'<rect x="{bar_x_start}" y="{y}" width="{bar_max_w}" height="{bar_h}" '
            f'fill="{_SVG_COLOR_LIGHT}" fill-opacity="0.3"/>'
        )
        # bar
        body.append(
            f'<rect x="{bar_x_start}" y="{y}" width="{bw:.2f}" height="{bar_h}" fill="{color}"/>'
        )
        # label (left-aligned, right of label area)
        body.append(
            f'<text x="{label_x_end}" y="{y + bar_h - 5}" text-anchor="end" '
            f'font-family="{_FONT}" font-size="{_FS_LABEL}" fill="{_SVG_COLOR_TEXT}">'
            f"{html.escape(label)}</text>"
        )
        # value label: inside bar (white) when it would overflow the axis line
        if bw > bar_max_w - 42:
            vx = bar_x_start + bw - 5
            vanchor = "end"
            vcol = "#ffffff"
        else:
            vx = bar_x_start + bw + 5
            vanchor = "start"
            vcol = _SVG_COLOR_TEXT_MUTED
        body.append(
            f'<text x="{vx:.2f}" y="{y + bar_h - 5}" text-anchor="{vanchor}" '
            f'font-family="{_FONT}" font-size="{_FS_VAL}" fill="{vcol}">'
            f"{val:.3f}</text>"
        )


def render_rf_importance() -> str:
    """Render two-panel RF feature importance horizontal bar chart."""
    W, H = 1120, 450
    bar_h, bar_gap = 24, 12
    n = 8
    y_top = 72
    y_ax = y_top + n * (bar_h + bar_gap) + 4

    # Panel 1 (quality, blue) — x: 0..545
    p1_bar_start = 248
    p1_bar_w = 272

    # Panel 2 (runtime, orange) — x: 570..1120
    p2_bar_start = 788
    p2_bar_w = 282

    body: list[str] = []

    # Panel divider
    body.append(
        f'<line x1="557" y1="24" x2="557" y2="{H - 24}" '
        f'stroke="{_SVG_COLOR_LIGHT}" stroke-width="1"/>'
    )

    _render_importance_panel(
        body,
        QUALITY_IMPORTANCE,
        _SVG_COLOR_PRIMARY,
        "Quality model: RF feature importance",
        0,
        p1_bar_start,
        p1_bar_w,
        y_top,
        bar_h,
        bar_gap,
        y_ax,
        W,
    )
    _render_importance_panel(
        body,
        RUNTIME_IMPORTANCE,
        _SVG_COLOR_ACCENT_ORANGE,
        "Runtime model: RF feature importance",
        570,
        p2_bar_start,
        p2_bar_w,
        y_top,
        bar_h,
        bar_gap,
        y_ax,
        W,
    )

    return _canvas(W, H, "".join(body))


# ── Figure 2: γ histogram + BSM validation ──────────────────────────────────


def _diamond(cx: float, cy: float, r: float, color: str, **kw: str) -> str:
    pts = f"{cx},{cy - r} {cx + r},{cy} {cx},{cy + r} {cx - r},{cy}"
    extra = " ".join(f'{k}="{v}"' for k, v in kw.items())
    return f'<polygon points="{pts}" fill="{color}" {extra}/>'


def render_bsm_validation(results_path: Path) -> str:
    """Render two-panel figure: γ histogram and BSM operating-point dot plot."""
    df = pd.read_csv(results_path)
    r_vals = df["nrmse_relative"].replace([np.inf, -np.inf], np.nan).dropna().values
    succ_mask = df["null_screened"].isna()
    r_succ = df.loc[succ_mask, "nrmse_relative"].replace([np.inf, -np.inf], np.nan).dropna().values
    n_successful = int(len(r_succ))
    n_null_screened = int(df["null_screened"].notna().sum())
    n_total = int(len(r_vals))
    median_successful = float(np.median(r_succ)) if n_successful else 0.0

    W, H = 1130, 410

    # ── Left panel: histogram ────────────────────────────────────────────────
    # Plot area x=25..535, y=74..312 → 510 × 238; matches right panel ax_y_r=312
    lp_x0, lp_y0 = 25, 74
    lp_w, lp_h = 510, 238

    bins = np.arange(-0.90, 0.021, 0.02)
    counts, edges = np.histogram(r_vals, bins=bins)
    n_bins = len(counts)
    x_data_min, x_data_max = edges[0], edges[-1]
    y_data_max = counts.max() * 1.08

    def lp_x(v: float) -> float:
        return lp_x0 + (v - x_data_min) / (x_data_max - x_data_min) * lp_w

    def lp_y(c: float) -> float:
        return lp_y0 + lp_h - (c / y_data_max) * lp_h

    body: list[str] = []

    # Title + rule — two lines, both starting at the same y as the right panel title
    body.append(
        f'<text x="{lp_x0 + 5}" y="24" font-family="{_FONT}" font-size="{_FS_TITLE}" '
        f'font-weight="700" fill="{_SVG_COLOR_TITLE}">'
        "Distribution of \u03b3 across sensitivity study runs</text>"
        f'<text x="{lp_x0 + 5}" y="50" font-family="{_FONT}" font-size="{_FS_TITLE}" '
        f'font-weight="700" fill="{_SVG_COLOR_TITLE}">'
        f"(n\u00a0=\u00a0{n_total:,})</text>"
        f'<line x1="{lp_x0}" y1="58" x2="{lp_x0 + lp_w}" y2="58" '
        f'stroke="{_SVG_COLOR_EDGE}" stroke-width="0.8"/>'
    )

    # Axes
    body.append(
        f'<line x1="{lp_x0}" y1="{lp_y0 + lp_h}" x2="{lp_x0 + lp_w}" y2="{lp_y0 + lp_h}" '
        f'stroke="{_SVG_COLOR_EDGE}" stroke-width="1.2"/>'
    )
    body.append(
        f'<line x1="{lp_x0}" y1="{lp_y0}" x2="{lp_x0}" y2="{lp_y0 + lp_h}" '
        f'stroke="{_SVG_COLOR_EDGE}" stroke-width="1.2"/>'
    )

    # Bars
    bin_px = lp_w / n_bins
    for i, count in enumerate(counts):
        bx = lp_x0 + i * bin_px
        by = lp_y(count)
        bh = lp_y0 + lp_h - by
        body.append(
            f'<rect x="{bx:.2f}" y="{by:.2f}" width="{bin_px - 0.5:.2f}" height="{bh:.2f}" '
            f'fill="{_SVG_COLOR_PRIMARY}" fill-opacity="0.85"/>'
        )

    # x-axis ticks
    for tv in [-0.8, -0.6, -0.4, -0.2, 0.0]:
        tx = lp_x(tv)
        body.append(
            f'<line x1="{tx:.1f}" y1="{lp_y0 + lp_h}" x2="{tx:.1f}" y2="{lp_y0 + lp_h + 4}" '
            f'stroke="{_SVG_COLOR_EDGE}" stroke-width="1"/>'
        )
        body.append(_tick_label(f"{tv:.1f}", tx, lp_y0 + lp_h + 15))

    # y-axis ticks
    for yv in [0, 100, 200, 300, 400, 500]:
        if yv > y_data_max:
            break
        ty = lp_y(yv)
        body.append(
            f'<line x1="{lp_x0 - 4}" y1="{ty:.1f}" x2="{lp_x0}" y2="{ty:.1f}" '
            f'stroke="{_SVG_COLOR_EDGE}" stroke-width="1"/>'
        )
        body.append(_tick_label(str(yv), lp_x0 - 7, ty + 4, anchor="end"))

    # Axis labels
    body.append(
        f'<text x="{lp_x0 + lp_w // 2}" y="{lp_y0 + lp_h + 30}" text-anchor="middle" '
        f'font-family="{_FONT}" font-size="{_FS_AXIS}" fill="{_SVG_COLOR_TEXT}">'
        "\u03b3 = (nRMSE"
        '<tspan baseline-shift="sub" font-size="75%">final</tspan>'
        " \u2212 nRMSE"
        '<tspan baseline-shift="sub" font-size="75%">null</tspan>) / nRMSE'
        '<tspan baseline-shift="sub" font-size="75%">null</tspan>'
        "</text>"
    )
    body.append(
        f'<text x="{lp_x0 - 35}" y="{lp_y0 + lp_h // 2}" '
        f'transform="rotate(-90 {lp_x0 - 35} {lp_y0 + lp_h // 2})" '
        f'text-anchor="middle" font-family="{_FONT}" font-size="{_FS_AXIS}" '
        f'fill="{_SVG_COLOR_TEXT}">Count</text>'
    )

    # Reference lines
    bsm_x = lp_x(BSM_R)
    med_x = lp_x(median_successful)

    body.append(
        f'<line x1="{bsm_x:.1f}" y1="{lp_y0}" x2="{bsm_x:.1f}" y2="{lp_y0 + lp_h}" '
        f'stroke="{_SVG_COLOR_DANGER}" stroke-width="1.8" stroke-dasharray="7 4"/>'
    )
    body.append(
        f'<line x1="{med_x:.1f}" y1="{lp_y0}" x2="{med_x:.1f}" y2="{lp_y0 + lp_h}" '
        f'stroke="{_SVG_COLOR_ACCENT_GREEN}" stroke-width="1.8" stroke-dasharray="3 4"/>'
    )

    # Legend — placed below the x-axis to avoid overlapping bars
    leg_y = lp_y0 + lp_h + 40
    body.append(
        f'<line x1="{lp_x0 + 10}" y1="{leg_y}" x2="{lp_x0 + 34}" y2="{leg_y}" '
        f'stroke="{_SVG_COLOR_DANGER}" stroke-width="1.8" stroke-dasharray="7 4"/>'
    )
    body.append(
        f'<text x="{lp_x0 + 38}" y="{leg_y + 4}" font-family="{_FONT}" '
        f'font-size="{_SVG_FS_LEGEND}" fill="{_SVG_COLOR_TEXT}">'
        f"BSM \u03b3 = {BSM_R:.3f}</text>"
    )
    body.append(
        f'<line x1="{lp_x0 + 10}" y1="{leg_y + 17}" x2="{lp_x0 + 34}" y2="{leg_y + 17}" '
        f'stroke="{_SVG_COLOR_ACCENT_GREEN}" stroke-width="1.8" stroke-dasharray="3 4"/>'
    )
    body.append(
        f'<text x="{lp_x0 + 38}" y="{leg_y + 21}" font-family="{_FONT}" '
        f'font-size="{_SVG_FS_LEGEND}" fill="{_SVG_COLOR_TEXT}">'
        f"Median (successful fits) = {median_successful:.3f}</text>"
    )

    # Null-screened annotation on the spike
    spike_bin_idx = np.argmax(counts)
    spike_top = lp_y(counts[spike_bin_idx])
    body.append(
        f'<text x="{lp_x0 + lp_w - 6}" y="{spike_top - 5:.1f}" text-anchor="end" '
        f'font-family="{_FONT}" font-size="{_SVG_FS_SMALL}" fill="{_SVG_COLOR_TEXT_MUTED}">'
        f"null-screened (n={n_null_screened})</text>"
    )

    # ── Right panel: BSM operating-point dot plot ────────────────────────────
    # Panel x: 580..1120, plot x: 635..1105 → 470px
    rp_left, rp_right = 580, 1120
    pp_x0 = rp_left + 58
    pp_w = rp_right - pp_x0 - 20
    pp_yc = 246  # center y for main row
    pp_ya = 198  # y for analog dots row
    ax_y_r = 312  # x-axis line y

    x_nrmse_min, x_nrmse_max = 0.067, 0.083
    nrmse_span = x_nrmse_max - x_nrmse_min

    def rp_x(v: float) -> float:
        return pp_x0 + (v - x_nrmse_min) / nrmse_span * pp_w

    # Title + rule
    body.append(
        _panel_title(
            "BSM operating-point validation",
            rp_left + 5,
            24,
            rp_left,
            rp_right - 10,
        )
    )

    # x-axis
    body.append(
        f'<line x1="{pp_x0}" y1="{ax_y_r}" x2="{pp_x0 + pp_w}" y2="{ax_y_r}" '
        f'stroke="{_SVG_COLOR_EDGE}" stroke-width="1.2"/>'
    )
    for tv in [0.068, 0.070, 0.072, 0.074, 0.076, 0.078, 0.080, 0.082]:
        tx = rp_x(tv)
        body.append(
            f'<line x1="{tx:.1f}" y1="{ax_y_r}" x2="{tx:.1f}" y2="{ax_y_r + 4}" '
            f'stroke="{_SVG_COLOR_EDGE}" stroke-width="1"/>'
        )
        body.append(
            f'<line x1="{tx:.1f}" y1="{pp_ya - 20}" x2="{tx:.1f}" y2="{ax_y_r}" '
            f'stroke="{_SVG_COLOR_LIGHT}" stroke-width="0.5" stroke-dasharray="2 3"/>'
        )
        body.append(_tick_label(f"{tv:.3f}", tx, ax_y_r + 14))

    body.append(_ax_label("Holdout macro nRMSE", int(pp_x0 + pp_w / 2), ax_y_r + 30))

    # ── RF 80% prediction interval (light orange rectangle)
    pi_x1 = rp_x(BSM_RF_P10)
    pi_x2 = rp_x(BSM_RF_P90)
    pi_h = 22
    body.append(
        f'<rect x="{pi_x1:.2f}" y="{pp_yc - pi_h / 2:.1f}" '
        f'width="{pi_x2 - pi_x1:.2f}" height="{pi_h}" '
        f'fill="{_SVG_COLOR_ACCENT_ORANGE}" fill-opacity="0.30" '
        f'stroke="{_SVG_COLOR_ACCENT_ORANGE}" stroke-width="1.2"/>'
    )

    # ── BSM analog dots (gray, row above)
    for val in BSM_ANALOG_NRMSE:
        ax_pos = rp_x(val)
        body.append(
            f'<circle cx="{ax_pos:.2f}" cy="{pp_ya:.1f}" r="3.5" '
            f'fill="{_SVG_COLOR_TEXT_MUTED}" fill-opacity="0.85"/>'
        )

    # ── RF prediction point
    rf_x = rp_x(BSM_RF_PRED)
    body.append(
        f'<circle cx="{rf_x:.2f}" cy="{pp_yc:.1f}" r="7" fill="{_SVG_COLOR_ACCENT_ORANGE}"/>'
    )

    # ── BSM actual (diamond)
    bsm_x2 = rp_x(BSM_NRMSE_ACTUAL)
    body.append(_diamond(bsm_x2, pp_yc, 8, _SVG_COLOR_DANGER))

    # ── Legend (right panel)
    leg_items = [
        (
            _SVG_COLOR_ACCENT_ORANGE,
            0.30,
            "rect",
            f"RF 80% PI [{BSM_RF_P10:.4f}, {BSM_RF_P90:.4f}]",
        ),
        (_SVG_COLOR_ACCENT_ORANGE, 1.0, "circle", f"RF prediction ({BSM_RF_PRED:.4f})"),
        (_SVG_COLOR_DANGER, 1.0, "diamond", f"BSM actual ({BSM_NRMSE_ACTUAL:.4f})"),
        (_SVG_COLOR_TEXT_MUTED, 0.85, "circle_sm", "BSM-structure analogs (n=25, n_runs=28,750)"),
    ]
    lex = rp_left + 10
    ley0 = 60
    for j, (col, alpha, shape, label) in enumerate(leg_items):
        ly = ley0 + j * 22
        if shape == "rect":
            body.append(
                f'<rect x="{lex}" y="{ly - 7}" width="22" height="10" '
                f'fill="{col}" fill-opacity="{alpha}" stroke="{col}" stroke-width="1"/>'
            )
        elif shape == "circle":
            body.append(
                f'<circle cx="{lex + 11}" cy="{ly - 2}" r="5" fill="{col}" fill-opacity="{alpha}"/>'
            )
        elif shape == "diamond":
            body.append(_diamond(lex + 11, ly - 2, 6, col))
        elif shape == "circle_sm":
            body.append(
                f'<circle cx="{lex + 11}" cy="{ly - 2}" r="3.5" '
                f'fill="{col}" fill-opacity="{alpha}"/>'
            )
        body.append(
            f'<text x="{lex + 28}" y="{ly + 2}" font-family="{_FONT}" '
            f'font-size="{_SVG_FS_LEGEND}" fill="{_SVG_COLOR_TEXT}">{html.escape(label)}</text>'
        )

    # Panel divider
    body.append(
        f'<line x1="565" y1="10" x2="565" y2="{H - 10}" '
        f'stroke="{_SVG_COLOR_LIGHT}" stroke-width="1"/>'
    )

    return _canvas(W, H, "".join(body))


# ── CLI ──────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "figures" / "sensitivity")
    return p.parse_args()


def main() -> int:
    """Generate RF meta-regression sensitivity figures and save as SVG + PDF."""
    args = _parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    fig1 = out / "fig_sensitivity_rf_importance.svg"
    _save_svg(fig1, render_rf_importance())
    _save_pdf(fig1)
    print("  fig_sensitivity_rf_importance.svg/pdf")

    fig2 = out / "fig_sensitivity_bsm_validation.svg"
    _save_svg(fig2, render_bsm_validation(args.results))
    _save_pdf(fig2)
    print("  fig_sensitivity_bsm_validation.svg/pdf")

    print(f"output_dir={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
