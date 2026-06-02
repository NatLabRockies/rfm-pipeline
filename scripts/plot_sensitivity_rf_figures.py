#!/usr/bin/env python3
# Copyright (c) 2026 Dylan Hettinger
"""Generate RF meta-regression figures for sensitivity section.

Figures produced:
  fig_sensitivity_rf_importance.svg   -- two-panel RF feature importance
  fig_sensitivity_formula_top10.svg   -- diverging top-10 t-statistics
  fig_sensitivity_bsm_validation.svg  -- r histogram + BSM validation dot plot
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
)

# ── Hardcoded sensitivity results ───────────────────────────────────────────

QUALITY_IMPORTANCE = [
    ("BH threshold (q)", 0.331),
    ("Screening permutations", 0.104),
    ("Sparsity (s)", 0.076),
    ("Stability subsamples", 0.069),
    ("Interaction p-threshold", 0.068),
    ("Nonlinearity strength (\u03ba)", 0.056),
    ("LASSO \u03b1 percentile", 0.054),
    ("Input count (d)", 0.054),
]

RUNTIME_IMPORTANCE = [
    ("Input count (d)", 0.314),
    ("Sparsity (s)", 0.267),
    ("Run count (n)", 0.103),
    ("Interaction density (\u03c1)", 0.092),
    ("Nonlinearity strength (\u03ba)", 0.078),
    ("Signal-to-noise ratio (\u03c3)", 0.058),
    ("Interaction permutations", 0.027),
    ("LASSO \u03b1 percentile", 0.016),
]

FORMULA_TOP10 = [
    # (label, t-statistic)  -- ordered by |t| descending
    ("log N(s) \u00d7 sparsity", +16.0),
    ("\u03c1 \u00d7 log d", +15.5),
    ("q \u00d7 sparsity", +15.0),
    ("LASSO \u03b1 (main effect)", -13.9),
    ("q \u00d7 \u03c1", +12.8),
    ("Var. threshold \u00d7 LASSO \u03b1", +11.8),
    ("q \u00d7 log d", +11.4),
    ("LASSO \u03b1 \u00d7 log \u03b4", -10.6),
    ("Var. threshold \u00d7 log d", -9.9),
    ("log N(s) \u00d7 LASSO \u03b1", -9.8),
]

# BSM production r_BSM: (0.0721 - 0.1653) / 0.1653 = -0.564
# (uses current pipeline nRMSE with production-dataset null nRMSE)
BSM_R = -0.564
BSM_MEDIAN_SUCCESSFUL = -0.462

BSM_NRMSE_ACTUAL = 0.0721
BSM_RF_PRED = 0.0762
BSM_RF_P10 = 0.0729
BSM_RF_P90 = 0.0774
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
_FS_TITLE = 15
_FS_AXIS = 11
_FS_TICK = 10
_FS_LABEL = 11
_FS_VAL = 10


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
    W, H = 1120, 420
    bar_h, bar_gap = 24, 12
    n = 8
    y_top = 65
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
        f'<line x1="557" y1="20" x2="557" y2="{H - 20}" '
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


# ── Figure 2: formula top-10 diverging bar chart ────────────────────────────


def render_formula_top10() -> str:
    """Render diverging horizontal bar chart of top-10 polynomial t-statistics."""
    W, H = 1000, 530
    label_x_end = 278
    plot_x_start = 288
    plot_w = 640
    bar_h, bar_gap = 26, 13
    stride = bar_h + bar_gap
    y_top = 65
    n = len(FORMULA_TOP10)
    y_ax = y_top + n * stride + 4

    # Data range symmetric enough for ±17
    x_min_data, x_max_data = -17.0, 17.0
    span = x_max_data - x_min_data
    x0 = plot_x_start + (abs(x_min_data) / span) * plot_w  # zero line x

    def xp(t: float) -> float:
        return plot_x_start + ((t - x_min_data) / span) * plot_w

    body: list[str] = []

    # Title + rule
    body.append(
        _panel_title(
            "Degree-2 meta-regression: top 10 terms by |t|",
            18,
            34,
            12,
            W - 12,
        )
    )

    # x-axis bottom line
    body.append(
        f'<line x1="{plot_x_start}" y1="{y_ax}" x2="{plot_x_start + plot_w}" y2="{y_ax}" '
        f'stroke="{_SVG_COLOR_EDGE}" stroke-width="1.2"/>'
    )

    # Ticks
    for tv in [-15, -10, -5, 0, 5, 10, 15]:
        tx = xp(tv)
        body.append(
            f'<line x1="{tx:.1f}" y1="{y_ax}" x2="{tx:.1f}" y2="{y_ax + 4}" '
            f'stroke="{_SVG_COLOR_EDGE}" stroke-width="1"/>'
        )
        body.append(
            f'<line x1="{tx:.1f}" y1="{y_top}" x2="{tx:.1f}" y2="{y_ax}" '
            f'stroke="{_SVG_COLOR_LIGHT}" stroke-width="0.5" stroke-dasharray="2 3"/>'
        )
        body.append(_tick_label(str(tv), tx, y_ax + 14))

    # Zero line (prominent)
    body.append(
        f'<line x1="{x0:.1f}" y1="{y_top - 5}" x2="{x0:.1f}" y2="{y_ax}" '
        f'stroke="{_SVG_COLOR_EDGE}" stroke-width="1.8"/>'
    )

    # x-axis label
    body.append(_ax_label("t-statistic", int(plot_x_start + plot_w / 2), y_ax + 28))

    # Annotation
    body.append(
        f'<text x="{plot_x_start + plot_w - 5}" y="{y_ax + 46}" text-anchor="end" '
        f'font-family="{_FONT}" font-size="9" fill="{_SVG_COLOR_TEXT_MUTED}">'
        f"Blue = positive coefficient; Red = negative coefficient</text>"
    )

    # Bars
    for i, (label, t) in enumerate(FORMULA_TOP10):
        y = y_top + i * stride
        color = _SVG_COLOR_PRIMARY if t >= 0 else _SVG_COLOR_DANGER
        bar_x_left = min(x0, xp(t))
        bar_x_right = max(x0, xp(t))
        bw = bar_x_right - bar_x_left

        # background track (full width)
        body.append(
            f'<rect x="{plot_x_start}" y="{y}" width="{plot_w}" height="{bar_h}" '
            f'fill="{_SVG_COLOR_LIGHT}" fill-opacity="0.25"/>'
        )
        # bar
        body.append(
            f'<rect x="{bar_x_left:.2f}" y="{y}" width="{bw:.2f}" height="{bar_h}" fill="{color}"/>'
        )
        # term label (right-aligned, left of plot)
        body.append(
            f'<text x="{label_x_end}" y="{y + bar_h - 5}" text-anchor="end" '
            f'font-family="{_FONT}" font-size="{_FS_LABEL}" fill="{_SVG_COLOR_TEXT}">'
            f"{html.escape(label)}</text>"
        )
        # value label (outside bar end)
        if t >= 0:
            vx = bar_x_right + 5
            anchor = "start"
        else:
            vx = bar_x_left - 5
            anchor = "end"
        body.append(
            f'<text x="{vx:.1f}" y="{y + bar_h - 5}" text-anchor="{anchor}" '
            f'font-family="{_FONT}" font-size="{_FS_VAL}" fill="{_SVG_COLOR_TEXT_MUTED}">'
            f"{t:+.1f}</text>"
        )

    return _canvas(W, H, "".join(body))


# ── Figure 3: r histogram + BSM validation ──────────────────────────────────


def _diamond(cx: float, cy: float, r: float, color: str, **kw: str) -> str:
    pts = f"{cx},{cy - r} {cx + r},{cy} {cx},{cy + r} {cx - r},{cy}"
    extra = " ".join(f'{k}="{v}"' for k, v in kw.items())
    return f'<polygon points="{pts}" fill="{color}" {extra}/>'


def render_bsm_validation(results_path: Path) -> str:
    """Render two-panel figure: r histogram and BSM operating-point dot plot."""
    df = pd.read_csv(results_path)
    r_vals = df["nrmse_relative"].replace([np.inf, -np.inf], np.nan).dropna().values

    W, H = 1130, 520

    # ── Left panel: histogram ────────────────────────────────────────────────
    # Plot area within x=25..535, y=60..360 → 510 × 300
    lp_x0, lp_y0 = 25, 60
    lp_w, lp_h = 510, 300

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

    # Title + rule
    body.append(
        _panel_title(
            "Distribution of r across sensitivity study runs (n\u00a0=\u00a02,583)",
            lp_x0 + 5,
            34,
            lp_x0,
            lp_x0 + lp_w,
        )
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
        _ax_label(
            "r = (nRMSE_final \u2212 nRMSE_null) / nRMSE_null",
            lp_x0 + lp_w // 2,
            lp_y0 + lp_h + 30,
        )
    )
    body.append(
        f'<text x="{lp_x0 - 35}" y="{lp_y0 + lp_h // 2}" '
        f'transform="rotate(-90 {lp_x0 - 35} {lp_y0 + lp_h // 2})" '
        f'text-anchor="middle" font-family="{_FONT}" font-size="{_FS_AXIS}" '
        f'fill="{_SVG_COLOR_TEXT}">Count</text>'
    )

    # Reference lines
    bsm_x = lp_x(BSM_R)
    med_x = lp_x(BSM_MEDIAN_SUCCESSFUL)

    body.append(
        f'<line x1="{bsm_x:.1f}" y1="{lp_y0}" x2="{bsm_x:.1f}" y2="{lp_y0 + lp_h}" '
        f'stroke="{_SVG_COLOR_DANGER}" stroke-width="1.8" stroke-dasharray="7 4"/>'
    )
    body.append(
        f'<line x1="{med_x:.1f}" y1="{lp_y0}" x2="{med_x:.1f}" y2="{lp_y0 + lp_h}" '
        f'stroke="{_SVG_COLOR_ACCENT_GREEN}" stroke-width="1.8" stroke-dasharray="3 4"/>'
    )

    # Legend — placed below the x-axis to avoid overlapping bars
    leg_y = lp_y0 + lp_h + 38
    body.append(
        f'<line x1="{lp_x0 + 10}" y1="{leg_y}" x2="{lp_x0 + 34}" y2="{leg_y}" '
        f'stroke="{_SVG_COLOR_DANGER}" stroke-width="1.8" stroke-dasharray="7 4"/>'
    )
    body.append(
        f'<text x="{lp_x0 + 38}" y="{leg_y + 4}" font-family="{_FONT}" '
        f'font-size="9.5" fill="{_SVG_COLOR_TEXT}">'
        f"BSM r = {BSM_R:.3f}</text>"
    )
    body.append(
        f'<line x1="{lp_x0 + 10}" y1="{leg_y + 17}" x2="{lp_x0 + 34}" y2="{leg_y + 17}" '
        f'stroke="{_SVG_COLOR_ACCENT_GREEN}" stroke-width="1.8" stroke-dasharray="3 4"/>'
    )
    body.append(
        f'<text x="{lp_x0 + 38}" y="{leg_y + 21}" font-family="{_FONT}" '
        f'font-size="9.5" fill="{_SVG_COLOR_TEXT}">'
        f"Median (successful fits) = {BSM_MEDIAN_SUCCESSFUL:.3f}</text>"
    )

    # Null-screened annotation on the spike
    spike_bin_idx = np.argmax(counts)
    spike_x = lp_x0 + spike_bin_idx * bin_px + bin_px / 2
    spike_top = lp_y(counts[spike_bin_idx])
    body.append(
        f'<text x="{spike_x:.1f}" y="{spike_top - 5:.1f}" text-anchor="middle" '
        f'font-family="{_FONT}" font-size="9" fill="{_SVG_COLOR_TEXT_MUTED}">'
        f"null-screened (n=534)</text>"
    )

    # ── Right panel: BSM operating-point dot plot ────────────────────────────
    # Panel x: 580..1120, plot x: 635..1105 → 470px
    rp_left, rp_right = 580, 1120
    pp_x0 = rp_left + 58
    pp_w = rp_right - pp_x0 - 20
    pp_yc = 230  # center y for main row
    pp_ya = 185  # y for analog dots row
    ax_y_r = 295  # x-axis line y

    x_nrmse_min, x_nrmse_max = 0.067, 0.083
    nrmse_span = x_nrmse_max - x_nrmse_min

    def rp_x(v: float) -> float:
        return pp_x0 + (v - x_nrmse_min) / nrmse_span * pp_w

    # Title + rule
    body.append(
        _panel_title(
            "BSM operating-point validation",
            rp_left + 5,
            34,
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
            f'fill="#9ca3af" fill-opacity="0.85"/>'
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
        ("#9ca3af", 0.85, "circle_sm", "BSM-structure analogs (n=25, n_runs=28,750)"),
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
            f'font-size="9.5" fill="{_SVG_COLOR_TEXT}">{html.escape(label)}</text>'
        )

    # Panel divider
    body.append(
        f'<line x1="565" y1="20" x2="565" y2="{H - 20}" '
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
    """Generate RF meta-regression sensitivity figures and save as SVG."""
    args = _parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    _save_svg(out / "fig_sensitivity_rf_importance.svg", render_rf_importance())
    print("  fig_sensitivity_rf_importance.svg")

    _save_svg(out / "fig_sensitivity_formula_top10.svg", render_formula_top10())
    print("  fig_sensitivity_formula_top10.svg")

    _save_svg(out / "fig_sensitivity_bsm_validation.svg", render_bsm_validation(args.results))
    print("  fig_sensitivity_bsm_validation.svg")

    print(f"output_dir={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
