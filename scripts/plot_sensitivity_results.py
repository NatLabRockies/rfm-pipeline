#!/usr/bin/env python3
# Copyright (c) 2026 Dylan Hettinger
"""Render lightweight SVG summary figures for sensitivity-study results."""

from __future__ import annotations

import argparse
import html
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

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
    _SVG_FS_AXIS,
    _SVG_FS_LABEL,
    _SVG_FS_LEGEND,
    _SVG_FS_SMALL,
    _SVG_FS_TICK,
    _SVG_FS_TITLE,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="Collected results CSV.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures"),
        help="Directory for generated SVG figures.",
    )
    return parser.parse_args()


def _svg_canvas(width: int, height: int, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        f'<rect width="100%" height="100%" fill="{_SVG_COLOR_BACKGROUND}"/>'
        f"{body}</svg>"
    )


def _title(text: str) -> str:
    return (
        f'<text x="40" y="34" font-family="{_SVG_FONT_FAMILY}" font-size="{_SVG_FS_TITLE}" '
        f'font-weight="700" fill="{_SVG_COLOR_TITLE}">{html.escape(text)}</text>'
    )


def _save_svg(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content + "\n", encoding="utf-8")


def _save_pdf(svg_path: Path) -> None:
    """Convert SVG to PDF using Chrome headless with proper page sizing."""
    import re
    import subprocess
    import tempfile

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
    chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
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


def _expand_config_overrides(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the config_overrides JSON column and add each key as a flat column."""
    if "config_overrides" not in df.columns:
        return df
    rows: list[dict] = []
    for v in df["config_overrides"]:
        try:
            rows.append(json.loads(v) if pd.notna(v) else {})
        except (json.JSONDecodeError, ValueError):
            rows.append({})
    expanded = pd.DataFrame(rows, index=df.index)
    df = df.copy()
    for col in expanded.columns:
        if col not in df.columns:
            df[col] = expanded[col]
    return df


# Feature order must match fit_rf_meta_regression.py FEATURES.
_RF_FEATURES = [
    "n_inputs",
    "n_runs",
    "sparsity",
    "interaction_density",
    "nonlinearity_strength",
    "noise_snr",
    "stages.empirical_null_screening.n_permutations",
    "stages.empirical_null_screening.bh_q_threshold",
    "stages.interaction_discovery.n_permutations",
    "stages.interaction_discovery.p_threshold",
    "stages.sparse_selection.n_stability_subsamples",
    "stages.sparse_selection.lasso_alpha_grid_size",
    "variance_threshold",
]


def _build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Return (X, feature_names) for rows where all RF features are available."""
    available = [c for c in _RF_FEATURES if c in df.columns]
    return df[available].values, available


def _group_cv_rf_predictions(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Honest group-blocked CV predictions for successful (non-null-screened) rows.

    Groups by (config_idx, dgp_idx) pair so that replicates of the same
    configuration never appear in both train and test.  Mirrors the actual
    training procedure: each fold trains on ALL rows (including null-screened)
    for the held-in groups, then predicts only successful rows in the test fold.
    This matches how the persisted wave12 RF quality pickle was trained and
    avoids inflated R² from restricting both training and evaluation to the
    successful-run subspace.

    Returns (observed, predicted, cv_r2) on the successful rows only.
    """
    X_all, _ = _build_feature_matrix(df)
    y_all = df["nrmse_relative"].values
    is_succ = df["null_screened"].isna().values  # True for the successful subset

    if "config_idx" in df.columns and "dgp_idx" in df.columns:
        pair_key = df["config_idx"].astype(str) + "_" + df["dgp_idx"].astype(str)
        groups = pair_key.astype("category").cat.codes.values
        n_groups = int(groups.max()) + 1
        n_splits = min(10, n_groups)
        cv = GroupKFold(n_splits=n_splits)
        split_iter = cv.split(X_all, y_all, groups)
    else:
        from sklearn.model_selection import KFold

        cv = KFold(n_splits=10, shuffle=True, random_state=42)
        split_iter = cv.split(X_all, y_all)

    obs_list: list[np.ndarray] = []
    pred_list: list[np.ndarray] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for tr, va in split_iter:
            rf = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
            rf.fit(X_all[tr], y_all[tr])
            # Evaluate only on successful rows in the test fold.
            va_succ = va[is_succ[va]]
            if len(va_succ) == 0:
                continue
            obs_list.append(y_all[va_succ])
            pred_list.append(rf.predict(X_all[va_succ]))

    obs = np.concatenate(obs_list)
    preds = np.concatenate(pred_list)
    cv_r2 = float(r2_score(obs, preds))
    return obs, preds, cv_r2


def _partial_dependence_n_runs(
    df: pd.DataFrame, rf_path: Path
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (n_runs_grid, pdp_mean) partial dependence using the fitted RF.

    Loads the joblib-serialised quality model.  Returns None when the model
    file is absent or fails to load.
    """
    if not rf_path.exists():
        return None
    try:
        rf = joblib.load(rf_path)
    except Exception:
        return None

    available = [c for c in _RF_FEATURES if c in df.columns]
    if len(available) < len(_RF_FEATURES):
        return None

    # Marginalize over successful runs only — matches the docstring of
    # ``_render_sample_size_curve`` and the raw-overlay band, which exclude
    # null-screened rows. Including them dampens PDP toward γ = 0 by the
    # null-screen fraction (~25% on wave12).
    if "null_screened" in df.columns:
        base_df = df.loc[df["null_screened"].isna()]
    else:
        base_df = df
    X_base = base_df[available].values.copy().astype(float)
    try:
        n_runs_idx = available.index("n_runs")
    except ValueError:
        return None

    n_runs_min = float(df["n_runs"].min())
    n_runs_max = float(df["n_runs"].max())
    grid = np.linspace(n_runs_min, n_runs_max, 40)
    pdp = np.empty(len(grid))
    for i, val in enumerate(grid):
        X_mod = X_base.copy()
        X_mod[:, n_runs_idx] = val
        pdp[i] = float(rf.predict(X_mod).mean())

    return grid, pdp


def _render_sample_size_curve(results: pd.DataFrame, rf_path: Path | None = None) -> str:
    """Marginal effect of training-run count on pipeline quality (γ = nrmse_relative).

    Uses the RF partial dependence if the model is available; otherwise falls
    back to a per-level mean ± CI band from the raw data.  All null-screened
    runs (γ = 0 by construction) are excluded so the y-axis reflects genuine
    quality variation.
    """
    # --- Partial dependence path (preferred) ---
    if rf_path is not None:
        pdp = _partial_dependence_n_runs(results, rf_path)
    else:
        pdp = None

    # --- Raw marginal (mean ± CI per n_runs level, successful runs only) ---
    x_col = "subsample_n" if "subsample_n" in results.columns else "n_runs"
    if x_col not in results.columns or "nrmse_relative" not in results.columns:
        return _svg_canvas(900, 540, _title("Sensitivity: effect of training-run count on quality"))

    succ = (
        results[results["null_screened"].isna()].copy()
        if "null_screened" in results.columns
        else results.copy()
    )  # noqa: E501
    if succ.empty:
        succ = results.copy()

    grouped = (
        succ.groupby(x_col)["nrmse_relative"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={x_col: "subsample_n"})
    )
    grouped["ci"] = 1.96 * grouped["std"].fillna(0.0) / np.sqrt(grouped["count"].clip(lower=1))
    if grouped.empty:
        return _svg_canvas(900, 540, _title("Sensitivity: effect of training-run count on quality"))

    width, height = 900, 540
    left, top, plot_w, plot_h = 90, 70, 760, 390

    n_vals = grouped["subsample_n"].to_numpy(dtype=float)
    raw_means = grouped["mean"].to_numpy(dtype=float)
    raw_ci = grouped["ci"].to_numpy(dtype=float)

    # y-axis: widen slightly so PDP and raw band fit
    y_min = float(min((raw_means - raw_ci).min(), pdp[1].min() if pdp else raw_means.min()))
    y_max = float(max((raw_means + raw_ci).max(), pdp[1].max() if pdp else raw_means.max()))
    y_pad = max((y_max - y_min) * 0.08, 0.01)
    y_min -= y_pad
    y_max += y_pad
    x_min_val, x_max_val = float(n_vals.min()), float(n_vals.max())
    y_span = max(y_max - y_min, 1e-9)
    x_span = max(x_max_val - x_min_val, 1e-9)

    def xm(v: float) -> float:
        return left + ((v - x_min_val) / x_span) * plot_w

    def ym(v: float) -> float:
        return top + plot_h - ((v - y_min) / y_span) * plot_h

    body = [
        _title("Sensitivity: effect of training-run count on quality"),
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" '
        f'stroke="{_SVG_COLOR_EDGE}" stroke-width="1.5"/>',
    ]

    # Raw CI band (light blue fill)
    upper_pts = " ".join(
        f"{xm(n):.2f},{ym(m + c):.2f}" for n, m, c in zip(n_vals, raw_means, raw_ci, strict=True)
    )  # noqa: E501
    lower_pts = " ".join(
        f"{xm(n):.2f},{ym(m - c):.2f}"
        for n, m, c in zip(n_vals[::-1], raw_means[::-1], raw_ci[::-1], strict=True)  # noqa: E501
    )
    body.append(
        f'<polygon points="{upper_pts} {lower_pts}" '
        f'fill="{_SVG_COLOR_PRIMARY}" fill-opacity="0.12" stroke="none"/>'
    )
    # Raw mean line (dashed, muted)
    raw_line = " ".join(f"{xm(n):.2f},{ym(m):.2f}" for n, m in zip(n_vals, raw_means, strict=True))
    body.append(
        f'<polyline points="{raw_line}" fill="none" stroke="{_SVG_COLOR_TEXT_MUTED}" '
        f'stroke-width="1.5" stroke-dasharray="4 3"/>'
    )

    # PDP line (solid, primary colour)
    if pdp is not None:
        pdp_grid, pdp_vals = pdp
        pdp_line = " ".join(
            f"{xm(n):.2f},{ym(v):.2f}" for n, v in zip(pdp_grid, pdp_vals, strict=True)
        )
        body.append(
            f'<polyline points="{pdp_line}" fill="none" stroke="{_SVG_COLOR_PRIMARY}" '
            f'stroke-width="3"/>'
        )

    # Legend — bottom-right inside plot bounding box
    n_legend_items = 3 if pdp is not None else 2
    legend_item_h = 22
    legend_pad_v = 10
    legend_pad_h = 12
    legend_swatch_w = 30
    legend_text_w = 222  # wide enough for "Observed mean (successful runs)"
    legend_total_w = legend_pad_h + legend_swatch_w + 6 + legend_text_w + legend_pad_h
    legend_total_h = n_legend_items * legend_item_h + 2 * legend_pad_v
    legend_right = left + plot_w - 12
    legend_bottom = top + plot_h - 12
    legend_bg_x = legend_right - legend_total_w
    legend_bg_y = legend_bottom - legend_total_h
    lx = legend_bg_x + legend_pad_h

    body.append(
        f'<rect x="{legend_bg_x}" y="{legend_bg_y}" '
        f'width="{legend_total_w}" height="{legend_total_h}" '
        f'fill="{_SVG_COLOR_BACKGROUND}" fill-opacity="0.85" '
        f'stroke="none"/>'
    )

    def _legy(i: int) -> float:
        return legend_bg_y + legend_pad_v + (i + 0.5) * legend_item_h

    # Row 0: CI band swatch
    ly0 = _legy(0)
    body += [
        f'<rect x="{lx}" y="{ly0 - 6:.1f}" width="{legend_swatch_w}" height="12" '
        f'fill="{_SVG_COLOR_PRIMARY}" fill-opacity="0.35" stroke="none"/>',
        f'<text x="{lx + legend_swatch_w + 6}" y="{ly0 + 4:.1f}" '
        f'font-family="{_SVG_FONT_FAMILY}" font-size="{_SVG_FS_LEGEND}" '
        f'fill="{_SVG_COLOR_TEXT}">95% CI (observed mean)</text>',
    ]
    # Row 1: observed mean (dashed)
    ly1 = _legy(1)
    body += [
        f'<line x1="{lx}" y1="{ly1:.1f}" x2="{lx + legend_swatch_w}" y2="{ly1:.1f}" '
        f'stroke="{_SVG_COLOR_TEXT_MUTED}" stroke-width="1.5" stroke-dasharray="4 3"/>',
        f'<text x="{lx + legend_swatch_w + 6}" y="{ly1 + 4:.1f}" '
        f'font-family="{_SVG_FONT_FAMILY}" font-size="{_SVG_FS_LEGEND}" '
        f'fill="{_SVG_COLOR_TEXT}">Observed mean (successful runs)</text>',
    ]
    # Row 2 (only when RF model available): PDP solid line
    if pdp is not None:
        ly2 = _legy(2)
        body += [
            f'<line x1="{lx}" y1="{ly2:.1f}" x2="{lx + legend_swatch_w}" y2="{ly2:.1f}" '
            f'stroke="{_SVG_COLOR_PRIMARY}" stroke-width="3"/>',
            f'<text x="{lx + legend_swatch_w + 6}" y="{ly2 + 4:.1f}" '
            f'font-family="{_SVG_FONT_FAMILY}" font-size="{_SVG_FS_LEGEND}" '
            f'fill="{_SVG_COLOR_TEXT}">Partial dependence (RF)</text>',
        ]

    # X ticks
    tick_candidates = np.arange(5000, 35000, 5000)
    tick_vals_x = [t for t in tick_candidates if x_min_val <= t <= x_max_val]
    for tv in tick_vals_x:
        tx = xm(float(tv))
        body += [
            f'<line x1="{tx:.2f}" y1="{top + plot_h}" x2="{tx:.2f}" y2="{top + plot_h + 5}" '
            f'stroke="{_SVG_COLOR_EDGE}" stroke-width="1"/>',
            f'<text x="{tx:.2f}" y="{top + plot_h + 20}" text-anchor="middle" '
            f'font-family="{_SVG_FONT_FAMILY}" font-size="{_SVG_FS_TICK}" '
            f'fill="{_SVG_COLOR_TEXT_MUTED}">{int(tv):,}</text>',
        ]

    # Y ticks (4 ticks)
    y_tick_vals = np.linspace(y_min, y_max, 5)[1:-1]
    for tv in y_tick_vals:
        ty = ym(float(tv))
        body += [
            f'<line x1="{left - 5}" y1="{ty:.2f}" x2="{left}" y2="{ty:.2f}" '
            f'stroke="{_SVG_COLOR_EDGE}" stroke-width="1"/>',
            f'<text x="{left - 8}" y="{ty + 4:.2f}" text-anchor="end" '
            f'font-family="{_SVG_FONT_FAMILY}" font-size="{_SVG_FS_TICK}" '
            f'fill="{_SVG_COLOR_TEXT_MUTED}">{tv:.2f}</text>',
        ]

    # Axis labels
    body += [
        f'<text x="470" y="510" text-anchor="middle" font-family="{_SVG_FONT_FAMILY}" '
        f'font-size="{_SVG_FS_AXIS}" fill="{_SVG_COLOR_TEXT}">Number of training runs</text>',
        f'<text x="28" y="265" transform="rotate(-90 28 265)" text-anchor="middle" '
        f'font-family="{_SVG_FONT_FAMILY}" font-size="{_SVG_FS_AXIS}" '
        f'fill="{_SVG_COLOR_TEXT}">\u03b3 = (nRMSE \u2013 nRMSE\u2080) / nRMSE\u2080</text>',
        # RF importance annotation
        f'<text x="{left + 12}" y="{top + plot_h - 10}" font-family="{_SVG_FONT_FAMILY}" '
        f'font-size="{_SVG_FS_SMALL}" fill="{_SVG_COLOR_TEXT_MUTED}">'
        f"RF importance: 0.029 (lowest of 13 parameters)</text>",
    ]
    return _svg_canvas(width, height, "".join(body))


def _render_horizontal_bar_chart(
    title: str, labels: list[str], values: list[float], color: str
) -> str:
    width, height = 900, 540
    left, top = 240, 80
    plot_w = 590
    bar_h = 26
    gap = 18
    max_value = max(values) if values else 1.0
    body = [_title(title)]
    for idx, (label, value) in enumerate(zip(labels, values, strict=True)):
        y_pos = top + idx * (bar_h + gap)
        bar_w = 0.0 if max_value == 0 else (value / max_value) * plot_w
        body.append(
            f'<text x="220" y="{y_pos + 18}" text-anchor="end" font-family="{_SVG_FONT_FAMILY}" font-size="{_SVG_FS_LABEL}" fill="{_SVG_COLOR_TEXT}">{html.escape(label)}</text>'  # noqa: E501
        )
        body.append(
            f'<rect x="{left}" y="{y_pos}" width="{plot_w}" height="{bar_h}" fill="{_SVG_COLOR_LIGHT}" opacity="0.3"/>'  # noqa: E501
        )
        body.append(
            f'<rect x="{left}" y="{y_pos}" width="{bar_w:.2f}" height="{bar_h}" fill="{color}"/>'
        )
        # Value label: inside bar (white) when it would overflow the right axis edge
        if bar_w > plot_w - 42:
            body.append(
                f'<text x="{left + bar_w - 6:.2f}" y="{y_pos + 18}" text-anchor="end" '
                f'font-family="{_SVG_FONT_FAMILY}" font-size="{_SVG_FS_LEGEND}" fill="#ffffff">'
                f"{value:.4f}</text>"
            )
        else:
            body.append(
                f'<text x="{left + bar_w + 8:.2f}" y="{y_pos + 18}" font-family="{_SVG_FONT_FAMILY}" font-size="{_SVG_FS_LEGEND}" fill="{_SVG_COLOR_TEXT_MUTED}">{value:.4f}</text>'  # noqa: E501
            )
    return _svg_canvas(width, height, "".join(body))


def _render_scatter(observed: np.ndarray, predicted: np.ndarray, cv_r2: float) -> str:
    """Scatter plot of RF group-CV predicted vs actual γ (successful runs only)."""
    width, height = 900, 540
    left, top, plot_w, plot_h = 100, 70, 740, 380

    x_min = float(np.percentile(observed, 1))
    x_max = float(np.percentile(observed, 99))
    y_min = float(np.percentile(predicted, 1))
    y_max = float(np.percentile(predicted, 99))
    x_pad = max((x_max - x_min) * 0.05, 0.005)
    y_pad = max((y_max - y_min) * 0.05, 0.005)
    x_min -= x_pad
    x_max += x_pad
    y_min -= y_pad
    y_max += y_pad
    x_span = max(x_max - x_min, 1e-9)
    y_span = max(y_max - y_min, 1e-9)

    def xm(v: float) -> float:
        return left + ((v - x_min) / x_span) * plot_w

    def ym(v: float) -> float:
        return top + plot_h - ((v - y_min) / y_span) * plot_h

    # 1:1 reference line mapped to shared diagonal in independent axes
    shared_lo = max(x_min, y_min)
    shared_hi = min(x_max, y_max)
    body = [
        _title("RF meta-regression: predicted vs actual \u03b3 (cross-validated)"),
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" '
        f'stroke="{_SVG_COLOR_EDGE}" stroke-width="1.5"/>',
    ]
    if shared_lo < shared_hi:
        body.append(
            f'<line x1="{xm(shared_lo):.1f}" y1="{ym(shared_lo):.1f}" '
            f'x2="{xm(shared_hi):.1f}" y2="{ym(shared_hi):.1f}" '
            f'stroke="{_SVG_COLOR_DANGER}" stroke-dasharray="6 4" stroke-width="2"/>'
        )

    # Scatter points
    for obs, pred in zip(observed, predicted, strict=True):
        if x_min <= obs <= x_max and y_min <= pred <= y_max:
            body.append(
                f'<circle cx="{xm(obs):.2f}" cy="{ym(pred):.2f}" r="3.5" '
                f'fill="{_SVG_COLOR_PRIMARY}" fill-opacity="0.40"/>'
            )

    # Axis labels
    body += [
        f'<text x="470" y="500" text-anchor="middle" font-family="{_SVG_FONT_FAMILY}" '
        f'font-size="{_SVG_FS_AXIS}" fill="{_SVG_COLOR_TEXT}">Observed \u03b3</text>',
        f'<text x="32" y="265" transform="rotate(-90 32 265)" text-anchor="middle" '
        f'font-family="{_SVG_FONT_FAMILY}" font-size="{_SVG_FS_AXIS}" '
        f'fill="{_SVG_COLOR_TEXT}">Predicted \u03b3</text>',
    ]

    # X ticks (4 evenly spaced)
    for tv in np.linspace(x_min, x_max, 5)[1:-1]:
        tx = xm(float(tv))
        body += [
            f'<line x1="{tx:.2f}" y1="{top + plot_h}" x2="{tx:.2f}" y2="{top + plot_h + 5}" '
            f'stroke="{_SVG_COLOR_EDGE}" stroke-width="1"/>',
            f'<text x="{tx:.2f}" y="{top + plot_h + 18}" text-anchor="middle" '
            f'font-family="{_SVG_FONT_FAMILY}" font-size="{_SVG_FS_TICK}" '
            f'fill="{_SVG_COLOR_TEXT_MUTED}">{tv:.2f}</text>',
        ]

    # Y ticks
    for tv in np.linspace(y_min, y_max, 5)[1:-1]:
        ty = ym(float(tv))
        body += [
            f'<line x1="{left - 5}" y1="{ty:.2f}" x2="{left}" y2="{ty:.2f}" '
            f'stroke="{_SVG_COLOR_EDGE}" stroke-width="1"/>',
            f'<text x="{left - 8}" y="{ty + 4:.2f}" text-anchor="end" '
            f'font-family="{_SVG_FONT_FAMILY}" font-size="{_SVG_FS_TICK}" '
            f'fill="{_SVG_COLOR_TEXT_MUTED}">{tv:.2f}</text>',
        ]

    # R² annotation (SVG tspan for superscript)
    body.append(
        f'<text x="{left + 16}" y="{top + 26}" font-family="{_SVG_FONT_FAMILY}" '
        f'font-size="{_SVG_FS_LABEL}" fill="{_SVG_COLOR_TEXT_MUTED}">'
        f'CV R<tspan dy="-5" font-size="{_SVG_FS_SMALL}">2</tspan>'
        f'<tspan dy="5"> = {cv_r2:.3f}</tspan></text>'
    )
    body.append(
        f'<text x="{left + 16}" y="{top + 44}" font-family="{_SVG_FONT_FAMILY}" '
        f'font-size="{_SVG_FS_LEGEND}" fill="{_SVG_COLOR_TEXT_MUTED}">'
        f"Group-blocked CV (config \u00d7 DGP pairs); n = {len(observed):,}</text>"
    )
    return _svg_canvas(width, height, "".join(body))


def main() -> int:
    """Render sensitivity study SVG figures from collected results."""
    args = _parse_args()
    results = pd.read_csv(args.results)
    results = _expand_config_overrides(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # RF model path (alongside the results CSV). Derive the prefix from the
    # input CSV stem so a future ``wave34_combined_clean.csv`` picks up
    # ``wave34_rf_quality.pkl`` instead of silently reusing the wave12 model
    # (mirrors the ``scripts/fit_sensitivity_rf.py --dump-models`` prefix
    # convention introduced in round 13). Missing models fall back to the raw
    # observed mean curve instead of silently reusing another wave's model.
    prefix = args.results.stem.split("_", 1)[0]
    rf_path = args.results.parent / f"{prefix}_rf_quality.pkl"

    # --- Sample-size curve (partial dependence if RF available) ---
    sample_curve = _render_sample_size_curve(results, rf_path=rf_path)

    # --- Main effects (Pearson |r| with nrmse_relative) ---
    analysis_frame = results.replace([np.inf, -np.inf], np.nan).dropna(subset=["nrmse_relative"])
    aidx = analysis_frame.index
    nrmse_rel = analysis_frame["nrmse_relative"]
    main_effects_labels = [
        "Holdout fraction",
        "Variance threshold",
        "Screening permutations",
        "BH threshold (q)",
        "Interaction permutations",
        "Interaction p-threshold",
        "Stability subsamples",
        "LASSO \u03b1 grid size",
        "\u03b4 threshold override",
    ]
    main_effects_values = [
        abs(analysis_frame.get("holdout_fraction", pd.Series(0.05, index=aidx)).corr(nrmse_rel)),
        abs(
            analysis_frame.get(
                "variance_threshold",
                analysis_frame.get("algorithm.variance_threshold", pd.Series(0.90, index=aidx)),
            ).corr(nrmse_rel)
        ),
        abs(
            analysis_frame.get(
                "stages.empirical_null_screening.n_permutations", pd.Series(201, index=aidx)
            ).corr(nrmse_rel)
        ),
        abs(
            analysis_frame.get(
                "stages.empirical_null_screening.bh_q_threshold", pd.Series(0.05, index=aidx)
            ).corr(nrmse_rel)
        ),
        abs(
            analysis_frame.get(
                "stages.interaction_discovery.n_permutations", pd.Series(31, index=aidx)
            ).corr(nrmse_rel)
        ),
        abs(
            analysis_frame.get(
                "stages.interaction_discovery.p_threshold", pd.Series(0.05, index=aidx)
            ).corr(nrmse_rel)
        ),
        abs(
            analysis_frame.get(
                "stages.sparse_selection.n_stability_subsamples", pd.Series(50, index=aidx)
            ).corr(nrmse_rel)
        ),
        abs(
            analysis_frame.get(
                "stages.sparse_selection.lasso_alpha_grid_size", pd.Series(40, index=aidx)
            ).corr(nrmse_rel)
        ),
        abs(
            analysis_frame.get(
                "stages.final_artifacts.delta_threshold_override", pd.Series(0.0, index=aidx)
            )
            .fillna(0.0)
            .corr(nrmse_rel)
        ),
    ]
    main_effects_values = [0.0 if pd.isna(v) else float(v) for v in main_effects_values]
    main_effects_svg = _render_horizontal_bar_chart(
        "Sensitivity: main effect correlations with nRMSE",
        main_effects_labels,
        main_effects_values,
        _SVG_COLOR_ACCENT_ORANGE,
    )

    # --- Runtime breakdown ---
    if "stage" in results.columns and "runtime_seconds" in results.columns:
        runtime = (
            results.groupby("stage", as_index=False)["runtime_seconds"]
            .mean()
            .sort_values("runtime_seconds", ascending=False)
        )
        runtime_labels = runtime["stage"].tolist()
        runtime_values = runtime["runtime_seconds"].astype(float).tolist()
    else:
        rt_col = next(
            (c for c in ("pipeline_seconds", "total_wall_seconds") if c in results.columns), None
        )
        if rt_col and "block" in results.columns:
            runtime = (
                results.groupby("block", as_index=False)[rt_col]
                .mean()
                .sort_values(rt_col, ascending=False)
            )
            _block_label_map = {
                "pure_synthetic": "Pure synthetic runs",
                "bsm_structure": "BSM-structure runs",
            }
            runtime_labels = [
                _block_label_map.get(b, b.replace("_", " ").title())
                for b in runtime["block"].tolist()
            ]
            runtime_values = runtime[rt_col].astype(float).tolist()
        elif rt_col:
            runtime_labels = ["Mean pipeline time"]
            runtime_values = [float(results[rt_col].mean())]
        else:
            runtime_labels, runtime_values = [], []
    runtime_svg = _render_horizontal_bar_chart(
        "Sensitivity: mean pipeline runtime by block (seconds)",
        runtime_labels,
        runtime_values,
        _SVG_COLOR_ACCENT_GREEN,
    )

    # --- Formula validation (RF group-CV predicted vs actual) ---
    print("Computing group-blocked CV predictions (this may take ~60 s)...")
    observed, predicted, cv_r2 = _group_cv_rf_predictions(results)
    print(f"  Group-CV R\u00b2 = {cv_r2:.4f} (successful runs, config\u00d7DGP groups)")
    validation_svg = _render_scatter(observed, predicted, cv_r2)

    figs = {
        "fig_sensitivity_sample_size_curve": sample_curve,
        "fig_sensitivity_main_effects": main_effects_svg,
        "fig_sensitivity_runtime_breakdown": runtime_svg,
        "fig_sensitivity_rf_validation": validation_svg,
    }
    for stem, svg in figs.items():
        p = args.output_dir / f"{stem}.svg"
        _save_svg(p, svg)
        _save_pdf(p)
        print(f"  {stem}.svg/pdf")
    print(f"output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
