#!/usr/bin/env python3
# Copyright (c) 2026 Dylan Hettinger
"""Render lightweight SVG summary figures for sensitivity-study results."""

from __future__ import annotations

import argparse
import html
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bsm_rfm.manuscript_stages import (  # noqa: E402
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
        f'<text x="40" y="34" font-family="{_SVG_FONT_FAMILY}" font-size="18" '
        f'font-weight="700" fill="{_SVG_COLOR_TITLE}">{html.escape(text)}</text>'
    )


def _save_svg(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content + "\n", encoding="utf-8")


def _fit_relative_model(results: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Fit a degree-2 OLS on nrmse_relative. Handles long and flat formats."""
    if "stage" in results.columns and "nrmse" in results.columns:
        # Legacy long format.
        final_rows = results.loc[results["stage"] == "final_ols"].copy()
        null_rows = results.loc[
            results["stage"] == "null_baseline", ["job_id", "subsample_n", "nrmse"]
        ].rename(columns={"nrmse": "nrmse_null"})
        frame = final_rows.merge(null_rows, on=["job_id", "subsample_n"], how="inner")
        frame["nrmse_relative"] = (frame["nrmse"] - frame["nrmse_null"]) / frame["nrmse_null"]
    else:
        # Flat format: nrmse_relative already present, or compute it.
        frame = results.copy()
        if "nrmse_relative" not in frame.columns:
            null_col = next(
                (c for c in ("null_mean_nrmse", "nrmse_null") if c in frame.columns), None
            )
            final_col = next(
                (c for c in ("nrmse_final", "final_ols_nrmse") if c in frame.columns), None
            )
            if null_col and final_col:
                frame["nrmse_relative"] = (frame[final_col] - frame[null_col]) / frame[
                    null_col
                ].abs()
        # Use n_runs as the sample-size predictor when subsample_n is absent.
        if "subsample_n" not in frame.columns and "n_runs" in frame.columns:
            frame["subsample_n"] = frame["n_runs"]

    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["nrmse_relative"])
    predictors = pd.DataFrame(
        {
            "log_subsample_n": np.log(
                frame.get("subsample_n", frame.get("n_runs", pd.Series(1000, index=frame.index)))
                .astype(float)
                .clip(lower=1)
            ),
            "holdout_fraction": frame.get("holdout_fraction", 0.05),
            "variance_threshold": frame.get(
                "variance_threshold", frame.get("algorithm.variance_threshold", 0.90)
            ),
            "log_n_perm_screen": np.log(
                frame.get("stages.empirical_null_screening.n_permutations", 201)
            ),
            "bh_q": frame.get("stages.empirical_null_screening.bh_q_threshold", 0.05),
            "log_n_perm_interaction": np.log(
                frame.get("stages.interaction_discovery.n_permutations", 31)
            ),
            "p_threshold": frame.get("stages.interaction_discovery.p_threshold", 0.05),
            "log_n_stability_subsamples": np.log(
                frame.get("stages.sparse_selection.n_stability_subsamples", 50)
            ),
            "lasso_alpha_percentile": frame.get(
                "stages.sparse_selection.lasso_alpha_percentile", 40
            ),
            "log_delta_threshold": np.log(
                frame.get(
                    "stages.final_artifacts.delta_threshold_override",
                    pd.Series(0.0, index=frame.index),
                ).fillna(0.0)
                + 0.001
            ),
        }
    )
    poly = PolynomialFeatures(degree=2, include_bias=True)
    design = poly.fit_transform(predictors.to_numpy(dtype=float))
    coef, _, _, _ = np.linalg.lstsq(
        design, frame["nrmse_relative"].to_numpy(dtype=float), rcond=None
    )
    predicted = design @ coef
    return frame, predicted


def _render_sample_size_curve(results: pd.DataFrame) -> str:
    """Render nRMSE vs sample size. Handles both long and flat result formats."""
    if "stage" in results.columns and "nrmse" in results.columns:
        frame = results.loc[results["stage"] == "final_ols", ["subsample_n", "nrmse"]].copy()
        x_col, y_col = "subsample_n", "nrmse"
    else:
        # Flat format: use n_runs as the x axis and nrmse_final as y.
        x_col = "subsample_n" if "subsample_n" in results.columns else "n_runs"
        y_col = next((c for c in ("nrmse_final", "final_ols_nrmse") if c in results.columns), None)
        if x_col not in results.columns or y_col is None:
            return _svg_canvas(900, 540, _title("Sensitivity sample-size curve"))
        frame = results[[x_col, y_col]].copy().rename(columns={y_col: "nrmse"})
        x_col = x_col  # keep local name consistent
    grouped = frame.groupby(x_col)["nrmse"].agg(["mean", "std", "count"]).reset_index()
    grouped = grouped.rename(columns={x_col: "subsample_n"})
    grouped["ci"] = 1.96 * grouped["std"].fillna(0.0) / np.sqrt(grouped["count"].clip(lower=1))
    if grouped.empty:
        return _svg_canvas(900, 540, _title("Sensitivity sample-size curve"))

    width, height = 900, 540
    left, top, plot_w, plot_h = 90, 70, 760, 380
    x_values = np.log(grouped["subsample_n"].to_numpy(dtype=float))
    y_min = float((grouped["mean"] - grouped["ci"]).min())
    y_max = float((grouped["mean"] + grouped["ci"]).max())
    x_min, x_max = float(x_values.min()), float(x_values.max())
    y_span = max(y_max - y_min, 1e-9)
    x_span = max(x_max - x_min, 1e-9)

    def x_map(value: float) -> float:
        return left + ((value - x_min) / x_span) * plot_w

    def y_map(value: float) -> float:
        return top + plot_h - ((value - y_min) / y_span) * plot_h

    upper = " ".join(
        f"{x_map(x):.2f},{y_map(y):.2f}"
        for x, y in zip(x_values, grouped["mean"] + grouped["ci"], strict=True)
    )
    lower = " ".join(
        f"{x_map(x):.2f},{y_map(y):.2f}"
        for x, y in zip(x_values[::-1], (grouped["mean"] - grouped["ci"])[::-1], strict=True)
    )
    line = " ".join(
        f"{x_map(x):.2f},{y_map(y):.2f}" for x, y in zip(x_values, grouped["mean"], strict=True)
    )
    body = [
        _title("Sensitivity sample-size curve"),
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="{_SVG_COLOR_EDGE}" stroke-width="1.5"/>',  # noqa: E501
        f'<polygon points="{upper} {lower}" fill="{_SVG_COLOR_PRIMARY}" fill-opacity="0.18" stroke="none"/>',  # noqa: E501
        f'<polyline points="{line}" fill="none" stroke="{_SVG_COLOR_PRIMARY}" stroke-width="3"/>',
    ]
    for _, row in grouped.iterrows():
        x_pos = x_map(math.log(float(row["subsample_n"])))
        y_pos = y_map(float(row["mean"]))
        body.append(
            f'<circle cx="{x_pos:.2f}" cy="{y_pos:.2f}" r="4" fill="{_SVG_COLOR_PRIMARY}"/>'
        )
        body.append(
            f'<text x="{x_pos:.2f}" y="{top + plot_h + 24}" text-anchor="middle" font-family="{_SVG_FONT_FAMILY}" font-size="11" fill="{_SVG_COLOR_TEXT_MUTED}">{int(row["subsample_n"]):d}</text>'  # noqa: E501
        )
    body.append(
        f'<text x="470" y="500" text-anchor="middle" font-family="{_SVG_FONT_FAMILY}" font-size="13" fill="{_SVG_COLOR_TEXT}">subsample_n</text>'  # noqa: E501
    )
    body.append(
        f'<text x="28" y="265" transform="rotate(-90 28 265)" text-anchor="middle" font-family="{_SVG_FONT_FAMILY}" font-size="13" fill="{_SVG_COLOR_TEXT}">nRMSE</text>'  # noqa: E501
    )
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
            f'<text x="220" y="{y_pos + 18}" text-anchor="end" font-family="{_SVG_FONT_FAMILY}" font-size="12" fill="{_SVG_COLOR_TEXT}">{html.escape(label)}</text>'  # noqa: E501
        )
        body.append(
            f'<rect x="{left}" y="{y_pos}" width="{plot_w}" height="{bar_h}" fill="{_SVG_COLOR_LIGHT}" opacity="0.3"/>'  # noqa: E501
        )
        body.append(
            f'<rect x="{left}" y="{y_pos}" width="{bar_w:.2f}" height="{bar_h}" fill="{color}"/>'
        )
        body.append(
            f'<text x="{left + bar_w + 8:.2f}" y="{y_pos + 18}" font-family="{_SVG_FONT_FAMILY}" font-size="11" fill="{_SVG_COLOR_TEXT_MUTED}">{value:.4f}</text>'  # noqa: E501
        )
    return _svg_canvas(width, height, "".join(body))


def _render_scatter(frame: pd.DataFrame, predicted: np.ndarray) -> str:
    width, height = 900, 540
    left, top, plot_w, plot_h = 90, 70, 760, 380
    observed = frame["nrmse_relative"].to_numpy(dtype=float)
    minimum = float(min(observed.min(), predicted.min()))
    maximum = float(max(observed.max(), predicted.max()))
    span = max(maximum - minimum, 1e-9)

    def x_map(value: float) -> float:
        return left + ((value - minimum) / span) * plot_w

    def y_map(value: float) -> float:
        return top + plot_h - ((value - minimum) / span) * plot_h

    body = [
        _title("Sensitivity formula validation"),
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="{_SVG_COLOR_EDGE}" stroke-width="1.5"/>',  # noqa: E501
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top}" stroke="{_SVG_COLOR_DANGER}" stroke-dasharray="6 4" stroke-width="2"/>',  # noqa: E501
    ]
    for obs, pred in zip(observed, predicted, strict=True):
        body.append(
            f'<circle cx="{x_map(obs):.2f}" cy="{y_map(pred):.2f}" r="3.5" fill="{_SVG_COLOR_PRIMARY}" fill-opacity="0.55"/>'  # noqa: E501
        )
    body.append(
        f'<text x="470" y="500" text-anchor="middle" font-family="{_SVG_FONT_FAMILY}" font-size="13" fill="{_SVG_COLOR_TEXT}">observed nRMSE relative</text>'  # noqa: E501
    )
    body.append(
        f'<text x="28" y="265" transform="rotate(-90 28 265)" text-anchor="middle" font-family="{_SVG_FONT_FAMILY}" font-size="13" fill="{_SVG_COLOR_TEXT}">predicted nRMSE relative</text>'  # noqa: E501
    )
    body.append(
        f'<text x="680" y="96" font-family="{_SVG_FONT_FAMILY}" font-size="12" fill="{_SVG_COLOR_TEXT_MUTED}">R² = {r2_score(observed, predicted):.4f}</text>'  # noqa: E501
    )
    return _svg_canvas(width, height, "".join(body))


def main() -> int:
    """Render sensitivity study SVG figures from collected results."""
    args = _parse_args()
    results = pd.read_csv(args.results)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame, predicted = _fit_relative_model(results)
    sample_curve = _render_sample_size_curve(results)
    # Build the analysis frame for correlations (flat format: use frame already filtered).
    analysis_frame = frame
    main_effects_labels = [
        "holdout_fraction",
        "variance_threshold",
        "screen_permutations",
        "screen_bh_q",
        "interaction_permutations",
        "interaction_p_threshold",
        "stability_subsamples",
        "lasso_alpha_percentile",
        "delta_threshold_override",
    ]
    aidx = analysis_frame.index
    nrmse_rel = analysis_frame["nrmse_relative"]
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
                "stages.sparse_selection.lasso_alpha_percentile", pd.Series(40, index=aidx)
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
    main_effects_values = [0.0 if pd.isna(value) else float(value) for value in main_effects_values]
    main_effects_svg = _render_horizontal_bar_chart(
        "Sensitivity main effects",
        main_effects_labels,
        main_effects_values,
        _SVG_COLOR_ACCENT_ORANGE,
    )

    # Runtime breakdown: long format uses stage/runtime_seconds; flat format uses pipeline_seconds.
    if "stage" in results.columns and "runtime_seconds" in results.columns:
        runtime = (
            results.groupby("stage", as_index=False)["runtime_seconds"]
            .mean()
            .sort_values("runtime_seconds", ascending=False)
        )
        runtime_labels = runtime["stage"].tolist()
        runtime_values = runtime["runtime_seconds"].astype(float).tolist()
    else:
        # Flat format: show mean wall time by block (pure_synthetic vs bsm_structure).
        rt_col = next(
            (c for c in ("pipeline_seconds", "total_wall_seconds") if c in results.columns), None
        )
        if rt_col and "block" in results.columns:
            runtime = (
                results.groupby("block", as_index=False)[rt_col]
                .mean()
                .sort_values(rt_col, ascending=False)
            )
            runtime_labels = runtime["block"].tolist()
            runtime_values = runtime[rt_col].astype(float).tolist()
        elif rt_col:
            runtime_labels = ["mean_pipeline"]
            runtime_values = [float(results[rt_col].mean())]
        else:
            runtime_labels, runtime_values = [], []
    runtime_svg = _render_horizontal_bar_chart(
        "Sensitivity runtime breakdown",
        runtime_labels,
        runtime_values,
        _SVG_COLOR_ACCENT_GREEN,
    )
    validation_svg = _render_scatter(frame, predicted)

    _save_svg(args.output_dir / "fig_sensitivity_sample_size_curve.svg", sample_curve)
    _save_svg(args.output_dir / "fig_sensitivity_main_effects.svg", main_effects_svg)
    _save_svg(args.output_dir / "fig_sensitivity_runtime_breakdown.svg", runtime_svg)
    _save_svg(args.output_dir / "fig_sensitivity_formula_validation.svg", validation_svg)
    print(f"output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
