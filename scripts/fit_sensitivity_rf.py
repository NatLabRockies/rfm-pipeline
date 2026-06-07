#!/usr/bin/env python3
# Copyright (c) 2026 Dylan Hettinger
"""Fit the RF meta-regression used in §6 / Figure 8.

Emits the top-8 feature importances for the quality (gamma) and runtime
(log wall-seconds) targets. The numbers printed here are pasted into
``scripts/plot_sensitivity_rf_figures.py`` (QUALITY_IMPORTANCE /
RUNTIME_IMPORTANCE) and the manuscript section 6 text.

Methodology mirrors the manuscript figure caption:
  - quality: all rows (null-screened gamma filled to 0), target = nrmse_relative
  - runtime: successful rows only, target = log(total_wall_seconds)
  - RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
  - features = the canonical _RF_FEATURES list from plot_sensitivity_results.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

_RF_FEATURES: list[tuple[str, str]] = [
    ("n_inputs", "Input count (d)"),
    ("n_runs", "Run count (n)"),
    ("sparsity", "Sparsity (s)"),
    ("interaction_density", "Interaction density (\u03c1)"),
    ("nonlinearity_strength", "Nonlinearity strength (\u03ba)"),
    ("noise_snr", "Signal-to-noise ratio (\u03c3)"),
    ("stages.empirical_null_screening.n_permutations", "Screening permutations"),
    ("stages.empirical_null_screening.bh_q_threshold", "BH threshold (q)"),
    ("stages.interaction_discovery.n_permutations", "Interaction permutations"),
    ("stages.interaction_discovery.p_threshold", "Interaction p-threshold"),
    ("stages.sparse_selection.n_stability_subsamples", "Stability subsamples"),
    ("stages.sparse_selection.lasso_alpha_grid_size", "LASSO \u03b1 grid size"),
    ("variance_threshold", "Variance threshold"),
]


def _fit(X: np.ndarray, y: np.ndarray) -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1).fit(X, y)


def main() -> int:
    """Fit RF on cleaned sensitivity CSV and print importance arrays."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--top-n", type=int, default=8)
    args = p.parse_args()

    df = pd.read_csv(args.results)
    cols = [c for c, _ in _RF_FEATURES]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"missing required columns: {missing}")
    labels = dict(_RF_FEATURES)

    # Quality
    y_q = df["nrmse_relative"].fillna(0.0).values
    X_q = df[cols].values
    rf_q = _fit(X_q, y_q)
    imp_q = sorted(zip(cols, rf_q.feature_importances_, strict=True), key=lambda kv: -kv[1])

    # Runtime
    is_succ = df["null_screened"].isna().values
    mask = is_succ & df["total_wall_seconds"].notna().values & (df["total_wall_seconds"] > 0).values
    y_r = np.log(df.loc[mask, "total_wall_seconds"].values)
    X_r = df.loc[mask, cols].values
    rf_r = _fit(X_r, y_r)
    imp_r = sorted(zip(cols, rf_r.feature_importances_, strict=True), key=lambda kv: -kv[1])

    print(f"# n_quality={len(y_q)}  n_runtime={len(y_r)}")
    print("QUALITY_IMPORTANCE = [")
    for c, v in imp_q[: args.top_n]:
        print(f'    ("{labels[c]}", {v:.3f}),')
    print("]\n")
    print("RUNTIME_IMPORTANCE = [")
    for c, v in imp_r[: args.top_n]:
        print(f'    ("{labels[c]}", {v:.3f}),')
    print("]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
