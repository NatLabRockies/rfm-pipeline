#!/usr/bin/env python3
# Copyright (c) 2026 Dylan Hettinger
"""Wave123 RF validation against the BSM production operating point.

Computes the four numbers cited in `docs/manuscripts/full_dataset_run_revision_notes.md`
§1.3 "RF runtime model — group-blocked CV R² and back-transform" and "BSM
production operating-point predictions from wave123 RF":

  [1] Runtime RF, 10-fold group-blocked CV R² on log wall-seconds
      (groups = config_idx × dgp_idx).
  [2] Back-transform factor exp(σ²/2) from group-CV log-residuals.
  [3] BSM runtime point + 80% per-tree PI (back-transform applied to point;
      quantile PI reported in the un-back-transformed space because
      quantiles commute with the strictly-monotone exp(·) transform).
  [4] BSM nRMSE point + 80% per-tree PI via γ = (nRMSE − ν)/ν with
      ν = null nRMSE for the BSM operating point (manuscript §7.5).

Inputs (all in artifacts/sensitivity/, relative to repo root):
  wave123_combined_clean.csv
  wave123_rf_quality.pkl
  wave123_rf_runtime.pkl

Re-run via:
    pixi run python scripts/compute_bsm_rf_validation.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/sensitivity"

_FEATS = [
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

# BSM production operating point (user-supplied, with variance_threshold=0.9
# read from the wave-1 bsm_structure config row in the clean CSV).
BSM_FEATS = {
    "n_inputs": 135,
    "n_runs": 28750,
    "sparsity": 0.28,
    "interaction_density": 0.15,
    "nonlinearity_strength": 0.20,
    "noise_snr": 22.3,
    "stages.empirical_null_screening.n_permutations": 201,
    "stages.empirical_null_screening.bh_q_threshold": 0.05,
    "stages.interaction_discovery.n_permutations": 31,
    "stages.interaction_discovery.p_threshold": 0.05,
    "stages.sparse_selection.n_stability_subsamples": 50,
    "stages.sparse_selection.lasso_alpha_grid_size": 40,
    "variance_threshold": 0.9,
}


def per_tree_predictions(rf: RandomForestRegressor, X: np.ndarray) -> np.ndarray:
    """Return shape (n_trees, n_samples) tree-by-tree predictions."""
    return np.stack([t.predict(X) for t in rf.estimators_], axis=0)


def main() -> int:
    """Run the wave-N RF BSM-validation pipeline and print all four numbers."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--prefix",
        default="wave123",
        help="Filename prefix for the cleaned CSV and the two RF pickles "
        "(default: wave123 -> wave123_combined_clean.csv + "
        "wave123_rf_quality.pkl + wave123_rf_runtime.pkl).",
    )
    args = p.parse_args()
    csv = ART / f"{args.prefix}_combined_clean.csv"
    rf_q_path = ART / f"{args.prefix}_rf_quality.pkl"
    rf_r_path = ART / f"{args.prefix}_rf_runtime.pkl"
    print(f"# prefix={args.prefix}  csv={csv.name}")
    df = pd.read_csv(csv)
    rf_q: RandomForestRegressor = joblib.load(rf_q_path)
    rf_r: RandomForestRegressor = joblib.load(rf_r_path)

    print(f"# rows={len(df)}  features={_FEATS}\n")

    # ── 1. Runtime group-blocked CV R² (groups = config_idx × dgp_idx) ─────
    is_succ = df["null_screened"].isna().values
    mask = is_succ & df["total_wall_seconds"].notna().values & (df["total_wall_seconds"] > 0).values
    sub = df.loc[mask].copy()
    y = np.log(sub["total_wall_seconds"].values)
    X = sub[_FEATS].values
    groups = (sub["config_idx"].astype(str) + "_" + sub["dgp_idx"].astype(str)).values

    gkf = GroupKFold(n_splits=10)
    preds = np.empty_like(y)
    for tr, te in gkf.split(X, y, groups):
        rf_cv = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
        rf_cv.fit(X[tr], y[tr])
        preds[te] = rf_cv.predict(X[te])
    ss_res = float(np.sum((y - preds) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2_run_gcv = 1.0 - ss_res / ss_tot
    print(f"[1] Runtime group-blocked CV R² (n={len(y)}, 10 folds): {r2_run_gcv:.4f}")

    # ── 2. Back-transform factor exp(σ²/2) from runtime RF log-residuals ───
    # In-sample residuals collapse for RFs (each leaf memorises its training
    # rows). Use the group-blocked CV out-of-sample residuals from [1].
    resid = y - preds
    sigma2 = float(np.var(resid, ddof=1))
    back = float(np.exp(sigma2 / 2.0))
    print(
        f"[2] Runtime back-transform (group-CV residuals): "
        f"σ²(log resid)={sigma2:.5f}  exp(σ²/2)={back:.4f}"
    )

    # ── 3. BSM runtime prediction + 80% leaf-weighted PI ───────────────────
    x_bsm = np.array([[BSM_FEATS[c] for c in _FEATS]], dtype=float)
    tree_logs = per_tree_predictions(rf_r, x_bsm)[:, 0]  # (n_trees,)
    tree_runs = np.exp(tree_logs)  # minutes? need /60
    point_log = float(rf_r.predict(x_bsm)[0])
    point_sec = float(np.exp(point_log))
    point_sec_bt = point_sec * back
    p10_sec = float(np.quantile(tree_runs, 0.10))
    p90_sec = float(np.quantile(tree_runs, 0.90))
    print(
        f"[3] BSM runtime: point={point_sec / 60:.2f} min "
        f"(back-transformed {point_sec_bt / 60:.2f} min); "
        f"80% PI [{p10_sec / 60:.2f}, {p90_sec / 60:.2f}] min"
    )
    print(f"    raw seconds: point={point_sec:.1f}s  PI [{p10_sec:.1f}, {p90_sec:.1f}]s")
    measured_mean_min = 101.4
    err_pct = (point_sec / 60 - measured_mean_min) / measured_mean_min * 100
    err_pct_bt = (point_sec_bt / 60 - measured_mean_min) / measured_mean_min * 100
    print(
        f"    vs measured analog mean 101.4 min: err={err_pct:+.2f}% "
        f"(back-transformed {err_pct_bt:+.2f}%)"
    )

    # ── 4. BSM quality (nRMSE) prediction + 80% leaf-weighted PI ───────────
    # Quality target was nrmse_relative (γ); convert γ_pred → nRMSE using
    # null_nRMSE for the BSM operating point. From manuscript: BSM uses
    # null nRMSE = 0.1653 and actual nRMSE = 0.0721.
    null_nrmse_bsm = 0.1653
    actual_nrmse_bsm = 0.0721
    tree_gamma = per_tree_predictions(rf_q, x_bsm)[:, 0]
    gamma_point = float(rf_q.predict(x_bsm)[0])
    gamma_p10 = float(np.quantile(tree_gamma, 0.10))
    gamma_p90 = float(np.quantile(tree_gamma, 0.90))

    # γ = (nrmse - nrmse_null) / nrmse_null  ⇒ nrmse = nrmse_null * (1 + γ)
    nrmse_point = null_nrmse_bsm * (1.0 + gamma_point)
    nrmse_p10 = null_nrmse_bsm * (1.0 + gamma_p10)
    nrmse_p90 = null_nrmse_bsm * (1.0 + gamma_p90)
    err_q = (nrmse_point - actual_nrmse_bsm) / actual_nrmse_bsm * 100
    print(
        f"[4] BSM quality: γ point={gamma_point:.4f}  γ 80% PI [{gamma_p10:.4f}, {gamma_p90:.4f}]"
    )
    print(
        f"    nRMSE point={nrmse_point:.4f}  "
        f"nRMSE 80% PI [{nrmse_p10:.4f}, {nrmse_p90:.4f}]  "
        f"actual={actual_nrmse_bsm:.4f}  err={err_q:+.2f}%"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
