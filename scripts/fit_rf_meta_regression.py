#!/usr/bin/env python3
# Copyright (c) 2026 Dylan Hettinger
r"""Fit Random Forest meta-regression models for sensitivity-study results.

Two models are fit and saved:
  - Quality model: predicts nrmse_relative (improvement over null baseline)
    Trained on ALL jobs (null-screened jobs contribute nrmse_relative=0).
  - Runtime model: predicts log(total_wall_seconds)
    Trained on SUCCESSFUL jobs only (null-screened jobs are a separate regime).

Both models use an 80/20 OOB score for unbiased performance estimation.
Runtime prediction intervals use QRF leaf-based weighting (Meinshausen 2006),
which avoids the polynomial surface pathology of OLS meta-regression.

Usage
-----
    pixi run python scripts/fit_rf_meta_regression.py \
        --results artifacts/sensitivity/wave1_results.csv \
        --output-dir artifacts/sensitivity/

Outputs
-------
    wave1_rf_quality.pkl       - sklearn RF quality model (joblib)
    wave1_rf_runtime.pkl       - sklearn RF runtime model (joblib)
    wave1_rf_quality_imp.csv   - feature importances, quality model
    wave1_rf_runtime_imp.csv   - feature importances, runtime model
    wave1_rf_cv_predictions.csv - honest group-blocked CV predictions (quality model)
    wave1_rf_summary.txt       - CV metrics and BSM operating-point predictions
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Canonical feature list (order matters for joblib model reuse).
FEATURES = [
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
    "stages.sparse_selection.lasso_alpha_percentile",
    "variance_threshold",
]

# BSM production operating point (from configs/bsm_config.yml).
BSM_POINT: dict[str, float] = {
    "n_inputs": 135,
    "n_runs": 27000,
    "sparsity": 0.28,
    "interaction_density": 0.15,
    "nonlinearity_strength": 0.20,
    "noise_snr": 22.3,
    "stages.empirical_null_screening.n_permutations": 201,
    "stages.empirical_null_screening.bh_q_threshold": 0.20,
    "stages.interaction_discovery.n_permutations": 21,
    "stages.interaction_discovery.p_threshold": 0.10,
    "stages.sparse_selection.n_stability_subsamples": 25,
    "stages.sparse_selection.lasso_alpha_percentile": 80,
    "variance_threshold": 0.85,
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True, help="Collected results CSV.")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/sensitivity"),
        help="Directory for output files (default: artifacts/sensitivity/).",
    )
    p.add_argument(
        "--n-estimators",
        type=int,
        default=500,
        help="Number of trees in each RF (default: 500).",
    )
    p.add_argument(
        "--cv-folds",
        type=int,
        default=10,
        help="Cross-validation folds (default: 10).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    return p.parse_args()


def _load_data(results_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (all_clean, successful) data frames with model features."""
    df = pd.read_csv(results_path)
    if "nrmse_relative" not in df.columns:
        raise KeyError("Missing nrmse_relative column; run collect_sensitivity_results.py first.")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + ["nrmse_relative"])
    all_clean = df.copy()
    successful = df[df["nrmse_relative"] < 0].copy()
    successful = successful.dropna(subset=["total_wall_seconds"])
    return all_clean, successful


def _qrf_prediction_interval(
    rf: RandomForestRegressor,
    X_train: np.ndarray,
    y_log_train: np.ndarray,
    X_query: np.ndarray,
    quantiles: tuple[float, ...] = (0.10, 0.50, 0.90),
) -> dict[str, np.ndarray]:
    """Quantile Regression Forest leaf-based prediction interval.

    For each query point, computes the weighted empirical CDF of training
    responses, where weights equal the fraction of trees in which each
    training sample shares a leaf with the query point (Meinshausen 2006).
    Returns predictions in original (non-log) units.
    """
    train_leaves = rf.apply(X_train)  # (n_train, n_trees)
    query_leaves = rf.apply(X_query)  # (n_query, n_trees)
    n_trees = train_leaves.shape[1]
    result: dict[str, np.ndarray] = {f"P{int(q * 100)}": np.empty(len(X_query)) for q in quantiles}

    for qi in range(len(X_query)):
        weights = np.zeros(len(X_train))
        for t in range(n_trees):
            weights += (train_leaves[:, t] == query_leaves[qi, t]).astype(float)
        weights /= n_trees

        order = np.argsort(y_log_train)
        y_sorted = y_log_train[order]
        w_sorted = weights[order]
        w_cumsum = np.cumsum(w_sorted) / max(w_sorted.sum(), 1e-12)

        for q in quantiles:
            idx = int(np.searchsorted(w_cumsum, q))
            idx = min(idx, len(y_sorted) - 1)
            result[f"P{int(q * 100)}"][qi] = float(np.exp(y_sorted[idx]))

    return result


def _fit_quality_model(
    all_clean: pd.DataFrame,
    n_estimators: int,
    cv_folds: int,
    seed: int,
) -> tuple[RandomForestRegressor, dict[str, float], np.ndarray]:
    X = all_clean[FEATURES].values
    y = all_clean["nrmse_relative"].values
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_features="sqrt",
        oob_score=True,
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(X, y)

    # Honest group-blocked CV: group by (config_idx, dgp_idx) pair so that
    # replicates of the same configuration cannot appear in both train and test.
    if "config_idx" in all_clean.columns and "dgp_idx" in all_clean.columns:
        pair_key = all_clean["config_idx"].astype(str) + "_" + all_clean["dgp_idx"].astype(str)
        groups = pair_key.astype("category").cat.codes.values
        n_splits = min(cv_folds, int(groups.max()) + 1)
        kf = GroupKFold(n_splits=n_splits)
        split_iter = kf.split(X, y, groups)
    else:
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        split_iter = kf.split(X, y)

    cv_preds = np.zeros_like(y, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for tr, va in split_iter:
            rf_fold = RandomForestRegressor(
                n_estimators=200, max_features="sqrt", random_state=seed, n_jobs=-1
            )
            rf_fold.fit(X[tr], y[tr])
            cv_preds[va] = rf_fold.predict(X[va])

    from sklearn.metrics import r2_score

    cv_r2 = float(r2_score(y, cv_preds))
    cv_rmse = float(np.sqrt(np.mean((y - cv_preds) ** 2)))
    metrics = {
        "oob_r2": float(rf.oob_score_),
        "cv_r2": cv_r2,
        "cv_rmse": cv_rmse,
        "n": len(y),
    }
    return rf, metrics, cv_preds


def _fit_runtime_model(
    successful: pd.DataFrame,
    n_estimators: int,
    cv_folds: int,
    seed: int,
) -> tuple[RandomForestRegressor, dict[str, float]]:
    X = successful[FEATURES].values
    y = np.log(successful["total_wall_seconds"].values)
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_features="sqrt",
        oob_score=True,
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(X, y)

    if "config_idx" in successful.columns and "dgp_idx" in successful.columns:
        pair_key = (
            successful["config_idx"].astype(str) + "_" + successful["dgp_idx"].astype(str)
        )
        groups = pair_key.astype("category").cat.codes.values
        n_splits = min(cv_folds, int(groups.max()) + 1)
        kf = GroupKFold(n_splits=n_splits)
        split_iter = kf.split(X, y, groups)
    else:
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        split_iter = kf.split(X, y)

    cv_preds = np.zeros_like(y, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for tr, va in split_iter:
            rf_fold = RandomForestRegressor(
                n_estimators=200, max_features="sqrt", random_state=seed, n_jobs=-1
            )
            rf_fold.fit(X[tr], y[tr])
            cv_preds[va] = rf_fold.predict(X[va])

    from sklearn.metrics import r2_score

    cv_r2 = float(r2_score(y, cv_preds))
    cv_rmse_log = float(np.sqrt(np.mean((y - cv_preds) ** 2)))
    metrics = {
        "oob_r2": float(rf.oob_score_),
        "cv_r2": cv_r2,
        "cv_rmse_log": cv_rmse_log,
        "cv_multiplicative_uncertainty": float(np.exp(cv_rmse_log)),
        "n": len(y),
    }
    return rf, metrics


def _predict_bsm(
    rf_quality: RandomForestRegressor,
    rf_runtime: RandomForestRegressor,
    successful: pd.DataFrame,
    seed: int = 42,
    bsm_null_nrmse: float = 0.1247,
) -> dict[str, object]:
    bsm_x = np.array([[BSM_POINT[f] for f in FEATURES]])

    # Quality prediction.
    q_pred = float(rf_quality.predict(bsm_x)[0])
    tree_q = np.array([t.predict(bsm_x)[0] for t in rf_quality.estimators_])
    q_nrmse = bsm_null_nrmse * (1 + q_pred)

    # Runtime prediction (point estimate from RF).
    r_log_pred = float(rf_runtime.predict(bsm_x)[0])
    r_pred_sec = float(np.exp(r_log_pred))

    # Runtime prediction interval from QRF leaf-based weighting.
    X_succ = successful[FEATURES].values
    y_log_succ = np.log(successful["total_wall_seconds"].values)
    qrf_preds = _qrf_prediction_interval(rf_runtime, X_succ, y_log_succ, bsm_x)

    return {
        "quality_nrmse_relative": q_pred,
        "quality_nrmse_relative_tree_std": float(tree_q.std()),
        "quality_nRMSE": q_nrmse,
        "quality_nRMSE_P10": bsm_null_nrmse * (1 + float(np.percentile(tree_q, 10))),
        "quality_nRMSE_P90": bsm_null_nrmse * (1 + float(np.percentile(tree_q, 90))),
        "runtime_P50_sec": qrf_preds["P50"][0],
        "runtime_P10_sec": qrf_preds["P10"][0],
        "runtime_P90_sec": qrf_preds["P90"][0],
        "runtime_RF_point_sec": r_pred_sec,
    }


def main() -> None:  # noqa: D103
    args = _parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data ...")
    all_clean, successful = _load_data(args.results)
    print(f"  All jobs: {len(all_clean)}, Successful (non-null-screened): {len(successful)}")

    print("\nFitting quality model ...")
    rf_q, q_metrics, q_cv_preds = _fit_quality_model(
        all_clean, args.n_estimators, args.cv_folds, args.seed
    )
    print(
        f"  OOB R²={q_metrics['oob_r2']:.4f} (biased; group-blocked CV below), "
        f"CV R²={q_metrics['cv_r2']:.4f} (group-blocked), "
        f"CV RMSE={q_metrics['cv_rmse']:.4f}"
    )

    print("\nFitting runtime model ...")
    rf_r, r_metrics = _fit_runtime_model(successful, args.n_estimators, args.cv_folds, args.seed)
    print(
        f"  OOB R²={r_metrics['oob_r2']:.4f}, CV R²={r_metrics['cv_r2']:.4f}, "
        f"CV RMSE (log-scale)={r_metrics['cv_rmse_log']:.4f} "
        f"(×{r_metrics['cv_multiplicative_uncertainty']:.2f} uncertainty)"
    )

    print("\nBSM operating-point prediction ...")
    bsm_pred = _predict_bsm(rf_q, rf_r, successful, seed=args.seed)
    print(
        f"  Quality nRMSE: {bsm_pred['quality_nRMSE']:.4f} "
        f"(P10={bsm_pred['quality_nRMSE_P10']:.4f}, P90={bsm_pred['quality_nRMSE_P90']:.4f})"
    )
    print("  Actual BSM nRMSE: 0.0721")
    print(
        f"  Runtime QRF P10={bsm_pred['runtime_P10_sec'] / 60:.0f}, "
        f"P50={bsm_pred['runtime_P50_sec'] / 60:.0f}, "
        f"P90={bsm_pred['runtime_P90_sec'] / 60:.0f} min (single-core)"
    )
    print(f"  Runtime RF point estimate: {bsm_pred['runtime_RF_point_sec'] / 60:.0f} min")

    # Feature importances.
    fi_q = pd.Series(rf_q.feature_importances_, index=FEATURES, name="importance").sort_values(
        ascending=False
    )
    fi_r = pd.Series(rf_r.feature_importances_, index=FEATURES, name="importance").sort_values(
        ascending=False
    )

    # Save artifacts.
    joblib.dump(rf_q, out_dir / "wave1_rf_quality.pkl")
    joblib.dump(rf_r, out_dir / "wave1_rf_runtime.pkl")
    fi_q.to_csv(out_dir / "wave1_rf_quality_imp.csv")
    fi_r.to_csv(out_dir / "wave1_rf_runtime_imp.csv")

    # Save honest CV predictions for use by plotting scripts.
    cv_df = all_clean[["nrmse_relative"]].copy()
    cv_df["cv_predicted"] = q_cv_preds
    cv_df.to_csv(out_dir / "wave1_rf_cv_predictions.csv", index=False)

    # Write summary.
    summary_lines = [
        "=== Random Forest Meta-Regression Summary ===",
        "",
        "Quality model (target: nrmse_relative, all jobs including null-screened):",
        f"  n_jobs             = {q_metrics['n']}",
        f"  OOB R² (biased)    = {q_metrics['oob_r2']:.4f}",
        f"  CV R² (group-blocked, config×DGP pairs) = {q_metrics['cv_r2']:.4f}",
        f"  CV RMSE            = {q_metrics['cv_rmse']:.4f}",
        "  NOTE: OOB is inflated by replicate leakage; use group-blocked CV R².",
        "",
        "Runtime model (target: log(total_wall_seconds), successful jobs only):",
        f"  n_jobs             = {r_metrics['n']}",
        f"  OOB R² (biased)    = {r_metrics['oob_r2']:.4f}",
        f"  CV R² (group-blocked) = {r_metrics['cv_r2']:.4f}",
        f"  CV RMSE (log)      = {r_metrics['cv_rmse_log']:.4f}",
        f"  Mult. uncert.      = ×{r_metrics['cv_multiplicative_uncertainty']:.2f}",
        "",
        "BSM operating-point predictions:",
        f"  nRMSE predicted  = {bsm_pred['quality_nRMSE']:.4f} "
        f"(P10={bsm_pred['quality_nRMSE_P10']:.4f}, P90={bsm_pred['quality_nRMSE_P90']:.4f})",
        "  nRMSE actual     = 0.0721",
        f"  nRMSE error      = {abs(bsm_pred['quality_nRMSE'] - 0.0721):.4f} "
        f"({abs(bsm_pred['quality_nRMSE'] - 0.0721) / 0.0721 * 100:.1f}% of actual)",
        f"  Runtime P10/P50/P90 = {bsm_pred['runtime_P10_sec'] / 60:.0f} / "
        f"{bsm_pred['runtime_P50_sec'] / 60:.0f} / "
        f"{bsm_pred['runtime_P90_sec'] / 60:.0f} min (single-core, QRF)",
        f"  Runtime RF point    = {bsm_pred['runtime_RF_point_sec'] / 60:.0f} min",
        "",
        "Top-5 quality drivers (RF importance):",
    ]
    for feat, imp in fi_q.head(5).items():
        summary_lines.append(f"  {feat:<55s} {imp:.4f}")
    summary_lines += ["", "Top-5 runtime drivers (RF importance):"]
    for feat, imp in fi_r.head(5).items():
        summary_lines.append(f"  {feat:<55s} {imp:.4f}")

    summary_path = out_dir / "wave1_rf_summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n")
    print(f"\nOutputs written to {out_dir}/")
    print("  wave1_rf_quality.pkl, wave1_rf_runtime.pkl")
    print("  wave1_rf_quality_imp.csv, wave1_rf_runtime_imp.csv")
    print("  wave1_rf_summary.txt")


if __name__ == "__main__":
    main()
