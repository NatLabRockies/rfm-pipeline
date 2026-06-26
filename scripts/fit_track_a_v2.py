"""Wave5 Track A v2: improved measurement-based meta-model fit.

Three improvements over fit_wave5_track_a.py (v1):

  Improvement 1 — Larger proxy output cap.
      v1 capped n_outputs at 1000 when regenerating DGPs for measurement.
      BSM has 9,954 outputs; the output spectrum top-1 share and decay
      exponent depend critically on having enough outputs. Proxy-capped
      measurements severely underestimate BSM's rank-1 structure.
      v2 defaults to proxy_n_outputs_cap=9000 (HPC has 220 GB RAM).

  Improvement 2 — Hybrid oracle Ridge as primary model.
      v1 fitted a direct RF and GP-ARD to predict nrmse_final.
      v2 makes the physics-constrained hybrid (η × γ_oracle, Ridge)
      the primary model — the same transformation that took Track B
      from 26 % to 2.7 % BSM error. RF is kept as a comparison baseline.

  Improvement 3 — Optimized GP-ARD.
      v1 used optimizer=None (fixed kernel, no hyperparameter learning).
      v2 uses optimizer='fmin_l_bfgs_b' so GP-ARD can learn which
      measurement features drive efficiency.  With 240 rows × 24 features
      on a compute node this is feasible in < 2 h.

Run on a Kestrel compute node only — regenerating 80 DGPs at n_outputs up
to 9000 + GP optimization is OOM / too slow for a login node or laptop.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from rfm_pipeline.wave5_track_a import (
    aggregate_wave5_replicates,
    build_wave5_measurement_frame,
    select_model_feature_columns,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = ROOT / "artifacts" / "sensitivity" / "wave5_results.csv"
DEFAULT_OUT = ROOT / "artifacts" / "sensitivity" / "wave5_measurement_models"

# BSM production operating-point feature vector (for manuscript validation)
BSM_OPS = {
    "n_inputs": 135,
    "n_runs": 28750,
    "n_outputs": 9954,
    "sparsity": 0.28,
    "interaction_density": 0.15,
    "nonlinearity_strength": 0.20,
    "noise_snr": 22.3,
    "holdout_fraction": 0.05,
    "variance_threshold": 0.90,
    "stages.empirical_null_screening.n_permutations": 201,
    "stages.empirical_null_screening.bh_q_threshold": 0.05,
    "stages.interaction_discovery.n_permutations": 31,
    "stages.interaction_discovery.p_threshold": 0.05,
    "stages.sparse_selection.n_stability_subsamples": 50,
    "stages.final_artifacts.delta_threshold_override": 0.002,
    # measurement-based features are NOT available for BSM at prediction time
    # — they are predicted from the dataset structure at inference.
}
BSM_NRMSE_OBS = 0.0721
BSM_GAMMA_OBS = -0.564
BSM_SNR = 22.3


def gamma_oracle(snr: np.ndarray) -> np.ndarray:
    """Closed-form Gaussian noise floor: sqrt(1/(snr+1)) - 1."""
    return np.sqrt(1.0 / (snr + 1.0)) - 1.0


def _metrics(pred: np.ndarray, obs: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(obs, pred)),
        "rmse": float(np.sqrt(mean_squared_error(obs, pred))),
        "mae": float(mean_absolute_error(obs, pred)),
    }


def _group_cv_predict(
    estimator_factory,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 8,
) -> np.ndarray:
    predictions = np.full_like(y, fill_value=np.nan, dtype=float)
    n_splits = min(n_splits, len(np.unique(groups)))
    splitter = GroupKFold(n_splits=n_splits)
    for train_idx, test_idx in splitter.split(X, y, groups):
        est = estimator_factory()
        est.fit(X[train_idx], y[train_idx])
        predictions[test_idx] = est.predict(X[test_idx])
    return predictions


def make_hybrid_ridge() -> Pipeline:
    """Improvement 2: hybrid oracle Ridge (predict η, not γ directly)."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 25))),
        ]
    )


def make_rf() -> RandomForestRegressor:
    """RF baseline for comparison with hybrid Ridge."""
    return RandomForestRegressor(n_estimators=500, min_samples_leaf=3, random_state=0, n_jobs=-1)


def make_gp_ard(n_features: int) -> Pipeline:
    """Improvement 3: optimized GP-ARD (optimizer enabled, not None)."""
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=np.ones(n_features),
        length_scale_bounds=(1e-2, 1e2),
        nu=1.5,
    ) + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 1.0))
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "gp",
                GaussianProcessRegressor(
                    kernel=kernel,
                    normalize_y=True,
                    # Improvement 3: enable kernel hyperparameter optimization
                    optimizer="fmin_l_bfgs_b",
                    n_restarts_optimizer=3,
                    random_state=0,
                ),
            ),
        ]
    )


def main() -> None:
    """Fit Track A v2 models on wave5 measurements with 3 improvements."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_IN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    # Improvement 1: larger default proxy caps; override via CLI for ablations
    parser.add_argument("--proxy-n-runs-cap", type=int, default=10_000)
    parser.add_argument("--proxy-n-outputs-cap", type=int, default=9_000)
    parser.add_argument("--max-outputs-for-pca", type=int, default=9_000)
    parser.add_argument("--max-inputs-for-corr", type=int, default=500)
    parser.add_argument("--max-rows-for-corr", type=int, default=10_000)
    parser.add_argument("--skip-gp", action="store_true", help="Skip GP-ARD (faster ablation)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[v2] Loading wave5 results from {args.input}")
    results = pd.read_csv(args.input)
    print(f"[v2] {len(results)} rows, {results['dgp_idx'].nunique()} unique DGPs")

    print(
        f"[v2] Building measurement frame: "
        f"proxy_runs={args.proxy_n_runs_cap}, proxy_outputs={args.proxy_n_outputs_cap}"
    )
    measured = build_wave5_measurement_frame(
        results,
        proxy_n_runs_cap=args.proxy_n_runs_cap,
        proxy_n_outputs_cap=args.proxy_n_outputs_cap,
        max_outputs_for_pca=args.max_outputs_for_pca,
        max_inputs_for_corr=args.max_inputs_for_corr,
        max_rows_for_corr=args.max_rows_for_corr,
    )
    aggregated = aggregate_wave5_replicates(measured)
    print(f"[v2] Aggregated to {len(aggregated)} rows ({aggregated['dgp_idx'].nunique()} DGPs)")

    feature_cols = [
        c for c in select_model_feature_columns(aggregated.columns) if c in aggregated.columns
    ]
    print(f"[v2] {len(feature_cols)} feature columns: {feature_cols}")

    groups = aggregated["dgp_idx"].to_numpy()
    X = aggregated[feature_cols].to_numpy(dtype=float)
    snr = aggregated["noise_snr"].to_numpy(dtype=float)
    g_oracle = gamma_oracle(snr)
    y_gamma = aggregated["nrmse_relative"].to_numpy(dtype=float)
    y_nrmse = aggregated["nrmse_final"].to_numpy(dtype=float)
    y_eta = np.where(np.abs(g_oracle) > 1e-9, np.clip(y_gamma / g_oracle, -3.0, 5.0), np.nan)
    y_runtime = np.log(aggregated["pipeline_seconds"].to_numpy(dtype=float))

    outputs: dict[str, dict] = {}

    # --- Improvement 2: Hybrid oracle Ridge as primary nRMSE model ---
    print("[v2] Fitting hybrid oracle Ridge (Improvement 2)...")
    valid = np.isfinite(y_eta)
    if valid.sum() > 0:
        eta_cv = _group_cv_predict(make_hybrid_ridge, X[valid], y_eta[valid], groups[valid])
        gamma_cv = eta_cv * g_oracle[valid]
        nrmse_cv = aggregated["nrmse_null"].to_numpy(dtype=float)[valid] * (1 + gamma_cv)
        outputs["hybrid_ridge_eta_cv"] = _metrics(eta_cv, y_eta[valid])
        outputs["hybrid_ridge_nrmse_cv"] = _metrics(nrmse_cv, y_nrmse[valid])
        # Refit on all data for BSM prediction
        hybrid_ridge = make_hybrid_ridge()
        hybrid_ridge.fit(X[valid], y_eta[valid])
        (args.output_dir / "hybrid_ridge_eta.pkl").write_bytes(pickle.dumps(hybrid_ridge))
        print(
            f"[v2]   hybrid_ridge_eta CV R²(η)={outputs['hybrid_ridge_eta_cv']['r2']:.3f}, "
            f"CV R²(nRMSE)={outputs['hybrid_ridge_nrmse_cv']['r2']:.3f}"
        )
    else:
        print("[v2]   WARNING: no valid eta rows")

    # --- RF comparison baseline ---
    print("[v2] Fitting RF baseline...")
    rf_eta_cv = _group_cv_predict(make_rf, X[valid], y_eta[valid], groups[valid])
    rf_gamma_cv = rf_eta_cv * g_oracle[valid]
    rf_nrmse_cv = aggregated["nrmse_null"].to_numpy(dtype=float)[valid] * (1 + rf_gamma_cv)
    outputs["rf_eta_cv"] = _metrics(rf_eta_cv, y_eta[valid])
    outputs["rf_nrmse_cv"] = _metrics(rf_nrmse_cv, y_nrmse[valid])
    rf_model = make_rf()
    rf_model.fit(X[valid], y_eta[valid])
    (args.output_dir / "rf_eta.pkl").write_bytes(pickle.dumps(rf_model))
    print(
        f"[v2]   rf_eta CV R²(η)={outputs['rf_eta_cv']['r2']:.3f}, "
        f"CV R²(nRMSE)={outputs['rf_nrmse_cv']['r2']:.3f}"
    )

    # --- Improvement 3: Optimized GP-ARD ---
    if not args.skip_gp:
        print("[v2] Fitting GP-ARD with kernel optimization (Improvement 3)...")
        gp_eta_cv = _group_cv_predict(
            lambda: make_gp_ard(X.shape[1]), X[valid], y_eta[valid], groups[valid]
        )
        gp_nrmse_cv = aggregated["nrmse_null"].to_numpy(dtype=float)[valid] * (
            1 + gp_eta_cv * g_oracle[valid]
        )
        outputs["gp_ard_eta_cv"] = _metrics(gp_eta_cv, y_eta[valid])
        outputs["gp_ard_nrmse_cv"] = _metrics(gp_nrmse_cv, y_nrmse[valid])
        gp_model = make_gp_ard(X.shape[1])
        gp_model.fit(X[valid], y_eta[valid])
        (args.output_dir / "gp_ard_eta.pkl").write_bytes(pickle.dumps(gp_model))
        print(
            f"[v2]   gp_ard_eta CV R²(η)={outputs['gp_ard_eta_cv']['r2']:.3f}, "
            f"CV R²(nRMSE)={outputs['gp_ard_nrmse_cv']['r2']:.3f}"
        )
        # Extract ARD length scales for manuscript Figure
        gp_kern = gp_model.named_steps["gp"].kernel_
        try:
            ls = gp_kern.k1.k2.length_scale
            ard_output = dict(zip(feature_cols, ls.tolist(), strict=True))
            (args.output_dir / "gp_ard_length_scales.json").write_text(
                json.dumps({"length_scales": ard_output}, indent=2) + "\n", encoding="utf-8"
            )
            print("[v2]   ARD length scales saved to gp_ard_length_scales.json")
        except Exception as exc:
            print(f"[v2]   Could not extract ARD length scales: {exc}")

    # --- Runtime: hybrid analytic + Ridge ---
    print("[v2] Fitting runtime model (analytic baseline + Ridge)...")
    knob_frame = results.drop_duplicates(["dgp_idx", "config_idx"]).copy()
    n_runs = knob_frame["n_runs"].to_numpy(dtype=float)
    n_inputs = knob_frame["n_inputs"].to_numpy(dtype=float)
    _empty = pd.Series(dtype=float)
    p_screen = knob_frame.get("stages.empirical_null_screening.n_permutations", _empty).to_numpy(
        dtype=float
    )
    p_int = knob_frame.get("stages.interaction_discovery.n_permutations", _empty).to_numpy(
        dtype=float
    )
    n_stab = knob_frame.get("stages.sparse_selection.n_stability_subsamples", _empty).to_numpy(
        dtype=float
    )
    lasso = knob_frame.get("stages.sparse_selection.lasso_alpha_grid_size", _empty).to_numpy(
        dtype=float
    )
    runtime_oracle = (
        p_screen * n_inputs * n_runs
        + p_int * n_inputs**2 * n_runs
        + n_stab * n_inputs * n_runs * lasso
        + n_inputs * n_runs
    )
    valid_rt = np.isfinite(y_runtime) & (runtime_oracle > 0) & np.isfinite(runtime_oracle)
    if valid_rt.sum() > 10:
        log_resid = y_runtime[valid_rt] - np.log(runtime_oracle[valid_rt])
        rt_cv = _group_cv_predict(make_hybrid_ridge, X[valid_rt], log_resid, groups[valid_rt])
        bt = float(np.exp(0.5 * np.var(log_resid - rt_cv)))
        pred_sec = np.exp(rt_cv + np.log(runtime_oracle[valid_rt])) * bt
        y_sec = np.exp(y_runtime[valid_rt])
        outputs["runtime_hybrid_ridge_cv"] = _metrics(pred_sec, y_sec)
        outputs["runtime_back_transform_factor"] = bt
        rt_model = make_hybrid_ridge()
        rt_model.fit(X[valid_rt], log_resid)
        (args.output_dir / "runtime_hybrid_ridge.pkl").write_bytes(pickle.dumps(rt_model))
        (args.output_dir / "runtime_hybrid_ridge_meta.json").write_text(
            json.dumps({"back_transform_factor": bt, "features": feature_cols}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"[v2]   runtime_hybrid_ridge CV R²(sec)={outputs['runtime_hybrid_ridge_cv']['r2']:.3f}"
        )

    # --- BSM prediction using measurement features ---
    # For BSM, we can only use DGP-knob features (no measurements available without running BSM)
    bsm_knob_cols = [
        "n_inputs",
        "n_runs",
        "n_outputs",
        "sparsity",
        "noise_snr",
        "holdout_fraction",
        "variance_threshold",
        "stages.empirical_null_screening.n_permutations",
        "stages.empirical_null_screening.bh_q_threshold",
        "stages.interaction_discovery.n_permutations",
        "stages.interaction_discovery.p_threshold",
        "stages.sparse_selection.n_stability_subsamples",
        "stages.final_artifacts.delta_threshold_override",
    ]
    bsm_knob_cols_in = [c for c in bsm_knob_cols if c in feature_cols]
    if valid.sum() > 0 and bsm_knob_cols_in:
        bsm_X_knob = np.array([[BSM_OPS.get(c, np.nan) for c in feature_cols]])
        # Predict with hybrid Ridge using only knob features (measurement columns → use their
        # training-set mean as fill — conservative, not ideal, but shows direction)
        meas_cols = [c for c in feature_cols if c not in bsm_knob_cols]
        if meas_cols:
            meas_means = aggregated[feature_cols].mean()
            for mc in meas_cols:
                bsm_X_knob[0, feature_cols.index(mc)] = meas_means[mc]
        bsm_oracle = float(gamma_oracle(np.array([BSM_SNR]))[0])
        bsm_eta_pred = float(hybrid_ridge.predict(bsm_X_knob)[0])
        bsm_nrmse_pred = float(0.165 * (1 + np.clip(bsm_eta_pred, -3, 1) * bsm_oracle))
        bsm_pct_err = 100.0 * (bsm_nrmse_pred - BSM_NRMSE_OBS) / BSM_NRMSE_OBS
        outputs["bsm_prediction"] = {
            "eta_pred": bsm_eta_pred,
            "nrmse_pred": bsm_nrmse_pred,
            "nrmse_obs": BSM_NRMSE_OBS,
            "pct_err": bsm_pct_err,
            "note": "measurement columns filled with training-set mean (conservative)",
        }
        print(
            f"[v2] BSM prediction: nRMSE={bsm_nrmse_pred:.4f} ({bsm_pct_err:+.1f}% vs obs 0.0721)"
        )

    # --- Predictions CSV ---
    pred_rows = aggregated[
        ["dgp_idx", "config_idx", "nrmse_final", "nrmse_null", "noise_snr"]
    ].copy()
    pred_rows["nrmse_obs"] = y_nrmse
    if valid.sum() > 0:
        full_hybrid = hybrid_ridge.predict(X)
        pred_rows["eta_pred_hybrid_ridge"] = full_hybrid
        pred_rows["nrmse_pred_hybrid_ridge"] = pred_rows["nrmse_null"] * (
            1 + np.clip(full_hybrid * g_oracle, -1, 0)
        )
        pred_rows["eta_pred_rf"] = rf_model.predict(X)
    pred_rows.to_csv(args.output_dir / "wave5_track_a_v2_predictions.csv", index=False)

    # --- Summary JSON ---
    summary = {
        "version": "v2",
        "improvements": [
            "1: proxy_n_outputs_cap raised to 9000 (BSM has 9954 outputs)",
            "2: hybrid oracle Ridge as primary model (predict eta, not nrmse)",
            "3: GP-ARD with optimizer enabled (learns measurement feature relevance)",
        ],
        "proxy_n_outputs_cap": args.proxy_n_outputs_cap,
        "proxy_n_runs_cap": args.proxy_n_runs_cap,
        "feature_cols": feature_cols,
        "row_count": int(len(aggregated)),
        "group_count": int(aggregated["dgp_idx"].nunique()),
        "metrics": outputs,
    }
    (args.output_dir / "wave5_track_a_v2_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print("[v2] Done. Summary written to wave5_track_a_v2_summary.json")
    print(json.dumps({k: v for k, v in outputs.items() if isinstance(v, dict)}, indent=2))


if __name__ == "__main__":
    main()
