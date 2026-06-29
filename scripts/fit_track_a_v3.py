"""Wave5/6 Track A v3: BSM-transfer-focused measurement meta-model fit."""

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

from rfm_pipeline.track_a_v3 import (
    build_runtime_model_frame,
    compute_near_bsm_sample_weights,
    estimate_bsm_measurements_knn,
)
from rfm_pipeline.wave5_track_a import (
    aggregate_wave5_replicates,
    build_wave5_measurement_frame,
    select_model_feature_columns,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = ROOT / "artifacts" / "sensitivity" / "wave5_results.csv"
DEFAULT_OUT = ROOT / "artifacts" / "sensitivity" / "wave5_measurement_models_v3"

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
    "stages.sparse_selection.lasso_alpha_grid_size": 40,
    "stages.final_artifacts.delta_threshold_override": 0.002,
}
BSM_NRMSE_NULL = 0.165
BSM_NRMSE_OBS = 0.0721
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
    sample_weight: np.ndarray | None = None,
    n_splits: int = 8,
) -> np.ndarray:
    predictions = np.full_like(y, fill_value=np.nan, dtype=float)
    n_splits = min(n_splits, len(np.unique(groups)))
    splitter = GroupKFold(n_splits=n_splits)
    for train_idx, test_idx in splitter.split(X, y, groups):
        est = estimator_factory()
        fit_kwargs: dict[str, np.ndarray] = {}
        if sample_weight is not None:
            w_train = sample_weight[train_idx]
            if isinstance(est, Pipeline):
                fit_kwargs["ridge__sample_weight"] = w_train
            else:
                fit_kwargs["sample_weight"] = w_train
        est.fit(X[train_idx], y[train_idx], **fit_kwargs)
        predictions[test_idx] = est.predict(X[test_idx])
    return predictions


def make_hybrid_ridge() -> Pipeline:
    """Hybrid eta-model used for primary quality prediction."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 25))),
        ]
    )


def make_rf() -> RandomForestRegressor:
    """Random-forest eta baseline."""
    return RandomForestRegressor(n_estimators=500, min_samples_leaf=3, random_state=0, n_jobs=-1)


def make_gp_ard(n_features: int) -> Pipeline:
    """GP-ARD eta model with optimizer enabled."""
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
                    optimizer="fmin_l_bfgs_b",
                    n_restarts_optimizer=3,
                    random_state=0,
                ),
            ),
        ]
    )


def _load_results(path_wave5: Path, path_wave6: Path | None) -> tuple[pd.DataFrame, dict[str, int]]:
    wave5 = pd.read_csv(path_wave5)
    counts = {"wave5_rows": int(len(wave5)), "wave6_rows": 0}
    if path_wave6 is None:
        return wave5, counts
    wave6 = pd.read_csv(path_wave6)
    counts["wave6_rows"] = int(len(wave6))
    combined = pd.concat([wave5, wave6], ignore_index=True)
    return combined, counts


def main() -> None:
    """Fit Track A v3 models and write prediction artifacts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_IN)
    parser.add_argument("--input-wave6", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--proxy-n-runs-cap", type=int, default=12_000)
    parser.add_argument("--proxy-n-outputs-cap", type=int, default=9_954)
    parser.add_argument("--max-outputs-for-pca", type=int, default=9_954)
    parser.add_argument("--max-inputs-for-corr", type=int, default=600)
    parser.add_argument("--max-rows-for-corr", type=int, default=12_000)
    parser.add_argument("--knn-measurement-neighbors", type=int, default=8)
    parser.add_argument("--near-bsm-weight-max", type=float, default=2.5)
    parser.add_argument("--near-bsm-distance-bandwidth", type=float, default=2.0)
    parser.add_argument("--skip-gp", action="store_true", help="Skip GP-ARD (faster ablation)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[v3] Loading results from {args.input}")
    results, counts = _load_results(args.input, args.input_wave6)
    print(f"[v3] rows={len(results)}, unique dgps={results['dgp_idx'].nunique()}")

    print(
        f"[v3] Building measurement frame: "
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
    print(f"[v3] Aggregated to {len(aggregated)} rows ({aggregated['dgp_idx'].nunique()} DGPs)")

    feature_cols = [
        c for c in select_model_feature_columns(aggregated.columns) if c in aggregated.columns
    ]
    groups = aggregated["dgp_idx"].to_numpy()
    X = aggregated[feature_cols].to_numpy(dtype=float)
    snr = aggregated["noise_snr"].to_numpy(dtype=float)
    g_oracle = gamma_oracle(snr)
    y_gamma = aggregated["nrmse_relative"].to_numpy(dtype=float)
    y_nrmse = aggregated["nrmse_final"].to_numpy(dtype=float)
    y_eta = np.where(np.abs(g_oracle) > 1e-9, np.clip(y_gamma / g_oracle, -3.0, 5.0), np.nan)

    sample_w = compute_near_bsm_sample_weights(
        frame=aggregated,
        feature_cols=feature_cols,
        bsm_ops=BSM_OPS,
        weight_max=args.near_bsm_weight_max,
        distance_bandwidth=args.near_bsm_distance_bandwidth,
    )

    outputs: dict[str, dict | float | int] = {}
    valid = np.isfinite(y_eta)
    if valid.sum() == 0:
        raise ValueError("No valid eta rows for Track A v3.")

    print("[v3] Fitting hybrid oracle Ridge with near-BSM weighting...")
    eta_cv = _group_cv_predict(
        make_hybrid_ridge,
        X[valid],
        y_eta[valid],
        groups[valid],
        sample_weight=sample_w[valid],
    )
    gamma_cv = eta_cv * g_oracle[valid]
    nrmse_cv = aggregated["nrmse_null"].to_numpy(dtype=float)[valid] * (1 + gamma_cv)
    outputs["hybrid_ridge_eta_cv"] = _metrics(eta_cv, y_eta[valid])
    outputs["hybrid_ridge_nrmse_cv"] = _metrics(nrmse_cv, y_nrmse[valid])
    hybrid_ridge = make_hybrid_ridge()
    hybrid_ridge.fit(X[valid], y_eta[valid], ridge__sample_weight=sample_w[valid])
    (args.output_dir / "hybrid_ridge_eta.pkl").write_bytes(pickle.dumps(hybrid_ridge))

    print("[v3] Fitting RF baseline with near-BSM weighting...")
    rf_eta_cv = _group_cv_predict(
        make_rf,
        X[valid],
        y_eta[valid],
        groups[valid],
        sample_weight=sample_w[valid],
    )
    rf_gamma_cv = rf_eta_cv * g_oracle[valid]
    rf_nrmse_cv = aggregated["nrmse_null"].to_numpy(dtype=float)[valid] * (1 + rf_gamma_cv)
    outputs["rf_eta_cv"] = _metrics(rf_eta_cv, y_eta[valid])
    outputs["rf_nrmse_cv"] = _metrics(rf_nrmse_cv, y_nrmse[valid])
    rf_model = make_rf()
    rf_model.fit(X[valid], y_eta[valid], sample_weight=sample_w[valid])
    (args.output_dir / "rf_eta.pkl").write_bytes(pickle.dumps(rf_model))

    if not args.skip_gp:
        print("[v3] Fitting GP-ARD...")
        gp_eta_cv = _group_cv_predict(
            lambda: make_gp_ard(X.shape[1]),
            X[valid],
            y_eta[valid],
            groups[valid],
        )
        gp_nrmse_cv = aggregated["nrmse_null"].to_numpy(dtype=float)[valid] * (
            1 + gp_eta_cv * g_oracle[valid]
        )
        outputs["gp_ard_eta_cv"] = _metrics(gp_eta_cv, y_eta[valid])
        outputs["gp_ard_nrmse_cv"] = _metrics(gp_nrmse_cv, y_nrmse[valid])
        gp_model = make_gp_ard(X.shape[1])
        gp_model.fit(X[valid], y_eta[valid])
        (args.output_dir / "gp_ard_eta.pkl").write_bytes(pickle.dumps(gp_model))
        gp_kern = gp_model.named_steps["gp"].kernel_
        if hasattr(gp_kern, "k1") and hasattr(gp_kern.k1, "k2"):
            ls = gp_kern.k1.k2.length_scale
            ard_output = dict(zip(feature_cols, ls.tolist(), strict=True))
            (args.output_dir / "gp_ard_length_scales.json").write_text(
                json.dumps({"length_scales": ard_output}, indent=2) + "\n", encoding="utf-8"
            )

    print("[v3] Fitting runtime model with stratified large-output complexity term...")
    runtime_source = results.drop_duplicates(["dgp_idx", "config_idx"]).copy()
    runtime_frame = build_runtime_model_frame(runtime_source)
    runtime_join = aggregated[["dgp_idx", "config_idx", "pipeline_seconds"]].merge(
        runtime_frame,
        on=["dgp_idx", "config_idx"],
        how="left",
        validate="one_to_one",
    )
    y_runtime = np.log(runtime_join["pipeline_seconds"].to_numpy(dtype=float))
    runtime_oracle = runtime_join["runtime_oracle"].to_numpy(dtype=float)
    runtime_extra_cols = [
        "runtime_screen_term",
        "runtime_interaction_term",
        "runtime_sparse_term",
        "runtime_base_term",
        "runtime_output_term",
        "runtime_large_output_term",
    ]
    runtime_extra = np.log1p(runtime_join[runtime_extra_cols].to_numpy(dtype=float))
    runtime_X = np.hstack([X, runtime_extra])
    valid_rt = np.isfinite(y_runtime) & np.isfinite(runtime_oracle) & (runtime_oracle > 0)
    if valid_rt.sum() > 10:
        log_resid = y_runtime[valid_rt] - np.log(runtime_oracle[valid_rt])
        rt_cv = _group_cv_predict(
            make_hybrid_ridge,
            runtime_X[valid_rt],
            log_resid,
            groups[valid_rt],
            sample_weight=sample_w[valid_rt],
        )
        bt = float(np.exp(0.5 * np.var(log_resid - rt_cv)))
        pred_sec = np.exp(rt_cv + np.log(runtime_oracle[valid_rt])) * bt
        y_sec = np.exp(y_runtime[valid_rt])
        outputs["runtime_hybrid_ridge_cv"] = _metrics(pred_sec, y_sec)
        outputs["runtime_back_transform_factor"] = bt
        rt_model = make_hybrid_ridge()
        rt_model.fit(runtime_X[valid_rt], log_resid, ridge__sample_weight=sample_w[valid_rt])
        (args.output_dir / "runtime_hybrid_ridge.pkl").write_bytes(pickle.dumps(rt_model))
        (args.output_dir / "runtime_hybrid_ridge_meta.json").write_text(
            json.dumps(
                {
                    "back_transform_factor": bt,
                    "features": feature_cols + runtime_extra_cols,
                    "runtime_extra_cols": runtime_extra_cols,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    print("[v3] Building BSM prediction with kNN measurement imputation and OOD gate...")
    imputed_meas, ood_diag = estimate_bsm_measurements_knn(
        frame=aggregated,
        feature_cols=feature_cols,
        bsm_ops=BSM_OPS,
        k_neighbors=args.knn_measurement_neighbors,
    )
    bsm_X = np.array(
        [[imputed_meas.get(c, float(BSM_OPS.get(c, np.nan))) for c in feature_cols]],
        dtype=float,
    )
    bsm_oracle = float(gamma_oracle(np.array([BSM_SNR]))[0])
    bsm_eta_pred = float(hybrid_ridge.predict(bsm_X)[0])
    bsm_nrmse_pred = float(BSM_NRMSE_NULL * (1 + np.clip(bsm_eta_pred, -3, 1) * bsm_oracle))
    bsm_pct_err = 100.0 * (bsm_nrmse_pred - BSM_NRMSE_OBS) / BSM_NRMSE_OBS
    residual = y_nrmse[valid] - nrmse_cv
    sigma = float(np.std(residual))
    ood_scale = 1.0 + 0.15 * float(ood_diag["nearest_knob_distance"])
    half_width = 1.96 * sigma * ood_scale
    outputs["bsm_prediction"] = {
        "eta_pred": bsm_eta_pred,
        "nrmse_pred": bsm_nrmse_pred,
        "nrmse_obs": BSM_NRMSE_OBS,
        "pct_err": bsm_pct_err,
        "nrmse_pred_low95": bsm_nrmse_pred - half_width,
        "nrmse_pred_high95": bsm_nrmse_pred + half_width,
        "ood_diagnostics": ood_diag,
        "note": "kNN measurement imputation + OOD-scaled uncertainty interval",
    }

    pred_rows = aggregated[
        ["dgp_idx", "config_idx", "nrmse_final", "nrmse_null", "noise_snr"]
    ].copy()
    pred_rows["nrmse_obs"] = y_nrmse
    full_hybrid = hybrid_ridge.predict(X)
    pred_rows["eta_pred_hybrid_ridge"] = full_hybrid
    pred_rows["nrmse_pred_hybrid_ridge"] = pred_rows["nrmse_null"] * (
        1 + np.clip(full_hybrid * g_oracle, -1, 0)
    )
    pred_rows["eta_pred_rf"] = rf_model.predict(X)
    pred_rows["sample_weight_near_bsm"] = sample_w
    pred_rows.to_csv(args.output_dir / "wave5_track_a_v3_predictions.csv", index=False)

    summary = {
        "version": "v3",
        "improvements": [
            "1: BSM measurement imputation via kNN in knob-space",
            "2: optional wave6 merge + near-BSM sample weighting",
            "3: OOD distance diagnostics + uncertainty interval",
            "4: runtime log-residual model with large-output regime term",
        ],
        "input_counts": counts,
        "proxy_n_outputs_cap": args.proxy_n_outputs_cap,
        "proxy_n_runs_cap": args.proxy_n_runs_cap,
        "feature_cols": feature_cols,
        "row_count": int(len(aggregated)),
        "group_count": int(aggregated["dgp_idx"].nunique()),
        "metrics": outputs,
    }
    (args.output_dir / "wave5_track_a_v3_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print("[v3] Done. Summary written to wave5_track_a_v3_summary.json")


if __name__ == "__main__":
    main()
