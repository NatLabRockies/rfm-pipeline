"""Track B extended analytic baselines.

Extends scripts/analytic_baselines.py with:

1. Stage-by-stage efficiency decomposition. Each pipeline stage closes some
   fraction of the (null - oracle) gap; we attribute the total efficiency
   (eta = gamma_obs / gamma_oracle) to screening, discovery, sparse-selection,
   and final-fit stages individually.

2. Hybrid analytic + learned predictor:  gamma_predicted =
   gamma_oracle(snr) * eta_predicted(features).  The oracle gamma is fixed by
   physics; only the dimensionless efficiency eta is learned.  This constrains
   the model to physical bounds and makes extrapolation to BSM more reliable.

3. Head-to-head on the BSM operating point: hybrid analytic baseline vs the
   wave123 RF predictor that the handoff bundle uses.

Outputs:
- artifacts/sensitivity/wave1234_stage_decomposition.csv
- artifacts/sensitivity/wave1234_hybrid_predictor.csv
- artifacts/sensitivity/track_b_extended_summary.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = ROOT / "artifacts" / "sensitivity" / "wave1234_combined_clean.csv"
OUT_DIR = ROOT / "artifacts" / "sensitivity"


def gamma_oracle(snr: np.ndarray) -> np.ndarray:
    """Oracle gamma = sqrt(1/(snr+1)) - 1."""
    return np.sqrt(1.0 / (snr + 1.0)) - 1.0


def main() -> None:
    """Run extended Track B baselines (stage decomp + hybrid predictor + BSM)."""
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df):,} rows × {len(df.columns)} cols")

    # ---- 1. Stage-by-stage decomposition ----
    # Each stage closes some part of the gap between the null nRMSE and the
    # oracle nRMSE.  The oracle nRMSE is nrmse_null * sqrt(1/(snr+1))
    # (derivation in docs/manuscripts/track_b_analytic_baselines.md, §1).
    snr = df["noise_snr"].to_numpy(dtype=float)
    nrmse_null = df["nrmse_null"].to_numpy(dtype=float)
    nrmse_oracle = nrmse_null * np.sqrt(1.0 / (snr + 1.0))
    gap = nrmse_null - nrmse_oracle  # > 0; "what the pipeline could close"
    # Guard against null_screened rows where the pipeline never ran past null
    null_screened = df["null_screened"].fillna(False).astype(bool).to_numpy()
    safe_gap = np.where(gap > 1e-9, gap, np.nan)

    # Stage nRMSEs (each falls back to nrmse_null when stage never executed)
    nrmse_null_s = df["nrmse_null"]
    nrmse_main = df["main_effects_ols_nrmse"].fillna(nrmse_null_s).to_numpy(dtype=float)
    nrmse_screened = df["screened_ols_nrmse"].fillna(pd.Series(nrmse_main)).to_numpy(dtype=float)
    nrmse_pen = df["penalized_ols_nrmse"].fillna(pd.Series(nrmse_screened)).to_numpy(dtype=float)
    nrmse_final = df["nrmse_final"].to_numpy(dtype=float)

    # Per-stage gap closure (can be negative if a stage hurt)
    eta_main = (nrmse_null - nrmse_main) / safe_gap
    eta_screen = (nrmse_main - nrmse_screened) / safe_gap
    eta_sparse = (nrmse_screened - nrmse_pen) / safe_gap
    eta_final = (nrmse_pen - nrmse_final) / safe_gap
    eta_total = (nrmse_null - nrmse_final) / safe_gap

    stage_df = pd.DataFrame(
        {
            "job_id": df["job_id"],
            "noise_snr": snr,
            "nrmse_null": nrmse_null,
            "nrmse_oracle": nrmse_oracle,
            "nrmse_final": nrmse_final,
            "gap": gap,
            "null_screened": null_screened,
            "eta_main_effects": eta_main,
            "eta_screening": eta_screen,
            "eta_sparse_selection": eta_sparse,
            "eta_final_debias": eta_final,
            "eta_total": eta_total,
            "n_features_retained": df["n_features_retained"],
            "sparsity": df["sparsity"],
            "n_inputs": df["n_inputs"],
            "n_runs": df["n_runs"],
            "n_outputs": df["n_outputs"],
        }
    )
    stage_df.to_csv(OUT_DIR / "wave1234_stage_decomposition.csv", index=False)

    # Aggregate by null_screened state (the two clusters in the data)
    print("\n=== Stage attribution (mean eta per stage; successful rows only) ===")
    success = stage_df[~stage_df["null_screened"]]
    agg = success[
        [
            "eta_main_effects",
            "eta_screening",
            "eta_sparse_selection",
            "eta_final_debias",
            "eta_total",
        ]
    ].describe(percentiles=[0.1, 0.5, 0.9])
    print(agg.round(3))

    # ---- 2. Hybrid analytic + learned predictor ----
    # gamma_predicted = gamma_oracle(snr) * eta_predicted(features)
    # Learn eta from features other than snr (which is already captured by
    # gamma_oracle), constrained to [0, 1] via clipping.  Compare to a vanilla
    # RF predicting gamma directly from the same features.

    # Use the same predictor set as wave123 RF (top-9 from manuscript)
    feature_cols = [
        "n_inputs",
        "n_runs",
        "n_outputs",
        "sparsity",
        "interaction_density",
        "nonlinearity_strength",
        "holdout_fraction",
        "variance_threshold",
        "stages.empirical_null_screening.n_permutations",
        "stages.empirical_null_screening.bh_q_threshold",
        "stages.interaction_discovery.n_permutations",
        "stages.interaction_discovery.p_threshold",
        "stages.sparse_selection.n_stability_subsamples",
        "stages.final_artifacts.delta_threshold_override",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]
    print(f"\nUsing {len(feature_cols)} features for eta-predictor")

    # Build target: eta_total (clip to [-0.5, 1.5] to keep outliers from dominating)
    mask = ~stage_df["null_screened"] & np.isfinite(eta_total)
    X = df.loc[mask, feature_cols].to_numpy(dtype=float)
    eta_target = np.clip(eta_total[mask], -0.5, 1.5)
    gamma_obs = df.loc[mask, "nrmse_relative"].to_numpy(dtype=float)
    gamma_oracle_row = gamma_oracle(snr[mask])
    groups = df.loc[mask, "dgp_idx"].to_numpy()

    # Hybrid: learn eta, multiply by oracle gamma
    eta_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 25))),
        ]
    )
    rf_eta = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=0, n_jobs=-1)

    # Direct: learn gamma straight
    rf_direct = RandomForestRegressor(
        n_estimators=300, min_samples_leaf=5, random_state=0, n_jobs=-1
    )

    gkf = GroupKFold(n_splits=10)
    # CV scores against gamma (the metric that matters for BSM prediction)
    # Hybrid linear: score eta * gamma_oracle vs observed gamma
    eta_pred_lin = np.full_like(gamma_obs, np.nan)
    eta_pred_rf = np.full_like(gamma_obs, np.nan)
    gamma_pred_direct = np.full_like(gamma_obs, np.nan)
    for tr, te in gkf.split(X, eta_target, groups):
        eta_pipe.fit(X[tr], eta_target[tr])
        eta_pred_lin[te] = eta_pipe.predict(X[te])
        rf_eta.fit(X[tr], eta_target[tr])
        eta_pred_rf[te] = rf_eta.predict(X[te])
        rf_direct.fit(X[tr], gamma_obs[tr])
        gamma_pred_direct[te] = rf_direct.predict(X[te])

    gamma_pred_hybrid_lin = np.clip(eta_pred_lin, 0.0, 1.2) * gamma_oracle_row
    gamma_pred_hybrid_rf = np.clip(eta_pred_rf, 0.0, 1.2) * gamma_oracle_row

    def metrics(pred: np.ndarray, obs: np.ndarray) -> dict[str, float]:
        ss_res = float(np.sum((pred - obs) ** 2))
        ss_tot = float(np.sum((obs - obs.mean()) ** 2))
        return {
            "r2": 1.0 - ss_res / ss_tot,
            "rmse": float(np.sqrt(np.mean((pred - obs) ** 2))),
            "mae": float(np.mean(np.abs(pred - obs))),
        }

    print("\n=== Group-blocked 10-fold CV: predicting gamma ===")
    m_hybrid_lin = metrics(gamma_pred_hybrid_lin, gamma_obs)
    m_hybrid_rf = metrics(gamma_pred_hybrid_rf, gamma_obs)
    m_direct = metrics(gamma_pred_direct, gamma_obs)
    print(f"Hybrid Ridge-eta:  R2={m_hybrid_lin['r2']:.3f}  RMSE={m_hybrid_lin['rmse']:.4f}")
    print(f"Hybrid RF-eta:     R2={m_hybrid_rf['r2']:.3f}  RMSE={m_hybrid_rf['rmse']:.4f}")
    print(f"Direct RF:         R2={m_direct['r2']:.3f}  RMSE={m_direct['rmse']:.4f}")

    # Save per-row predictions
    pred_df = pd.DataFrame(
        {
            "job_id": df.loc[mask, "job_id"].to_numpy(),
            "noise_snr": snr[mask],
            "gamma_obs": gamma_obs,
            "gamma_oracle": gamma_oracle_row,
            "eta_pred_lin": eta_pred_lin,
            "eta_pred_rf": eta_pred_rf,
            "gamma_pred_hybrid_lin": gamma_pred_hybrid_lin,
            "gamma_pred_hybrid_rf": gamma_pred_hybrid_rf,
            "gamma_pred_direct_rf": gamma_pred_direct,
        }
    )
    pred_df.to_csv(OUT_DIR / "wave1234_hybrid_predictor.csv", index=False)

    # ---- 3. BSM head-to-head ----
    # BSM operating-point feature vector (from handoff bundle item 4):
    bsm_features = {
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
    bsm_features_full = bsm_features
    bsm_features = {k: bsm_features_full[k] for k in feature_cols if k in bsm_features_full}
    bsm_snr = 22.3  # from manuscript R10
    bsm_gamma_obs = -0.564
    bsm_nrmse_obs = 0.0721
    bsm_nrmse_null = 0.165
    bsm_gamma_oracle = gamma_oracle(np.array([bsm_snr]))[0]

    # Refit all three models on the FULL successful set (no CV holdout) and predict BSM
    eta_pipe_full = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 25))),
        ]
    ).fit(X, eta_target)
    rf_eta_full = RandomForestRegressor(
        n_estimators=500, min_samples_leaf=5, random_state=0, n_jobs=-1
    ).fit(X, eta_target)
    rf_direct_full = RandomForestRegressor(
        n_estimators=500, min_samples_leaf=5, random_state=0, n_jobs=-1
    ).fit(X, gamma_obs)

    bsm_X = np.array([[bsm_features[c] for c in feature_cols]])
    eta_pred_lin_bsm = float(np.clip(eta_pipe_full.predict(bsm_X)[0], 0.0, 1.2))
    eta_pred_rf_bsm = float(np.clip(rf_eta_full.predict(bsm_X)[0], 0.0, 1.2))
    gamma_pred_lin_bsm = eta_pred_lin_bsm * bsm_gamma_oracle
    gamma_pred_rf_bsm = eta_pred_rf_bsm * bsm_gamma_oracle
    gamma_pred_direct_bsm = float(rf_direct_full.predict(bsm_X)[0])

    nrmse_pred_lin_bsm = bsm_nrmse_null * (1.0 + gamma_pred_lin_bsm)
    nrmse_pred_rf_bsm = bsm_nrmse_null * (1.0 + gamma_pred_rf_bsm)
    nrmse_pred_direct_bsm = bsm_nrmse_null * (1.0 + gamma_pred_direct_bsm)

    # Wave123 RF (existing model from handoff); use joblib + canonical _RF_FEATURES order
    wave123_rf_path = OUT_DIR / "wave123_rf_quality.pkl"
    nrmse_pred_wave123_bsm: float | None = None
    if wave123_rf_path.exists():
        import joblib

        wave123_rf = joblib.load(wave123_rf_path)
        # _RF_FEATURES from scripts/fit_sensitivity_rf.py (target = log(nrmse_final))
        rf_feat_cols = [
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
        # Target is nrmse_relative (gamma directly), not log(nrmse). No back-transform.
        bsm_X_w123 = np.array([[bsm_features_full[c] for c in rf_feat_cols]])
        gamma_pred_wave123 = float(wave123_rf.predict(bsm_X_w123)[0])
        nrmse_pred_wave123_bsm = float(bsm_nrmse_null * (1.0 + gamma_pred_wave123))
        print(
            f"  wave123 RF gamma_pred={gamma_pred_wave123:+.4f}, "
            f"nRMSE_pred={nrmse_pred_wave123_bsm:.4f}"
        )

    def pct_err(pred: float, obs: float) -> float:
        return 100.0 * (pred - obs) / obs

    print("\n=== BSM operating-point head-to-head ===")
    print(f"  Observed nRMSE                       : {bsm_nrmse_obs:.4f}")
    print(f"  Observed gamma                       : {bsm_gamma_obs:+.4f}")
    print(f"  Oracle  gamma (theoretical max)      : {bsm_gamma_oracle:+.4f}")
    print(f"  Observed eta = gamma_obs/gamma_oracle: {bsm_gamma_obs / bsm_gamma_oracle:.3f}")
    print()
    print("  Hybrid Ridge:")
    err_l = pct_err(nrmse_pred_lin_bsm, bsm_nrmse_obs)
    print(
        f"     eta_pred={eta_pred_lin_bsm:.3f}  gamma_pred={gamma_pred_lin_bsm:+.4f}"
        f"  nRMSE_pred={nrmse_pred_lin_bsm:.4f}  err={err_l:+.1f}%"
    )
    print("  Hybrid RF:")
    err_r = pct_err(nrmse_pred_rf_bsm, bsm_nrmse_obs)
    print(
        f"     eta_pred={eta_pred_rf_bsm:.3f}  gamma_pred={gamma_pred_rf_bsm:+.4f}"
        f"  nRMSE_pred={nrmse_pred_rf_bsm:.4f}  err={err_r:+.1f}%"
    )
    print("  Direct RF:")
    print(
        f"     gamma_pred={gamma_pred_direct_bsm:+.4f}"
        f"  nRMSE_pred={nrmse_pred_direct_bsm:.4f}"
        f"  err={pct_err(nrmse_pred_direct_bsm, bsm_nrmse_obs):+.1f}%"
    )
    if nrmse_pred_wave123_bsm is not None:
        print("  Wave123 RF:")
        print(
            f"     nRMSE_pred={nrmse_pred_wave123_bsm:.4f}"
            f"  err={pct_err(nrmse_pred_wave123_bsm, bsm_nrmse_obs):+.1f}%"
        )

    # Summary JSON
    summary = {
        "stage_attribution_mean_successful_rows": {k: float(v) for k, v in agg.loc["mean"].items()},
        "stage_attribution_median_successful_rows": {
            k: float(v) for k, v in agg.loc["50%"].items()
        },
        "cv_metrics_gamma": {
            "hybrid_ridge_eta": m_hybrid_lin,
            "hybrid_rf_eta": m_hybrid_rf,
            "direct_rf": m_direct,
        },
        "bsm_head_to_head": {
            "observed_nrmse": bsm_nrmse_obs,
            "observed_gamma": bsm_gamma_obs,
            "oracle_gamma": float(bsm_gamma_oracle),
            "observed_eta": bsm_gamma_obs / float(bsm_gamma_oracle),
            "hybrid_ridge_eta_pred": eta_pred_lin_bsm,
            "hybrid_ridge_nrmse_pred": nrmse_pred_lin_bsm,
            "hybrid_ridge_pct_err": pct_err(nrmse_pred_lin_bsm, bsm_nrmse_obs),
            "hybrid_rf_eta_pred": eta_pred_rf_bsm,
            "hybrid_rf_nrmse_pred": nrmse_pred_rf_bsm,
            "hybrid_rf_pct_err": pct_err(nrmse_pred_rf_bsm, bsm_nrmse_obs),
            "direct_rf_nrmse_pred": nrmse_pred_direct_bsm,
            "direct_rf_pct_err": pct_err(nrmse_pred_direct_bsm, bsm_nrmse_obs),
            "wave123_rf_nrmse_pred": nrmse_pred_wave123_bsm,
            "wave123_rf_pct_err": (
                pct_err(nrmse_pred_wave123_bsm, bsm_nrmse_obs)
                if nrmse_pred_wave123_bsm is not None
                else None
            ),
        },
    }
    with open(OUT_DIR / "track_b_extended_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nWrote {OUT_DIR / 'track_b_extended_summary.json'}")


if __name__ == "__main__":
    main()
