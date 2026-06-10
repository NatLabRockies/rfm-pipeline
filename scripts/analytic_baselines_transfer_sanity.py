"""Sanity check on the Ridge transfer test finding.

Verifies the 2.7% BSM error from pure-synthetic-trained Ridge by running:
1. Same with RF and ExtraTrees: do they also predict BSM ~0.07?
2. Pure-synthetic CV: is the model stable in-distribution?
3. Sensitivity: re-fit excluding random 10% of pure rows; does BSM pred drift?
4. Holdout pure→pure transfer: predict 10% pure → check that Ridge is well-calibrated on pure.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
IN_CSV = ROOT / "artifacts" / "sensitivity" / "wave1234_combined_clean.csv"


def gamma_oracle(snr: np.ndarray) -> np.ndarray:
    """Oracle gamma."""
    return np.sqrt(1.0 / (snr + 1.0)) - 1.0


def main() -> None:
    """Run."""
    df = pd.read_csv(IN_CSV)
    success = ~df["null_screened"].fillna(False).astype(bool).to_numpy()
    snr = df["noise_snr"].to_numpy(dtype=float)
    gamma_obs = df["nrmse_relative"].to_numpy(dtype=float)
    g_oracle = gamma_oracle(snr)
    eta_all = np.where(np.abs(g_oracle) > 1e-9, gamma_obs / g_oracle, 0.0)

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
    bsm_features_full = {
        "n_inputs": 135,
        "n_runs": 28750,
        "n_outputs": 9954,
        "sparsity": 0.28,
        "interaction_density": 0.15,
        "nonlinearity_strength": 0.20,
        "holdout_fraction": 0.05,
        "variance_threshold": 0.90,
        "stages.empirical_null_screening.n_permutations": 201,
        "stages.empirical_null_screening.bh_q_threshold": 0.05,
        "stages.interaction_discovery.n_permutations": 31,
        "stages.interaction_discovery.p_threshold": 0.05,
        "stages.sparse_selection.n_stability_subsamples": 50,
        "stages.final_artifacts.delta_threshold_override": 0.002,
    }
    bsm_snr = 22.3
    bsm_oracle = float(gamma_oracle(np.array([bsm_snr]))[0])
    bsm_nrmse_null = 0.165
    bsm_nrmse_obs = 0.0721
    bsm_X = np.array([[bsm_features_full[c] for c in feature_cols]])

    pure_mask = success & (df["block"] == "pure_synthetic").to_numpy()
    bsm_mask = success & (df["block"] == "bsm_structure").to_numpy()
    print(f"Pure: {pure_mask.sum()}, BSM-family: {bsm_mask.sum()}")

    X_pure = df.loc[pure_mask, feature_cols].to_numpy(dtype=float)
    eta_pure = np.clip(eta_all[pure_mask], -0.5, 1.5)
    gamma_pure_obs = gamma_obs[pure_mask]
    groups_pure = df.loc[pure_mask, "dgp_idx"].to_numpy()

    # === 1. Different model classes on pure → BSM ===
    print("\n=== BSM prediction from pure-synthetic-only training ===")
    models = {
        "Hybrid Ridge": (
            "eta",
            Pipeline([("s", StandardScaler()), ("r", RidgeCV(alphas=np.logspace(-3, 3, 25)))]),
        ),
        "Hybrid RF": (
            "eta",
            RandomForestRegressor(n_estimators=500, min_samples_leaf=5, random_state=0, n_jobs=-1),
        ),
        "Hybrid ExtraTrees": (
            "eta",
            ExtraTreesRegressor(n_estimators=500, min_samples_leaf=5, random_state=0, n_jobs=-1),
        ),
        "Direct Ridge": (
            "gamma",
            Pipeline([("s", StandardScaler()), ("r", RidgeCV(alphas=np.logspace(-3, 3, 25)))]),
        ),
        "Direct RF": (
            "gamma",
            RandomForestRegressor(n_estimators=500, min_samples_leaf=5, random_state=0, n_jobs=-1),
        ),
    }
    bsm_results = {}
    for name, (kind, mdl) in models.items():
        y_train = eta_pure if kind == "eta" else gamma_pure_obs
        mdl.fit(X_pure, y_train)
        pred = float(mdl.predict(bsm_X)[0])
        if kind == "eta":
            eta_pred = max(0.0, min(1.2, pred))
            gamma_pred = eta_pred * bsm_oracle
        else:
            gamma_pred = pred
            eta_pred = pred / bsm_oracle if abs(bsm_oracle) > 1e-9 else 0.0
        nrmse_pred = bsm_nrmse_null * (1.0 + gamma_pred)
        pct = 100.0 * (nrmse_pred - bsm_nrmse_obs) / bsm_nrmse_obs
        bsm_results[name] = {
            "eta_pred": eta_pred,
            "gamma_pred": gamma_pred,
            "nrmse_pred": nrmse_pred,
            "pct_err": pct,
        }
        print(
            f"  {name:22s}  eta={eta_pred:.3f}  gamma={gamma_pred:+.4f}  "
            f"nRMSE={nrmse_pred:.4f}  err={pct:+.1f}%"
        )

    # === 2. Pure-synthetic group-blocked CV (sanity: is hybrid Ridge stable in-distribution?) ===
    print("\n=== Pure-synthetic group-blocked 10-fold CV (in-distribution) ===")
    pipe = Pipeline([("s", StandardScaler()), ("r", RidgeCV(alphas=np.logspace(-3, 3, 25)))])
    rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=5, random_state=0, n_jobs=-1)
    gkf = GroupKFold(n_splits=10)
    gamma_pred_ridge = np.full_like(gamma_pure_obs, np.nan)
    gamma_pred_rf = np.full_like(gamma_pure_obs, np.nan)
    g_oracle_pure = g_oracle[pure_mask]
    for tr, te in gkf.split(X_pure, eta_pure, groups_pure):
        pipe.fit(X_pure[tr], eta_pure[tr])
        eta_p = np.clip(pipe.predict(X_pure[te]), 0.0, 1.2)
        gamma_pred_ridge[te] = eta_p * g_oracle_pure[te]
        rf.fit(X_pure[tr], gamma_pure_obs[tr])
        gamma_pred_rf[te] = rf.predict(X_pure[te])
    for name, pred in [("Hybrid Ridge", gamma_pred_ridge), ("Direct RF", gamma_pred_rf)]:
        r2 = 1.0 - np.sum((pred - gamma_pure_obs) ** 2) / np.sum(
            (gamma_pure_obs - gamma_pure_obs.mean()) ** 2
        )
        rmse = float(np.sqrt(np.mean((pred - gamma_pure_obs) ** 2)))
        print(f"  {name:14s}  R²={r2:.3f}  RMSE={rmse:.4f}  (in-distribution CV)")

    # === 3. Bootstrap sensitivity ===
    print("\n=== Bootstrap sensitivity: BSM prediction across 20 resamples ===")
    rng = np.random.default_rng(0)
    preds = []
    for _ in range(20):
        idx = rng.choice(len(X_pure), size=int(0.9 * len(X_pure)), replace=False)
        p = Pipeline([("s", StandardScaler()), ("r", RidgeCV(alphas=np.logspace(-3, 3, 25)))])
        p.fit(X_pure[idx], eta_pure[idx])
        ep = max(0.0, min(1.2, float(p.predict(bsm_X)[0])))
        preds.append(bsm_nrmse_null * (1.0 + ep * bsm_oracle))
    preds = np.array(preds)
    print(
        f"  Hybrid Ridge BSM nRMSE: mean={preds.mean():.4f}, std={preds.std():.4f}, "
        f"range=[{preds.min():.4f}, {preds.max():.4f}]"
    )
    print(f"  vs observed = {bsm_nrmse_obs:.4f}")

    # === 4. Compare bsm_structure observed vs predicted (on pure-trained model) ===
    print("\n=== bsm_structure family: observed vs pure-model predicted ===")
    X_bsm = df.loc[bsm_mask, feature_cols].to_numpy(dtype=float)
    eta_bsm_obs = eta_all[bsm_mask]
    # gamma_bsm_obs = gamma_obs[bsm_mask]
    pipe2 = Pipeline([("s", StandardScaler()), ("r", RidgeCV(alphas=np.logspace(-3, 3, 25)))]).fit(
        X_pure, eta_pure
    )
    eta_bsm_pred = np.clip(pipe2.predict(X_bsm), 0.0, 1.2)
    # gamma_bsm_pred = eta_bsm_pred * g_oracle_bsm
    print(
        f"  observed eta:   mean={eta_bsm_obs.mean():.3f}, median={np.median(eta_bsm_obs):.3f}, "
        f"p5={np.quantile(eta_bsm_obs, 0.05):.3f}, p95={np.quantile(eta_bsm_obs, 0.95):.3f}"
    )
    print(
        f"  predicted eta:  mean={eta_bsm_pred.mean():.3f}, median={np.median(eta_bsm_pred):.3f}, "
        f"p5={np.quantile(eta_bsm_pred, 0.05):.3f}, p95={np.quantile(eta_bsm_pred, 0.95):.3f}"
    )
    print(
        f"  bias on bsm_structure family: predicted - observed = "
        f"{(eta_bsm_pred - eta_bsm_obs).mean():+.3f}"
    )
    print(
        f"  (real BSM eta_obs = 0.711; predicted from pure-model = "
        f"{bsm_results['Hybrid Ridge']['eta_pred']:.3f})"
    )

    out = {
        "bsm_predictions_from_pure_only": bsm_results,
        "bootstrap_mean": float(preds.mean()),
        "bootstrap_std": float(preds.std()),
        "bootstrap_min": float(preds.min()),
        "bootstrap_max": float(preds.max()),
        "bsm_structure_observed_eta_mean": float(eta_bsm_obs.mean()),
        "bsm_structure_predicted_eta_mean": float(eta_bsm_pred.mean()),
        "real_bsm_observed_eta": 0.711,
        "real_bsm_predicted_eta_hybrid_ridge": bsm_results["Hybrid Ridge"]["eta_pred"],
    }
    (ROOT / "artifacts" / "sensitivity" / "track_b_transfer_sanity.json").write_text(
        json.dumps(out, indent=2)
    )


if __name__ == "__main__":
    main()
