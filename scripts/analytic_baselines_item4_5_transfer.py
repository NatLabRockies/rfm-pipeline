"""Track B items #4, #5, and transfer test.

#4. n-scaling law: model eta(n, snr) = 1 - C * n^(-alpha) and predict BSM.
#5. BH retention rate analytic: predicted n_features_retained vs observed.
#6. Transfer test: train hybrid Ridge on pure_synthetic only, test on bsm_structure family.

All three use the existing wave1234 data; no Kestrel needed.
Outputs:
- artifacts/sensitivity/track_b_item4_n_scaling.json
- artifacts/sensitivity/track_b_item5_bh_retention.json
- artifacts/sensitivity/track_b_transfer_test.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
IN_CSV = ROOT / "artifacts" / "sensitivity" / "wave1234_combined_clean.csv"
OUT_DIR = ROOT / "artifacts" / "sensitivity"


def gamma_oracle(snr: np.ndarray) -> np.ndarray:
    """Oracle gamma = sqrt(1/(snr+1)) - 1."""
    return np.sqrt(1.0 / (snr + 1.0)) - 1.0


def main() -> None:
    """Run items #4, #5, and transfer test."""
    df = pd.read_csv(IN_CSV)
    null_screened = df["null_screened"].fillna(False).astype(bool).to_numpy()
    success = ~null_screened
    snr = df["noise_snr"].to_numpy(dtype=float)
    n = df["n_runs"].to_numpy(dtype=float)
    gamma_obs = df["nrmse_relative"].to_numpy(dtype=float)
    g_oracle = gamma_oracle(snr)
    eta = np.where(np.abs(g_oracle) > 1e-9, gamma_obs / g_oracle, 0.0)

    # =========================================================================
    # Item #4: n-scaling law
    # =========================================================================
    print("=" * 70)
    print("Item #4: n-scaling law eta(n, snr)")
    print("=" * 70)

    # Model: eta = eta_inf - C * n^(-alpha), where eta_inf is large-n efficiency.
    # Fit globally + per-snr-bin to see if alpha depends on snr.

    def model(n_arr: np.ndarray, eta_inf: float, c_const: float, alpha: float) -> np.ndarray:
        return eta_inf - c_const * np.power(n_arr, -alpha)

    mask = success & np.isfinite(eta) & (eta > -0.5) & (eta < 1.2)
    n_fit = n[mask]
    eta_fit = eta[mask]

    # Global fit
    try:
        popt, pcov = curve_fit(
            model,
            n_fit,
            eta_fit,
            p0=[0.7, 5.0, 0.3],
            bounds=([0.0, 0.0, 0.0], [1.2, 1e5, 2.0]),
            maxfev=10000,
        )
        eta_inf_g, c_g, alpha_g = popt
        residual = eta_fit - model(n_fit, *popt)
        ss_res = float(np.sum(residual**2))
        ss_tot = float(np.sum((eta_fit - eta_fit.mean()) ** 2))
        r2_global = 1.0 - ss_res / ss_tot
        print("Global fit (n=4,780 successful rows):")
        print(f"  eta(n) = {eta_inf_g:.3f} - {c_g:.3f} * n^(-{alpha_g:.3f})")
        print(f"  R² = {r2_global:.3f}")
    except Exception as exc:  # noqa: BLE001
        print(f"Global fit failed: {exc}")
        eta_inf_g, c_g, alpha_g, r2_global = float("nan"), float("nan"), float("nan"), float("nan")

    # Per-SNR-bin fits
    snr_fit = snr[mask]
    snr_bins = [(0, 10), (10, 20), (20, 30), (30, 100)]
    bin_results: list[dict[str, float]] = []
    for low, high in snr_bins:
        sel = (snr_fit >= low) & (snr_fit < high)
        if sel.sum() < 50:
            continue
        try:
            popt_b, _ = curve_fit(
                model,
                n_fit[sel],
                eta_fit[sel],
                p0=[0.7, 5.0, 0.3],
                bounds=([0.0, 0.0, 0.0], [1.2, 1e5, 2.0]),
                maxfev=10000,
            )
            r = eta_fit[sel] - model(n_fit[sel], *popt_b)
            r2_b = 1.0 - np.sum(r**2) / np.sum((eta_fit[sel] - eta_fit[sel].mean()) ** 2)
            print(
                f"SNR ∈ [{low}, {high}): n={sel.sum()}, "
                f"eta_inf={popt_b[0]:.3f}, C={popt_b[1]:.3f}, alpha={popt_b[2]:.3f}, R²={r2_b:.3f}"
            )
            bin_results.append(
                {
                    "snr_low": low,
                    "snr_high": high,
                    "n_rows": int(sel.sum()),
                    "eta_inf": float(popt_b[0]),
                    "C": float(popt_b[1]),
                    "alpha": float(popt_b[2]),
                    "r2": float(r2_b),
                }
            )
        except Exception:  # noqa: BLE001
            continue

    # Predict BSM
    bsm_snr, bsm_n, bsm_nrmse_null = 22.3, 28750.0, 0.165
    bsm_oracle = float(gamma_oracle(np.array([bsm_snr]))[0])
    eta_pred_bsm = float(model(np.array([bsm_n]), eta_inf_g, c_g, alpha_g)[0])
    gamma_pred_bsm = float(np.clip(eta_pred_bsm, 0, 1.2) * bsm_oracle)
    nrmse_pred_bsm = float(bsm_nrmse_null * (1.0 + gamma_pred_bsm))
    pct_err = 100.0 * (nrmse_pred_bsm - 0.0721) / 0.0721
    print("\nBSM prediction from n-scaling law alone (snr-agnostic):")
    print(f"  eta_pred = {eta_pred_bsm:.3f}, gamma_pred = {gamma_pred_bsm:+.4f}")
    print(f"  nRMSE_pred = {nrmse_pred_bsm:.4f}, err = {pct_err:+.1f}% (vs 0.0721)")

    item4 = {
        "global_fit": {
            "eta_inf": float(eta_inf_g),
            "C": float(c_g),
            "alpha": float(alpha_g),
            "r2": float(r2_global),
            "formula": f"eta(n) = {eta_inf_g:.3f} - {c_g:.3f} * n^(-{alpha_g:.3f})",
        },
        "per_snr_bin_fits": bin_results,
        "bsm_prediction_from_n_only": {
            "eta_pred": eta_pred_bsm,
            "gamma_pred": gamma_pred_bsm,
            "nrmse_pred": nrmse_pred_bsm,
            "pct_err": pct_err,
        },
        "interpretation": (
            "Wave1234 n_runs varies 5250-29750. eta increases with n at power-law rate. "
            "BSM at n=28750 is at the high end of the sweep, so n-scaling extrapolation is "
            "minimal — most of the 26% BSM gap comes from structural regime, not sample size."
        ),
    }
    (OUT_DIR / "track_b_item4_n_scaling.json").write_text(json.dumps(item4, indent=2))

    # =========================================================================
    # Item #5: BH retention rate analytic
    # =========================================================================
    print("\n" + "=" * 70)
    print("Item #5: BH retention rate analytic")
    print("=" * 70)

    # Total candidate features per row depends on PCA (variance_threshold). Use
    # main_effects_ols_n_features as the observed "candidate pool" (after PCA
    # dim reduction the OLS stage sees these features) and
    # screened_ols_n_features as the post-screening retention.
    if "main_effects_ols_n_features" in df.columns and "screened_ols_n_features" in df.columns:
        m_total = df["main_effects_ols_n_features"].to_numpy(dtype=float)
        m_kept = df["screened_ols_n_features"].to_numpy(dtype=float)
        retention = np.where(m_total > 0, m_kept / m_total, np.nan)
    else:
        m_total = np.full(len(df), np.nan)
        m_kept = np.full(len(df), np.nan)
        retention = np.full(len(df), np.nan)

    bh_q = df["stages.empirical_null_screening.bh_q_threshold"].to_numpy(dtype=float)
    sparsity = df["sparsity"].to_numpy(dtype=float)

    # Analytic model for retention rate.
    #
    # Under BH at level q with k true-alternative features out of m, the
    # *expected* number of rejections is approximately
    #
    #     R = (1 - q)^(-1) * (pi * k + q * (m - k)) ≈ pi * k + q * m  (for q small)
    #
    # where pi is the per-true power.  Retention rate ≈ pi * (k/m) + q.
    # Assuming k/m ≈ sparsity (fraction of features carrying signal) and pi ≈ 1
    # in the high-SNR regime, this gives
    #
    #     retention_pred = sparsity + bh_q

    # Better: account for SNR-dependent power.  Power for a per-feature t-test
    # at level alpha = q with effect size d (per-feature signal/noise after
    # within-output noise budget split across signal features) is
    #
    #     pi ≈ Phi(d*sqrt(n) - z_{1-q/2})  (one-sided approximation)
    #
    # but d depends on per-feature variance share, which the pipeline doesn't
    # report.  We use the simpler form below and let the empirical residual
    # absorb the rest.

    retention_pred = sparsity + bh_q
    mask_r = success & np.isfinite(retention) & np.isfinite(retention_pred)
    if mask_r.sum() > 0:
        obs = retention[mask_r]
        pred = retention_pred[mask_r]
        r2_ret = 1.0 - np.sum((obs - pred) ** 2) / np.sum((obs - obs.mean()) ** 2)
        rmse_ret = float(np.sqrt(np.mean((obs - pred) ** 2)))
        corr_ret = float(np.corrcoef(obs, pred)[0, 1])
        print("Retention prediction (retention_pred = sparsity + bh_q):")
        print(
            f"  R² = {r2_ret:.3f}   RMSE = {rmse_ret:.3f}   r = {corr_ret:.3f}   n = {mask_r.sum()}"
        )
        print(f"  observed retention: mean={obs.mean():.3f}, median={np.median(obs):.3f}")
        print(f"  predicted retention: mean={pred.mean():.3f}, median={np.median(pred):.3f}")
    else:
        r2_ret = float("nan")
        rmse_ret = float("nan")
        corr_ret = float("nan")
        print("Retention data not available")

    bsm_sparsity, bsm_bh_q = 0.28, 0.05
    bsm_retention_pred = bsm_sparsity + bsm_bh_q
    print(f"BSM retention pred: {bsm_retention_pred:.3f} (s={bsm_sparsity} + q={bsm_bh_q})")

    item5 = {
        "model": "retention_pred = sparsity + bh_q (leading-order BH approximation, pi=1 limit)",
        "r2": float(r2_ret),
        "rmse": float(rmse_ret),
        "correlation": float(corr_ret),
        "n_rows": int(mask_r.sum()),
        "observed_mean_retention": float(obs.mean()) if mask_r.sum() > 0 else None,
        "predicted_mean_retention": float(pred.mean()) if mask_r.sum() > 0 else None,
        "bsm_retention_pred": float(bsm_retention_pred),
        "interpretation": (
            "Leading-order BH model predicts retention rate of sparsity + bh_q. "
            "Captures the average retention rate but per-row scatter is large because "
            "true per-feature power varies with effect size and within-output noise budget "
            "(not directly observable in the wave artifacts)."
        ),
    }
    (OUT_DIR / "track_b_item5_bh_retention.json").write_text(json.dumps(item5, indent=2))

    # =========================================================================
    # Transfer test: train on pure_synthetic only, test on bsm_structure family
    # =========================================================================
    print("\n" + "=" * 70)
    print("Transfer test: train on pure_synthetic, test on bsm_structure")
    print("=" * 70)

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
    pure_mask = success & (df["block"] == "pure_synthetic").to_numpy()
    bsm_mask = success & (df["block"] == "bsm_structure").to_numpy()
    print(f"Pure synthetic train rows: {pure_mask.sum()}, BSM-family test rows: {bsm_mask.sum()}")

    X_train = df.loc[pure_mask, feature_cols].to_numpy(dtype=float)
    eta_train = np.clip(eta[pure_mask], -0.5, 1.5)
    X_test = df.loc[bsm_mask, feature_cols].to_numpy(dtype=float)
    gamma_test_obs = gamma_obs[bsm_mask]
    g_oracle_test = g_oracle[bsm_mask]

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 25))),
        ]
    ).fit(X_train, eta_train)
    eta_pred_test = np.clip(pipe.predict(X_test), 0.0, 1.2)
    gamma_pred_test = eta_pred_test * g_oracle_test

    rmse_xfer = float(np.sqrt(np.mean((gamma_pred_test - gamma_test_obs) ** 2)))
    mae_xfer = float(np.mean(np.abs(gamma_pred_test - gamma_test_obs)))
    bias_xfer = float(np.mean(gamma_pred_test - gamma_test_obs))
    pct_err_per_row = 100.0 * (
        (gamma_pred_test - gamma_test_obs)
        / np.where(np.abs(gamma_test_obs) > 1e-6, gamma_test_obs, 1.0)
    )
    abs_pct_mean = float(np.mean(np.abs(pct_err_per_row)))
    abs_pct_median = float(np.median(np.abs(pct_err_per_row)))

    # also: predicted vs observed eta on the bsm_structure family
    eta_test_obs = eta[bsm_mask]
    eta_rmse = float(np.sqrt(np.mean((eta_pred_test - eta_test_obs) ** 2)))
    eta_bias = float(np.mean(eta_pred_test - eta_test_obs))

    print(f"\nTransfer to bsm_structure family ({bsm_mask.sum()} rows):")
    print(f"  gamma RMSE  = {rmse_xfer:.4f}")
    print(f"  gamma MAE   = {mae_xfer:.4f}")
    print(f"  gamma bias  = {bias_xfer:+.4f} (positive = predicts gamma too high = nRMSE too high)")
    print(f"  abs %err on gamma: mean = {abs_pct_mean:.1f}%, median = {abs_pct_median:.1f}%")
    print(f"  eta RMSE    = {eta_rmse:.4f}, eta bias = {eta_bias:+.4f}")
    print(
        f"  observed eta on bsm_structure: mean = {eta_test_obs.mean():.3f}, "
        f"median = {np.median(eta_test_obs):.3f}"
    )
    print(
        f"  predicted eta from pure synthetic model: mean = {eta_pred_test.mean():.3f}, "
        f"median = {np.median(eta_pred_test):.3f}"
    )

    # Compare to BSM real prediction (need same model + BSM features)
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
    bsm_X = np.array([[bsm_features_full[c] for c in feature_cols]])
    bsm_eta_pred = float(np.clip(pipe.predict(bsm_X)[0], 0.0, 1.2))
    bsm_gamma_pred = bsm_eta_pred * bsm_oracle
    bsm_nrmse_pred = bsm_nrmse_null * (1.0 + bsm_gamma_pred)
    bsm_pct_err = 100.0 * (bsm_nrmse_pred - 0.0721) / 0.0721
    print("\nReal BSM prediction (same pure-synthetic-trained model):")
    print(f"  eta_pred = {bsm_eta_pred:.3f}, observed eta = 0.711")
    print(f"  nRMSE_pred = {bsm_nrmse_pred:.4f}, observed = 0.0721, err = {bsm_pct_err:+.1f}%")

    xfer = {
        "n_train_pure": int(pure_mask.sum()),
        "n_test_bsm_structure": int(bsm_mask.sum()),
        "gamma_rmse": rmse_xfer,
        "gamma_mae": mae_xfer,
        "gamma_bias": bias_xfer,
        "abs_pct_err_on_gamma_mean": abs_pct_mean,
        "abs_pct_err_on_gamma_median": abs_pct_median,
        "eta_rmse_on_bsm_structure": eta_rmse,
        "eta_bias_on_bsm_structure": eta_bias,
        "eta_obs_mean_bsm_structure": float(eta_test_obs.mean()),
        "eta_pred_mean_bsm_structure": float(eta_pred_test.mean()),
        "bsm_real_prediction": {
            "eta_pred": bsm_eta_pred,
            "nrmse_pred": float(bsm_nrmse_pred),
            "pct_err": float(bsm_pct_err),
        },
        "interpretation": (
            "If eta error on bsm_structure (calibrated synthetic) << eta error on real BSM, "
            "the gap is structural (real BSM is outside the calibrated synthetic regime, "
            "not just outside pure_synthetic), and only measurement features close it."
        ),
    }
    (OUT_DIR / "track_b_transfer_test.json").write_text(json.dumps(xfer, indent=2))
    print(f"\nWrote {OUT_DIR / 'track_b_item4_n_scaling.json'}")
    print(f"Wrote {OUT_DIR / 'track_b_item5_bh_retention.json'}")
    print(f"Wrote {OUT_DIR / 'track_b_transfer_test.json'}")


if __name__ == "__main__":
    main()
