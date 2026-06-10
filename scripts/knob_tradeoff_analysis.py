"""Knob-tradeoff analysis for the user-facing tuning guide.

For each tuning knob the user can change, compute:
  - marginal effect on gamma (nRMSE improvement over null)
  - marginal effect on pipeline_seconds (runtime per core)
  - efficient frontier (gamma improvement per core-second)
  - recommended setting bands by dataset regime

Also fits the single chosen runtime model: hybrid analytic baseline x
learned ridge correction, mirroring the gamma_hybrid_ridge structure.

Inputs:
  artifacts/sensitivity/wave1234_combined_clean.csv

Outputs:
  artifacts/sensitivity/knob_tradeoff_summary.json
  artifacts/sensitivity/knob_partial_dependence.csv
  artifacts/sensitivity/runtime_hybrid_model.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "artifacts" / "sensitivity" / "wave1234_combined_clean.csv"
OUT = ROOT / "artifacts" / "sensitivity"

KNOBS = {
    "n_screening_perms": "stages.empirical_null_screening.n_permutations",
    "bh_q": "stages.empirical_null_screening.bh_q_threshold",
    "n_interaction_perms": "stages.interaction_discovery.n_permutations",
    "interaction_p_threshold": "stages.interaction_discovery.p_threshold",
    "n_stability_subsamples": "stages.sparse_selection.n_stability_subsamples",
    "delta_threshold": "stages.final_artifacts.delta_threshold_override",
}
DATASET = ["n_runs", "n_inputs", "sparsity", "noise_snr"]


def load_clean() -> pd.DataFrame:
    """Load wave1234 cleaned CSV, keep successful (null_screened is NaN) rows."""
    df = pd.read_csv(IN)
    keep = df["null_screened"].isna()  # NaN = success
    df = df.loc[keep].copy()
    df = df.dropna(subset=["nrmse_relative", "pipeline_seconds", *DATASET, *KNOBS.values()])
    return df


def gamma_oracle(snr: np.ndarray) -> np.ndarray:
    """Closed-form Gaussian noise floor: sqrt(1/(snr+1)) - 1."""
    return np.sqrt(1.0 / (snr + 1.0)) - 1.0


def fit_hybrid_ridge(
    df: pd.DataFrame, target_col: str, oracle_col: str, group_col: str = "dgp_idx"
) -> tuple[Pipeline, dict]:
    """Hybrid model: target = oracle * eta_ridge(features).

    For gamma: oracle = gamma_oracle(snr); target = nrmse_relative.
    """
    feats = [*DATASET, *KNOBS.values()]
    X = df[feats].to_numpy(dtype=float)
    y = df[target_col].to_numpy(dtype=float)
    oracle = df[oracle_col].to_numpy(dtype=float)
    safe = np.where(np.abs(oracle) < 1e-9, np.sign(oracle) * 1e-9 + 1e-9, oracle)
    eta = np.clip(y / safe, -3.0, 5.0)
    pipe = Pipeline(
        [("scaler", StandardScaler()), ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 25)))]
    )
    groups = df[group_col].to_numpy()
    cv = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    cv_r2 = cross_val_score(pipe, X, eta, groups=groups, cv=cv, scoring="r2").mean()
    pipe.fit(X, eta)
    pred = pipe.predict(X) * safe
    in_r2 = 1.0 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    return pipe, {
        "form": "y = oracle * ridge(features)",
        "features": feats,
        "cv_r2_eta": float(cv_r2),
        "in_sample_r2_target": float(in_r2),
        "n_train": int(len(df)),
    }


def fit_runtime_loglog(
    df: pd.DataFrame, oracle_col: str = "runtime_oracle", group_col: str = "dgp_idx"
) -> tuple[Pipeline, dict, float]:
    """log(pipeline_sec) = log(oracle) + ridge(features) + bias.

    Standard scientific-runtime form: machine constant and stage-specific
    correction factors fit on log-residuals.
    """
    feats = [*DATASET, *KNOBS.values()]
    X = df[feats].to_numpy(dtype=float)
    y_log = np.log(df["pipeline_seconds"].to_numpy(dtype=float))
    oracle_log = np.log(df[oracle_col].to_numpy(dtype=float))
    resid = y_log - oracle_log
    pipe = Pipeline(
        [("scaler", StandardScaler()), ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 25)))]
    )
    groups = df[group_col].to_numpy()
    cv = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    cv_r2 = cross_val_score(pipe, X, resid, groups=groups, cv=cv, scoring="r2").mean()
    pipe.fit(X, resid)
    pred_log = pipe.predict(X) + oracle_log
    bt = float(np.exp(0.5 * np.var(y_log - pred_log)))  # lognormal back-transform
    pred_sec = np.exp(pred_log) * bt
    y = np.exp(y_log)
    in_r2 = 1.0 - np.sum((y - pred_sec) ** 2) / np.sum((y - y.mean()) ** 2)
    return (
        pipe,
        {
            "form": "log(sec) = log(oracle) + ridge(features); back_transform = exp(sigma^2/2)",
            "features": feats,
            "cv_r2_log_residual": float(cv_r2),
            "in_sample_r2_seconds": float(in_r2),
            "back_transform_factor": bt,
            "n_train": int(len(df)),
        },
        bt,
    )


def analytic_runtime_baseline(df: pd.DataFrame) -> np.ndarray:
    """Sum of per-stage analytic costs.

    Per-stage scaling (leading order):
      screening:    n_screen_perms * n_inputs * n_runs
      interactions: n_int_perms * n_inputs^2 * n_runs
      stability:    n_stab_subsamples * n_inputs * n_runs * lasso_grid
      final:        n_inputs * n_runs   (delta_threshold has negligible cost)
      base OLS:     n_inputs * n_runs (constant per fit)
    Sum scaled by a single constant; the learned eta absorbs the constant +
    machine-specific overhead.
    """
    n = df["n_runs"].to_numpy(dtype=float)
    d = df["n_inputs"].to_numpy(dtype=float)
    p_screen = df[KNOBS["n_screening_perms"]].to_numpy(dtype=float)
    p_int = df[KNOBS["n_interaction_perms"]].to_numpy(dtype=float)
    n_stab = df[KNOBS["n_stability_subsamples"]].to_numpy(dtype=float)
    lasso = df["stages.sparse_selection.lasso_alpha_grid_size"].to_numpy(dtype=float)
    return p_screen * d * n + p_int * d * d * n + n_stab * d * n * lasso + d * n + d * n


def partial_dependence(
    df: pd.DataFrame,
    model: Pipeline,
    oracle: np.ndarray,
    knob: str,
    log_form: bool = False,
    back_transform: float = 1.0,
) -> list[dict]:
    """Vary knob over observed quantiles; hold other features at row values.

    Reports mean predicted target across the dataset at each knob level.
    log_form=True: model predicts log-residual; pred = exp(model + log(oracle)) * bt.
    log_form=False: pred = model * oracle.
    """
    feats = [*DATASET, *KNOBS.values()]
    col = KNOBS[knob]
    levels = sorted(df[col].dropna().unique().tolist())
    if len(levels) > 8:
        qs = np.linspace(0, 1, 8)
        levels = sorted(set(np.quantile(df[col].dropna(), qs).tolist()))
    rows = []
    for v in levels:
        X = df[feats].copy()
        X[col] = v
        if log_form:
            pred = np.exp(model.predict(X.to_numpy(dtype=float)) + np.log(oracle)) * back_transform
        else:
            pred = model.predict(X.to_numpy(dtype=float)) * oracle
        rows.append({"knob": knob, "value": float(v), "mean_pred": float(np.mean(pred))})
    return rows


def main() -> None:
    """Run the full knob-tradeoff pipeline and write artifacts."""
    df = load_clean()
    print(f"Loaded {len(df)} successful rows")

    # 1. gamma model: hybrid ridge with gamma_oracle baseline
    df["gamma_oracle"] = gamma_oracle(df["noise_snr"].to_numpy(dtype=float))
    gmod, ginfo = fit_hybrid_ridge(df, "nrmse_relative", "gamma_oracle")
    print(
        f"gamma hybrid ridge: cv_R2(eta)={ginfo['cv_r2_eta']:.3f}, "
        f"in_R2(gamma)={ginfo['in_sample_r2_target']:.3f}"
    )

    # 2. runtime model: log-log hybrid with analytic per-stage baseline
    df["runtime_oracle"] = analytic_runtime_baseline(df)
    rmod, rinfo, bt = fit_runtime_loglog(df)
    print(
        f"runtime loglog hybrid: cv_R2(resid)={rinfo['cv_r2_log_residual']:.3f}, "
        f"in_R2(sec)={rinfo['in_sample_r2_seconds']:.3f}, bt={bt:.3f}"
    )

    # 3. per-knob partial dependence on gamma and runtime
    pd_rows = []
    for knob in KNOBS:
        pd_rows.extend(
            {"target": "gamma", **r}
            for r in partial_dependence(df, gmod, df["gamma_oracle"].to_numpy(), knob)
        )
        pd_rows.extend(
            {"target": "runtime_sec", **r}
            for r in partial_dependence(
                df, rmod, df["runtime_oracle"].to_numpy(), knob, log_form=True, back_transform=bt
            )
        )
    pd_df = pd.DataFrame(pd_rows)
    pd_df.to_csv(OUT / "knob_partial_dependence.csv", index=False)
    print(f"Wrote {OUT / 'knob_partial_dependence.csv'}")

    # 3. CONTROLLED marginal effects.
    # gamma: from hybrid ridge eta-coefs (knobs only appear in eta correction).
    # runtime: fit a SEPARATE plain log-linear model so coefficients are
    # total marginal effects in log-seconds per std-dev (not mixed with
    # the analytic baseline). This is the model used to inform users; the
    # hybrid loglog model is for point prediction.
    feats = [*DATASET, *KNOBS.values()]
    X_full = df[feats].to_numpy(dtype=float)
    y_logsec = np.log(df["pipeline_seconds"].to_numpy(dtype=float))
    rmod_marg = Pipeline(
        [("scaler", StandardScaler()), ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 25)))]
    )
    rmod_marg.fit(X_full, y_logsec)

    knob_labels = {v: k for k, v in KNOBS.items()}
    g_coef = gmod.named_steps["ridge"].coef_  # eta units per std-dev
    r_coef = rmod_marg.named_steps["ridge"].coef_  # log-sec per std-dev (TOTAL)
    mean_g_oracle = float(df["gamma_oracle"].mean())

    knob_table = []
    summary = {"knobs": {}, "gamma_model": ginfo, "runtime_model": rinfo}
    for fi, f in enumerate(feats):
        if f not in knob_labels:
            continue
        label = knob_labels[f]
        std = float(df[f].std())
        d_gamma_per_sd = float(g_coef[fi] * mean_g_oracle)
        d_logsec_per_sd = float(r_coef[fi])
        d_pct_runtime_per_sd = (np.exp(d_logsec_per_sd) - 1.0) * 100
        vmin = float(df[f].min())
        vmax = float(df[f].max())
        n_sd = (vmax - vmin) / std if std > 0 else 0.0
        d_gamma_swing = d_gamma_per_sd * n_sd
        d_pct_runtime_swing = (np.exp(d_logsec_per_sd * n_sd) - 1.0) * 100
        # improvement-per-pct-runtime: positive when scaling up the knob
        # both improves gamma (more negative) AND costs more runtime
        improvement_per_pct = (
            (-d_gamma_swing) / d_pct_runtime_swing if d_pct_runtime_swing > 1e-6 else 0.0
        )
        if d_pct_runtime_swing <= 0:
            rec = "scale up: improves gamma without runtime cost"
        elif d_gamma_swing < -0.005 and improvement_per_pct > 1e-4:
            rec = "scale up: meaningful gamma gain per runtime cost"
        elif abs(d_gamma_swing) < 0.005:
            rec = "use minimum: no detectable gamma benefit"
        elif d_gamma_swing > 0:
            rec = "use minimum: scaling up HURTS gamma"
        else:
            rec = "ambiguous: check dataset regime"
        knob_table.append(
            {
                "knob": label,
                "value_min": vmin,
                "value_max": vmax,
                "d_gamma_per_sd_controlled": d_gamma_per_sd,
                "d_logsec_per_sd_controlled": d_logsec_per_sd,
                "pct_runtime_per_sd": d_pct_runtime_per_sd,
                "gamma_swing_min_to_max": d_gamma_swing,
                "pct_runtime_swing_min_to_max": d_pct_runtime_swing,
                "gamma_improvement_per_pct_runtime": improvement_per_pct,
                "recommendation": rec,
            }
        )
        summary["knobs"][label] = knob_table[-1]

    pd.DataFrame(knob_table).to_csv(OUT / "knob_controlled_marginal_effects.csv", index=False)
    print(f"Wrote {OUT / 'knob_controlled_marginal_effects.csv'}")

    (OUT / "knob_tradeoff_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {OUT / 'knob_tradeoff_summary.json'}")


if __name__ == "__main__":
    main()
