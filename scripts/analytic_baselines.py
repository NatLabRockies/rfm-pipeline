"""Compute analytic baseline predictions for sensitivity-study runs (Track B).

For each run, derives closed-form predictions for the maximum pipeline reduction
achievable under perfect feature identification (the "oracle bound") and reports
how the observed pipeline performance compares.

The oracle bound assumes:
  * Pipeline perfectly identifies true active features (perfect screening).
  * All residual variance is irreducible Gaussian noise.
  * nRMSE_oracle / nRMSE_null = sqrt(noise_var / (signal_var + noise_var))
    which, for the synthetic DGP where signal is standardized to unit variance
    and noise has variance 1/snr, evaluates to sqrt(1 / (snr + 1)).

So gamma_oracle = sqrt(1 / (snr + 1)) - 1, and "pipeline efficiency" is the
fraction of this theoretical reduction actually achieved.

Pipeline efficiency is itself a useful summary statistic for the manuscript:
it normalizes performance by what is intrinsically achievable on each problem.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent


def compute_analytic_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame augmented with analytic baseline columns and efficiency metric."""
    out = df.copy()
    snr = out["noise_snr"].astype(float).to_numpy()

    out["gamma_oracle"] = np.sqrt(1.0 / (snr + 1.0)) - 1.0
    out["nrmse_oracle"] = out["nrmse_null"].astype(float) * np.sqrt(1.0 / (snr + 1.0))

    n_arr = out["n_runs"].astype(float).to_numpy()
    e_max = np.sqrt(2.0 * np.log(n_arr)) - 0.5 * (
        np.log(np.log(n_arr)) + np.log(4.0 * np.pi)
    ) / np.sqrt(2.0 * np.log(n_arr))
    out["nrmse_null_predicted"] = 1.0 / (2.0 * e_max)

    obs_gamma = out["nrmse_relative"].astype(float).to_numpy()
    oracle_gamma = out["gamma_oracle"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        out["pipeline_efficiency"] = obs_gamma / oracle_gamma
    return out


def summarize(out: pd.DataFrame) -> dict[str, float | int]:
    """Compute scalar summary statistics for the report."""
    mask_eff = np.isfinite(out["pipeline_efficiency"]) & (out["gamma_oracle"] < -1e-6)
    eff = out.loc[mask_eff, "pipeline_efficiency"].to_numpy()
    obs_g = out.loc[mask_eff, "nrmse_relative"].astype(float).to_numpy()
    orc_g = out.loc[mask_eff, "gamma_oracle"].to_numpy()

    pred_null = out["nrmse_null_predicted"].astype(float).to_numpy()
    obs_null = out["nrmse_null"].astype(float).to_numpy()
    mask_null = np.isfinite(pred_null) & np.isfinite(obs_null)

    return {
        "n_rows": int(len(out)),
        "n_rows_efficiency": int(mask_eff.sum()),
        "gamma_oracle_mean": float(orc_g.mean()),
        "gamma_oracle_p05": float(np.percentile(orc_g, 5)),
        "gamma_oracle_p95": float(np.percentile(orc_g, 95)),
        "gamma_observed_mean": float(obs_g.mean()),
        "pipeline_efficiency_mean": float(eff.mean()),
        "pipeline_efficiency_median": float(np.median(eff)),
        "pipeline_efficiency_p05": float(np.percentile(eff, 5)),
        "pipeline_efficiency_p95": float(np.percentile(eff, 95)),
        "nrmse_null_pred_obs_correlation": float(
            np.corrcoef(pred_null[mask_null], obs_null[mask_null])[0, 1]
        ),
        "nrmse_null_pred_obs_rmse": float(
            np.sqrt(np.mean((pred_null[mask_null] - obs_null[mask_null]) ** 2))
        ),
    }


def main() -> int:
    """Run analytic baseline computation against a wave results CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        default=str(REPO_ROOT / "artifacts" / "sensitivity" / "wave1234_combined_clean.csv"),
    )
    parser.add_argument(
        "--out-prefix",
        default=str(REPO_ROOT / "artifacts" / "sensitivity" / "wave1234_analytic_baselines"),
    )
    args = parser.parse_args()

    df = pd.read_csv(args.results)
    df = df.dropna(subset=["nrmse_null", "nrmse_final", "noise_snr", "n_runs"])
    out = compute_analytic_baselines(df)

    csv_path = Path(args.out_prefix + ".csv")
    summary_path = Path(args.out_prefix + "_summary.json")

    out_cols = [
        "job_id",
        "block",
        "n_runs",
        "noise_snr",
        "nrmse_null",
        "nrmse_final",
        "nrmse_relative",
        "nrmse_null_predicted",
        "gamma_oracle",
        "nrmse_oracle",
        "pipeline_efficiency",
    ]
    available = [c for c in out_cols if c in out.columns]
    out[available].to_csv(csv_path, index=False)

    summary = summarize(out)
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
