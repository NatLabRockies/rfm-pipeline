#!/usr/bin/env python3
# Copyright (c) 2026 Dylan Hettinger
"""Fit a degree-2 OLS response surface for sensitivity-study results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PREDICTOR_CANDIDATES = {
    # DGP characteristics
    "n_runs": ["n_runs"],
    "n_inputs": ["n_inputs"],
    "sparsity": ["sparsity"],
    "interaction_density": ["interaction_density"],
    "nonlinearity_strength": ["nonlinearity_strength"],
    "noise_snr": ["noise_snr"],
    # Pipeline config parameters
    "holdout_fraction": ["holdout_fraction"],
    "variance_threshold": ["variance_threshold", "algorithm.variance_threshold"],
    "n_perm_screen": ["stages.empirical_null_screening.n_permutations"],
    "bh_q": ["stages.empirical_null_screening.bh_q_threshold"],
    "n_perm_interaction": ["stages.interaction_discovery.n_permutations"],
    "p_threshold": ["stages.interaction_discovery.p_threshold"],
    "n_stability_subsamples": ["stages.sparse_selection.n_stability_subsamples"],
    "lasso_alpha_percentile": ["stages.sparse_selection.lasso_alpha_percentile"],
    "delta_threshold": ["stages.final_artifacts.delta_threshold_override"],
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="Collected results CSV.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("formula_table.csv"),
        help="Output coefficient table CSV path.",
    )
    return parser.parse_args()


def _first_present(frame: pd.DataFrame, candidates: list[str]) -> str:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise KeyError(f"Missing required predictor column. Tried: {candidates}")


def _prepare_analysis_frame(results: pd.DataFrame) -> pd.DataFrame:
    """Build the regression-ready frame from either long or flat format.

    Long format (legacy): has ``stage``, ``subsample_n``, ``nrmse`` columns.
    Flat format (current): has ``nrmse_final``, ``null_mean_nrmse`` columns written
    by ``run_sensitivity_job.py``.
    """
    if "stage" in results.columns and "nrmse" in results.columns:
        # Legacy long format.
        final_rows = results.loc[results["stage"] == "final_ols"].copy()
        null_rows = results.loc[
            results["stage"] == "null_baseline", ["job_id", "subsample_n", "nrmse"]
        ].copy()
        null_rows = null_rows.rename(columns={"nrmse": "nrmse_null"})
        analysis = final_rows.merge(null_rows, on=["job_id", "subsample_n"], how="inner")
        analysis["nrmse_relative"] = (analysis["nrmse"] - analysis["nrmse_null"]) / analysis[
            "nrmse_null"
        ]
    else:
        # Flat format from run_sensitivity_job.py.
        # nrmse_relative may already be present; if not, compute from available columns.
        analysis = results.copy()
        if "nrmse_relative" not in analysis.columns:
            null_col = next(
                (c for c in ("null_mean_nrmse", "nrmse_null") if c in analysis.columns), None
            )
            final_col = next(
                (c for c in ("nrmse_final", "final_ols_nrmse") if c in analysis.columns), None
            )
            if null_col and final_col:
                analysis["nrmse_relative"] = (analysis[final_col] - analysis[null_col]) / analysis[
                    null_col
                ].abs()
            else:
                raise KeyError("Cannot determine nrmse_relative: missing nrmse_final/nrmse_null")

    analysis = analysis.replace([np.inf, -np.inf], np.nan).dropna(subset=["nrmse_relative"])

    # DGP predictors (from jobs.csv join, may be absent in legacy data).
    if "n_runs" in analysis.columns:
        analysis["log_n_runs"] = np.log(analysis["n_runs"].astype(float).clip(lower=1))
    else:
        analysis["log_n_runs"] = np.nan

    if "subsample_n" in analysis.columns:
        analysis["log_subsample_n"] = np.log(analysis["subsample_n"].astype(float).clip(lower=1))
    elif "n_runs" in analysis.columns:
        analysis["log_subsample_n"] = analysis["log_n_runs"]
    else:
        analysis["log_subsample_n"] = np.nan

    analysis["holdout_fraction_model"] = analysis[
        _first_present(analysis, PREDICTOR_CANDIDATES["holdout_fraction"])
    ]
    analysis["variance_threshold_model"] = analysis[
        _first_present(analysis, PREDICTOR_CANDIDATES["variance_threshold"])
    ]
    analysis["log_n_perm_screen"] = np.log(
        analysis[_first_present(analysis, PREDICTOR_CANDIDATES["n_perm_screen"])]
    )
    analysis["bh_q_model"] = analysis[_first_present(analysis, PREDICTOR_CANDIDATES["bh_q"])]
    analysis["log_n_perm_interaction"] = np.log(
        analysis[_first_present(analysis, PREDICTOR_CANDIDATES["n_perm_interaction"])]
    )
    analysis["p_threshold_model"] = analysis[
        _first_present(analysis, PREDICTOR_CANDIDATES["p_threshold"])
    ]
    analysis["log_n_stability_subsamples"] = np.log(
        analysis[_first_present(analysis, PREDICTOR_CANDIDATES["n_stability_subsamples"])]
    )
    analysis["lasso_alpha_percentile_model"] = analysis[
        _first_present(analysis, PREDICTOR_CANDIDATES["lasso_alpha_percentile"])
    ]
    delta_column = _first_present(analysis, PREDICTOR_CANDIDATES["delta_threshold"])
    analysis["delta_threshold_model"] = analysis[delta_column].fillna(0.0)
    analysis["log_delta_threshold"] = np.log(analysis["delta_threshold_model"] + 0.001)
    return analysis


def main() -> int:
    """Fit and report the degree-2 meta-regression response surface."""
    args = _parse_args()
    results = pd.read_csv(args.results)
    analysis = _prepare_analysis_frame(results)
    predictors = [
        "log_subsample_n",
        "holdout_fraction_model",
        "variance_threshold_model",
        "log_n_perm_screen",
        "bh_q_model",
        "log_n_perm_interaction",
        "p_threshold_model",
        "log_n_stability_subsamples",
        "lasso_alpha_percentile_model",
        "log_delta_threshold",
    ]
    # Add DGP predictors when available (from jobs.csv join).
    dgp_predictors = []
    for col in ("sparsity", "interaction_density", "nonlinearity_strength"):
        if col in analysis.columns and analysis[col].notna().any():
            dgp_predictors.append(col)
    for col in ("noise_snr", "n_inputs"):
        if col in analysis.columns and analysis[col].notna().any():
            log_col = f"log_{col}"
            analysis[log_col] = np.log(analysis[col].astype(float).clip(lower=1e-9))
            dgp_predictors.append(log_col)
    predictors = predictors + dgp_predictors
    # Drop rows with NaN in any predictor.
    analysis = analysis.dropna(subset=predictors)
    x = analysis[predictors].to_numpy(dtype=float)
    y = analysis["nrmse_relative"].to_numpy(dtype=float)

    poly = PolynomialFeatures(degree=2, include_bias=True)
    design = poly.fit_transform(x)
    coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coef
    n_obs, n_terms = design.shape
    dof = max(1, n_obs - n_terms)
    rss = float(np.sum((y - fitted) ** 2))
    sigma2 = rss / dof
    xtx_inv = np.linalg.pinv(design.T @ design)
    standard_errors = np.sqrt(np.maximum(np.diag(xtx_inv) * sigma2, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stats = np.divide(
            coef, standard_errors, out=np.zeros_like(coef), where=standard_errors > 0
        )
    p_values = 2.0 * stats.t.sf(np.abs(t_stats), df=dof)

    terms = poly.get_feature_names_out(predictors)
    formula_table = pd.DataFrame(
        {
            "term": terms,
            "coefficient": coef,
            "se": standard_errors,
            "t_stat": t_stats,
            "p_value": p_values,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    formula_table.to_csv(args.output, index=False)

    top_terms = formula_table.assign(abs_t=np.abs(formula_table["t_stat"]))
    top_terms = top_terms.sort_values("abs_t", ascending=False).head(10)

    print(f"rows={len(analysis)}")
    print(f"r2={r2_score(y, fitted):.6f}")
    print(f"rmse={mean_squared_error(y, fitted, squared=False):.6f}")
    print("top_terms=")
    for _, row in top_terms.iterrows():
        print(f"  {row['term']}: coef={row['coefficient']:.6g}, t={row['t_stat']:.4f}")
    print(f"formula_table={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
