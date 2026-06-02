#!/usr/bin/env python3
# Copyright (c) 2026 Dylan Hettinger
"""Fit an OLS polynomial response surface for sensitivity-study results.

For degree=2 (default), plain OLS is used and t-statistics are reliable.
For degree>=3, LASSO-then-OLS is used: LASSO selects a parsimonious subset
of terms (via cross-validated alpha), then OLS is refit on those terms for
interpretable coefficients. Cross-validated R² and RMSE are always reported.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

try:
    from sklearn.metrics import root_mean_squared_error as _rmse_fn
except ImportError:

    def _rmse_fn(y_true: np.ndarray, y_pred: np.ndarray) -> float:  # type: ignore[misc]
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))


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
    parser.add_argument(
        "--degree",
        type=int,
        default=2,
        choices=[2, 3, 4],
        help="Polynomial degree (default: 2). Degree>=3 uses LASSO+OLS.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Report cross-validated performance across degrees 2 and 3 then exit.",
    )
    parser.add_argument(
        "--save-scaler",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Save scaler metadata (means, stds, predictor names, degree) to a JSON file "
            "alongside the formula CSV. Required for degree>=3 predictions on new data. "
            "Ignored for degree<=2 (raw-scale OLS needs no scaler)."
        ),
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


def _cv_r2_rmse(
    x: np.ndarray,
    y: np.ndarray,
    degree: int,
    n_splits: int = 10,
    random_state: int = 42,
) -> tuple[float, float]:
    """Return cross-validated R² and RMSE using the appropriate estimator."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    preds = np.zeros_like(y, dtype=float)
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    for tr, va in kf.split(x):
        D_tr = poly.fit_transform(x[tr])
        D_va = poly.transform(x[va])
        if degree <= 2:
            coef, _, _, _ = np.linalg.lstsq(D_tr, y[tr], rcond=None)
            preds[va] = D_va @ coef
        else:
            sc = StandardScaler()
            D_tr_s = sc.fit_transform(D_tr)
            D_va_s = sc.transform(D_va)
            lc = LassoCV(cv=5, max_iter=5000, n_jobs=-1)
            lc.fit(D_tr_s, y[tr])
            sel = np.where(np.abs(lc.coef_) > 0)[0]
            if len(sel) == 0:
                preds[va] = y[tr].mean()
            else:
                lr = LinearRegression(fit_intercept=True)
                lr.fit(D_tr_s[:, sel], y[tr])
                preds[va] = lr.predict(D_va_s[:, sel])
    return float(r2_score(y, preds)), float(_rmse_fn(y, preds))


def _fit_lasso_ols(
    design_scaled: np.ndarray,
    y: np.ndarray,
    feature_names: np.ndarray,
) -> pd.DataFrame:
    """LASSO term selection + OLS refit; return coefficient table."""
    lasso = LassoCV(cv=10, max_iter=10000, n_jobs=-1, random_state=42)
    lasso.fit(design_scaled, y)
    sel = np.where(np.abs(lasso.coef_) > 0)[0]

    D_sel = np.hstack([np.ones((len(y), 1)), design_scaled[:, sel]])
    coef, _, _, _ = np.linalg.lstsq(D_sel, y, rcond=None)
    fitted = D_sel @ coef
    n_obs, n_terms = D_sel.shape
    dof = max(1, n_obs - n_terms)
    rss = float(np.sum((y - fitted) ** 2))
    sigma2 = rss / dof
    xtx_inv = np.linalg.pinv(D_sel.T @ D_sel)
    se = np.sqrt(np.maximum(np.diag(xtx_inv) * sigma2, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_vals = np.divide(coef, se, out=np.zeros_like(coef), where=se > 0)
    p_vals = 2.0 * stats.t.sf(np.abs(t_vals), df=dof)

    selected_names = np.concatenate([["intercept"], feature_names[sel]])
    table = pd.DataFrame(
        {
            "term": selected_names,
            "coefficient": coef,
            "se": se,
            "t_stat": t_vals,
            "p_value": p_vals,
            "lasso_selected": True,
        }
    )
    table.attrs["n_candidates"] = len(feature_names)
    table.attrs["n_selected"] = len(sel)
    table.attrs["lasso_alpha"] = float(lasso.alpha_)
    return table, fitted


def main() -> int:
    """Fit and report the polynomial meta-regression response surface."""
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
    analysis = analysis.dropna(subset=predictors)
    x = analysis[predictors].to_numpy(dtype=float)
    y = analysis["nrmse_relative"].to_numpy(dtype=float)
    n_obs = len(y)

    if args.compare:
        print(f"rows={n_obs}")
        for deg in [2, 3]:
            poly_tmp = PolynomialFeatures(degree=deg, include_bias=False)
            n_terms = poly_tmp.fit_transform(x).shape[1]
            r2_cv, rmse_cv = _cv_r2_rmse(x, y, deg)
            method = "OLS" if deg <= 2 else "LASSO+OLS"
            print(
                f"degree={deg} ({method}): terms={n_terms}"
                f"  CV_R2={r2_cv:.4f}  CV_RMSE={rmse_cv:.5f}"
            )
        return 0

    degree = args.degree
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    design = poly.fit_transform(x)
    poly_noconst = PolynomialFeatures(degree=degree, include_bias=False)
    poly_noconst.fit(x)
    feature_names = poly_noconst.get_feature_names_out(predictors)

    print(f"rows={n_obs}  degree={degree}  terms_total={design.shape[1]}")

    # Cross-validated performance (always reported).
    r2_cv, rmse_cv = _cv_r2_rmse(x, y, degree)
    print(f"CV_R2={r2_cv:.4f}  CV_RMSE={rmse_cv:.5f}")

    if degree <= 2:
        # Plain OLS — reliable t-statistics.
        coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        fitted = design @ coef
        n_terms = design.shape[1]
        dof = max(1, n_obs - n_terms)
        rss = float(np.sum((y - fitted) ** 2))
        sigma2 = rss / dof
        xtx_inv = np.linalg.pinv(design.T @ design)
        se = np.sqrt(np.maximum(np.diag(xtx_inv) * sigma2, 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            t_vals = np.divide(coef, se, out=np.zeros_like(coef), where=se > 0)
        p_vals = 2.0 * stats.t.sf(np.abs(t_vals), df=dof)
        terms = poly.get_feature_names_out(predictors)
        formula_table = pd.DataFrame(
            {
                "term": terms,
                "coefficient": coef,
                "se": se,
                "t_stat": t_vals,
                "p_value": p_vals,
            }
        )
        top_col = "t_stat"
    else:
        # Degree ≥ 3: LASSO selection then OLS refit on selected terms.
        scaler = StandardScaler()
        design_scaled = scaler.fit_transform(design[:, 1:])  # skip bias col
        formula_table, fitted = _fit_lasso_ols(design_scaled, y, feature_names)
        n_sel = formula_table.attrs.get("n_selected", "?")
        n_cand = formula_table.attrs.get("n_candidates", "?")
        alpha_lasso = formula_table.attrs.get("lasso_alpha", "?")
        print(f"LASSO selected {n_sel}/{n_cand} terms  (alpha={alpha_lasso:.5f})")
        top_col = "t_stat"

        if args.save_scaler is not None:
            scaler_meta = {
                "degree": degree,
                "predictors": predictors,
                "poly_feature_names": feature_names.tolist(),
                "mean_": scaler.mean_.tolist(),
                "scale_": scaler.scale_.tolist(),
                "var_": scaler.var_.tolist(),
                "n_features_in_": int(scaler.n_features_in_),
                "note": (
                    f"Scaler applies to the degree-{degree} polynomial features EXCLUDING the "
                    f"intercept column.  To apply: build PolynomialFeatures(degree={degree}, "
                    "include_bias=True).fit_transform(x), drop column 0 (bias), then "
                    "StandardScaler with these mean_/scale_ values."
                ),
            }
            scaler_path = Path(args.save_scaler)
            scaler_path.parent.mkdir(parents=True, exist_ok=True)
            scaler_path.write_text(json.dumps(scaler_meta, indent=2))
            print(f"scaler_metadata={scaler_path}")

    r2_insample = float(r2_score(y, fitted))
    rmse_insample = float(_rmse_fn(y, fitted))
    print(f"R2_insample={r2_insample:.4f}  RMSE_insample={rmse_insample:.5f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    formula_table.to_csv(args.output, index=False)

    top_terms = formula_table.assign(abs_t=np.abs(formula_table[top_col]))
    top_terms = top_terms.sort_values("abs_t", ascending=False).head(10)
    print("top_terms=")
    for _, row in top_terms.iterrows():
        print(f"  {row['term']}: coef={row['coefficient']:.6g}, t={row[top_col]:.4f}")
    print(f"formula_table={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
