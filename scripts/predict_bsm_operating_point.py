#!/usr/bin/env python3
# Copyright (c) 2026 Dylan Hettinger
"""Predict nRMSE at the BSM case-study operating point using the meta-regression formula.

Two formulas are available:
  - degree-2 (wave1_formula_table.csv): plain OLS on raw polynomial features.
    Reliable t-statistics for interpretation; NOT suitable for out-of-sample
    prediction because mixed-scale predictors (lasso_alpha_percentile O(50)
    vs log-scale predictors O(1-10)) create large interaction terms that
    extrapolate unstably outside the training joint distribution.
  - degree-3 (wave1_formula_d3.csv + wave1_formula_d3_scaler.json): LASSO+OLS
    on StandardScaler-normalized polynomial features.  More stable for
    prediction; used here for the BSM operating-point validation.

The BSM operating point is constructed from:
  - Config:  configs/validation_full_dataset_final_cost_04.yml
  - Run:     artifacts/publication_full_dataset_distributed_20260526_short_hp1
  - DGP properties: estimated from the BSM's retained model structure

The actual holdout nRMSE (from the ablation table) is used for comparison.

Usage
-----
    pixi run python scripts/predict_bsm_operating_point.py
    pixi run python scripts/predict_bsm_operating_point.py \
        --formula-d3 artifacts/sensitivity/wave1_formula_d3.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Predictor order must match what fit_meta_regression.py used for the wave1 fit.
# ---------------------------------------------------------------------------
PREDICTORS = [
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
    "sparsity",
    "interaction_density",
    "nonlinearity_strength",
    "log_noise_snr",
    "log_n_inputs",
]

# ---------------------------------------------------------------------------
# BSM operating point
# Source of each value is documented inline.
# ---------------------------------------------------------------------------
#
# Config file: configs/validation_full_dataset_final_cost_04.yml
#   holdout_fraction   = 0.10
#   variance_threshold = 0.90  (PCA retained-variance fraction)
#   n_perm_screen      = 101   (empirical_null_screening.n_permutations)
#   bh_q               = 0.10  (empirical_null_screening.bh_q_threshold)
#   n_perm_interaction = 21    (interaction_discovery.n_permutations)
#   p_threshold        = 0.05  (interaction_discovery.p_threshold)
#   n_stability        = 12    (sparse_selection.n_stability_subsamples)
#   lasso_alpha_pct    = 50    (sparse_selection.lasso_alpha_percentile)
#
# BSM run metadata (bsm-public-rf configs/manuscript_case_study_fast_sparse.yml):
#   delta_threshold_override = 0.002
#   n_exogenous_inputs       = 160
#   n_runs                   = 30 000  →  n_training = 30 000 × 0.90 = 27 000
#
# DGP properties (estimated from the BSM's retained model structure):
#   sparsity            ≈ 0.61  (98 of 160 inputs not retained as main effects)
#   interaction_density ≈ 0.032 (403 retained pairs / C(160,2) = 12 720)
#   nonlinearity_strength ≈ 0.25 (40 transformations retained / 160 inputs,
#                                  mapped to sensitivity-sweep [0,1] scale)
#   noise_snr           ≈ 97.0  (BSM is deterministic; using sweep maximum
#                                 as a practical approximation of infinite SNR)


def _bsm_point_dict() -> dict[str, float]:
    n_training = 30_000 * 0.90  # holdout_fraction = 0.10
    delta = 0.002
    return {
        "log_subsample_n": float(np.log(n_training)),
        "holdout_fraction_model": 0.10,
        "variance_threshold_model": 0.90,
        "log_n_perm_screen": float(np.log(101)),
        "bh_q_model": 0.10,
        "log_n_perm_interaction": float(np.log(21)),
        "p_threshold_model": 0.05,
        "log_n_stability_subsamples": float(np.log(12)),
        "lasso_alpha_percentile_model": 50.0,
        "log_delta_threshold": float(np.log(delta + 0.001)),
        # DGP properties (estimated)
        "sparsity": 0.61,
        "interaction_density": 0.032,
        "nonlinearity_strength": 0.25,
        "log_noise_snr": float(np.log(97.0)),
        "log_n_inputs": float(np.log(160)),
    }


def _apply_formula_d3(
    x: np.ndarray,
    formula: pd.DataFrame,
    scaler_meta: dict,
) -> float:
    """Apply degree-3 LASSO+OLS formula on StandardScaler-normalized polynomial features."""
    poly3 = PolynomialFeatures(degree=3, include_bias=True)
    design3 = poly3.fit_transform(x)
    design3_scaled = (design3[:, 1:] - np.array(scaler_meta["mean_"])) / np.array(
        scaler_meta["scale_"]
    )
    poly3_nc = PolynomialFeatures(degree=3, include_bias=False)
    poly3_nc.fit(x)
    poly_names = poly3_nc.get_feature_names_out(PREDICTORS)
    coef_map = {
        row["term"]: row["coefficient"]
        for _, row in formula.iterrows()
        if row["term"] != "intercept"
    }
    intercept = float(
        formula.loc[formula["term"] == "intercept", "coefficient"].values[0]
    )
    pred = intercept + sum(
        coef_map[name] * design3_scaled[0, i]
        for i, name in enumerate(poly_names)
        if name in coef_map
    )
    return float(pred)


def _sensitivity_table(
    formula: pd.DataFrame,
    scaler_meta: dict,
    base: dict[str, float],
    bsm_null: float,
) -> pd.DataFrame:
    """Show how the prediction varies with plausible BSM DGP property assumptions."""
    scenarios = {
        "Base estimate (sparsity=0.61, density=0.032, nonlinearity=0.25)": base,
        "Low nonlinearity (0.10)": {**base, "nonlinearity_strength": 0.10},
        "High nonlinearity (0.50)": {**base, "nonlinearity_strength": 0.50},
        "Higher interaction density (0.10)": {**base, "interaction_density": 0.10},
        "Lower sparsity (0.40)": {**base, "sparsity": 0.40},
        "Very high SNR (log(9999))": {**base, "log_noise_snr": float(np.log(9999))},
    }
    rows = []
    for label, overrides in scenarios.items():
        x_row = np.array([[overrides[p] for p in PREDICTORS]])
        pred_rel = _apply_formula_d3(x_row, formula, scaler_meta)
        pred_abs = bsm_null * (1.0 + pred_rel)
        rows.append(
            {
                "scenario": label,
                "predicted_nrmse_relative": pred_rel,
                "predicted_nrmse_abs": pred_abs,
            }
        )
    return pd.DataFrame(rows)


def main() -> int:  # noqa: D103
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--formula-d3",
        type=Path,
        default=REPO_ROOT / "artifacts/sensitivity/wave1_formula_d3.csv",
        help="Path to the degree-3 LASSO formula CSV.",
    )
    parser.add_argument(
        "--scaler",
        type=Path,
        default=REPO_ROOT / "artifacts/sensitivity/wave1_formula_d3_scaler.json",
        help="Path to the degree-3 scaler JSON.",
    )
    parser.add_argument(
        "--ablation-table",
        type=Path,
        default=REPO_ROOT
        / "artifacts/publication_full_dataset_distributed_results"
        / "publication_full_dataset_distributed_20260526_short_hp1"
        / "artifacts/final_manuscript_artifacts/tables/ablation_table.csv",
        help="Ablation table CSV for BSM actual nRMSE values.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "artifacts/sensitivity/bsm_operating_point_prediction.csv",
        help="Output CSV with prediction vs actual comparison.",
    )
    args = parser.parse_args()

    formula_d3 = pd.read_csv(args.formula_d3)
    scaler_meta = json.loads(args.scaler.read_text())
    ablation = pd.read_csv(args.ablation_table)

    nrmse = {row["model_name"]: row["nrmse"] for _, row in ablation.iterrows()}
    bsm_null = nrmse["null_mean"]
    bsm_final = nrmse["final_ols"]
    bsm_actual_rel = (bsm_final - bsm_null) / bsm_null

    print("=== BSM actual performance ===")
    print(f"  null_mean nRMSE  : {bsm_null:.4f}")
    print(f"  final_ols nRMSE  : {bsm_final:.4f}")
    print(f"  nrmse_relative   : {bsm_actual_rel:.4f}  "
          f"({bsm_actual_rel * 100:.1f}% change vs null)")

    base_dict = _bsm_point_dict()
    x_bsm = np.array([[base_dict[p] for p in PREDICTORS]])
    pred_rel = _apply_formula_d3(x_bsm, formula_d3, scaler_meta)
    pred_abs = bsm_null * (1.0 + pred_rel)

    print()
    print("=== Degree-3 LASSO meta-regression prediction at BSM operating point ===")
    print("  (degree-3 LASSO+OLS on StandardScaler-normalized polynomial features)")
    print("  (wave 1 sensitivity data: 2,583 runs across 200 DGPs × 50-pt LHC configs)")
    print()
    print("  Hyperparameters (from configs/validation_full_dataset_final_cost_04.yml):")
    print(f"    n_training             : 27,000  (log = {np.log(27000):.3f})")
    print("    holdout_fraction       : 0.10")
    print("    variance_threshold     : 0.90  (PCA retained-variance fraction)")
    print(f"    n_perm_screen          : 101   (log = {np.log(101):.3f})")
    print("    bh_q                   : 0.10")
    print(f"    n_perm_interaction     : 21    (log = {np.log(21):.3f})")
    print("    p_threshold            : 0.05")
    print(f"    n_stability_subsamples : 12    (log = {np.log(12):.3f})")
    print("    lasso_alpha_percentile : 50")
    print(f"    delta_threshold        : 0.002 (log(0.002+0.001) = {np.log(0.003):.3f})")
    print()
    print("  DGP properties (estimated from retained BSM model structure):")
    print("    sparsity               : 0.61  (98/160 inputs not retained)")
    print("    interaction_density    : 0.032 (403/12,720 input pairs retained)")
    print("    nonlinearity_strength  : 0.25  (40/160 inputs have nonlinear transforms)")
    print("    noise_snr              : 97.0  (deterministic; sweep maximum used)")
    print(f"    n_inputs               : 160   (log = {np.log(160):.3f})")
    print()
    print(f"  Predicted nrmse_relative : {pred_rel:.4f}  ({pred_rel * 100:.1f}%)")
    print(f"  Predicted nRMSE (abs)    : {pred_abs:.4f}")
    print(f"  Actual nRMSE (abs)       : {bsm_final:.4f}")
    abs_err = abs(pred_abs - bsm_final)
    rel_err = abs_err / bsm_final
    print(f"  Absolute error           : {abs_err:.4f}  ({rel_err * 100:.1f}% of actual)")

    print()
    print("=== Sensitivity to DGP property assumptions ===")
    sens = _sensitivity_table(formula_d3, scaler_meta, base_dict, bsm_null)
    for _, row in sens.iterrows():
        print(f"  [{row['scenario'][:65]}]")
        print(f"    rel={row['predicted_nrmse_relative']:.4f}"
              f"  abs_nrmse={row['predicted_nrmse_abs']:.4f}")

    result = pd.DataFrame(
        [
            {
                "source": "meta_regression_degree3_lasso_wave1",
                "null_nrmse_actual": bsm_null,
                "final_nrmse_actual": bsm_final,
                "nrmse_relative_actual": bsm_actual_rel,
                "nrmse_relative_predicted": pred_rel,
                "final_nrmse_predicted": pred_abs,
                "absolute_error": abs_err,
                "relative_error_pct": rel_err * 100,
                "n_training": 27_000,
                "holdout_fraction": 0.10,
                "variance_threshold": 0.90,
                "n_perm_screen": 101,
                "bh_q": 0.10,
                "n_perm_interaction": 21,
                "p_threshold": 0.05,
                "n_stability_subsamples": 12,
                "lasso_alpha_percentile": 50,
                "delta_threshold": 0.002,
                "sparsity_estimated": 0.61,
                "interaction_density_estimated": 0.032,
                "nonlinearity_strength_estimated": 0.25,
                "noise_snr_estimated": 97.0,
                "n_inputs": 160,
            }
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"\nbsm_prediction={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
