"""Fit wave5 Track A measurement-based and efficiency-target models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from rfm_pipeline.wave5_track_a import (
    aggregate_wave5_replicates,
    build_wave5_measurement_frame,
    select_model_feature_columns,
)


def _group_cv_predict(
    estimator_factory,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    predictions = np.full_like(y, fill_value=np.nan, dtype=float)
    n_splits = min(8, len(np.unique(groups)))
    splitter = GroupKFold(n_splits=n_splits)
    for train_idx, test_idx in splitter.split(X, y, groups):
        estimator = estimator_factory()
        estimator.fit(X[train_idx], y[train_idx])
        predictions[test_idx] = estimator.predict(X[test_idx])
    return predictions


def _metrics(pred: np.ndarray, obs: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(obs, pred)),
        "rmse": float(np.sqrt(mean_squared_error(obs, pred))),
        "mae": float(mean_absolute_error(obs, pred)),
    }


def _gp_factory(n_features: int) -> Pipeline:
    kernel = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(
        length_scale=np.ones(n_features), length_scale_bounds=(1e-2, 1e2), nu=1.5
    ) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1e0))
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "gp",
                GaussianProcessRegressor(
                    kernel=kernel,
                    normalize_y=True,
                    optimizer=None,
                    random_state=0,
                ),
            ),
        ]
    )


def _rf_factory() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=500,
        min_samples_leaf=5,
        random_state=0,
        n_jobs=-1,
    )


def main() -> None:
    """Fit Track A models on wave5 measurements and write summaries."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("artifacts/sensitivity/wave5_results.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/sensitivity"),
    )
    parser.add_argument("--max-outputs-for-pca", type=int, default=800)
    parser.add_argument("--max-inputs-for-corr", type=int, default=200)
    parser.add_argument("--max-rows-for-corr", type=int, default=10_000)
    parser.add_argument("--proxy-n-runs-cap", type=int, default=4_000)
    parser.add_argument("--proxy-n-outputs-cap", type=int, default=1_000)
    args = parser.parse_args()

    results = pd.read_csv(args.input)
    measured = build_wave5_measurement_frame(
        results,
        proxy_n_runs_cap=args.proxy_n_runs_cap,
        proxy_n_outputs_cap=args.proxy_n_outputs_cap,
        max_outputs_for_pca=args.max_outputs_for_pca,
        max_inputs_for_corr=args.max_inputs_for_corr,
        max_rows_for_corr=args.max_rows_for_corr,
    )
    aggregated = aggregate_wave5_replicates(measured)

    feature_cols = [
        column
        for column in select_model_feature_columns(aggregated.columns)
        if column in aggregated.columns
    ]
    groups = aggregated["dgp_idx"].to_numpy()
    X = aggregated[feature_cols].to_numpy(dtype=float)

    outputs: dict[str, dict[str, float]] = {}

    y_nrmse = aggregated["nrmse_final"].to_numpy(dtype=float)
    gp_pred = _group_cv_predict(lambda: _gp_factory(X.shape[1]), X, y_nrmse, groups)
    rf_pred = _group_cv_predict(_rf_factory, X, y_nrmse, groups)
    outputs["nrmse_final_gp"] = _metrics(gp_pred, y_nrmse)
    outputs["nrmse_final_rf"] = _metrics(rf_pred, y_nrmse)

    y_eta = aggregated["eta_total"].to_numpy(dtype=float)
    eta_gp_pred = _group_cv_predict(lambda: _gp_factory(X.shape[1]), X, y_eta, groups)
    eta_rf_pred = _group_cv_predict(_rf_factory, X, y_eta, groups)
    outputs["eta_total_gp"] = _metrics(eta_gp_pred, y_eta)
    outputs["eta_total_rf"] = _metrics(eta_rf_pred, y_eta)

    eta_gp_clipped = np.clip(eta_gp_pred, -0.5, 1.5)
    eta_rf_clipped = np.clip(eta_rf_pred, -0.5, 1.5)
    nrmse_oracle = aggregated["nrmse_oracle"].to_numpy(dtype=float)
    nrmse_null = aggregated["nrmse_null"].to_numpy(dtype=float)
    gap = nrmse_null - nrmse_oracle
    outputs["nrmse_from_eta_gp"] = _metrics(nrmse_null - eta_gp_clipped * gap, y_nrmse)
    outputs["nrmse_from_eta_rf"] = _metrics(nrmse_null - eta_rf_clipped * gap, y_nrmse)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "dgp_idx": aggregated["dgp_idx"],
            "config_idx": aggregated["config_idx"],
            "nrmse_final": y_nrmse,
            "nrmse_pred_gp": gp_pred,
            "nrmse_pred_rf": rf_pred,
            "eta_total": y_eta,
            "eta_pred_gp": eta_gp_pred,
            "eta_pred_rf": eta_rf_pred,
        }
    ).to_csv(args.output_dir / "wave5_track_a_predictions.csv", index=False)
    (args.output_dir / "wave5_track_a_summary.json").write_text(
        json.dumps(
            {
                "feature_cols": feature_cols,
                "row_count": int(len(aggregated)),
                "group_count": int(len(np.unique(groups))),
                "metrics": outputs,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(outputs, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
