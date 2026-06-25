"""Wave 5 Track A measurement-based meta-model helpers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import fields

import numpy as np
import pandas as pd

from .dgp_measurements import DGPMeasurements, measure_dataset
from .synthetic_dgp import SyntheticDGPSpec, generate_calibrated_structure_synthetic

MEASUREMENT_COLUMNS = [field.name for field in fields(DGPMeasurements)]
MODEL_FEATURE_COLUMNS = [
    "n_rows",
    "n_inputs",
    "n_outputs",
    "output_std_over_range_mean",
    "output_std_over_range_std",
    "output_kurtosis_mean",
    "output_kurtosis_median",
    "output_skewness_abs_mean",
    "output_spectrum_top1_share",
    "output_spectrum_top5_share",
    "output_spectrum_effective_rank",
    "output_spectrum_decay_exponent",
    "input_correlation_offdiag_mean_abs",
    "input_correlation_effective_rank",
    "marginal_xy_corr_mean_abs",
    "marginal_xy_corr_top1pct_mean_abs",
    "marginal_xy_corr_pareto_alpha",
]
TRACK_A_CONFIG_COLUMNS = [
    "holdout_fraction",
    "variance_threshold",
    "stages.empirical_null_screening.n_permutations",
    "stages.empirical_null_screening.bh_q_threshold",
    "stages.interaction_discovery.n_permutations",
    "stages.interaction_discovery.p_threshold",
    "stages.sparse_selection.n_stability_subsamples",
    "stages.sparse_selection.lasso_alpha_grid_size",
    "stages.final_artifacts.delta_threshold_override",
]


def compute_efficiency_targets(frame: pd.DataFrame) -> pd.DataFrame:
    """Add oracle-gap efficiency targets to a wave5 results frame."""
    required = {"nrmse_null", "nrmse_final", "noise_snr"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"missing required columns for efficiency targets: {missing}")

    out = frame.copy()
    out["nrmse_oracle"] = out["nrmse_null"] * np.sqrt(1.0 / (out["noise_snr"] + 1.0))
    out["oracle_gap"] = out["nrmse_null"] - out["nrmse_oracle"]
    with np.errstate(divide="ignore", invalid="ignore"):
        out["eta_total"] = (out["nrmse_null"] - out["nrmse_final"]) / out["oracle_gap"]
    out.loc[~np.isfinite(out["eta_total"]), "eta_total"] = np.nan
    return out


def select_model_feature_columns(columns: Iterable[str]) -> list[str]:
    """Return the Track A feature set present in the provided columns."""
    column_set = set(columns)
    return [
        column for column in MODEL_FEATURE_COLUMNS + TRACK_A_CONFIG_COLUMNS if column in column_set
    ]


def build_wave5_measurement_frame(
    results: pd.DataFrame,
    *,
    proxy_n_runs_cap: int = 4_000,
    proxy_n_outputs_cap: int = 1_000,
    max_outputs_for_pca: int = 800,
    max_inputs_for_corr: int = 200,
    max_rows_for_corr: int = 10_000,
) -> pd.DataFrame:
    """Measure each unique DGP in wave5 and merge the measurements back.

    The synthetic DGPs are generated with capped proxy sizes so the local fit
    stays tractable while preserving the structural knobs that drive the
    measurement vector.
    """
    required = {
        "dgp_idx",
        "n_inputs",
        "n_runs",
        "n_outputs",
        "sparsity",
        "interaction_density",
        "nonlinearity_strength",
        "noise_snr",
        "holdout_fraction",
        "dgp_family",
        "seed",
        "factor_model_rank",
        "input_correlation_strength",
        "factor_signal_weight",
        "output_scale_heterogeneity",
        "output_nonlinearity_strength",
        "per_output_snr_heterogeneity",
        "active_input_beta_concentration",
    }
    missing = sorted(required.difference(results.columns))
    if missing:
        raise KeyError(f"missing required wave5 columns: {missing}")

    unique_dgps = results.sort_values("dgp_idx").drop_duplicates("dgp_idx")
    measurement_rows: list[dict[str, float | int]] = []
    for _, row in unique_dgps.iterrows():
        spec = SyntheticDGPSpec(
            n_inputs=int(row["n_inputs"]),
            n_runs=min(int(row["n_runs"]), proxy_n_runs_cap),
            n_outputs=min(int(row["n_outputs"]), proxy_n_outputs_cap),
            sparsity=float(row["sparsity"]),
            interaction_density=float(row["interaction_density"]),
            nonlinearity_strength=float(row["nonlinearity_strength"]),
            noise_snr=float(row["noise_snr"]),
            holdout_fraction=float(row["holdout_fraction"]),
            dgp_family=str(row["dgp_family"]),
            seed=int(row["seed"]),
            factor_model_rank=int(row["factor_model_rank"]),
            input_correlation_strength=float(row["input_correlation_strength"]),
            factor_signal_weight=float(row["factor_signal_weight"]),
            output_scale_heterogeneity=float(row["output_scale_heterogeneity"]),
            output_nonlinearity_strength=float(row["output_nonlinearity_strength"]),
            per_output_snr_heterogeneity=float(row["per_output_snr_heterogeneity"]),
            active_input_beta_concentration=float(row["active_input_beta_concentration"]),
        )
        dataset = generate_calibrated_structure_synthetic(spec)
        measurements = measure_dataset(
            dataset.input_matrix,
            dataset.output_matrix,
            max_outputs_for_pca=max_outputs_for_pca,
            max_inputs_for_corr=max_inputs_for_corr,
            max_rows_for_corr=max_rows_for_corr,
            random_state=spec.seed,
        ).to_dict()
        measurement_rows.append({"dgp_idx": int(row["dgp_idx"]), **measurements})

    measurements_frame = pd.DataFrame(measurement_rows)
    merged = results.merge(measurements_frame, on="dgp_idx", how="left", validate="many_to_one")
    return compute_efficiency_targets(merged)


def aggregate_wave5_replicates(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse replicate rows to one row per DGP/config pair."""
    group_cols = ["dgp_idx", "config_idx"]
    numeric_cols = frame.select_dtypes(include=["number"]).columns.difference(group_cols)
    grouped = frame.groupby(group_cols, as_index=False)[list(numeric_cols)].mean(numeric_only=True)
    keep_cols = [col for col in frame.columns if col not in numeric_cols and col not in group_cols]
    keep_cols = [col for col in keep_cols if col in frame.columns]
    return grouped.merge(
        frame[group_cols + keep_cols].drop_duplicates(group_cols),
        on=group_cols,
        how="left",
    )
