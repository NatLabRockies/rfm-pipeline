"""Executable manuscript-reproduction stages for the BSM case study.

The functions in this module implement source-level stage computations that are shared by the
tracked manuscript notebooks and automated tests. They operate on the artifact tables resolved by
``bsm_rfm.manuscript_runtime`` and write deterministic intermediate artifacts under the resolved
manuscript output root.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OutputConditioningSpec:
    """Frozen output-conditioning settings from the manuscript case-study contract.

    Parameters
    ----------
    epsilon_var
        Minimum training-set output variance required for a scalar output to be retained.
    epsilon_snr
        Minimum relative dynamic range required for a scalar output to be retained.
    snr_delta
        Stability constant in the denominator of the relative dynamic-range calculation.
    method
        Temporary response-reduction method. Phase 3 currently implements ``"pca"``.
    retained_components
        Case-study maximum number of principal components retained for screening.
    retained_variance_fraction
        Target cumulative explained-variance fraction for the retained PCA scores.
    """

    epsilon_var: float
    epsilon_snr: float
    snr_delta: float
    method: str
    retained_components: int
    retained_variance_fraction: float


@dataclass(frozen=True)
class OutputConditioningResult:
    """Materialized output-conditioning artifacts for one manuscript run.

    Parameters
    ----------
    retained_output_names
        Output columns retained after variance and relative dynamic-range filtering.
    culled_output_names
        Output columns removed by the conditioning filters.
    output_filter_diagnostics
        Per-output variance, range, relative dynamic range, and retention diagnostics.
    pca_scores
        Principal-component scores for every row in the response artifact.
    pca_loadings
        PCA loading matrix, with one row per retained scalar output.
    pca_explained_variance
        Explained-variance table for retained components.
    summary
        One-row summary of the conditioning stage.
    """

    retained_output_names: tuple[str, ...]
    culled_output_names: tuple[str, ...]
    output_filter_diagnostics: pd.DataFrame
    pca_scores: pd.DataFrame
    pca_loadings: pd.DataFrame
    pca_explained_variance: pd.DataFrame
    summary: pd.DataFrame


@dataclass(frozen=True)
class OutputConditioningStageResult:
    """Output-conditioning result plus paths written for notebook handoff.

    Parameters
    ----------
    conditioning
        In-memory output-conditioning result.
    artifact_paths
        Mapping from stable artifact name to the CSV path written under the output root.
    """

    conditioning: OutputConditioningResult
    artifact_paths: dict[str, Path]


def output_conditioning_spec_from_case_study_config(
    case_study_config: dict[str, Any],
) -> OutputConditioningSpec:
    """Build the output-conditioning specification from the case-study YAML payload.

    Parameters
    ----------
    case_study_config
        Parsed ``configs/manuscript_case_study.yml`` mapping.

    Returns
    -------
    OutputConditioningSpec
        Typed output-conditioning specification.
    """
    section = case_study_config["case_study"]["output_conditioning"]
    variance_filter = section["variance_filter"]
    snr_filter = section["snr_filter"]
    reduction = section["temporary_reduction"]
    return OutputConditioningSpec(
        epsilon_var=float(variance_filter["epsilon_var"]),
        epsilon_snr=float(snr_filter["epsilon_snr"]),
        snr_delta=float(snr_filter["delta"]),
        method=str(reduction["method"]),
        retained_components=int(reduction["retained_components"]),
        retained_variance_fraction=float(reduction["retained_variance_fraction"]),
    )


def condition_manuscript_outputs(
    output_matrix: pd.DataFrame,
    holdout_assignments: pd.DataFrame,
    spec: OutputConditioningSpec,
) -> OutputConditioningResult:
    """Filter scalar outputs and compute the manuscript PCA reduced response.

    The conditioning rule is intentionally deterministic and train-only: variance, relative dynamic
    range, standardization parameters, PCA loadings, and the retained component count are
    learned from rows whose split is ``"train"`` in ``holdout_assignments``. The fitted
    loadings are then applied to every row in ``output_matrix`` so downstream notebooks can
    join the scores back to the fixed holdout split.

    Parameters
    ----------
    output_matrix
        Case-study scalar-output table with a ``sample_id`` column and one column per scalar output.
    holdout_assignments
        Table with ``sample_id`` and ``split`` columns.
    spec
        Output-conditioning specification.

    Returns
    -------
    OutputConditioningResult
        Materialized filter diagnostics, PCA scores, loadings, explained variance, and summary.

    Raises
    ------
    ValueError
        Raised when inputs are malformed, all outputs are culled, or PCA is not identifiable.
    """
    if spec.method != "pca":
        raise ValueError(f"Unsupported output-conditioning method: {spec.method}")

    output_columns = _output_columns(output_matrix)
    train_sample_ids = _train_sample_ids(holdout_assignments)
    all_response = _align_output_matrix(output_matrix, output_matrix["sample_id"], output_columns)
    train_response = _align_output_matrix(output_matrix, train_sample_ids, output_columns)
    if train_response.shape[0] < 2:
        raise ValueError("Output conditioning requires at least two training rows for PCA.")

    variances = train_response.var(axis=0, ddof=0)
    dynamic_ranges = train_response.max(axis=0) - train_response.min(axis=0)
    relative_dynamic_ranges = dynamic_ranges / (train_response.mean(axis=0).abs() + spec.snr_delta)
    retained_mask = (variances > spec.epsilon_var) & (relative_dynamic_ranges >= spec.epsilon_snr)
    retained_output_names = tuple(str(name) for name in variances.index[retained_mask])
    culled_output_names = tuple(str(name) for name in variances.index[~retained_mask])
    if not retained_output_names:
        raise ValueError("Output conditioning culled every scalar output.")

    diagnostics = _build_filter_diagnostics(
        output_columns=output_columns,
        variances=variances,
        dynamic_ranges=dynamic_ranges,
        relative_dynamic_ranges=relative_dynamic_ranges,
        retained_mask=retained_mask,
        spec=spec,
    )
    train_retained = train_response.loc[:, list(retained_output_names)]
    all_retained = all_response.loc[:, list(retained_output_names)]
    standardized_train, means, scales = _standardize_train_response(train_retained)
    standardized_all = (all_retained - means) / scales

    scores, loadings, explained = _fit_pca_reduction(
        standardized_train=standardized_train,
        standardized_all=standardized_all,
        sample_ids=output_matrix["sample_id"].reset_index(drop=True),
        retained_output_names=retained_output_names,
        spec=spec,
    )
    summary = _build_output_conditioning_summary(
        n_outputs_total=len(output_columns),
        n_outputs_retained=len(retained_output_names),
        n_training_rows=len(train_response),
        explained=explained,
        spec=spec,
    )
    return OutputConditioningResult(
        retained_output_names=retained_output_names,
        culled_output_names=culled_output_names,
        output_filter_diagnostics=diagnostics,
        pca_scores=scores,
        pca_loadings=loadings,
        pca_explained_variance=explained,
        summary=summary,
    )


def write_output_conditioning_artifacts(
    result: OutputConditioningResult,
    output_root: Path,
) -> dict[str, Path]:
    """Write output-conditioning artifacts under ``output_root/output_conditioning``.

    Parameters
    ----------
    result
        Materialized output-conditioning result.
    output_root
        Resolved manuscript output root from the runtime context.

    Returns
    -------
    dict[str, pathlib.Path]
        Paths keyed by stable artifact name.
    """
    stage_root = output_root / "output_conditioning"
    stage_root.mkdir(parents=True, exist_ok=True)
    tables = {
        "output_filter_diagnostics": result.output_filter_diagnostics,
        "pca_scores": result.pca_scores,
        "pca_loadings": result.pca_loadings,
        "pca_explained_variance": result.pca_explained_variance,
        "output_conditioning_summary": result.summary,
    }
    written: dict[str, Path] = {}
    for name, table in tables.items():
        path = stage_root / f"{name}.csv"
        table.to_csv(path, index=False)
        written[name] = path
    return written


def run_output_conditioning_stage(context: Any) -> OutputConditioningStageResult:
    """Run the manuscript output-conditioning stage from a notebook runtime context.

    Parameters
    ----------
    context
        ``bsm_rfm.manuscript_runtime.ManuscriptNotebookContext``. It is typed as ``Any`` here to
        avoid an import cycle between the runtime and stage modules.

    Returns
    -------
    OutputConditioningStageResult
        In-memory result and written artifact paths.
    """
    spec = output_conditioning_spec_from_case_study_config(context.case_study_config)
    conditioning = condition_manuscript_outputs(
        context.tables["case_study_output_matrix"],
        context.tables["fixed_holdout_assignments"],
        spec,
    )
    artifact_paths = write_output_conditioning_artifacts(
        conditioning,
        context.runtime.output_root,
    )
    return OutputConditioningStageResult(conditioning=conditioning, artifact_paths=artifact_paths)


def _output_columns(output_matrix: pd.DataFrame) -> list[str]:
    """Return scalar-output columns from an output artifact table."""
    if "sample_id" not in output_matrix.columns:
        raise ValueError("output_matrix must include a sample_id column.")
    output_columns = [str(column) for column in output_matrix.columns if column != "sample_id"]
    if not output_columns:
        raise ValueError("output_matrix must include at least one scalar output column.")
    return output_columns


def _train_sample_ids(holdout_assignments: pd.DataFrame) -> pd.Series:
    """Return sample IDs assigned to the train split."""
    required = {"sample_id", "split"}
    missing = sorted(required.difference(holdout_assignments.columns))
    if missing:
        raise ValueError(f"holdout_assignments is missing required columns: {', '.join(missing)}")
    duplicate_ids = holdout_assignments["sample_id"].duplicated(keep=False)
    if duplicate_ids.any():
        raise ValueError("holdout_assignments must contain unique sample_id values.")
    split = holdout_assignments["split"].astype(str).str.lower()
    train_ids = holdout_assignments.loc[split == "train", "sample_id"]
    if train_ids.empty:
        raise ValueError("holdout_assignments must contain at least one train row.")
    return train_ids.reset_index(drop=True)


def _align_output_matrix(
    output_matrix: pd.DataFrame,
    sample_ids: pd.Series,
    output_columns: list[str],
) -> pd.DataFrame:
    """Align and coerce an output matrix to a requested sample-id order."""
    duplicate_ids = output_matrix["sample_id"].duplicated(keep=False)
    if duplicate_ids.any():
        raise ValueError("output_matrix must contain unique sample_id values.")
    indexed = output_matrix.set_index("sample_id", drop=False)
    missing_ids = [sample_id for sample_id in sample_ids if sample_id not in indexed.index]
    if missing_ids:
        preview = ", ".join(str(value) for value in missing_ids[:5])
        raise ValueError(f"output_matrix is missing sample_id values: {preview}")
    aligned = indexed.loc[list(sample_ids), output_columns].reset_index(drop=True)
    numeric = aligned.apply(pd.to_numeric, errors="raise")
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("output_matrix scalar-output values must be finite numeric values.")
    return numeric


def _build_filter_diagnostics(
    *,
    output_columns: list[str],
    variances: pd.Series,
    dynamic_ranges: pd.Series,
    relative_dynamic_ranges: pd.Series,
    retained_mask: pd.Series,
    spec: OutputConditioningSpec,
) -> pd.DataFrame:
    """Build per-output conditioning-filter diagnostics."""
    rows = []
    for output_name in output_columns:
        reasons = []
        if variances.loc[output_name] <= spec.epsilon_var:
            reasons.append("variance")
        if relative_dynamic_ranges.loc[output_name] < spec.epsilon_snr:
            reasons.append("relative_dynamic_range")
        rows.append(
            {
                "output_name": output_name,
                "training_variance": float(variances.loc[output_name]),
                "training_dynamic_range": float(dynamic_ranges.loc[output_name]),
                "training_relative_dynamic_range": float(relative_dynamic_ranges.loc[output_name]),
                "retained": bool(retained_mask.loc[output_name]),
                "cull_reason": "+".join(reasons) if reasons else "retained",
            }
        )
    return pd.DataFrame.from_records(rows)


def _standardize_train_response(
    train_response: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Standardize retained training responses using train-only means and scales."""
    means = train_response.mean(axis=0)
    scales = train_response.std(axis=0, ddof=0).replace(0.0, 1.0)
    standardized = (train_response - means) / scales
    return standardized, means, scales


def _fit_pca_reduction(
    *,
    standardized_train: pd.DataFrame,
    standardized_all: pd.DataFrame,
    sample_ids: pd.Series,
    retained_output_names: tuple[str, ...],
    spec: OutputConditioningSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit train-only PCA loadings and score all rows."""
    matrix = standardized_train.to_numpy(dtype=float)
    _, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)
    eigenvalues = singular_values**2 / max(matrix.shape[0] - 1, 1)
    total_variance = float(eigenvalues.sum())
    if total_variance <= 0.0:
        raise ValueError("Retained outputs have zero total standardized variance.")
    ratios = eigenvalues / total_variance
    cumulative = np.cumsum(ratios)
    needed_for_target = int(np.searchsorted(cumulative, spec.retained_variance_fraction) + 1)
    max_components = min(spec.retained_components, vt.shape[0], vt.shape[1])
    n_components = max(1, min(max_components, needed_for_target))

    loading_values = vt[:n_components].T
    score_values = standardized_all.to_numpy(dtype=float) @ loading_values
    component_names = [f"PC{index}" for index in range(1, n_components + 1)]
    scores = pd.DataFrame(score_values, columns=component_names)
    scores.insert(0, "sample_id", sample_ids.reset_index(drop=True))

    loadings = pd.DataFrame(loading_values, columns=component_names)
    loadings.insert(0, "output_name", list(retained_output_names))
    explained = pd.DataFrame(
        {
            "component": component_names,
            "explained_variance": eigenvalues[:n_components],
            "explained_variance_ratio": ratios[:n_components],
            "cumulative_variance_ratio": cumulative[:n_components],
        }
    )
    return scores, loadings, explained


def _build_output_conditioning_summary(
    *,
    n_outputs_total: int,
    n_outputs_retained: int,
    n_training_rows: int,
    explained: pd.DataFrame,
    spec: OutputConditioningSpec,
) -> pd.DataFrame:
    """Build the one-row output-conditioning summary table."""
    return pd.DataFrame(
        [
            {
                "stage": "output_conditioning",
                "method": spec.method,
                "n_training_rows": int(n_training_rows),
                "n_outputs_total": int(n_outputs_total),
                "n_outputs_retained": int(n_outputs_retained),
                "n_outputs_culled": int(n_outputs_total - n_outputs_retained),
                "n_components_retained": int(len(explained)),
                "variance_fraction_retained": float(
                    explained["cumulative_variance_ratio"].iloc[-1]
                ),
                "epsilon_var": float(spec.epsilon_var),
                "epsilon_snr": float(spec.epsilon_snr),
                "snr_delta": float(spec.snr_delta),
                "target_variance_fraction": float(spec.retained_variance_fraction),
                "configured_component_cap": int(spec.retained_components),
            }
        ]
    )
