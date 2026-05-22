"""Executable manuscript-reproduction stages for the BSM case study.

The functions in this module implement source-level stage computations that are shared by the
tracked manuscript notebooks and automated tests. They operate on the artifact tables resolved by
``bsm_rfm.manuscript_runtime`` and write deterministic intermediate artifacts under the resolved
manuscript output root.
"""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing as mp
import os
import re
import subprocess
import tempfile
import time
import warnings
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, MultiTaskElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd

from .features import canonical_module_from_factor_name
from .final_ols import (
    fit_final_ols,
    make_coefficient_matrix_frame,
    make_standardization_frame,
    predict_final_ols,
)
from .metrics import (
    bootstrap_macro_nrmse_ci,
    macro_nrmse_with_ref,
    make_null_mean_prediction,
    per_output_nrmse_frame,
)
from .parallel import get_executor

_PROGRESS_TELEMETRY_PATH: Path | None = None


def configure_progress_telemetry(path: Path | None) -> None:
    """Configure optional JSON progress telemetry output path."""
    global _PROGRESS_TELEMETRY_PATH
    _PROGRESS_TELEMETRY_PATH = path


def _report_progress(
    *,
    stage: str,
    completed: int,
    total: int,
    unit: str,
    detail: str | None = None,
) -> None:
    """Write best-effort fine-grained stage progress telemetry."""
    if _PROGRESS_TELEMETRY_PATH is None:
        return
    payload = {
        "recorded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stage": stage,
        "completed": int(completed),
        "total": int(total),
        "unit": unit,
        "progress_fraction": (float(completed) / float(total)) if total > 0 else 1.0,
    }
    if detail:
        payload["detail"] = detail
    path = _PROGRESS_TELEMETRY_PATH
    assert path is not None
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _progress_batch_size(total: int, default_cap: int) -> int:
    """Choose progress batch size with optional high-fidelity override."""
    if total <= 0:
        return 1
    override = os.getenv("BSM_PROGRESS_BATCH_SIZE")
    if override:
        try:
            return max(1, min(total, int(override)))
        except ValueError:
            pass
    return max(1, min(default_cap, total // 20 if total > 20 else total))


def _coerce_bool(value: Any, *, default: bool) -> bool:
    """Coerce mixed config values to bool using predictable string handling."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
        return default
    return bool(value)


def _append_parallel_diagnostic(stage: str, payload: dict[str, Any]) -> None:
    """Append structured parallel diagnostics for post-mortem hang analysis."""
    if _PROGRESS_TELEMETRY_PATH is None:
        return
    diagnostics_path = _PROGRESS_TELEMETRY_PATH.parent / f"{stage}_parallel_diagnostics.jsonl"
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "recorded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **payload,
    }
    with diagnostics_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _worker_process_snapshot(parent_pid: int) -> list[dict[str, str]]:
    """Capture lightweight worker-process snapshot for diagnostics."""
    try:
        output = subprocess.check_output(
            ["ps", "-ax", "-o", "pid,ppid,%cpu,%mem,command"],
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    snapshot: list[dict[str, str]] = []
    for line in output.splitlines()[1:]:
        parts = line.strip().split(None, 4)
        if len(parts) < 5:
            continue
        pid_s, ppid_s, cpu_s, mem_s, cmd = parts
        try:
            if int(ppid_s) != parent_pid:
                continue
        except ValueError:
            continue
        snapshot.append(
            {
                "pid": pid_s,
                "ppid": ppid_s,
                "cpu_percent": cpu_s,
                "mem_percent": mem_s,
                "command": cmd,
            }
        )
    return snapshot


def _checkpoint_run_dir(
    checkpoint_dir: Path | None,
    *,
    stage: str,
    signature_payload: dict[str, Any],
) -> Path | None:
    """Resolve deterministic checkpoint directory for a stage/signature payload."""
    if checkpoint_dir is None:
        return None
    payload = {"stage": stage, **signature_payload}
    signature = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    run_dir = checkpoint_dir / signature
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = run_dir / "checkpoint_metadata.json"
    if not metadata_path.exists():
        metadata_path.write_text(
            json.dumps(
                {
                    "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "signature": signature,
                    **payload,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return run_dir


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically to avoid partial checkpoint writes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(path)


def _read_json_dict(path: Path) -> dict[str, Any] | None:
    """Read JSON dictionary payload; return None on malformed or missing files."""
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


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


@dataclass(frozen=True)
class EmpiricalNullScreeningSpec:
    """Frozen empirical-null screening settings from the manuscript contract.

    Parameters
    ----------
    statistic
        Screening statistic. The BSM case study uses ``"coefficient_row_l2_norm"``.
    permutation_count_B
        Number of response permutations used to estimate the empirical null.
    bh_q_screen
        Benjamini--Hochberg false-discovery-rate threshold for screening.
    retained_terms_reference
        Manuscript-reported retained-term count for the full case study.
    implementation_method
        Public implementation used by this repository for empirical-null screening.
    implementation_status
        Alignment status of the public implementation relative to the manuscript contract.
    source_script_reference
        Name or identifier for the private/upstream screening script, if known.
    source_script_equivalence_status
        Validation status against the private/upstream screening script.
    random_seed
        Deterministic random seed for the permutation sequence.
    """

    statistic: str
    permutation_count_B: int
    bh_q_screen: float
    retained_terms_reference: int
    implementation_method: str = "coefficient_row_l2_permutation"
    implementation_status: str = "source_backed_public_surrogate"
    source_script_reference: str = "private_delta_null_screening_script"
    source_script_equivalence_status: str = "not_yet_validated"
    random_seed: int = 123
    n_jobs: int = 1
    max_retained_terms: int | None = None


@dataclass(frozen=True)
class EmpiricalNullScreeningResult:
    """Materialized empirical-null screening artifacts for one manuscript run.

    Parameters
    ----------
    feature_screening_statistics
        Per-feature observed statistic, empirical p-value, BH rank, and retention flag.
    component_coefficients
        Per-feature, per-component standardized coefficient table used in the row norm.
    permutation_null_summary
        Per-feature summary of the permutation-null statistic distribution.
    retained_terms
        Retained term table after BH screening.
    provenance
        One-row provenance table describing the public implementation and private-script
        equivalence status.
    summary
        One-row summary of the screening stage.
    """

    feature_screening_statistics: pd.DataFrame
    component_coefficients: pd.DataFrame
    permutation_null_summary: pd.DataFrame
    retained_terms: pd.DataFrame
    provenance: pd.DataFrame
    summary: pd.DataFrame


@dataclass(frozen=True)
class EmpiricalNullScreeningStageResult:
    """Empirical-null screening result plus paths written for notebook handoff.

    Parameters
    ----------
    screening
        In-memory empirical-null screening result.
    artifact_paths
        Mapping from stable artifact name to the CSV path written under the output root.
    """

    screening: EmpiricalNullScreeningResult
    artifact_paths: dict[str, Path]


@dataclass(frozen=True)
class InteractionDiscoverySpec:
    """Frozen interaction-discovery settings from the manuscript contract.

    Parameters
    ----------
    method
        Manuscript interaction-discovery method recorded in the frozen contract.
    aggregation_rule
        Rule used to collapse component-level interaction evidence to one score per pair.
    null_threshold_quantile
        Quantile of the empirical-null score distribution used as the retention threshold.
    retained_pairs_reference
        Manuscript-reported retained interaction-pair count for the full case study.
    permutation_count_B
        Number of response permutations used for the public deterministic null threshold.
    random_seed
        Deterministic random seed for the permutation sequence.
    implementation_method
        Public implementation method actually run by this package.
    implementation_status
        Alignment status for the public implementation relative to the manuscript method.
    source_workflow_reference
        Reference label for the manuscript interaction workflow to reconcile against.
    source_workflow_equivalence_status
        Validation status for equivalence to the manuscript interaction workflow.
    """

    method: str
    aggregation_rule: str
    null_threshold_quantile: float
    retained_pairs_reference: int
    permutation_count_B: int
    random_seed: int = 123
    implementation_method: str = "tree_shap_gradient_boosting"
    implementation_status: str = "manuscript_aligned"
    source_workflow_reference: str = "private_tree_shap_interaction_workflow"
    source_workflow_equivalence_status: str = "manuscript_aligned_via_shap_gradient_boosting"
    n_tree_estimators: int = 100
    max_tree_depth: int = 3
    max_shap_samples: int = 500
    n_jobs: int = 1
    elasticnet_l1_ratio: float = 0.5
    elasticnet_cv_folds: int = 5
    min_component_variance_fraction: float = 0.01  # Skip components below 1% variance
    parallel_batch_timeout_seconds: int = 900
    parallel_backend: str = "threading"
    dask_workers: int | None = None
    dask_cores_per_worker: int = 1
    dask_memory_per_worker: str = "4 GB"


@dataclass(frozen=True)
class InteractionDiscoveryResult:
    """Materialized interaction-discovery artifacts for one manuscript run.

    Parameters
    ----------
    pair_scores
        Per-pair observed score, null threshold, empirical p-value, and retention flag.
    component_interaction_scores
        Long-form component-level residual interaction coefficients.
    interaction_null_summary
        Per-pair summary of empirical-null interaction-score distributions.
    retained_pairs
        Retained interaction-pair table after thresholding.
    provenance
        One-row provenance table recording the manuscript method, public surrogate method, and
        equivalence-validation status.
    summary
        One-row summary of the interaction-discovery stage.
    """

    pair_scores: pd.DataFrame
    component_interaction_scores: pd.DataFrame
    interaction_null_summary: pd.DataFrame
    retained_pairs: pd.DataFrame
    provenance: pd.DataFrame
    summary: pd.DataFrame


@dataclass(frozen=True)
class InteractionDiscoveryStageResult:
    """Interaction-discovery result plus paths written for notebook handoff.

    Parameters
    ----------
    interactions
        In-memory interaction-discovery result.
    artifact_paths
        Mapping from stable artifact name to the CSV path written under the output root.
    """

    interactions: InteractionDiscoveryResult
    artifact_paths: dict[str, Path]


@dataclass(frozen=True)
class NonlinearDiscoverySpec:
    """Frozen nonlinear-discovery settings from the manuscript contract.

    Parameters
    ----------
    method
        Manuscript nonlinear-discovery method recorded in the frozen contract.
    curvature_rule
        Rule used by the full case study to identify nonlinear response shapes.
    replacement_selection_rule
        Rule used to choose restricted parametric replacements for smooth nonlinear terms.
    identified_transformations_reference
        Manuscript-reported number of transformations identified before final support filtering.
    final_support_transformations_reference
        Manuscript-reported number of transformations retained in the final model support.
    minimum_curvature_score
        Deterministic public-stage threshold for the residualized nonlinear coefficient score.
    implementation_method
        Public implementation method actually run by this package.
    implementation_status
        Alignment status for the public implementation relative to the manuscript method.
    source_workflow_reference
        Reference label for the manuscript nonlinear workflow to reconcile against.
    source_workflow_equivalence_status
        Validation status for equivalence to the manuscript nonlinear workflow.
    """

    method: str
    curvature_rule: str
    replacement_selection_rule: str
    identified_transformations_reference: int
    final_support_transformations_reference: int
    minimum_curvature_score: float = 2.0
    implementation_method: str = "gam_cubic_smoothing_spline"
    implementation_status: str = "manuscript_aligned"
    source_workflow_reference: str = "private_gam_nonlinear_discovery_workflow"
    source_workflow_equivalence_status: str = "manuscript_aligned_via_scipy_smoothing_spline"
    n_jobs: int = 1


@dataclass(frozen=True)
class NonlinearDiscoveryResult:
    """Materialized nonlinear-discovery artifacts for one manuscript run.

    Parameters
    ----------
    transformation_scores
        Per-transformation residualized curvature score and replacement diagnostics.
    component_transformation_scores
        Long-form component-level residualized nonlinear coefficients.
    retained_transformations
        Transformations retained by the deterministic public nonlinear-discovery rule.
    provenance
        One-row provenance table recording the manuscript method, public surrogate method, and
        equivalence-validation status.
    summary
        One-row summary of the nonlinear-discovery stage.
    """

    transformation_scores: pd.DataFrame
    component_transformation_scores: pd.DataFrame
    retained_transformations: pd.DataFrame
    provenance: pd.DataFrame
    summary: pd.DataFrame


@dataclass(frozen=True)
class NonlinearDiscoveryStageResult:
    """Nonlinear-discovery result plus paths written for notebook handoff.

    Parameters
    ----------
    nonlinear
        In-memory nonlinear-discovery result.
    artifact_paths
        Mapping from stable artifact name to the CSV path written under the output root.
    """

    nonlinear: NonlinearDiscoveryResult
    artifact_paths: dict[str, Path]


@dataclass(frozen=True)
class SparseSelectionStabilitySpec:
    """Frozen sparse-selection and stability settings from the manuscript contract.

    Parameters
    ----------
    model_class
        Sparse model class recorded in the frozen manuscript contract.
    ebic_gamma
        Extended Bayesian information criterion gamma used to select the L1 penalty.
    support_aggregation_rule
        Rule used to aggregate per-component supports into one candidate support.
    resampling_scheme
        Human-readable stability-resampling scheme from the frozen contract.
    subsample_count
        Number of deterministic stability subsamples.
    subsample_fraction
        Fraction of training rows used in each stability subsample.
    jaccard_threshold
        Minimum mean support Jaccard similarity used for the stable-support flag.
    spearman_threshold
        Minimum mean feature-rank Spearman correlation used for the stable-rank flag.
    implementation_method
        Public sparse-selection implementation method used by this package.
    implementation_status
        Manuscript-alignment status for the public implementation.
    source_workflow_reference
        Recovered private/notebook workflow this stage is compared against.
    source_workflow_equivalence_status
        Whether equivalence to the recovered workflow has been validated.
    source_artifact
        Recovered notebook or script artifact referenced by the provenance record.
    source_selected_feature_count_reference
        Recovered selected-feature count for the source sparse-screening workflow.
    random_seed
        Deterministic random seed for stability subsampling.
    """

    model_class: str
    ebic_gamma: float
    support_aggregation_rule: str
    resampling_scheme: str
    subsample_count: int
    subsample_fraction: float
    jaccard_threshold: float
    spearman_threshold: float
    implementation_method: str = "ebic_l1_component_union_with_subsample_stability"
    implementation_status: str = "source_backed_public_surrogate"
    source_workflow_reference: str = "notebook_pca_debiased_lasso"
    source_workflow_equivalence_status: str = "not_yet_validated"
    source_artifact: str = "LASSO_to_OLS_v9.ipynb"
    source_selected_feature_count_reference: int = 346
    random_seed: int = 123
    n_jobs: int = 1
    max_candidate_terms: int | None = None
    adaptive_early_stopping_enabled: bool = False
    stability_convergence_window: int = 3


@dataclass(frozen=True)
class SparseSelectionStabilityResult:
    """Materialized sparse-selection and stability artifacts for one manuscript run.

    Parameters
    ----------
    support_candidates
        Candidate terms entering the L1/EBIC sparse-selection stage.
    component_model_selection
        Per-component EBIC-selected penalty and support-size diagnostics.
    component_coefficients
        Per-candidate, per-component standardized sparse coefficients.
    stability_resample_summary
        Per-resample support-overlap and rank-correlation diagnostics.
    stability_feature_summary
        Per-candidate selection-frequency and full-model importance diagnostics.
    final_stable_support
        Terms selected by the full-data sparse model and passing stability filters.
    provenance
        One-row provenance table describing the public implementation and recovered
        de-biased-LASSO workflow equivalence status.
    summary
        One-row summary of the sparse-selection and stability stage.
    """

    support_candidates: pd.DataFrame
    component_model_selection: pd.DataFrame
    component_coefficients: pd.DataFrame
    stability_resample_summary: pd.DataFrame
    stability_feature_summary: pd.DataFrame
    final_stable_support: pd.DataFrame
    provenance: pd.DataFrame
    summary: pd.DataFrame


@dataclass(frozen=True)
class SparseSelectionStabilityStageResult:
    """Sparse-selection/stability result plus paths written for notebook handoff.

    Parameters
    ----------
    sparse_selection
        In-memory sparse-selection and stability result.
    artifact_paths
        Mapping from stable artifact name to the CSV path written under the output root.
    """

    sparse_selection: SparseSelectionStabilityResult
    artifact_paths: dict[str, Path]


@dataclass(frozen=True)
class FinalManuscriptArtifactsSpec:
    """Frozen final-model and manuscript-artifact settings.

    Parameters
    ----------
    final_predictor_count_reference
        Manuscript-reported final predictor count for the full case study.
    final_first_order_input_count_reference
        Manuscript-reported count of first-order inputs represented in the final support.
    intermediate_penalized_holdout_nrmse_reference
        Manuscript-reported intermediate penalized-model holdout nRMSE.
    final_ols_holdout_nrmse_reference
        Manuscript-reported final OLS holdout nRMSE.
    nrmse_denominator_definition
        Frozen definition of the nRMSE denominator.
    nrmse_min_range
        Minimum training-response range used when computing macro nRMSE.
    nrmse_reference_matrix
        Frozen source matrix used for nRMSE normalization ranges.
    bootstrap_count
        Number of deterministic row-bootstrap replicates used for demo uncertainty.
    bootstrap_alpha
        Two-sided bootstrap error level.
    random_seed
        Deterministic random seed for bootstrap resampling.
    inferential_filter_interval_method
        Frozen final inferential-filter rule from the manuscript contract.
    inferential_filter_alpha
        Two-sided error level for the HC3 Wald intervals.
    n_jobs
        Number of parallel workers for bootstrap CI computation. ``1`` is serial;
        ``-1`` uses all available CPUs.
    """

    final_predictor_count_reference: int
    final_first_order_input_count_reference: int
    intermediate_penalized_holdout_nrmse_reference: float
    final_ols_holdout_nrmse_reference: float
    nrmse_denominator_definition: str
    nrmse_min_range: float
    nrmse_reference_matrix: str
    bootstrap_count: int = 200
    bootstrap_alpha: float = 0.05
    random_seed: int = 123
    inferential_filter_interval_method: str = (
        "hc3_wald_95_percent_drop_if_zero_compatible_for_all_outputs"
    )
    inferential_filter_alpha: float = 0.05
    hc3_output_subset_mode: str = "all"
    hc3_output_fraction: float | None = None
    hc3_output_names: tuple[str, ...] = ()
    hc3_output_max_outputs: int | None = None
    hc3_output_random_seed: int = 123
    hc3_output_subset_metric: str = "variance"
    pruning_error_scale_quantile: float = 0.95
    pruning_delta_threshold_override: float | None = None
    pruning_remove_count_override: int | None = None
    n_jobs: int = 1


@dataclass(frozen=True)
class FinalManuscriptArtifactsResult:
    """Materialized final-model, manuscript-table, and figure-source artifacts.

    Parameters
    ----------
    prefilter_support_features
        Sparse/stability support before applying the final HC3 inferential filter.
    final_support_features
        Final selected support after applying the HC3 inferential filter.
    hc3_wald_intervals
        Long-form per-output HC3 Wald interval diagnostics for each prefilter feature.
    hc3_inferential_filter_summary
        Per-feature summary of the final inferential-filter decision.
    final_ols_summary
        One-row summary of the final OLS fit and holdout uncertainty.
    model_performance
        Manuscript-facing performance table including demo and manuscript-reference rows.
    workflow_stage_summary
        Manuscript-facing summary of retained counts by workflow stage.
    coefficient_matrix_raw_scale
        Raw-scale final OLS coefficient matrix.
    coefficient_matrix_standardized
        Standardized final OLS coefficient matrix.
    x_standardization
        Final-feature standardization parameters estimated on training rows.
    y_standardization
        Final-output standardization parameters estimated on training rows.
    figure_model_performance_data
        Source data for the model-performance figure.
    figure_support_composition_data
        Source data for the final-support composition figure.
    figure_selected_by_module_data
        Source data for module-level selected-feature count/share figures.
    figure_nrmse_summary_data
        Source data for bootstrap nRMSE summary figure.
    legacy_feature_type_counts
        Legacy-compatible feature-type summary used by influential-factor figures.
    legacy_influential_counts_by_module
        Legacy-compatible influential-feature count summary by module.
    legacy_interaction_counts_by_module_pair
        Legacy-compatible interaction pair count summary by module pair.
    legacy_interaction_density_module_matrix
        Legacy-compatible symmetric interaction-density matrix by module.
    legacy_module_total_interactions
        Legacy-compatible per-module total interaction counts.
    figure_specs
        Registry of generated figure assets.
    svg_figures
        SVG text keyed by stable figure name.
    summary
        One-row artifact-regeneration summary.
    """

    prefilter_support_features: pd.DataFrame
    final_support_features: pd.DataFrame
    hc3_wald_intervals: pd.DataFrame
    hc3_inferential_filter_summary: pd.DataFrame
    final_ols_summary: pd.DataFrame
    model_performance: pd.DataFrame
    workflow_stage_summary: pd.DataFrame
    coefficient_matrix_raw_scale: pd.DataFrame
    coefficient_matrix_standardized: pd.DataFrame
    x_standardization: pd.DataFrame
    y_standardization: pd.DataFrame
    figure_model_performance_data: pd.DataFrame
    figure_support_composition_data: pd.DataFrame
    figure_selected_by_module_data: pd.DataFrame
    figure_nrmse_summary_data: pd.DataFrame
    legacy_feature_type_counts: pd.DataFrame
    legacy_influential_counts_by_module: pd.DataFrame
    legacy_interaction_counts_by_module_pair: pd.DataFrame
    legacy_interaction_density_module_matrix: pd.DataFrame
    legacy_module_total_interactions: pd.DataFrame
    feature_pruning_impact: pd.DataFrame
    figure_feature_pruning_curve_data: pd.DataFrame
    feature_pruning_summary: pd.DataFrame
    figure_specs: pd.DataFrame
    svg_figures: dict[str, str]
    ablation_table: pd.DataFrame
    per_output_nrmse: pd.DataFrame
    per_output_nrmse_summary: pd.DataFrame
    summary: pd.DataFrame


@dataclass(frozen=True)
class FinalManuscriptArtifactsStageResult:
    """Final manuscript-artifact result plus paths written for notebook handoff.

    Parameters
    ----------
    final_artifacts
        In-memory final manuscript artifacts.
    artifact_paths
        Mapping from stable artifact name to CSV or SVG paths under the output root.
    """

    final_artifacts: FinalManuscriptArtifactsResult
    artifact_paths: dict[str, Path]


@dataclass(frozen=True)
class ManuscriptReproductionStageChainResult:
    """End-to-end manuscript reproduction chain result.

    Parameters
    ----------
    output_conditioning
        Output conditioning stage result and written artifact paths.
    empirical_null_screening
        Empirical-null screening stage result and written artifact paths.
    interaction_discovery
        Interaction-discovery stage result and written artifact paths.
    nonlinear_discovery
        Nonlinear-discovery stage result and written artifact paths.
    sparse_selection_stability
        Sparse-selection and stability stage result and written artifact paths.
    final_manuscript_artifacts
        Final table and figure regeneration result and written artifact paths.
    artifact_paths
        Nested mapping from stage family to that stage's written artifact paths.
    """

    output_conditioning: OutputConditioningStageResult
    empirical_null_screening: EmpiricalNullScreeningStageResult
    interaction_discovery: InteractionDiscoveryStageResult
    nonlinear_discovery: NonlinearDiscoveryStageResult
    sparse_selection_stability: SparseSelectionStabilityStageResult
    final_manuscript_artifacts: FinalManuscriptArtifactsStageResult
    artifact_paths: dict[str, dict[str, Path]]


@dataclass(frozen=True)
class ManuscriptReproductionAuditResult:
    """QA manifest and metric checks for a manuscript reproduction run.

    Parameters
    ----------
    artifact_manifest
        One row per written artifact with path, file size, SHA-256 digest, and presence checks.
    metric_checks
        Explicit pass/fail metric and consistency checks for the demo reproduction outputs.
    summary
        One-row audit summary with aggregate artifact and metric-check status.
    """

    artifact_manifest: pd.DataFrame
    metric_checks: pd.DataFrame
    summary: pd.DataFrame


@dataclass(frozen=True)
class ManuscriptReproductionAuditStageResult:
    """End-to-end reproduction chain plus written QA audit artifacts.

    Parameters
    ----------
    reproduction
        Complete manuscript reproduction stage-chain result.
    audit
        Materialized audit manifest, metric checks, and summary.
    artifact_paths
        Paths to written audit CSV artifacts under the output root.
    """

    reproduction: ManuscriptReproductionStageChainResult
    audit: ManuscriptReproductionAuditResult
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


def empirical_null_screening_spec_from_case_study_config(
    case_study_config: dict[str, Any],
) -> EmpiricalNullScreeningSpec:
    """Build the empirical-null screening specification from the case-study config.

    Parameters
    ----------
    case_study_config
        Parsed ``configs/manuscript_case_study.yml`` mapping.

    Returns
    -------
    EmpiricalNullScreeningSpec
        Typed empirical-null screening specification.
    """
    section = case_study_config["case_study"]["empirical_null_screen"]
    interface = case_study_config["case_study"].get("interface", {})
    runtime = case_study_config["case_study"].get("runtime", {})
    return EmpiricalNullScreeningSpec(
        statistic=str(section["statistic"]),
        permutation_count_B=int(section["permutation_count_B"]),
        bh_q_screen=float(section["bh_q_screen"]),
        retained_terms_reference=int(section["retained_terms"]),
        implementation_method=str(
            section.get("public_implementation_method", "coefficient_row_l2_permutation")
        ),
        implementation_status=str(
            section.get("public_implementation_status", "source_backed_public_surrogate")
        ),
        source_script_reference=str(
            section.get("source_script_reference", "private_delta_null_screening_script")
        ),
        source_script_equivalence_status=str(
            section.get("source_script_equivalence_status", "not_yet_validated")
        ),
        random_seed=int(interface.get("holdout_random_seed", 123)),
        n_jobs=int(runtime.get("n_jobs", 1)),
        max_retained_terms=(
            int(section["max_retained_terms"])
            if section.get("max_retained_terms") is not None
            else None
        ),
    )


def build_manuscript_feature_design(
    input_matrix: pd.DataFrame,
    feature_catalog: pd.DataFrame,
) -> pd.DataFrame:
    """Materialize the manuscript feature catalog against an input matrix.

    The public demo catalog includes direct first-order inputs, simple colon-delimited
    interactions, and algebraic transformation names. The same deterministic parser is
    used by the empirical-null stage so notebooks never need to re-create feature logic.

    Parameters
    ----------
    input_matrix
        Input table with ``sample_id`` and source input columns.
    feature_catalog
        Feature catalog with at least ``feature_name`` and ``feature_type`` columns.

    Returns
    -------
    pandas.DataFrame
        Design matrix with ``sample_id`` and one column per materialized feature.
    """
    if "sample_id" not in input_matrix.columns:
        raise ValueError("input_matrix must include a sample_id column.")
    if "feature_name" not in feature_catalog.columns:
        raise ValueError("feature_catalog must include a feature_name column.")
    if input_matrix["sample_id"].duplicated(keep=False).any():
        raise ValueError("input_matrix must contain unique sample_id values.")

    sample_id = input_matrix["sample_id"].reset_index(drop=True)
    feature_columns = {}
    for feature_name in feature_catalog["feature_name"].astype(str):
        if feature_name in feature_columns:
            raise ValueError(f"Duplicate feature name in feature_catalog: {feature_name}")
        values = _materialize_feature_column(input_matrix, feature_name)
        numeric = pd.to_numeric(values, errors="raise").to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise ValueError(f"Feature {feature_name!r} produced non-finite values.")
        feature_columns[feature_name] = numeric

    feature_df = pd.DataFrame(feature_columns)
    design = pd.concat([pd.DataFrame({"sample_id": sample_id}), feature_df], axis=1)
    return design


def _first_order_feature_catalog(feature_catalog: pd.DataFrame) -> pd.DataFrame:
    """Return first-order-only catalog rows for empirical-null screening."""
    if "feature_name" not in feature_catalog.columns:
        raise ValueError("feature_catalog must include a feature_name column.")
    if "feature_type" not in feature_catalog.columns:
        raise ValueError("feature_catalog must include a feature_type column.")
    feature_types = feature_catalog["feature_type"].astype(str).str.lower()
    first_order = feature_catalog.loc[feature_types.isin({"first_order", "numeric"})].copy()
    if first_order.empty:
        raise ValueError("feature_catalog does not contain first-order features for screening.")
    return first_order.reset_index(drop=True)


def screen_manuscript_empirical_null_terms(
    input_matrix: pd.DataFrame,
    feature_catalog: pd.DataFrame,
    holdout_assignments: pd.DataFrame,
    pca_scores: pd.DataFrame,
    spec: EmpiricalNullScreeningSpec,
    *,
    checkpoint_dir: Path | None = None,
    active_permutation_indices: set[int] | None = None,
) -> EmpiricalNullScreeningResult:
    """Screen manuscript terms with a coefficient-row-norm empirical null.

    The observed statistic for each feature is the Euclidean norm of its standardized
    coefficient row across retained PCA component scores. Empirical p-values are computed
    from featurewise response permutations, and retained terms are selected with the
    Benjamini--Hochberg threshold frozen in the manuscript contract.

    Parameters
    ----------
    input_matrix
        Case-study input table with ``sample_id`` and source input columns.
    feature_catalog
        Manuscript feature catalog. Only first-order rows are screened.
    holdout_assignments
        Table with ``sample_id`` and ``split`` columns. Only train rows are screened.
    pca_scores
        Output-conditioning PCA score table with ``sample_id`` and component columns.
    spec
        Empirical-null screening specification.

    Returns
    -------
    EmpiricalNullScreeningResult
        Materialized screening tables and summary.
    """
    if spec.statistic != "coefficient_row_l2_norm":
        raise ValueError(f"Unsupported empirical-null statistic: {spec.statistic}")
    if spec.permutation_count_B < 1:
        raise ValueError("permutation_count_B must be positive.")
    if not 0.0 < spec.bh_q_screen <= 1.0:
        raise ValueError("bh_q_screen must be in the interval (0, 1].")
    if spec.max_retained_terms is not None and spec.max_retained_terms < 1:
        raise ValueError("max_retained_terms must be positive when provided.")

    first_order_catalog = _first_order_feature_catalog(feature_catalog)
    design = build_manuscript_feature_design(input_matrix, first_order_catalog)
    feature_names = [str(column) for column in design.columns if column != "sample_id"]
    component_names = _component_columns(pca_scores)
    train_ids = _train_sample_ids(holdout_assignments)
    x_train = _align_table_by_sample_id(design, train_ids, feature_names, "feature design")
    y_train = _align_table_by_sample_id(pca_scores, train_ids, component_names, "PCA scores")
    if len(x_train) < 3:
        raise ValueError("Empirical-null screening requires at least three training rows.")

    x_scaled, feature_active = _standardize_for_screening(x_train)
    y_scaled, component_active = _standardize_for_screening(y_train)
    if not component_active.any():
        raise ValueError("All retained PCA components have zero training variance.")

    coefficients = (x_scaled.T @ y_scaled) / float(len(x_scaled))
    observed = np.linalg.norm(coefficients, axis=1)
    observed[~feature_active] = 0.0
    train_ids_digest = hashlib.sha256(
        "|".join(str(sample_id) for sample_id in train_ids).encode("utf-8")
    ).hexdigest()
    null_statistics = _permutation_row_norm_null(
        x_scaled,
        y_scaled,
        n_permutations=spec.permutation_count_B,
        random_seed=spec.random_seed,
        n_jobs=spec.n_jobs,
        active_permutation_indices=active_permutation_indices,
        checkpoint_dir=(
            None if checkpoint_dir is None else checkpoint_dir / "_permutation_checkpoints"
        ),
        checkpoint_signature_payload={
            "feature_names": feature_names,
            "component_names": component_names,
            "train_ids_sha256": train_ids_digest,
            "statistic": spec.statistic,
        },
    )
    null_statistics[:, ~feature_active] = 0.0
    p_values = (1.0 + (null_statistics >= observed[None, :]).sum(axis=0)) / (
        spec.permutation_count_B + 1.0
    )
    retained, bh_rank, bh_critical = _benjamini_hochberg_retention(
        p_values,
        q=spec.bh_q_screen,
    )
    feature_stats = _build_feature_screening_statistics(
        feature_names=feature_names,
        observed=observed,
        p_values=p_values,
        retained=retained,
        bh_rank=bh_rank,
        bh_critical=bh_critical,
        feature_active=feature_active,
    )
    component_coefficients = _build_component_coefficients(
        feature_names=feature_names,
        component_names=component_names,
        coefficients=coefficients,
    )
    null_summary = _build_permutation_null_summary(feature_names, null_statistics)
    retained_terms = feature_stats.loc[feature_stats["retained"]].copy()
    retained_terms = retained_terms.sort_values(
        ["empirical_p_value", "observed_statistic", "feature_name"],
        ascending=[True, False, True],
        ignore_index=True,
    )
    if spec.max_retained_terms is not None:
        retained_terms = retained_terms.head(spec.max_retained_terms).reset_index(drop=True)
        retained_names = set(retained_terms["feature_name"].astype(str))
        feature_stats["retained"] = feature_stats["feature_name"].astype(str).isin(retained_names)
    summary = _build_empirical_null_screening_summary(
        n_training_rows=len(x_train),
        n_candidate_terms=len(feature_names),
        n_active_terms=int(feature_active.sum()),
        n_components=len(component_names),
        n_retained_terms=len(retained_terms),
        min_p_value=float(p_values.min()),
        spec=spec,
    )
    provenance = _build_empirical_null_screening_provenance(spec)
    return EmpiricalNullScreeningResult(
        feature_screening_statistics=feature_stats,
        component_coefficients=component_coefficients,
        permutation_null_summary=null_summary,
        retained_terms=retained_terms,
        provenance=provenance,
        summary=summary,
    )


def write_empirical_null_screening_artifacts(
    result: EmpiricalNullScreeningResult,
    output_root: Path,
) -> dict[str, Path]:
    """Write empirical-null screening artifacts under ``output_root``.

    Parameters
    ----------
    result
        Materialized empirical-null screening result.
    output_root
        Resolved manuscript output root from the runtime context.

    Returns
    -------
    dict[str, pathlib.Path]
        Paths keyed by stable artifact name.
    """
    stage_root = output_root / "empirical_null_screen"
    stage_root.mkdir(parents=True, exist_ok=True)
    tables = {
        "feature_screening_statistics": result.feature_screening_statistics,
        "component_coefficients": result.component_coefficients,
        "permutation_null_summary": result.permutation_null_summary,
        "retained_terms": result.retained_terms,
        "empirical_null_provenance": result.provenance,
        "empirical_null_screen_summary": result.summary,
    }
    written: dict[str, Path] = {}
    for name, table in tables.items():
        path = stage_root / f"{name}.csv"
        table.to_csv(path, index=False)
        written[name] = path
    return written


def run_empirical_null_screening_stage(context: Any) -> EmpiricalNullScreeningStageResult:
    """Run empirical-null screening from a manuscript notebook runtime context.

    Parameters
    ----------
    context
        ``bsm_rfm.manuscript_runtime.ManuscriptNotebookContext``. It is typed as ``Any`` here to
        avoid an import cycle between the runtime and stage modules.

    Returns
    -------
    EmpiricalNullScreeningStageResult
        In-memory result and written artifact paths.
    """
    conditioning_spec = output_conditioning_spec_from_case_study_config(context.case_study_config)
    conditioning = condition_manuscript_outputs(
        context.tables["case_study_output_matrix"],
        context.tables["fixed_holdout_assignments"],
        conditioning_spec,
    )
    screening_spec = empirical_null_screening_spec_from_case_study_config(context.case_study_config)
    screening = screen_manuscript_empirical_null_terms(
        context.tables["case_study_input_matrix"],
        context.tables["manuscript_feature_catalog"],
        context.tables["fixed_holdout_assignments"],
        conditioning.pca_scores,
        screening_spec,
        checkpoint_dir=context.runtime.output_root / "empirical_null_screen",
    )
    artifact_paths = write_empirical_null_screening_artifacts(
        screening,
        context.runtime.output_root,
    )
    return EmpiricalNullScreeningStageResult(screening=screening, artifact_paths=artifact_paths)


def interaction_discovery_spec_from_case_study_config(
    case_study_config: dict[str, Any],
) -> InteractionDiscoverySpec:
    """Build the interaction-discovery specification from the case-study config.

    Parameters
    ----------
    case_study_config
        Parsed ``configs/manuscript_case_study.yml`` mapping.

    Returns
    -------
    InteractionDiscoverySpec
        Typed interaction-discovery specification.
    """
    case_study = case_study_config["case_study"]
    interaction = case_study["interaction_discovery"]
    empirical_null = case_study["empirical_null_screen"]
    interface = case_study.get("interface", {})
    runtime = case_study.get("runtime", {})
    dask_workers_raw = interaction.get("dask_workers")
    return InteractionDiscoverySpec(
        method=str(interaction["method"]),
        aggregation_rule=str(interaction["aggregation_rule"]),
        null_threshold_quantile=float(interaction["null_threshold_quantile"]),
        retained_pairs_reference=int(interaction["retained_pairs"]),
        permutation_count_B=int(
            interaction.get("permutation_count_B", empirical_null["permutation_count_B"])
        ),
        random_seed=int(interface.get("holdout_random_seed", 123)),
        n_tree_estimators=int(interaction.get("n_tree_estimators", 100)),
        max_tree_depth=int(interaction.get("max_tree_depth", 3)),
        max_shap_samples=int(interaction.get("max_shap_samples", 500)),
        implementation_method=str(
            interaction.get(
                "public_implementation_method",
                "residualized_product_permutation_surrogate",
            )
        ),
        implementation_status=str(
            interaction.get("public_implementation_status", "source_backed_public_surrogate")
        ),
        source_workflow_reference=str(
            interaction.get("source_workflow_reference", "private_tree_shap_interaction_workflow")
        ),
        source_workflow_equivalence_status=str(
            interaction.get("source_workflow_equivalence_status", "not_yet_validated")
        ),
        n_jobs=int(runtime.get("n_jobs", 1)),
        elasticnet_l1_ratio=float(interaction.get("elasticnet_l1_ratio", 0.5)),
        elasticnet_cv_folds=int(interaction.get("elasticnet_cv_folds", 5)),
        min_component_variance_fraction=float(
            interaction.get("min_component_variance_fraction", 0.01)
        ),
        parallel_batch_timeout_seconds=int(interaction.get("parallel_batch_timeout_seconds", 900)),
        parallel_backend=str(interaction.get("parallel_backend", "threading")),
        dask_workers=(int(dask_workers_raw) if dask_workers_raw is not None else None),
        dask_cores_per_worker=int(interaction.get("dask_cores_per_worker", 1)),
        dask_memory_per_worker=str(interaction.get("dask_memory_per_worker", "4 GB")),
    )


def _discover_elasticnet_interactions(
    input_matrix: pd.DataFrame,
    holdout_assignments: pd.DataFrame,
    pca_scores: pd.DataFrame,
    retained_terms: pd.DataFrame,
    spec: InteractionDiscoverySpec,
) -> InteractionDiscoveryResult:
    """Discover interaction pairs via ElasticNet coefficient selection (original archived approach).

    Candidate pairs are generated from first-order terms retained by empirical null screen.
    All candidate interactions are computed, then MultiTaskElasticNetCV selects which have
    non-zero coefficients. This matches the original archived workflow approach.

    Parameters
    ----------
    input_matrix
        Case-study input table with ``sample_id`` and source input columns.
    holdout_assignments
        Table with ``sample_id`` and ``split`` columns. Only train rows are used.
    pca_scores
        Output-conditioning PCA score table with ``sample_id`` and component columns.
    retained_terms
        Empirical-null retained-term table with at least a ``feature_name`` column.
    spec
        Interaction-discovery specification.

    Returns
    -------
    InteractionDiscoveryResult
        Materialized pair-score, component-score, null-summary, retained-pair, and summary tables.
    """
    if spec.method != "elasticnet_interactions":
        raise ValueError(f"Expected method 'elasticnet_interactions', got {spec.method}")
    if not 0.0 < spec.null_threshold_quantile < 1.0:
        raise ValueError("null_threshold_quantile must be in the interval (0, 1).")

    # Generate candidate interaction pairs from retained screening features.
    candidates = _generate_pairwise_interactions(
        _retained_first_order_term_names(retained_terms, input_matrix, minimum_count=2)
    )
    component_names = _component_columns(pca_scores)
    train_ids = _train_sample_ids(holdout_assignments)
    y_train = _align_table_by_sample_id(pca_scores, train_ids, component_names, "PCA scores")

    if len(y_train) < 4:
        raise ValueError("Interaction discovery requires at least four training rows.")
    y_scaled, component_active = _standardize_for_screening(y_train)
    if not component_active.any():
        raise ValueError("All retained PCA components have zero training variance.")
    if not candidates:
        raise ValueError("No candidate interaction pairs available.")

    # Build feature matrix: first-order features + all interaction candidates.
    feature_names = sorted({name for _, left, right in candidates for name in [left, right]})
    pair_names = [name for name, _, _ in candidates]
    interaction_indices = {
        name: (feature_names.index(left), feature_names.index(right))
        for name, left, right in candidates
    }

    indexed = input_matrix.set_index("sample_id", drop=False)
    missing_ids = [s for s in train_ids if s not in indexed.index]
    if missing_ids:
        preview = ", ".join(str(v) for v in missing_ids[:5])
        raise ValueError(f"input matrix is missing sample_id values: {preview}")

    train_rows = indexed.loc[list(train_ids)].reset_index(drop=True)
    x_first_order = np.column_stack(
        [_source_input_column(train_rows, f, f).to_numpy(dtype=float) for f in feature_names]
    )

    # Compute all interaction products.
    x_interactions = np.column_stack(
        [x_first_order[:, i] * x_first_order[:, j] for i, j in interaction_indices.values()]
    )

    # Combine first-order and interaction features.
    x_all = np.column_stack([x_first_order, x_interactions])

    # Standardize for ElasticNet.
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_all)

    # Fit MultiTaskElasticNetCV to select relevant interactions.
    try:
        model = MultiTaskElasticNetCV(
            l1_ratio=spec.elasticnet_l1_ratio,
            cv=spec.elasticnet_cv_folds,
            fit_intercept=True,
            max_iter=10000,
            random_state=spec.random_seed,
            n_jobs=spec.n_jobs,
        )
        model.fit(x_scaled, y_scaled)
    except Exception as e:
        raise ValueError(f"ElasticNet fitting failed: {e}") from e

    # Extract selected features (non-zero coefficients).
    coef_matrix = model.coef_.T  # n_features x n_outputs
    selected_mask = (np.abs(coef_matrix) > 1e-10).any(axis=1)

    # Determine which interaction pairs were selected.
    n_first_order = len(feature_names)
    selected_pairs = []
    for i, sel in enumerate(selected_mask[n_first_order:], start=n_first_order):
        if sel:
            pair_idx = i - n_first_order
            if pair_idx < len(pair_names):
                selected_pairs.append(pair_names[pair_idx])

    # Build output dataframes in same format as GBT approach.
    # Compute a "score" for each interaction based on max coefficient magnitude across outputs.
    pair_scores_data = []
    for pair_name, left, right in candidates:
        pair_idx = n_first_order + pair_names.index(pair_name)
        coefs = coef_matrix[pair_idx, :]
        max_coef = np.abs(coefs).max()
        is_selected = pair_name in selected_pairs

        # Create a pseudo-score based on coefficient magnitude (higher = more important).
        score = float(max_coef)

        pair_scores_data.append(
            {
                "pair_name": pair_name,
                "left_feature": left,
                "right_feature": right,
                "interaction_score": score,
                "null_threshold": 0.0,  # ElasticNet doesn't use null thresholds
                "p_value": 0.0 if is_selected else 1.0,
                "empirical_null_retained": is_selected,
                "retained": is_selected,
                "mean_coefficient_magnitude": float(np.abs(coefs).mean()),
            }
        )

    pair_scores = pd.DataFrame(pair_scores_data)

    # Component-level scores: assign max coefficient magnitude to each component.
    component_scores_data = []
    for comp_idx, comp_name in enumerate(component_names):
        if not component_active[comp_idx]:
            continue
        for pair_name, _left, _right in candidates:
            pair_idx = n_first_order + pair_names.index(pair_name)
            coef_value = float(coef_matrix[pair_idx, comp_idx])
            component_scores_data.append(
                {
                    "pair_name": pair_name,
                    "component": comp_name,
                    "mean_absolute_interaction": float(np.abs(coef_value)),
                }
            )

    component_interaction_scores = pd.DataFrame(component_scores_data)

    # Null summary: ElasticNet doesn't use permutation nulls, so use simplified summary.
    null_summary_data = []
    for pair_name, _, _ in candidates:
        null_summary_data.append(
            {
                "pair_name": pair_name,
                "null_mean": 0.0,
                "null_std": 0.0,
                "null_count": 0,
            }
        )
    interaction_null_summary = pd.DataFrame(null_summary_data)

    # Retained pairs: those with non-zero ElasticNet coefficients.
    retained_pairs = pair_scores.loc[pair_scores["retained"]].copy()
    retained_pairs = retained_pairs.sort_values(
        ["interaction_score", "pair_name"],
        ascending=[False, True],
        ignore_index=True,
    )

    # Provenance.
    provenance = pd.DataFrame(
        {
            "interaction_method": [spec.method],
            "interaction_implementation": ["elasticnet_coefficient_selection"],
            "n_tree_estimators": [spec.n_tree_estimators],
            "max_tree_depth": [spec.max_tree_depth],
            "elasticnet_l1_ratio": [spec.elasticnet_l1_ratio],
            "elasticnet_cv_folds": [spec.elasticnet_cv_folds],
            "null_threshold_quantile": [spec.null_threshold_quantile],
        }
    )

    # Summary.
    summary = pd.DataFrame(
        {
            "n_training_rows": [len(y_train)],
            "n_candidate_pairs": [len(candidates)],
            "n_empirical_null_retained_pairs": [len(retained_pairs)],
            "n_retained_pairs": [len(retained_pairs)],
            "n_active_components": [int(component_active.sum())],
            "max_interaction_score": [
                float(pair_scores["interaction_score"].max()) if len(pair_scores) > 0 else 0.0
            ],
            "elasticnet_alpha": [float(model.alpha_)],
        }
    )

    return InteractionDiscoveryResult(
        pair_scores=pair_scores,
        component_interaction_scores=component_interaction_scores,
        interaction_null_summary=interaction_null_summary,
        retained_pairs=retained_pairs,
        provenance=provenance,
        summary=summary,
    )


def discover_manuscript_interactions(
    input_matrix: pd.DataFrame,
    feature_catalog: pd.DataFrame,
    holdout_assignments: pd.DataFrame,
    pca_scores: pd.DataFrame,
    retained_terms: pd.DataFrame,
    spec: InteractionDiscoverySpec,
    *,
    checkpoint_dir: Path | None = None,
    pair_start_idx: int | None = None,
    pair_end_idx: int | None = None,
) -> InteractionDiscoveryResult:
    """Discover candidate interaction pairs via specified method (GBT+SHAP or ElasticNet).

    Supports two interaction discovery approaches:
    1. tree_shap_interaction_values: GBT+SHAP-based dynamic interaction discovery
    2. elasticnet_interactions: Original archived ElasticNet coefficient selection

    Parameters
    ----------
    input_matrix
        Case-study input table with ``sample_id`` and source input columns.
    feature_catalog
        Manuscript feature catalog (kept for interface compatibility).
    holdout_assignments
        Table with ``sample_id`` and ``split`` columns. Only train rows are scored.
    pca_scores
        Output-conditioning PCA score table with ``sample_id`` and component columns.
    retained_terms
        Empirical-null retained-term table with at least a ``feature_name`` column.
    spec
        Interaction-discovery specification.
    checkpoint_dir
        Optional directory for per-permutation checkpoints. When provided, completed permutation
        scores are persisted and reused on subsequent reruns with the same discovery signature.
    pair_start_idx
        Optional zero-based start index (inclusive) for candidate-pair slicing.
        When provided with ``pair_end_idx``, only that pair range is scored.
    pair_end_idx
        Optional zero-based end index (exclusive) for candidate-pair slicing.

    Returns
    -------
    InteractionDiscoveryResult
        Materialized pair-score, component-score, null-summary, retained-pair, and summary tables.
    """
    if spec.method == "elasticnet_interactions":
        return _discover_elasticnet_interactions(
            input_matrix, holdout_assignments, pca_scores, retained_terms, spec
        )
    elif spec.method != "tree_shap_interaction_values":
        raise ValueError(f"Unsupported interaction-discovery method: {spec.method}")

    expected_rule = "max_over_components_of_mean_absolute_shap_interaction"
    if spec.aggregation_rule != expected_rule:
        raise ValueError(f"Unsupported interaction aggregation rule: {spec.aggregation_rule}")
    if spec.permutation_count_B < 1:
        raise ValueError("permutation_count_B must be positive.")
    if not 0.0 < spec.null_threshold_quantile < 1.0:
        raise ValueError("null_threshold_quantile must be in the interval (0, 1).")

    def _run_interaction_jobs(
        work_items: list[tuple[int, int]],
        *,
        n_jobs: int,
        backend: str,
        timeout_seconds: int | None,
        temp_folder: str | None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        def _score_from_item(item: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
            score_index, seed = item
            return _score_interaction_permutation(
                y_base=y_scaled,
                permute_response=(score_index > 0),
                seed=seed,
                **_score_kwargs,
            )

        if backend == "dask":
            requested_workers = spec.dask_workers if spec.dask_workers is not None else n_jobs
            if requested_workers <= 0:
                requested_workers = os.cpu_count() or 1
            dask_timeout = int(timeout_seconds) if timeout_seconds is not None else 3600
            executor = None
            try:
                executor = get_executor(
                    "dask",
                    n_workers=max(1, int(requested_workers)),
                    cores_per_worker=max(1, int(spec.dask_cores_per_worker)),
                    memory_per_worker=spec.dask_memory_per_worker,
                    timeout=dask_timeout,
                )
                return executor.map(_score_from_item, work_items)
            except (ImportError, RuntimeError, OSError, TimeoutError, ValueError) as exc:
                _append_parallel_diagnostic(
                    "interaction_discovery",
                    {
                        "event": "dask_executor_fallback",
                        "fallback_backend": "threading",
                        "n_jobs": n_jobs,
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc),
                    },
                )
                backend = "threading"
            finally:
                if executor is not None:
                    executor.close()

        parallel_kwargs: dict[str, Any] = {}
        if n_jobs > 1 and timeout_seconds is not None and timeout_seconds > 0:
            parallel_kwargs["timeout"] = int(timeout_seconds)
        if n_jobs > 1:
            parallel_kwargs["backend"] = backend
            parallel_kwargs["prefer"] = "processes" if backend == "loky" else "threads"
            if backend == "loky" and temp_folder:
                parallel_kwargs["temp_folder"] = temp_folder
        jobs = [delayed(_score_from_item)(item) for item in work_items]
        with Parallel(n_jobs=n_jobs, **parallel_kwargs) as parallel:
            return parallel(jobs)

    candidates = _generate_pairwise_interactions(
        _retained_first_order_term_names(retained_terms, input_matrix, minimum_count=2)
    )
    component_names = _component_columns(pca_scores)
    train_ids = _train_sample_ids(holdout_assignments)
    y_train = _align_table_by_sample_id(pca_scores, train_ids, component_names, "PCA scores")
    if len(y_train) < 4:
        raise ValueError("Interaction discovery requires at least four training rows.")
    y_scaled, component_active = _standardize_for_screening(y_train)
    if not component_active.any():
        raise ValueError("All retained PCA components have zero training variance.")

    if not candidates:
        raise ValueError("feature_catalog does not contain any two-factor interaction candidates.")
    total_candidate_pairs = len(candidates)
    if pair_start_idx is not None or pair_end_idx is not None:
        start = int(pair_start_idx) if pair_start_idx is not None else 0
        end = int(pair_end_idx) if pair_end_idx is not None else total_candidate_pairs
        start = max(0, min(start, total_candidate_pairs))
        end = max(start, min(end, total_candidate_pairs))
        candidates = candidates[start:end]
        if not candidates:
            raise ValueError(
                "interaction_discovery shard candidate range is empty: "
                f"pair_start_idx={pair_start_idx}, pair_end_idx={pair_end_idx}, "
                f"total_candidate_pairs={total_candidate_pairs}"
            )

    # Build feature matrix from all unique features appearing in any pair.
    feature_names = sorted({name for _, left, right in candidates for name in [left, right]})
    pair_to_indices = {
        pair_name: (feature_names.index(left), feature_names.index(right))
        for pair_name, left, right in candidates
    }
    indexed = input_matrix.set_index("sample_id", drop=False)
    missing_ids = [s for s in train_ids if s not in indexed.index]
    if missing_ids:
        preview = ", ".join(str(v) for v in missing_ids[:5])
        raise ValueError(f"input matrix is missing sample_id values: {preview}")
    train_rows = indexed.loc[list(train_ids)].reset_index(drop=True)
    x_feat = np.column_stack(
        [_source_input_column(train_rows, f, f).to_numpy(dtype=float) for f in feature_names]
    )

    rng = np.random.default_rng(spec.random_seed)
    active_comp_indices = [i for i, a in enumerate(component_active) if a]

    # Component pruning: skip components with very low variance
    # (below min_component_variance_fraction). This reduces SHAP computation for
    # uninformative components while retaining signal components.
    if spec.min_component_variance_fraction > 0:
        component_variances = y_train.var(axis=0).values  # Convert to numpy array
        max_variance = component_variances.max()
        if max_variance > 0:
            variance_fractions = component_variances / max_variance
            pruned_indices = [
                i
                for i in active_comp_indices
                if variance_fractions[i] >= spec.min_component_variance_fraction
            ]
            if pruned_indices:  # Only prune if we keep at least one component
                active_comp_indices = pruned_indices

    n_pairs = len(candidates)
    n_comp = len(component_names)

    # Adaptive SHAP sampling: use 30% of training set or max 250 samples, whichever is lower.
    # This reduces SHAP computation cost while maintaining interaction estimate quality.
    n_train_samples = len(y_train)
    adaptive_shap_samples = min(250, max(100, int(0.3 * n_train_samples)))
    effective_max_shap_samples = min(spec.max_shap_samples, adaptive_shap_samples)

    # Pre-generate per-call seeds only. Each worker materializes its own response permutation.
    # This avoids shipping large permuted matrices through joblib IPC, which can destabilize
    # process-based parallel execution on large runs.
    seeds = [int(rng.integers(0, 2**31)) for _ in range(spec.permutation_count_B + 1)]

    _score_kwargs = dict(
        x_feat=x_feat,
        n_pairs=n_pairs,
        n_comp=n_comp,
        active_comp_indices=active_comp_indices,
        pair_to_indices=pair_to_indices,
        candidates=candidates,
        n_estimators=spec.n_tree_estimators,
        max_depth=spec.max_tree_depth,
        max_shap_samples=effective_max_shap_samples,
    )

    total_scores = len(seeds)
    checkpoint_root: Path | None = checkpoint_dir
    if checkpoint_root is None:
        env_checkpoint_dir = os.getenv("BSM_INTERACTION_CHECKPOINT_DIR")
        if env_checkpoint_dir:
            checkpoint_root = Path(env_checkpoint_dir)
    checkpoint_run_dir: Path | None = None
    if checkpoint_root is not None:
        train_ids_digest = hashlib.sha256(
            "|".join(str(sample_id) for sample_id in train_ids).encode("utf-8")
        ).hexdigest()
        checkpoint_signature_payload = {
            "method": spec.method,
            "aggregation_rule": spec.aggregation_rule,
            "null_threshold_quantile": float(spec.null_threshold_quantile),
            "permutation_count_B": int(spec.permutation_count_B),
            "random_seed": int(spec.random_seed),
            "n_tree_estimators": int(spec.n_tree_estimators),
            "max_tree_depth": int(spec.max_tree_depth),
            "n_pairs": int(n_pairs),
            "pair_names": [pair_name for pair_name, _, _ in candidates],
            "component_names": [str(name) for name in component_names],
            "train_ids_sha256": train_ids_digest,
        }
        checkpoint_signature = hashlib.sha256(
            json.dumps(checkpoint_signature_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        checkpoint_run_dir = checkpoint_root / checkpoint_signature
        checkpoint_run_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = checkpoint_run_dir / "checkpoint_metadata.json"
        if not metadata_path.exists():
            metadata_path.write_text(
                json.dumps(
                    {
                        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "signature": checkpoint_signature,
                        **checkpoint_signature_payload,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

    def _score_checkpoint_path(score_index: int) -> Path | None:
        if checkpoint_run_dir is None:
            return None
        return checkpoint_run_dir / f"score_{score_index:06d}.npz"

    def _load_checkpoint_score(score_index: int) -> tuple[np.ndarray, np.ndarray] | None:
        score_path = _score_checkpoint_path(score_index)
        if score_path is None or not score_path.exists():
            return None
        try:
            with np.load(score_path, allow_pickle=False) as data:
                scores = np.asarray(data["scores"], dtype=float)
                component_scores = np.asarray(data["component_scores"], dtype=float)
        except (OSError, ValueError, KeyError) as exc:
            _append_parallel_diagnostic(
                "interaction_discovery",
                {
                    "event": "checkpoint_read_failure",
                    "score_index": score_index,
                    "checkpoint_path": str(score_path),
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                },
            )
            return None
        if scores.shape != (n_pairs,) or component_scores.shape != (n_pairs, n_comp):
            _append_parallel_diagnostic(
                "interaction_discovery",
                {
                    "event": "checkpoint_shape_mismatch",
                    "score_index": score_index,
                    "checkpoint_path": str(score_path),
                    "scores_shape": list(scores.shape),
                    "component_scores_shape": list(component_scores.shape),
                    "expected_scores_shape": [n_pairs],
                    "expected_component_scores_shape": [n_pairs, n_comp],
                },
            )
            return None
        return scores, component_scores

    def _save_checkpoint_score(score_index: int, result: tuple[np.ndarray, np.ndarray]) -> None:
        score_path = _score_checkpoint_path(score_index)
        if score_path is None:
            return
        scores, component_scores = result
        temp_path = score_path.with_suffix(".tmp.npz")
        np.savez_compressed(
            temp_path,
            scores=np.asarray(scores, dtype=float),
            component_scores=np.asarray(component_scores, dtype=float),
        )
        temp_path.replace(score_path)

    cached_prefix = 0
    for score_index in range(total_scores):
        score_path = _score_checkpoint_path(score_index)
        if score_path is None or not score_path.exists():
            break
        cached_prefix += 1

    indexed_results: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    # Use divisor 8 instead of 25 for larger batches (reduce parallelization overhead)
    batch_size = _progress_batch_size(total_scores, 8)
    batch_timeout_seconds = int(
        os.getenv("BSM_INTERACTION_BATCH_TIMEOUT_SECONDS", spec.parallel_batch_timeout_seconds)
    )
    parallel_backend = os.getenv("BSM_INTERACTION_PARALLEL_BACKEND", spec.parallel_backend)
    if parallel_backend not in {"loky", "threading", "dask"}:
        raise ValueError(
            "interaction_discovery.parallel_backend must be 'loky', 'threading', or 'dask'."
        )
    start_detail = "starting interaction score permutations"
    if cached_prefix > 0:
        start_detail = (
            "resuming interaction score permutations from checkpoints "
            f"({cached_prefix}/{total_scores})"
        )
    _report_progress(
        stage="interaction_discovery",
        completed=cached_prefix,
        total=total_scores,
        unit="permutation_scores",
        detail=start_detail,
    )
    with tempfile.TemporaryDirectory(prefix="bsm-interaction-joblib-") as temp_dir:
        for start in range(0, total_scores, batch_size):
            stop = min(start + batch_size, total_scores)
            batch_items = list(enumerate(seeds[start:stop], start=start))
            missing_batch_items: list[tuple[int, int]] = []
            for score_index, seed in batch_items:
                checkpointed = _load_checkpoint_score(score_index)
                if checkpointed is not None:
                    indexed_results[score_index] = checkpointed
                else:
                    missing_batch_items.append((score_index, seed))
            if missing_batch_items:
                batch_result: list[tuple[np.ndarray, np.ndarray]] | None = None
                try:
                    batch_result = _run_interaction_jobs(
                        missing_batch_items,
                        n_jobs=1 if spec.n_jobs <= 1 else spec.n_jobs,
                        backend=parallel_backend,
                        timeout_seconds=(None if spec.n_jobs <= 1 else batch_timeout_seconds),
                        temp_folder=(None if spec.n_jobs <= 1 else temp_dir),
                    )
                except (TimeoutError, mp.TimeoutError, RuntimeError, OSError) as exc:
                    _append_parallel_diagnostic(
                        "interaction_discovery",
                        {
                            "event": "parallel_batch_failure",
                            "batch_start": start,
                            "batch_stop": stop,
                            "batch_size": len(missing_batch_items),
                            "backend": parallel_backend,
                            "n_jobs": spec.n_jobs,
                            "timeout_seconds": batch_timeout_seconds,
                            "exception_type": type(exc).__name__,
                            "exception_message": str(exc),
                            "controller_pid": os.getpid(),
                            "worker_snapshot": _worker_process_snapshot(os.getpid()),
                        },
                    )
                    raise RuntimeError(
                        "Interaction discovery parallel batch failed; "
                        "no serial fallback or retries are allowed for n_jobs>1."
                    ) from exc
                assert batch_result is not None
                for (score_index, _), result in zip(missing_batch_items, batch_result, strict=True):
                    indexed_results[score_index] = result
                    _save_checkpoint_score(score_index, result)
            _report_progress(
                stage="interaction_discovery",
                completed=stop,
                total=total_scores,
                unit="permutation_scores",
            )

    all_results = [indexed_results[idx] for idx in range(total_scores)]
    observed_scores, observed_comp = all_results[0]
    null_statistics = np.zeros((spec.permutation_count_B, n_pairs))
    for b in range(spec.permutation_count_B):
        null_statistics[b] = all_results[b + 1][0]

    thresholds = np.quantile(null_statistics, spec.null_threshold_quantile, axis=0)
    p_values = (1.0 + (null_statistics >= observed_scores[None, :]).sum(axis=0)) / (
        spec.permutation_count_B + 1.0
    )
    retained = observed_scores > thresholds
    retained_term_names = {pair_name for pair_name, _, _ in candidates}
    pair_scores = _build_interaction_pair_scores(
        candidates=candidates,
        observed_scores=observed_scores,
        thresholds=thresholds,
        p_values=p_values,
        retained=retained,
        retained_term_names=retained_term_names,
        spec=spec,
    )
    component_scores = _build_component_interaction_scores(
        candidates=candidates,
        component_names=component_names,
        coefficients=observed_comp,
    )
    null_summary = _build_interaction_null_summary(candidates, null_statistics)
    retained_pairs = pair_scores.loc[pair_scores["retained"]].copy()
    retained_pairs = retained_pairs.sort_values(
        ["interaction_score", "pair_name"],
        ascending=[False, True],
        ignore_index=True,
    )
    provenance = _build_interaction_provenance(spec)
    summary = _build_interaction_discovery_summary(
        n_training_rows=len(y_train),
        n_candidate_pairs=len(candidates),
        n_empirical_null_retained_pairs=int(pair_scores["empirical_null_retained"].sum()),
        n_retained_pairs=len(retained_pairs),
        n_components=len(component_names),
        max_score=float(observed_scores.max()),
        spec=spec,
    )
    return InteractionDiscoveryResult(
        pair_scores=pair_scores,
        component_interaction_scores=component_scores,
        interaction_null_summary=null_summary,
        retained_pairs=retained_pairs,
        provenance=provenance,
        summary=summary,
    )


def write_interaction_discovery_artifacts(
    result: InteractionDiscoveryResult,
    output_root: Path,
) -> dict[str, Path]:
    """Write interaction-discovery artifacts under ``output_root``.

    Parameters
    ----------
    result
        Materialized interaction-discovery result.
    output_root
        Resolved manuscript output root from the runtime context.

    Returns
    -------
    dict[str, pathlib.Path]
        Paths keyed by stable artifact name.
    """
    stage_root = output_root / "interaction_discovery"
    stage_root.mkdir(parents=True, exist_ok=True)
    tables = {
        "interaction_pair_scores": result.pair_scores,
        "component_interaction_scores": result.component_interaction_scores,
        "interaction_null_summary": result.interaction_null_summary,
        "retained_interaction_pairs": result.retained_pairs,
        "interaction_discovery_provenance": result.provenance,
        "interaction_discovery_summary": result.summary,
    }
    written: dict[str, Path] = {}
    for name, table in tables.items():
        path = stage_root / f"{name}.csv"
        table.to_csv(path, index=False)
        written[name] = path
    return written


def run_interaction_discovery_stage(context: Any) -> InteractionDiscoveryStageResult:
    """Run interaction discovery from a manuscript notebook runtime context.

    Parameters
    ----------
    context
        ``bsm_rfm.manuscript_runtime.ManuscriptNotebookContext``. It is typed as ``Any`` here to
        avoid an import cycle between the runtime and stage modules.

    Returns
    -------
    InteractionDiscoveryStageResult
        In-memory result and written artifact paths.
    """
    conditioning_spec = output_conditioning_spec_from_case_study_config(context.case_study_config)
    conditioning = condition_manuscript_outputs(
        context.tables["case_study_output_matrix"],
        context.tables["fixed_holdout_assignments"],
        conditioning_spec,
    )
    screening_spec = empirical_null_screening_spec_from_case_study_config(context.case_study_config)
    screening = screen_manuscript_empirical_null_terms(
        context.tables["case_study_input_matrix"],
        context.tables["manuscript_feature_catalog"],
        context.tables["fixed_holdout_assignments"],
        conditioning.pca_scores,
        screening_spec,
        checkpoint_dir=context.runtime.output_root / "empirical_null_screen",
    )
    interaction_spec = interaction_discovery_spec_from_case_study_config(context.case_study_config)
    interactions = discover_manuscript_interactions(
        context.tables["case_study_input_matrix"],
        context.tables["manuscript_feature_catalog"],
        context.tables["fixed_holdout_assignments"],
        conditioning.pca_scores,
        screening.retained_terms,
        interaction_spec,
        checkpoint_dir=context.runtime.output_root / "interaction_discovery" / "_batch_checkpoints",
    )
    artifact_paths = write_interaction_discovery_artifacts(
        interactions,
        context.runtime.output_root,
    )
    return InteractionDiscoveryStageResult(interactions=interactions, artifact_paths=artifact_paths)


def nonlinear_discovery_spec_from_case_study_config(
    case_study_config: dict[str, Any],
) -> NonlinearDiscoverySpec:
    """Build the nonlinear-discovery specification from the case-study config.

    Parameters
    ----------
    case_study_config
        Parsed ``configs/manuscript_case_study.yml`` mapping.

    Returns
    -------
    NonlinearDiscoverySpec
        Typed nonlinear-discovery specification.
    """
    section = case_study_config["case_study"]["nonlinear_discovery"]
    runtime = case_study_config["case_study"].get("runtime", {})
    return NonlinearDiscoverySpec(
        method=str(section["method"]),
        curvature_rule=str(section["curvature_rule"]),
        replacement_selection_rule=str(section["replacement_selection_rule"]),
        identified_transformations_reference=int(section["identified_transformations"]),
        final_support_transformations_reference=int(section["final_support_transformations"]),
        implementation_method=str(
            section.get(
                "public_implementation_method",
                "residualized_parametric_transform_surrogate",
            )
        ),
        implementation_status=str(
            section.get("public_implementation_status", "source_backed_public_surrogate")
        ),
        source_workflow_reference=str(
            section.get(
                "source_workflow_reference",
                "private_gam_nonlinear_discovery_workflow",
            )
        ),
        source_workflow_equivalence_status=str(
            section.get("source_workflow_equivalence_status", "not_yet_validated")
        ),
        n_jobs=int(runtime.get("n_jobs", 1)),
    )


def discover_manuscript_nonlinear_transformations(
    input_matrix: pd.DataFrame,
    feature_catalog: pd.DataFrame,
    holdout_assignments: pd.DataFrame,
    pca_scores: pd.DataFrame,
    retained_terms: pd.DataFrame,
    spec: NonlinearDiscoverySpec,
    *,
    checkpoint_dir: Path | None = None,
    active_feature_indices: set[int] | None = None,
) -> NonlinearDiscoveryResult:
    """Detect nonlinear transformations via GAM curvature diagnostics.

    Candidate transformations are generated dynamically from first-order terms retained by empirical
    null screening, with domain guards applied for inverse, log, and square-root families. For each
    retained first-order feature, a cubic smoothing spline (GAM surrogate) is fitted against each
    active PCA component. The smooth's effective degrees of freedom (EDF) and an F-test p-value are
    used to declare curvature when EDF > ``minimum_curvature_score`` and p < 0.01 for any component.
    For each nonlinear feature, the algebraic transform family with minimum RMSE against the fitted
    GAM smooth is retained; at most one transform is retained per base feature.

    Parameters
    ----------
    input_matrix
        Case-study input table with ``sample_id`` and source input columns.
    feature_catalog
        Manuscript feature catalog (kept for interface compatibility).
    holdout_assignments
        Table with ``sample_id`` and ``split`` columns. Only train rows are scored.
    pca_scores
        Output-conditioning PCA score table with ``sample_id`` and component columns.
    retained_terms
        Empirical-null retained-term table with at least a ``feature_name`` column.
    spec
        Nonlinear-discovery specification.

    Returns
    -------
    NonlinearDiscoveryResult
        Materialized transformation-score, component-score, retained-transformation, and summary
        tables.
    """
    if spec.method != "gam_plus_restricted_parametric_replacement":
        raise ValueError(f"Unsupported nonlinear-discovery method: {spec.method}")
    expected_curvature_rule = "edf_gt_1_and_smooth_pvalue_lt_0p01"
    if spec.curvature_rule != expected_curvature_rule:
        raise ValueError(f"Unsupported nonlinear curvature rule: {spec.curvature_rule}")
    expected_replacement_rule = "minimum_training_rmse_against_gam_smooth"
    if spec.replacement_selection_rule != expected_replacement_rule:
        raise ValueError(
            f"Unsupported nonlinear replacement selection rule: {spec.replacement_selection_rule}"
        )
    if spec.minimum_curvature_score < 0.0:
        raise ValueError("minimum_curvature_score must be non-negative.")

    candidates = _generate_supported_nonlinear_candidates(
        _retained_first_order_term_names(retained_terms, input_matrix),
        input_matrix,
    )
    if not candidates:
        raise ValueError("feature_catalog does not contain supported nonlinear candidates.")
    component_names = _component_columns(pca_scores)
    train_ids = _train_sample_ids(holdout_assignments)
    y_train = _align_table_by_sample_id(pca_scores, train_ids, component_names, "PCA scores")
    if len(y_train) < 3:
        raise ValueError("Nonlinear discovery requires at least three training rows.")
    y_scaled, component_active = _standardize_for_screening(y_train)
    if not component_active.any():
        raise ValueError("All retained PCA components have zero training variance.")

    # Extract training rows for feature materialization.
    indexed = input_matrix.set_index("sample_id", drop=False)
    missing_ids = [s for s in train_ids if s not in indexed.index]
    if missing_ids:
        preview = ", ".join(str(v) for v in missing_ids[:5])
        raise ValueError(f"input matrix is missing sample_id values: {preview}")
    train_rows = indexed.loc[list(train_ids)].reset_index(drop=True)

    # Group candidates by base feature for per-feature GAM tests.
    candidates_by_base: dict[str, list[tuple[str, str, str]]] = {}
    for feat_name, base_feat, family in candidates:
        candidates_by_base.setdefault(base_feat, []).append((feat_name, base_feat, family))

    # Pre-fetch raw base feature values for all base features.
    x_by_base: dict[str, np.ndarray] = {
        base: _source_input_column(train_rows, base, base).to_numpy(dtype=float)
        for base in candidates_by_base
    }

    # Run GAM tests in parallel; each base feature is independent.
    gam_cache: dict[tuple[str, str], tuple[float, float]] = {}
    active_comp_indices = [i for i, a in enumerate(component_active) if a]
    edf_threshold = max(spec.minimum_curvature_score, 2.0)
    gam_p_threshold = 0.01

    base_feat_list = list(candidates_by_base.keys())
    base_feature_index = {name: idx for idx, name in enumerate(base_feat_list)}
    feature_results: dict[str, dict] = {}
    total_features = len(base_feat_list)
    active_feature_set = (
        set(range(total_features))
        if active_feature_indices is None
        else {
            int(index) for index in active_feature_indices if 0 <= int(index) < int(total_features)
        }
    )
    if not active_feature_set:
        raise ValueError("active_feature_indices must include at least one valid feature index.")
    active_feature_total = max(1, len(active_feature_set))
    batch_size = _progress_batch_size(total_features, 50)
    train_ids_digest = hashlib.sha256(
        "|".join(str(sample_id) for sample_id in train_ids).encode("utf-8")
    ).hexdigest()
    checkpoint_run_dir = _checkpoint_run_dir(
        checkpoint_dir,
        stage="nonlinear_discovery",
        signature_payload={
            "base_features": base_feat_list,
            "component_names": component_names,
            "train_ids_sha256": train_ids_digest,
            "minimum_curvature_score": float(spec.minimum_curvature_score),
        },
    )

    def _feature_checkpoint_path(base_index: int) -> Path | None:
        if checkpoint_run_dir is None:
            return None
        return checkpoint_run_dir / f"feature_{base_index:06d}.json"

    cached_feature_count = 0
    cached_active_feature_count = 0
    for base_index, base_feat in enumerate(base_feat_list):
        checkpoint_path = _feature_checkpoint_path(base_index)
        payload = _read_json_dict(checkpoint_path) if checkpoint_path is not None else None
        if payload is None:
            continue
        if payload.get("base_feature") != base_feat:
            continue
        result_payload = payload.get("feature_result")
        cache_payload = payload.get("cache_entries")
        if not isinstance(result_payload, dict) or not isinstance(cache_payload, list):
            continue
        feature_results[base_feat] = result_payload
        for entry in cache_payload:
            if not isinstance(entry, dict):
                continue
            component = entry.get("component")
            if not isinstance(component, str):
                continue
            try:
                edf = float(entry["edf"])
                p_value = float(entry["p_value"])
            except (KeyError, TypeError, ValueError):
                continue
            gam_cache[(base_feat, component)] = (edf, p_value)
        cached_feature_count += 1
        if base_index in active_feature_set:
            cached_active_feature_count += 1

    start_detail = "starting nonlinear base-feature scoring"
    if cached_feature_count > 0:
        start_detail = (
            "resuming nonlinear base-feature scoring "
            f"({cached_active_feature_count}/{active_feature_total})"
        )
    _report_progress(
        stage="nonlinear_discovery",
        completed=cached_active_feature_count,
        total=active_feature_total,
        unit="base_features",
        detail=start_detail,
    )
    with Parallel(n_jobs=spec.n_jobs) as parallel:
        for start in range(0, total_features, batch_size):
            stop = min(start + batch_size, total_features)
            batch_indices = list(range(start, stop))
            missing_batch = [
                idx
                for idx in batch_indices
                if idx in active_feature_set and base_feat_list[idx] not in feature_results
            ]
            jobs = [
                delayed(_score_one_nonlinear_feature)(
                    base_feat_list[idx],
                    candidates_by_base[base_feat],
                    x_by_base[base_feat],
                    y_scaled,
                    component_names,
                    active_comp_indices,
                    edf_threshold,
                    gam_p_threshold,
                )
                for idx in missing_batch
                for base_feat in [base_feat_list[idx]]
            ]
            if jobs:
                for base_feat, feat_result, cache_entries in parallel(jobs):
                    feature_results[base_feat] = feat_result
                    serialized_cache_entries: list[dict[str, Any]] = []
                    for key, val in cache_entries:
                        gam_cache[key] = val
                        serialized_cache_entries.append(
                            {
                                "component": str(key[1]),
                                "edf": float(val[0]),
                                "p_value": float(val[1]),
                            }
                        )
                    base_index = base_feature_index[base_feat]
                    checkpoint_path = _feature_checkpoint_path(base_index)
                    if checkpoint_path is not None:
                        _write_json_atomic(
                            checkpoint_path,
                            {
                                "base_feature": base_feat,
                                "feature_result": feat_result,
                                "cache_entries": serialized_cache_entries,
                            },
                        )
            _report_progress(
                stage="nonlinear_discovery",
                completed=sum(
                    1
                    for idx, base_feat in enumerate(base_feat_list)
                    if idx in active_feature_set and base_feat in feature_results
                ),
                total=active_feature_total,
                unit="base_features",
            )

    for base_feat in base_feat_list:
        if base_feat in feature_results:
            continue
        default_transform = candidates_by_base[base_feat][0][0]
        default_component = component_names[active_comp_indices[0] if active_comp_indices else 0]
        feature_results[base_feat] = {
            "nonlinear": False,
            "best_transform_name": default_transform,
            "best_edf": 2.0,
            "best_p": 1.0,
            "best_comp_name": default_component,
            "best_rmse": float("nan"),
        }
        for component_name in component_names:
            gam_cache.setdefault((base_feat, component_name), (2.0, 1.0))

    nonlinear_bases = {b for b, fr in feature_results.items() if fr["nonlinear"]}

    # Build transformation_scores table (all domain-valid candidates with GAM diagnostics).
    rows = []
    for feat_name, base_feat, family in candidates:
        fr = feature_results[base_feat]
        is_best = feat_name == fr["best_transform_name"]
        retained = bool(fr["nonlinear"] and is_best)
        rows.append(
            {
                "feature_name": feat_name,
                "base_feature": base_feat,
                "transformation_family": family,
                "curvature_score": fr["best_edf"],
                "gam_p_value": fr["best_p"],
                "best_component": fr["best_comp_name"],
                "replacement_training_rmse": fr["best_rmse"] if is_best else float("nan"),
                "active_transform": True,
                "empirical_null_retained": base_feat in nonlinear_bases,
                "retained": retained,
                "curvature_rule": spec.curvature_rule,
                "replacement_selection_rule": spec.replacement_selection_rule,
            }
        )

    transformation_scores = pd.DataFrame.from_records(rows).sort_values(
        ["retained", "curvature_score", "feature_name"],
        ascending=[False, False, True],
        ignore_index=True,
    )

    # Component-level GAM scores table (EDF and p-value per feature per component).
    comp_rows = []
    for feat_name, base_feat, _ in candidates:
        for comp_name in component_names:
            edf, p = gam_cache.get((base_feat, comp_name), (2.0, 1.0))
            comp_rows.append(
                {
                    "feature_name": feat_name,
                    "component": comp_name,
                    "smooth_edf": edf,
                    "gam_p_value": p,
                }
            )
    component_scores = pd.DataFrame.from_records(comp_rows)

    retained_transformations = transformation_scores.loc[transformation_scores["retained"]].copy()
    retained_transformations = retained_transformations.sort_values(
        ["curvature_score", "feature_name"],
        ascending=[False, True],
        ignore_index=True,
    )
    provenance = _build_nonlinear_provenance(spec)
    summary = _build_nonlinear_discovery_summary(
        n_training_rows=len(y_train),
        n_candidate_transformations=len(candidates),
        n_active_transformations=len(candidates),
        n_empirical_null_retained_transformations=int(
            transformation_scores["empirical_null_retained"].sum()
        ),
        n_retained_transformations=len(retained_transformations),
        n_components=len(component_names),
        max_curvature_score=float(transformation_scores["curvature_score"].max()),
        spec=spec,
    )
    return NonlinearDiscoveryResult(
        transformation_scores=transformation_scores,
        component_transformation_scores=component_scores,
        retained_transformations=retained_transformations,
        provenance=provenance,
        summary=summary,
    )


def write_nonlinear_discovery_artifacts(
    result: NonlinearDiscoveryResult,
    output_root: Path,
) -> dict[str, Path]:
    """Write nonlinear-discovery artifacts under ``output_root``.

    Parameters
    ----------
    result
        Materialized nonlinear-discovery result.
    output_root
        Resolved manuscript output root from the runtime context.

    Returns
    -------
    dict[str, pathlib.Path]
        Paths keyed by stable artifact name.
    """
    stage_root = output_root / "nonlinear_discovery"
    stage_root.mkdir(parents=True, exist_ok=True)
    tables = {
        "transformation_scores": result.transformation_scores,
        "component_transformation_scores": result.component_transformation_scores,
        "retained_transformations": result.retained_transformations,
        "nonlinear_discovery_provenance": result.provenance,
        "nonlinear_discovery_summary": result.summary,
    }
    written: dict[str, Path] = {}
    for name, table in tables.items():
        path = stage_root / f"{name}.csv"
        table.to_csv(path, index=False)
        written[name] = path
    return written


def run_nonlinear_discovery_stage(context: Any) -> NonlinearDiscoveryStageResult:
    """Run nonlinear discovery from a manuscript notebook runtime context.

    Parameters
    ----------
    context
        ``bsm_rfm.manuscript_runtime.ManuscriptNotebookContext``. It is typed as ``Any`` here to
        avoid an import cycle between the runtime and stage modules.

    Returns
    -------
    NonlinearDiscoveryStageResult
        In-memory result and written artifact paths.
    """
    conditioning_spec = output_conditioning_spec_from_case_study_config(context.case_study_config)
    conditioning = condition_manuscript_outputs(
        context.tables["case_study_output_matrix"],
        context.tables["fixed_holdout_assignments"],
        conditioning_spec,
    )
    screening_spec = empirical_null_screening_spec_from_case_study_config(context.case_study_config)
    screening = screen_manuscript_empirical_null_terms(
        context.tables["case_study_input_matrix"],
        context.tables["manuscript_feature_catalog"],
        context.tables["fixed_holdout_assignments"],
        conditioning.pca_scores,
        screening_spec,
        checkpoint_dir=context.runtime.output_root / "empirical_null_screen",
    )
    nonlinear_spec = nonlinear_discovery_spec_from_case_study_config(context.case_study_config)
    nonlinear = discover_manuscript_nonlinear_transformations(
        context.tables["case_study_input_matrix"],
        context.tables["manuscript_feature_catalog"],
        context.tables["fixed_holdout_assignments"],
        conditioning.pca_scores,
        screening.retained_terms,
        nonlinear_spec,
        checkpoint_dir=context.runtime.output_root / "nonlinear_discovery",
    )
    artifact_paths = write_nonlinear_discovery_artifacts(
        nonlinear,
        context.runtime.output_root,
    )
    return NonlinearDiscoveryStageResult(nonlinear=nonlinear, artifact_paths=artifact_paths)


def sparse_selection_stability_spec_from_case_study_config(
    case_study_config: dict[str, Any],
) -> SparseSelectionStabilitySpec:
    """Build the sparse-selection/stability specification from the case-study config.

    Parameters
    ----------
    case_study_config
        Parsed ``configs/manuscript_case_study.yml`` mapping.

    Returns
    -------
    SparseSelectionStabilitySpec
        Typed sparse-selection and stability specification.
    """
    case_study = case_study_config["case_study"]
    sparse = case_study["sparse_selection"]
    stability = case_study["stability"]
    runtime = case_study.get("runtime", {})
    count, fraction, seed = _parse_stability_resampling_scheme(str(stability["resampling_scheme"]))
    return SparseSelectionStabilitySpec(
        model_class=str(sparse["model_class"]),
        ebic_gamma=float(sparse["ebic_gamma"]),
        support_aggregation_rule=str(sparse["support_aggregation_rule"]),
        resampling_scheme=str(stability["resampling_scheme"]),
        subsample_count=count,
        subsample_fraction=fraction,
        jaccard_threshold=float(stability["jaccard_threshold"]),
        spearman_threshold=float(stability["spearman_threshold"]),
        implementation_method=str(
            sparse.get(
                "public_implementation_method",
                "ebic_l1_component_union_with_subsample_stability",
            )
        ),
        implementation_status=str(
            sparse.get("public_implementation_status", "source_backed_public_surrogate")
        ),
        source_workflow_reference=str(
            sparse.get("source_workflow_reference", "notebook_pca_debiased_lasso")
        ),
        source_workflow_equivalence_status=str(
            sparse.get("source_workflow_equivalence_status", "not_yet_validated")
        ),
        source_artifact=str(sparse.get("source_artifact", "LASSO_to_OLS_v9.ipynb")),
        source_selected_feature_count_reference=int(
            sparse.get("source_selected_feature_count_reference", 346)
        ),
        random_seed=seed,
        n_jobs=int(runtime.get("n_jobs", 1)),
        max_candidate_terms=(
            int(sparse["max_candidate_terms"])
            if sparse.get("max_candidate_terms") is not None
            else None
        ),
        adaptive_early_stopping_enabled=bool(
            stability.get("adaptive_early_stopping_enabled", False)
        ),
        stability_convergence_window=int(stability.get("stability_convergence_window", 3)),
    )


def select_manuscript_sparse_support(
    input_matrix: pd.DataFrame,
    feature_catalog: pd.DataFrame,
    holdout_assignments: pd.DataFrame,
    pca_scores: pd.DataFrame,
    retained_terms: pd.DataFrame,
    retained_interaction_pairs: pd.DataFrame,
    retained_transformations: pd.DataFrame,
    spec: SparseSelectionStabilitySpec,
    *,
    checkpoint_dir: Path | None = None,
    active_resample_indices: set[int] | None = None,
) -> SparseSelectionStabilityResult:
    """Run EBIC-selected L1 sparse selection plus deterministic stability filtering.

    Candidate terms are the feature-catalog-order union of terms retained by empirical-null
    screening, retained interaction pairs, and retained nonlinear transformations. For each
    retained PCA component, the public stage fits a deterministic L1 path and selects the penalty
    minimizing EBIC with the frozen gamma. The full support is the union of nonzero component
    supports. Stability is assessed with deterministic row subsamples without replacement.

    Parameters
    ----------
    input_matrix
        Case-study input table with ``sample_id`` and source input columns.
    feature_catalog
        Manuscript feature catalog used to materialize candidate terms.
    holdout_assignments
        Table with ``sample_id`` and ``split`` columns. Only train rows are fit.
    pca_scores
        Output-conditioning PCA score table with ``sample_id`` and component columns.
    retained_terms
        Empirical-null retained-term table.
    retained_interaction_pairs
        Interaction-discovery retained-pair table.
    retained_transformations
        Nonlinear-discovery retained-transformation table.
    spec
        Sparse-selection and stability specification.

    Returns
    -------
    SparseSelectionStabilityResult
        Materialized sparse-selection, stability, final-support, and summary tables.
    """
    _validate_sparse_selection_spec(spec)
    candidate_names = _ordered_sparse_candidate_names(
        feature_catalog=feature_catalog,
        retained_terms=retained_terms,
        retained_interaction_pairs=retained_interaction_pairs,
        retained_transformations=retained_transformations,
    )
    candidate_names = _apply_sparse_candidate_cap(
        ordered_candidates=candidate_names,
        retained_terms=retained_terms,
        retained_interaction_pairs=retained_interaction_pairs,
        retained_transformations=retained_transformations,
        max_candidate_terms=spec.max_candidate_terms,
    )
    candidate_catalog = _feature_catalog_subset(feature_catalog, candidate_names)
    design = build_manuscript_feature_design(input_matrix, candidate_catalog)
    component_names = _component_columns(pca_scores)
    train_ids = _train_sample_ids(holdout_assignments)
    x_train = _align_table_by_sample_id(design, train_ids, candidate_names, "candidate design")
    y_train = _align_table_by_sample_id(pca_scores, train_ids, component_names, "PCA scores")
    if len(x_train) < 4:
        raise ValueError("Sparse selection requires at least four training rows.")

    x_scaled, feature_active = _standardize_for_screening(x_train)
    y_scaled, component_active = _standardize_for_screening(y_train)
    if not component_active.any():
        raise ValueError("All retained PCA components have zero training variance.")
    x_scaled[:, ~feature_active] = 0.0

    full_selection = _fit_sparse_l1_ebic_models(
        x_scaled,
        y_scaled,
        feature_names=candidate_names,
        component_names=component_names,
        spec=spec,
        active_features=feature_active,
    )
    full_support_mask = np.any(
        np.abs(full_selection["coefficient_matrix"]) > 0.0,
        axis=1,
    )
    full_importance = np.max(np.abs(full_selection["coefficient_matrix"]), axis=1)
    resample_summary, resample_supports, resample_importances = _run_stability_resamples(
        x_scaled=x_scaled,
        y_scaled=y_scaled,
        feature_names=candidate_names,
        component_names=component_names,
        full_support_mask=full_support_mask,
        full_importance=full_importance,
        spec=spec,
        active_features=feature_active,
        active_resample_indices=active_resample_indices,
        checkpoint_dir=(
            None if checkpoint_dir is None else checkpoint_dir / "_resample_checkpoints"
        ),
    )

    mean_jaccard = float(resample_summary["jaccard_with_full_support"].mean())
    mean_spearman = float(resample_summary["spearman_with_full_importance"].mean())
    stability_feature_summary = _build_stability_feature_summary(
        feature_names=candidate_names,
        feature_active=feature_active,
        full_support_mask=full_support_mask,
        full_importance=full_importance,
        resample_supports=resample_supports,
        resample_importances=resample_importances,
        mean_jaccard=mean_jaccard,
        mean_spearman=mean_spearman,
        spec=spec,
    )
    final_stable_support = stability_feature_summary.loc[
        stability_feature_summary["final_stable_support"]
    ].copy()
    final_stable_support = final_stable_support.sort_values(
        ["full_support_importance", "feature_name"],
        ascending=[False, True],
        ignore_index=True,
    )
    support_candidates = _build_sparse_support_candidates(
        candidate_names=candidate_names,
        candidate_catalog=candidate_catalog,
        retained_terms=retained_terms,
        retained_interaction_pairs=retained_interaction_pairs,
        retained_transformations=retained_transformations,
    )
    component_model_selection = full_selection["component_model_selection"]
    component_coefficients = _build_sparse_component_coefficients(
        candidate_names=candidate_names,
        component_names=component_names,
        coefficients=full_selection["coefficient_matrix"],
    )
    summary = _build_sparse_selection_summary(
        n_training_rows=len(x_train),
        n_candidate_terms=len(candidate_names),
        n_active_candidate_terms=int(feature_active.sum()),
        n_components=len(component_names),
        n_full_support_terms=int(full_support_mask.sum()),
        n_final_stable_support_terms=len(final_stable_support),
        mean_jaccard=mean_jaccard,
        mean_spearman=mean_spearman,
        spec=spec,
    )
    provenance = _build_sparse_selection_provenance(
        n_candidate_terms=len(candidate_names),
        n_full_support_terms=int(full_support_mask.sum()),
        n_final_stable_support_terms=len(final_stable_support),
        spec=spec,
    )
    return SparseSelectionStabilityResult(
        support_candidates=support_candidates,
        component_model_selection=component_model_selection,
        component_coefficients=component_coefficients,
        stability_resample_summary=resample_summary,
        stability_feature_summary=stability_feature_summary,
        final_stable_support=final_stable_support,
        provenance=provenance,
        summary=summary,
    )


def write_sparse_selection_stability_artifacts(
    result: SparseSelectionStabilityResult,
    output_root: Path,
) -> dict[str, Path]:
    """Write sparse-selection and stability artifacts under ``output_root``.

    Parameters
    ----------
    result
        Materialized sparse-selection and stability result.
    output_root
        Resolved manuscript output root from the runtime context.

    Returns
    -------
    dict[str, pathlib.Path]
        Paths keyed by stable artifact name.
    """
    stage_root = output_root / "sparse_selection"
    stage_root.mkdir(parents=True, exist_ok=True)
    tables = {
        "support_candidates": result.support_candidates,
        "component_model_selection": result.component_model_selection,
        "component_coefficients": result.component_coefficients,
        "stability_resample_summary": result.stability_resample_summary,
        "stability_feature_summary": result.stability_feature_summary,
        "final_stable_support": result.final_stable_support,
        "sparse_selection_provenance": result.provenance,
        "sparse_selection_summary": result.summary,
    }

    # Consolidated diagnostics for quick CI/local parity inspection
    summary_row = result.summary.iloc[0]
    diagnostics = {
        "n_candidate_terms": int(summary_row["n_candidate_terms"]),
        "n_full_support_terms": int(summary_row["n_full_support_terms"]),
        "n_final_stable_support_terms": int(summary_row["n_final_stable_support_terms"]),
        "mean_resample_jaccard": float(summary_row["mean_resample_jaccard"]),
        "mean_resample_spearman": float(summary_row["mean_resample_spearman"]),
        "jaccard_threshold": float(summary_row["jaccard_threshold"]),
        "spearman_threshold": float(summary_row["spearman_threshold"]),
        "passes_jaccard_threshold": float(summary_row["mean_resample_jaccard"])
        >= float(summary_row["jaccard_threshold"]),
        "passes_spearman_threshold": float(summary_row["mean_resample_spearman"])
        >= float(summary_row["spearman_threshold"]),
        "final_stable_support_nonempty": bool(len(result.final_stable_support) > 0),
    }
    diagnostics_df = pd.DataFrame([diagnostics])
    tables["sparse_selection_diagnostics"] = diagnostics_df

    written: dict[str, Path] = {}
    for name, table in tables.items():
        path = stage_root / f"{name}.csv"
        table.to_csv(path, index=False)
        written[name] = path
    return written


def run_sparse_selection_stability_stage(context: Any) -> SparseSelectionStabilityStageResult:
    """Run sparse selection and stability from a manuscript notebook runtime context.

    Parameters
    ----------
    context
        ``bsm_rfm.manuscript_runtime.ManuscriptNotebookContext``. It is typed as ``Any`` here to
        avoid an import cycle between the runtime and stage modules.

    Returns
    -------
    SparseSelectionStabilityStageResult
        In-memory result and written artifact paths.
    """
    conditioning_spec = output_conditioning_spec_from_case_study_config(context.case_study_config)
    conditioning = condition_manuscript_outputs(
        context.tables["case_study_output_matrix"],
        context.tables["fixed_holdout_assignments"],
        conditioning_spec,
    )
    screening_spec = empirical_null_screening_spec_from_case_study_config(context.case_study_config)
    screening = screen_manuscript_empirical_null_terms(
        context.tables["case_study_input_matrix"],
        context.tables["manuscript_feature_catalog"],
        context.tables["fixed_holdout_assignments"],
        conditioning.pca_scores,
        screening_spec,
        checkpoint_dir=context.runtime.output_root / "empirical_null_screen",
    )
    interaction_spec = interaction_discovery_spec_from_case_study_config(context.case_study_config)
    interactions = discover_manuscript_interactions(
        context.tables["case_study_input_matrix"],
        context.tables["manuscript_feature_catalog"],
        context.tables["fixed_holdout_assignments"],
        conditioning.pca_scores,
        screening.retained_terms,
        interaction_spec,
        checkpoint_dir=context.runtime.output_root / "interaction_discovery" / "_batch_checkpoints",
    )
    nonlinear_spec = nonlinear_discovery_spec_from_case_study_config(context.case_study_config)
    nonlinear = discover_manuscript_nonlinear_transformations(
        context.tables["case_study_input_matrix"],
        context.tables["manuscript_feature_catalog"],
        context.tables["fixed_holdout_assignments"],
        conditioning.pca_scores,
        screening.retained_terms,
        nonlinear_spec,
        checkpoint_dir=context.runtime.output_root / "nonlinear_discovery",
    )
    sparse_spec = sparse_selection_stability_spec_from_case_study_config(context.case_study_config)
    sparse_selection = select_manuscript_sparse_support(
        context.tables["case_study_input_matrix"],
        context.tables["manuscript_feature_catalog"],
        context.tables["fixed_holdout_assignments"],
        conditioning.pca_scores,
        screening.retained_terms,
        interactions.retained_pairs,
        nonlinear.retained_transformations,
        sparse_spec,
        checkpoint_dir=context.runtime.output_root / "sparse_selection",
    )
    artifact_paths = write_sparse_selection_stability_artifacts(
        sparse_selection,
        context.runtime.output_root,
    )
    return SparseSelectionStabilityStageResult(
        sparse_selection=sparse_selection,
        artifact_paths=artifact_paths,
    )


def final_manuscript_artifacts_spec_from_case_study_config(
    case_study_config: dict[str, Any],
) -> FinalManuscriptArtifactsSpec:
    """Build final-model and manuscript-artifact settings from the case-study config.

    Parameters
    ----------
    case_study_config
        Parsed ``configs/manuscript_case_study.yml`` mapping.

    Returns
    -------
    FinalManuscriptArtifactsSpec
        Typed final artifact-regeneration specification.
    """
    case_study = case_study_config["case_study"]
    final_model = case_study["final_model"]
    inferential_filter = case_study["final_inferential_filter"]
    feature_pruning = inferential_filter.get("feature_pruning", {})
    interface = case_study.get("interface", {})
    runtime = case_study.get("runtime", {})
    return FinalManuscriptArtifactsSpec(
        final_predictor_count_reference=int(final_model["final_predictor_count"]),
        final_first_order_input_count_reference=int(final_model["final_first_order_input_count"]),
        intermediate_penalized_holdout_nrmse_reference=float(
            final_model["intermediate_penalized_holdout_nrmse"]
        ),
        final_ols_holdout_nrmse_reference=float(final_model["final_ols_holdout_nrmse"]),
        nrmse_denominator_definition=str(final_model["nrmse_denominator_definition"]),
        nrmse_min_range=float(final_model["nrmse_min_range"]),
        nrmse_reference_matrix=str(final_model["nrmse_reference_matrix"]),
        bootstrap_count=int(final_model.get("bootstrap_count", 200)),
        bootstrap_alpha=float(final_model.get("bootstrap_alpha", 0.05)),
        random_seed=int(interface.get("holdout_random_seed", 123)),
        inferential_filter_interval_method=str(inferential_filter["interval_method"]),
        inferential_filter_alpha=float(inferential_filter.get("alpha", 0.05)),
        hc3_output_subset_mode=str(inferential_filter.get("output_subset_mode", "all")),
        hc3_output_fraction=(
            float(inferential_filter["output_fraction"])
            if inferential_filter.get("output_fraction") is not None
            else None
        ),
        hc3_output_names=tuple(str(name) for name in inferential_filter.get("output_names", [])),
        hc3_output_max_outputs=(
            int(inferential_filter["max_outputs"])
            if inferential_filter.get("max_outputs") is not None
            else None
        ),
        hc3_output_random_seed=int(inferential_filter.get("random_seed", 123)),
        hc3_output_subset_metric=str(inferential_filter.get("subset_metric", "variance")),
        pruning_error_scale_quantile=float(feature_pruning.get("error_scale_quantile", 0.95)),
        pruning_delta_threshold_override=(
            float(feature_pruning["delta_threshold_override"])
            if feature_pruning.get("delta_threshold_override") is not None
            else None
        ),
        pruning_remove_count_override=(
            int(feature_pruning["remove_count_override"])
            if feature_pruning.get("remove_count_override") is not None
            else None
        ),
        n_jobs=int(runtime.get("n_jobs", 1)),
    )


def _fit_ablation_ols_nrmse(
    feature_names: list[str],
    feature_catalog: pd.DataFrame,
    input_matrix: pd.DataFrame,
    train_ids: pd.Series,
    holdout_ids: pd.Series,
    y_train: pd.DataFrame,
    y_holdout: pd.DataFrame,
    null_predictions: np.ndarray,
    spec: FinalManuscriptArtifactsSpec,
    *,
    prebuilt_design: pd.DataFrame | None = None,
    checkpoint_dir: Path | None = None,
    active_bootstrap_indices: set[int] | None = None,
) -> dict[str, Any]:
    """Fit an OLS ablation model and return a performance row dict.

    Falls back to null-mean predictions when ``feature_names`` is empty or
    when no requested features appear in the materialized design.

    Parameters
    ----------
    prebuilt_design
        Optional pre-materialized design matrix (e.g. a superset built by the caller).
        When provided, columns are subset to ``feature_names`` without rebuilding.
    """
    if not feature_names:
        predictions: np.ndarray = null_predictions
        n_features = 0
    else:
        if prebuilt_design is not None:
            design = prebuilt_design
        else:
            catalog_sub = _feature_catalog_subset(feature_catalog, feature_names)
            design = build_manuscript_feature_design(input_matrix, catalog_sub)
        names_present = [n for n in feature_names if n in design.columns]
        if not names_present:
            predictions = null_predictions
            n_features = 0
        else:
            x_tr = _indexed_by_sample_id(
                _align_table_by_sample_id(design, train_ids, names_present, "abl_train"),
                train_ids,
            )
            x_ho = _indexed_by_sample_id(
                _align_table_by_sample_id(design, holdout_ids, names_present, "abl_holdout"),
                holdout_ids,
            )
            fit = fit_final_ols(x_tr, y_train)
            predictions = predict_final_ols(fit, x_ho).to_numpy(dtype=float)
            n_features = len(names_present)

    metric = bootstrap_macro_nrmse_ci(
        y_holdout.to_numpy(dtype=float),
        predictions,
        y_train.to_numpy(dtype=float),
        min_range=spec.nrmse_min_range,
        n_boot=spec.bootstrap_count,
        alpha=spec.bootstrap_alpha,
        random_state=spec.random_seed,
        n_jobs=spec.n_jobs,
        checkpoint_dir=checkpoint_dir,
        active_bootstrap_indices=active_bootstrap_indices,
    )
    return {
        "n_features": n_features,
        "nrmse": metric["point_estimate"],
        "ci_lower": metric["ci_lower"],
        "ci_upper": metric["ci_upper"],
    }


def _compute_ablation_table(
    feature_catalog: pd.DataFrame,
    input_matrix: pd.DataFrame,
    train_ids: pd.Series,
    holdout_ids: pd.Series,
    y_train: pd.DataFrame,
    y_holdout: pd.DataFrame,
    screening_retained_terms: pd.DataFrame,
    prefilter_feature_names: list[str],
    final_feature_names: list[str],
    null_predictions: np.ndarray,
    final_metric: dict[str, Any],
    spec: FinalManuscriptArtifactsSpec,
    *,
    checkpoint_dir: Path | None = None,
    active_bootstrap_indices: set[int] | None = None,
) -> pd.DataFrame:
    """Build a five-row ablation performance table.

    Models (in workflow order):
    - ``null_mean``: training-mean baseline
    - ``main_effects_ols``: OLS on all first-order catalog features
    - ``screened_ols``: OLS on empirical-null screened features
    - ``penalized_ols``: OLS on sparse/stability support before HC3 filter
    - ``final_ols``: OLS on HC3-filtered final support

    The design matrix is built once from the union of all ablation feature sets and
    subset per model, avoiding redundant materialization of interaction and nonlinear
    feature columns.
    """
    ablation_total = 5
    ablation_done = 0
    _report_progress(
        stage="final_ablation",
        completed=ablation_done,
        total=ablation_total,
        unit="models",
        detail="start",
    )

    null_metric = bootstrap_macro_nrmse_ci(
        y_holdout.to_numpy(dtype=float),
        null_predictions,
        y_train.to_numpy(dtype=float),
        min_range=spec.nrmse_min_range,
        n_boot=spec.bootstrap_count,
        alpha=spec.bootstrap_alpha,
        random_state=spec.random_seed,
        n_jobs=spec.n_jobs,
        checkpoint_dir=(None if checkpoint_dir is None else checkpoint_dir / "null_mean"),
        active_bootstrap_indices=active_bootstrap_indices,
    )
    ablation_done = 1
    _report_progress(
        stage="final_ablation",
        completed=ablation_done,
        total=ablation_total,
        unit="models",
        detail="null_mean complete",
    )

    first_order_names = _ablation_main_effect_feature_names(feature_catalog, input_matrix)
    screened_names = list(screening_retained_terms["feature_name"])

    # Build one superset design covering all ablation models; subset columns per model.
    all_ablation_names = list(
        dict.fromkeys(first_order_names + screened_names + prefilter_feature_names)
    )
    superset_catalog = _feature_catalog_subset(feature_catalog, all_ablation_names)
    superset_design = build_manuscript_feature_design(input_matrix, superset_catalog)

    common_kwargs: dict[str, Any] = dict(
        feature_catalog=feature_catalog,
        input_matrix=input_matrix,
        train_ids=train_ids,
        holdout_ids=holdout_ids,
        y_train=y_train,
        y_holdout=y_holdout,
        null_predictions=null_predictions,
        spec=spec,
        prebuilt_design=superset_design,
    )

    rows = [
        {
            "model_name": "null_mean",
            "n_features": 0,
            "nrmse": null_metric["point_estimate"],
            "ci_lower": null_metric["ci_lower"],
            "ci_upper": null_metric["ci_upper"],
        }
    ]

    rows.append(
        {
            "model_name": "main_effects_ols",
            **_fit_ablation_ols_nrmse(
                first_order_names,
                **common_kwargs,
                checkpoint_dir=(
                    None if checkpoint_dir is None else checkpoint_dir / "main_effects_ols"
                ),
                active_bootstrap_indices=active_bootstrap_indices,
            ),
        }
    )
    _report_progress(
        stage="final_ablation",
        completed=2,
        total=ablation_total,
        unit="models",
        detail="main_effects_ols complete",
    )

    rows.append(
        {
            "model_name": "screened_ols",
            **_fit_ablation_ols_nrmse(
                screened_names,
                **common_kwargs,
                checkpoint_dir=(
                    None if checkpoint_dir is None else checkpoint_dir / "screened_ols"
                ),
                active_bootstrap_indices=active_bootstrap_indices,
            ),
        }
    )
    _report_progress(
        stage="final_ablation",
        completed=3,
        total=ablation_total,
        unit="models",
        detail="screened_ols complete",
    )

    rows.append(
        {
            "model_name": "penalized_ols",
            **_fit_ablation_ols_nrmse(
                prefilter_feature_names,
                **common_kwargs,
                checkpoint_dir=(
                    None if checkpoint_dir is None else checkpoint_dir / "penalized_ols"
                ),
                active_bootstrap_indices=active_bootstrap_indices,
            ),
        }
    )
    _report_progress(
        stage="final_ablation",
        completed=4,
        total=ablation_total,
        unit="models",
        detail="penalized_ols complete",
    )

    rows.append(
        {
            "model_name": "final_ols",
            "n_features": len(final_feature_names),
            "nrmse": final_metric["point_estimate"],
            "ci_lower": final_metric["ci_lower"],
            "ci_upper": final_metric["ci_upper"],
        }
    )
    _report_progress(
        stage="final_ablation",
        completed=5,
        total=ablation_total,
        unit="models",
        detail="all models complete",
    )
    return pd.DataFrame(rows)


def _ablation_main_effect_feature_names(
    feature_catalog: pd.DataFrame,
    input_matrix: pd.DataFrame,
) -> list[str]:
    """Return direct-input feature names for the main-effects ablation model.

    Main effects should include first-order terms regardless of whether the catalog labels
    them as ``first_order`` (legacy) or ``numeric`` (current publication catalog).
    """
    if "feature_name" not in feature_catalog.columns:
        raise ValueError("feature_catalog must include a feature_name column.")
    if "sample_id" not in input_matrix.columns:
        raise ValueError("input_matrix must include a sample_id column.")

    feature_types = (
        feature_catalog["feature_type"].astype(str).str.lower()
        if "feature_type" in feature_catalog.columns
        else pd.Series([""] * len(feature_catalog), index=feature_catalog.index)
    )
    available_inputs = {
        str(column) for column in input_matrix.columns if str(column) != "sample_id"
    }

    names: list[str] = []
    for feature_name, feature_type in zip(
        feature_catalog["feature_name"].astype(str),
        feature_types,
        strict=False,
    ):
        if feature_name not in available_inputs:
            continue
        if feature_type in {"first_order", "numeric"}:
            names.append(feature_name)

    deduped = list(dict.fromkeys(names))
    if deduped:
        return deduped
    return [str(column) for column in input_matrix.columns if str(column) != "sample_id"]


def _build_per_output_nrmse_summary(per_output_df: pd.DataFrame) -> pd.DataFrame:
    """Summarise per-output nRMSE with quantiles and worst-output diagnostics.

    Returns a single-row DataFrame with columns:
    ``n_total``, ``n_included``, ``p10``, ``p25``, ``p50``, ``p75``, ``p90``,
    and up to three worst-output pairs ``worst_1_name`` / ``worst_1_nrmse`` etc.
    """
    included = per_output_df.loc[per_output_df["included_in_macro"]].copy()
    n_total = int(len(per_output_df))
    n_included = int(len(included))

    if n_included == 0:
        row: dict[str, Any] = {
            "n_total": n_total,
            "n_included": 0,
            "p10": float("nan"),
            "p25": float("nan"),
            "p50": float("nan"),
            "p75": float("nan"),
            "p90": float("nan"),
        }
        for rank in range(1, 4):
            row[f"worst_{rank}_name"] = None
            row[f"worst_{rank}_nrmse"] = float("nan")
        return pd.DataFrame([row])

    nrmse_vals = included["nrmse"].to_numpy(dtype=float)
    quantiles = (
        float(np.nanquantile(nrmse_vals, 0.10)),
        float(np.nanquantile(nrmse_vals, 0.25)),
        float(np.nanquantile(nrmse_vals, 0.50)),
        float(np.nanquantile(nrmse_vals, 0.75)),
        float(np.nanquantile(nrmse_vals, 0.90)),
    )

    worst = included.nlargest(min(3, n_included), "nrmse").reset_index(drop=True)
    row = {
        "n_total": n_total,
        "n_included": n_included,
        "p10": quantiles[0],
        "p25": quantiles[1],
        "p50": quantiles[2],
        "p75": quantiles[3],
        "p90": quantiles[4],
    }
    for rank in range(1, 4):
        if rank - 1 < len(worst):
            row[f"worst_{rank}_name"] = worst.loc[rank - 1, "output_name"]
            row[f"worst_{rank}_nrmse"] = float(worst.loc[rank - 1, "nrmse"])
        else:
            row[f"worst_{rank}_name"] = None
            row[f"worst_{rank}_nrmse"] = float("nan")
    return pd.DataFrame([row])


def regenerate_final_manuscript_artifacts(
    input_matrix: pd.DataFrame,
    output_matrix: pd.DataFrame,
    feature_catalog: pd.DataFrame,
    holdout_assignments: pd.DataFrame,
    conditioning: OutputConditioningResult,
    screening: EmpiricalNullScreeningResult,
    interactions: InteractionDiscoveryResult,
    nonlinear: NonlinearDiscoveryResult,
    sparse_selection: SparseSelectionStabilityResult,
    spec: FinalManuscriptArtifactsSpec,
    *,
    checkpoint_dir: Path | None = None,
    active_bootstrap_indices: set[int] | None = None,
) -> FinalManuscriptArtifactsResult:
    """Regenerate final OLS, manuscript tables, and figure-source artifacts.

    The final support is the stable sparse-selection support. The OLS model is fit on
    training rows and evaluated on the fixed holdout rows against retained scalar outputs
    from output conditioning. Macro nRMSE is normalized by the training response range,
    matching the frozen manuscript contract.

    Parameters
    ----------
    input_matrix
        Case-study input table with ``sample_id`` and source input columns.
    output_matrix
        Case-study scalar-output table with ``sample_id`` and output columns.
    feature_catalog
        Manuscript feature catalog used to materialize final support features.
    holdout_assignments
        Table with ``sample_id`` and ``split`` columns.
    conditioning
        Output-conditioning result used to identify retained scalar outputs.
    screening
        Empirical-null screening result used for workflow-stage summaries.
    interactions
        Interaction-discovery result used for workflow-stage summaries.
    nonlinear
        Nonlinear-discovery result used for workflow-stage summaries.
    sparse_selection
        Sparse-selection result whose stable support defines the final OLS feature set.
    spec
        Final artifact-regeneration specification.

    Returns
    -------
    FinalManuscriptArtifactsResult
        Materialized final-model, manuscript-table, and SVG figure artifacts.
    """
    _validate_final_manuscript_artifacts_spec(spec)
    final_steps_total = 10
    final_step = 0

    def _final_progress(detail: str) -> None:
        _report_progress(
            stage="final_manuscript_artifacts",
            completed=final_step,
            total=final_steps_total,
            unit="substeps",
            detail=detail,
        )

    _final_progress("start")
    prefilter_feature_names = _final_support_feature_names(sparse_selection.final_stable_support)
    prefilter_catalog = _feature_catalog_subset(feature_catalog, prefilter_feature_names)
    prefilter_support_features = _build_final_support_features(
        final_feature_names=prefilter_feature_names,
        final_catalog=prefilter_catalog,
        sparse_selection=sparse_selection,
    )
    prefilter_design = build_manuscript_feature_design(input_matrix, prefilter_catalog)
    retained_outputs = list(conditioning.retained_output_names)
    train_ids = _train_sample_ids(holdout_assignments)
    holdout_ids = _holdout_sample_ids(holdout_assignments)
    if holdout_ids.empty:
        raise ValueError("Final manuscript artifacts require at least one holdout row.")
    final_step = 1
    _final_progress(
        f"prefilter prepared; train_rows={len(train_ids)}, holdout_rows={len(holdout_ids)}"
    )
    x_prefilter_train = _indexed_by_sample_id(
        _align_table_by_sample_id(
            prefilter_design,
            train_ids,
            prefilter_feature_names,
            "prefilter final design",
        ),
        train_ids,
    )
    y_train = _indexed_by_sample_id(
        _align_output_matrix(output_matrix, train_ids, retained_outputs),
        train_ids,
    )
    hc3_wald_intervals, hc3_filter_summary = _build_hc3_inferential_filter_tables(
        x_prefilter_train,
        y_train,
        alpha=spec.inferential_filter_alpha,
        interval_method=spec.inferential_filter_interval_method,
        output_subset_mode=spec.hc3_output_subset_mode,
        output_fraction=spec.hc3_output_fraction,
        output_names=spec.hc3_output_names,
        max_outputs=spec.hc3_output_max_outputs,
        random_seed=spec.hc3_output_random_seed,
        subset_metric=spec.hc3_output_subset_metric,
    )
    final_step = 2
    _final_progress("hc3 inferential filter complete")
    hc3_feature_names = _hc3_retained_feature_names(hc3_filter_summary)
    hc3_catalog = _feature_catalog_subset(feature_catalog, hc3_feature_names)
    hc3_support_features = _build_final_support_features(
        final_feature_names=hc3_feature_names,
        final_catalog=hc3_catalog,
        sparse_selection=sparse_selection,
    )
    hc3_support_features = hc3_support_features.merge(
        hc3_filter_summary,
        on="feature_name",
        how="left",
        validate="one_to_one",
    )
    final_design = build_manuscript_feature_design(input_matrix, hc3_catalog)
    x_hc3_train = _indexed_by_sample_id(
        _align_table_by_sample_id(final_design, train_ids, hc3_feature_names, "final design"),
        train_ids,
    )
    x_hc3_holdout = _indexed_by_sample_id(
        _align_table_by_sample_id(
            final_design,
            holdout_ids,
            hc3_feature_names,
            "final holdout design",
        ),
        holdout_ids,
    )
    y_holdout = _indexed_by_sample_id(
        _align_output_matrix(output_matrix, holdout_ids, retained_outputs),
        holdout_ids,
    )
    final_step = 3
    _final_progress(
        "final design aligned; "
        f"outputs={len(retained_outputs)}, hc3_features={len(hc3_feature_names)}"
    )

    hc3_fit = fit_final_ols(x_hc3_train, y_train)
    hc3_predictions = predict_final_ols(hc3_fit, x_hc3_holdout)
    feature_pruning_impact, figure_feature_pruning_curve_data, feature_pruning_summary = (
        _build_feature_pruning_diagnostics(
            final_fit=hc3_fit,
            x_holdout=x_hc3_holdout,
            y_holdout=y_holdout,
            y_train=y_train,
            final_support_features=hc3_support_features,
            spec=spec,
        )
    )
    auto_remove_count = int(feature_pruning_summary.loc[0, "auto_remove_count"])
    effective_remove_count = int(feature_pruning_summary.loc[0, "effective_remove_count"])
    removed_feature_names = set(
        feature_pruning_impact.loc[
            feature_pruning_impact["selected_by_effective_cutoff"].astype(bool),
            "feature_name",
        ].astype(str)
    )
    final_feature_names = [
        str(name) for name in hc3_feature_names if str(name) not in removed_feature_names
    ]
    if not final_feature_names:
        raise ValueError(
            "Feature pruning removed all HC3-retained features; at least one is required."
        )
    final_catalog = _feature_catalog_subset(feature_catalog, final_feature_names)
    final_support_features = _build_final_support_features(
        final_feature_names=final_feature_names,
        final_catalog=final_catalog,
        sparse_selection=sparse_selection,
    )
    final_support_features = final_support_features.merge(
        hc3_filter_summary,
        on="feature_name",
        how="left",
        validate="one_to_one",
    )
    pruning_columns = [
        "feature_name",
        "delta_nrmse_when_feature_removed",
        "delta_nrmse_nonnegative",
        "remove_rank",
        "selected_by_auto_cutoff",
        "selected_by_effective_cutoff",
    ]
    final_support_features = final_support_features.merge(
        feature_pruning_impact.loc[:, pruning_columns],
        on="feature_name",
        how="left",
        validate="one_to_one",
    )
    x_train = x_hc3_train.loc[:, final_feature_names].copy()
    x_holdout = x_hc3_holdout.loc[:, final_feature_names].copy()
    if len(final_feature_names) == len(hc3_feature_names):
        final_fit = hc3_fit
        final_predictions = hc3_predictions
    else:
        final_fit = fit_final_ols(x_train, y_train)
        final_predictions = predict_final_ols(final_fit, x_holdout)
    final_step = 4
    _final_progress(
        "final OLS fit + predictions complete; "
        f"pruned_features={len(hc3_feature_names) - len(final_feature_names)}"
    )
    final_metric = bootstrap_macro_nrmse_ci(
        y_holdout.to_numpy(dtype=float),
        final_predictions.to_numpy(dtype=float),
        y_train.to_numpy(dtype=float),
        min_range=spec.nrmse_min_range,
        n_boot=spec.bootstrap_count,
        alpha=spec.bootstrap_alpha,
        random_state=spec.random_seed,
        n_jobs=spec.n_jobs,
        active_bootstrap_indices=active_bootstrap_indices,
        checkpoint_dir=(
            None if checkpoint_dir is None else checkpoint_dir / "_bootstrap_final_ols"
        ),
    )
    final_step = 5
    _final_progress(f"final metric bootstrap complete; n_boot={spec.bootstrap_count}")
    null_predictions = make_null_mean_prediction(
        y_train.to_numpy(dtype=float),
        n_rows=len(y_holdout),
    )
    null_metric = bootstrap_macro_nrmse_ci(
        y_holdout.to_numpy(dtype=float),
        null_predictions,
        y_train.to_numpy(dtype=float),
        min_range=spec.nrmse_min_range,
        n_boot=spec.bootstrap_count,
        alpha=spec.bootstrap_alpha,
        random_state=spec.random_seed,
        n_jobs=spec.n_jobs,
        active_bootstrap_indices=active_bootstrap_indices,
        checkpoint_dir=(
            None if checkpoint_dir is None else checkpoint_dir / "_bootstrap_null_mean"
        ),
    )
    final_step = 6
    _final_progress("null metric bootstrap complete")

    ablation_table = _compute_ablation_table(
        feature_catalog=feature_catalog,
        input_matrix=input_matrix,
        train_ids=train_ids,
        holdout_ids=holdout_ids,
        y_train=y_train,
        y_holdout=y_holdout,
        screening_retained_terms=screening.retained_terms,
        prefilter_feature_names=list(prefilter_feature_names),
        final_feature_names=list(final_feature_names),
        null_predictions=null_predictions,
        final_metric=final_metric,
        spec=spec,
        checkpoint_dir=(None if checkpoint_dir is None else checkpoint_dir / "_ablation"),
        active_bootstrap_indices=active_bootstrap_indices,
    )
    final_step = 7
    _final_progress("ablation table complete")

    per_output_nrmse = per_output_nrmse_frame(
        y_holdout.to_numpy(dtype=float),
        final_predictions.to_numpy(dtype=float),
        y_train.to_numpy(dtype=float),
        output_names=list(retained_outputs),
        min_range=spec.nrmse_min_range,
    )
    per_output_nrmse_summary = _build_per_output_nrmse_summary(per_output_nrmse)
    final_step = 8
    _final_progress("per-output metrics complete")
    feature_pruning_summary = feature_pruning_summary.copy()
    feature_pruning_summary["final_refit_holdout_nrmse"] = float(final_metric["point_estimate"])
    feature_pruning_summary["final_refit_n_features"] = int(len(final_feature_names))
    feature_pruning_summary["final_refit_removed_features"] = int(
        len(hc3_feature_names) - len(final_feature_names)
    )

    coefficient_matrix_raw_scale = make_coefficient_matrix_frame(
        final_fit.coef_raw_scale,
        output_names=list(final_fit.output_names),
        feature_names=list(final_fit.feature_names),
    )
    coefficient_matrix_standardized = make_coefficient_matrix_frame(
        final_fit.coef_standardized,
        output_names=list(final_fit.output_names),
        feature_names=list(final_fit.feature_names),
    )
    x_standardization = make_standardization_frame(
        list(final_fit.feature_names),
        final_fit.x_means,
        final_fit.x_scales,
        name_column="feature_name",
    )
    y_standardization = make_standardization_frame(
        list(final_fit.output_names),
        final_fit.y_means,
        final_fit.y_scales,
        name_column="output_name",
    )

    final_ols_summary = _build_final_ols_summary(
        n_training_rows=len(x_train),
        n_holdout_rows=len(x_holdout),
        n_prefilter_features=len(prefilter_feature_names),
        n_hc3_features=len(hc3_feature_names),
        n_features=len(final_feature_names),
        n_outputs=len(retained_outputs),
        final_metric=final_metric,
        null_metric=null_metric,
        spec=spec,
    )
    model_performance = _build_model_performance_table(
        final_metric=final_metric,
        null_metric=null_metric,
        spec=spec,
    )
    workflow_stage_summary = _build_workflow_stage_summary(
        conditioning=conditioning,
        screening=screening,
        interactions=interactions,
        nonlinear=nonlinear,
        sparse_selection=sparse_selection,
        hc3_filter_summary=hc3_filter_summary,
        final_ols_summary=final_ols_summary,
        spec=spec,
    )
    figure_model_performance_data = _build_model_performance_figure_data(model_performance)
    figure_support_composition_data = _build_support_composition_figure_data(final_support_features)
    figure_selected_by_module_data = _build_selected_by_module_figure_data(final_support_features)
    figure_nrmse_summary_data = _build_nrmse_summary_figure_data(ablation_table)
    legacy_feature_type_counts = _build_legacy_feature_type_counts(final_support_features)
    legacy_influential_counts_by_module = _build_legacy_influential_counts_by_module(
        final_support_features
    )
    (
        legacy_interaction_counts_by_module_pair,
        legacy_interaction_density_matrix_indexed,
    ) = _build_legacy_interaction_counts_by_module_pair(final_support_features)
    legacy_interaction_density_module_matrix = (
        legacy_interaction_density_matrix_indexed.reset_index().rename(columns={"index": "module"})
    )
    legacy_module_total_interactions = _build_legacy_module_total_interactions(
        legacy_interaction_density_matrix_indexed
    )
    svg_figures = {
        "figure_model_performance": _render_horizontal_bar_svg(
            figure_model_performance_data,
            label_column="display_name",
            value_column="nrmse",
            title="Holdout macro nRMSE",
        ),
        "figure_support_composition": _render_horizontal_bar_svg(
            figure_support_composition_data,
            label_column="feature_type",
            value_column="n_features",
            title="Final support composition",
        ),
        "figure_selected_by_module_count": _render_horizontal_bar_svg(
            figure_selected_by_module_data,
            label_column="module",
            value_column="n_selected_inputs",
            title="Selected inputs by module (count)",
        ),
        "figure_selected_by_module_share": _render_horizontal_bar_svg(
            figure_selected_by_module_data,
            label_column="module",
            value_column="share_selected_support",
            title="Selected inputs by module (share)",
        ),
        "figure_nrmse_bootstrap_summary": _render_nrmse_summary_svg(figure_nrmse_summary_data),
        "fig_feature_type_distribution": _render_horizontal_bar_svg(
            legacy_feature_type_counts,
            label_column="feature_type",
            value_column="count",
            title="Distribution of Influential Feature Types",
        ),
        "fig_influential_by_module": _render_horizontal_bar_svg(
            legacy_influential_counts_by_module.sort_values("module", ignore_index=True),
            label_column="module",
            value_column="count",
            title="Influential Features by Module",
        ),
        "fig_module_pair_heatmap": _render_module_pair_heatmap_svg(
            legacy_interaction_density_matrix_indexed
        ),
        "fig_module_total_interactions": _render_horizontal_bar_svg(
            legacy_module_total_interactions,
            label_column="module",
            value_column="total_interactions",
            title="Total Interactions by Module",
        ),
        "figure_feature_pruning_curve": _render_feature_pruning_curve_svg(
            figure_feature_pruning_curve_data,
            auto_remove_count=auto_remove_count,
            effective_remove_count=effective_remove_count,
        ),
    }
    figure_specs = _build_figure_specs(
        figure_model_performance_data=figure_model_performance_data,
        figure_support_composition_data=figure_support_composition_data,
        figure_selected_by_module_data=figure_selected_by_module_data,
        figure_nrmse_summary_data=figure_nrmse_summary_data,
        figure_feature_pruning_curve_data=figure_feature_pruning_curve_data,
        legacy_feature_type_counts=legacy_feature_type_counts,
        legacy_influential_counts_by_module=legacy_influential_counts_by_module,
        legacy_interaction_counts_by_module_pair=legacy_interaction_counts_by_module_pair,
        legacy_interaction_density_module_matrix=legacy_interaction_density_module_matrix,
        legacy_module_total_interactions=legacy_module_total_interactions,
        svg_figures=svg_figures,
    )
    summary = _build_final_artifact_summary(
        prefilter_support_features=prefilter_support_features,
        final_support_features=final_support_features,
        hc3_filter_summary=hc3_filter_summary,
        final_ols_summary=final_ols_summary,
        workflow_stage_summary=workflow_stage_summary,
        figure_specs=figure_specs,
        spec=spec,
    )
    final_step = 9
    _final_progress("final summaries complete")
    final_step = 10
    _final_progress("done")
    return FinalManuscriptArtifactsResult(
        prefilter_support_features=prefilter_support_features,
        final_support_features=final_support_features,
        hc3_wald_intervals=hc3_wald_intervals,
        hc3_inferential_filter_summary=hc3_filter_summary,
        final_ols_summary=final_ols_summary,
        model_performance=model_performance,
        workflow_stage_summary=workflow_stage_summary,
        coefficient_matrix_raw_scale=coefficient_matrix_raw_scale,
        coefficient_matrix_standardized=coefficient_matrix_standardized,
        x_standardization=x_standardization,
        y_standardization=y_standardization,
        figure_model_performance_data=figure_model_performance_data,
        figure_support_composition_data=figure_support_composition_data,
        figure_selected_by_module_data=figure_selected_by_module_data,
        figure_nrmse_summary_data=figure_nrmse_summary_data,
        legacy_feature_type_counts=legacy_feature_type_counts,
        legacy_influential_counts_by_module=legacy_influential_counts_by_module,
        legacy_interaction_counts_by_module_pair=legacy_interaction_counts_by_module_pair,
        legacy_interaction_density_module_matrix=legacy_interaction_density_module_matrix,
        legacy_module_total_interactions=legacy_module_total_interactions,
        feature_pruning_impact=feature_pruning_impact,
        figure_feature_pruning_curve_data=figure_feature_pruning_curve_data,
        feature_pruning_summary=feature_pruning_summary,
        figure_specs=figure_specs,
        svg_figures=svg_figures,
        ablation_table=ablation_table,
        per_output_nrmse=per_output_nrmse,
        per_output_nrmse_summary=per_output_nrmse_summary,
        summary=summary,
    )


def write_final_manuscript_artifacts(
    result: FinalManuscriptArtifactsResult,
    output_root: Path,
) -> dict[str, Path]:
    """Write final manuscript tables and SVG figures under ``output_root``.

    Parameters
    ----------
    result
        Materialized final manuscript artifact result.
    output_root
        Resolved manuscript output root from the runtime context.

    Returns
    -------
    dict[str, pathlib.Path]
        Paths keyed by stable artifact name.
    """
    stage_root = output_root / "final_manuscript_artifacts"
    final_model_root = stage_root / "final_model"
    table_root = stage_root / "tables"
    figure_root = stage_root / "figures"
    for root in (final_model_root, table_root, figure_root):
        root.mkdir(parents=True, exist_ok=True)

    tables = {
        "prefilter_support_features": (
            final_model_root / "prefilter_support_features.csv",
            result.prefilter_support_features,
        ),
        "final_support_features": (
            final_model_root / "final_support_features.csv",
            result.final_support_features,
        ),
        "hc3_wald_intervals": (
            final_model_root / "hc3_wald_intervals.csv",
            result.hc3_wald_intervals,
        ),
        "hc3_inferential_filter_summary": (
            final_model_root / "hc3_inferential_filter_summary.csv",
            result.hc3_inferential_filter_summary,
        ),
        "final_ols_summary": (
            final_model_root / "final_ols_summary.csv",
            result.final_ols_summary,
        ),
        "coefficient_matrix_raw_scale": (
            final_model_root / "coefficient_matrix_raw_scale.csv",
            result.coefficient_matrix_raw_scale,
        ),
        "coefficient_matrix_standardized": (
            final_model_root / "coefficient_matrix_standardized.csv",
            result.coefficient_matrix_standardized,
        ),
        "x_standardization": (
            final_model_root / "x_standardization.csv",
            result.x_standardization,
        ),
        "y_standardization": (
            final_model_root / "y_standardization.csv",
            result.y_standardization,
        ),
        "model_performance": (table_root / "model_performance.csv", result.model_performance),
        "workflow_stage_summary": (
            table_root / "workflow_stage_summary.csv",
            result.workflow_stage_summary,
        ),
        "ablation_table": (table_root / "ablation_table.csv", result.ablation_table),
        "per_output_nrmse": (table_root / "per_output_nrmse.csv", result.per_output_nrmse),
        "per_output_nrmse_summary": (
            table_root / "per_output_nrmse_summary.csv",
            result.per_output_nrmse_summary,
        ),
        "feature_pruning_impact": (
            final_model_root / "feature_pruning_impact.csv",
            result.feature_pruning_impact,
        ),
        "feature_pruning_summary": (
            table_root / "feature_pruning_summary.csv",
            result.feature_pruning_summary,
        ),
        "figure_model_performance_data": (
            figure_root / "figure_model_performance_data.csv",
            result.figure_model_performance_data,
        ),
        "figure_support_composition_data": (
            figure_root / "figure_support_composition_data.csv",
            result.figure_support_composition_data,
        ),
        "figure_selected_by_module_data": (
            figure_root / "figure_selected_by_module_data.csv",
            result.figure_selected_by_module_data,
        ),
        "figure_nrmse_summary_data": (
            figure_root / "figure_nrmse_summary_data.csv",
            result.figure_nrmse_summary_data,
        ),
        "figure_feature_pruning_curve_data": (
            figure_root / "figure_feature_pruning_curve_data.csv",
            result.figure_feature_pruning_curve_data,
        ),
        "feature_type_counts": (
            figure_root / "feature_type_counts.csv",
            result.legacy_feature_type_counts,
        ),
        "influential_counts_by_module": (
            figure_root / "influential_counts_by_module.csv",
            result.legacy_influential_counts_by_module,
        ),
        "interaction_counts_by_module_pair": (
            figure_root / "interaction_counts_by_module_pair.csv",
            result.legacy_interaction_counts_by_module_pair,
        ),
        "interaction_density_module_matrix": (
            figure_root / "interaction_density_module_matrix.csv",
            result.legacy_interaction_density_module_matrix,
        ),
        "module_total_interactions": (
            figure_root / "module_total_interactions.csv",
            result.legacy_module_total_interactions,
        ),
        "figure_specs": (figure_root / "figure_specs.csv", result.figure_specs),
        "final_artifact_summary": (
            stage_root / "final_artifact_summary.csv",
            result.summary,
        ),
    }
    written: dict[str, Path] = {}
    for name, (path, table) in tables.items():
        table.to_csv(path, index=False)
        written[name] = path
    for name, svg_text in result.svg_figures.items():
        path = figure_root / f"{name}.svg"
        path.write_text(svg_text, encoding="utf-8")
        written[f"{name}_svg"] = path
    return written


def run_final_manuscript_artifacts_stage(
    context: Any,
) -> FinalManuscriptArtifactsStageResult:
    """Run final OLS and manuscript table/figure regeneration from a notebook context.

    Parameters
    ----------
    context
        ``bsm_rfm.manuscript_runtime.ManuscriptNotebookContext``. It is typed as ``Any`` here to
        avoid an import cycle between the runtime and stage modules.

    Returns
    -------
    FinalManuscriptArtifactsStageResult
        In-memory result and written artifact paths.
    """
    conditioning_spec = output_conditioning_spec_from_case_study_config(context.case_study_config)
    conditioning = condition_manuscript_outputs(
        context.tables["case_study_output_matrix"],
        context.tables["fixed_holdout_assignments"],
        conditioning_spec,
    )
    screening_spec = empirical_null_screening_spec_from_case_study_config(context.case_study_config)
    screening = screen_manuscript_empirical_null_terms(
        context.tables["case_study_input_matrix"],
        context.tables["manuscript_feature_catalog"],
        context.tables["fixed_holdout_assignments"],
        conditioning.pca_scores,
        screening_spec,
        checkpoint_dir=context.runtime.output_root / "empirical_null_screen",
    )
    interaction_spec = interaction_discovery_spec_from_case_study_config(context.case_study_config)
    interactions = discover_manuscript_interactions(
        context.tables["case_study_input_matrix"],
        context.tables["manuscript_feature_catalog"],
        context.tables["fixed_holdout_assignments"],
        conditioning.pca_scores,
        screening.retained_terms,
        interaction_spec,
        checkpoint_dir=context.runtime.output_root / "interaction_discovery" / "_batch_checkpoints",
    )
    nonlinear_spec = nonlinear_discovery_spec_from_case_study_config(context.case_study_config)
    nonlinear = discover_manuscript_nonlinear_transformations(
        context.tables["case_study_input_matrix"],
        context.tables["manuscript_feature_catalog"],
        context.tables["fixed_holdout_assignments"],
        conditioning.pca_scores,
        screening.retained_terms,
        nonlinear_spec,
        checkpoint_dir=context.runtime.output_root / "nonlinear_discovery",
    )
    sparse_spec = sparse_selection_stability_spec_from_case_study_config(context.case_study_config)
    sparse_selection = select_manuscript_sparse_support(
        context.tables["case_study_input_matrix"],
        context.tables["manuscript_feature_catalog"],
        context.tables["fixed_holdout_assignments"],
        conditioning.pca_scores,
        screening.retained_terms,
        interactions.retained_pairs,
        nonlinear.retained_transformations,
        sparse_spec,
        checkpoint_dir=context.runtime.output_root / "sparse_selection",
    )
    final_spec = final_manuscript_artifacts_spec_from_case_study_config(context.case_study_config)
    final_artifacts = regenerate_final_manuscript_artifacts(
        context.tables["case_study_input_matrix"],
        context.tables["case_study_output_matrix"],
        context.tables["manuscript_feature_catalog"],
        context.tables["fixed_holdout_assignments"],
        conditioning,
        screening,
        interactions,
        nonlinear,
        sparse_selection,
        final_spec,
        checkpoint_dir=context.runtime.output_root / "final_manuscript_artifacts",
    )
    artifact_paths = write_final_manuscript_artifacts(
        final_artifacts,
        context.runtime.output_root,
    )
    return FinalManuscriptArtifactsStageResult(
        final_artifacts=final_artifacts,
        artifact_paths=artifact_paths,
    )


def run_manuscript_reproduction_stage_chain(
    context: Any,
) -> ManuscriptReproductionStageChainResult:
    """Run the complete manuscript reproduction stage chain and write all artifacts.

    Parameters
    ----------
    context
        ``bsm_rfm.manuscript_runtime.ManuscriptNotebookContext``. It is typed as ``Any`` here to
        avoid an import cycle between the runtime and stage modules.

    Returns
    -------
    ManuscriptReproductionStageChainResult
        In-memory stage results and nested artifact paths for every Phase 3 stage.
    """
    conditioning_spec = output_conditioning_spec_from_case_study_config(context.case_study_config)
    conditioning = condition_manuscript_outputs(
        context.tables["case_study_output_matrix"],
        context.tables["fixed_holdout_assignments"],
        conditioning_spec,
    )
    output_paths = write_output_conditioning_artifacts(
        conditioning,
        context.runtime.output_root,
    )
    output_result = OutputConditioningStageResult(
        conditioning=conditioning,
        artifact_paths=output_paths,
    )

    screening_spec = empirical_null_screening_spec_from_case_study_config(context.case_study_config)
    screening = screen_manuscript_empirical_null_terms(
        context.tables["case_study_input_matrix"],
        context.tables["manuscript_feature_catalog"],
        context.tables["fixed_holdout_assignments"],
        conditioning.pca_scores,
        screening_spec,
        checkpoint_dir=context.runtime.output_root / "empirical_null_screen",
    )
    screening_paths = write_empirical_null_screening_artifacts(
        screening,
        context.runtime.output_root,
    )
    screening_result = EmpiricalNullScreeningStageResult(
        screening=screening,
        artifact_paths=screening_paths,
    )

    interaction_spec = interaction_discovery_spec_from_case_study_config(context.case_study_config)
    interactions = discover_manuscript_interactions(
        context.tables["case_study_input_matrix"],
        context.tables["manuscript_feature_catalog"],
        context.tables["fixed_holdout_assignments"],
        conditioning.pca_scores,
        screening.retained_terms,
        interaction_spec,
        checkpoint_dir=context.runtime.output_root / "interaction_discovery" / "_batch_checkpoints",
    )
    interaction_paths = write_interaction_discovery_artifacts(
        interactions,
        context.runtime.output_root,
    )
    interaction_result = InteractionDiscoveryStageResult(
        interactions=interactions,
        artifact_paths=interaction_paths,
    )

    nonlinear_spec = nonlinear_discovery_spec_from_case_study_config(context.case_study_config)
    nonlinear = discover_manuscript_nonlinear_transformations(
        context.tables["case_study_input_matrix"],
        context.tables["manuscript_feature_catalog"],
        context.tables["fixed_holdout_assignments"],
        conditioning.pca_scores,
        screening.retained_terms,
        nonlinear_spec,
        checkpoint_dir=context.runtime.output_root / "nonlinear_discovery",
    )
    nonlinear_paths = write_nonlinear_discovery_artifacts(
        nonlinear,
        context.runtime.output_root,
    )
    nonlinear_result = NonlinearDiscoveryStageResult(
        nonlinear=nonlinear,
        artifact_paths=nonlinear_paths,
    )

    sparse_spec = sparse_selection_stability_spec_from_case_study_config(context.case_study_config)
    sparse_selection = select_manuscript_sparse_support(
        context.tables["case_study_input_matrix"],
        context.tables["manuscript_feature_catalog"],
        context.tables["fixed_holdout_assignments"],
        conditioning.pca_scores,
        screening.retained_terms,
        interactions.retained_pairs,
        nonlinear.retained_transformations,
        sparse_spec,
        checkpoint_dir=context.runtime.output_root / "sparse_selection",
    )
    sparse_paths = write_sparse_selection_stability_artifacts(
        sparse_selection,
        context.runtime.output_root,
    )
    sparse_result = SparseSelectionStabilityStageResult(
        sparse_selection=sparse_selection,
        artifact_paths=sparse_paths,
    )

    final_spec = final_manuscript_artifacts_spec_from_case_study_config(context.case_study_config)
    final_artifacts = regenerate_final_manuscript_artifacts(
        context.tables["case_study_input_matrix"],
        context.tables["case_study_output_matrix"],
        context.tables["manuscript_feature_catalog"],
        context.tables["fixed_holdout_assignments"],
        conditioning,
        screening,
        interactions,
        nonlinear,
        sparse_selection,
        final_spec,
        checkpoint_dir=context.runtime.output_root / "final_manuscript_artifacts",
    )
    final_paths = write_final_manuscript_artifacts(
        final_artifacts,
        context.runtime.output_root,
    )
    final_result = FinalManuscriptArtifactsStageResult(
        final_artifacts=final_artifacts,
        artifact_paths=final_paths,
    )

    artifact_paths = {
        "output_conditioning": output_paths,
        "empirical_null_screen": screening_paths,
        "interaction_discovery": interaction_paths,
        "nonlinear_discovery": nonlinear_paths,
        "sparse_selection": sparse_paths,
        "final_manuscript_artifacts": final_paths,
    }
    return ManuscriptReproductionStageChainResult(
        output_conditioning=output_result,
        empirical_null_screening=screening_result,
        interaction_discovery=interaction_result,
        nonlinear_discovery=nonlinear_result,
        sparse_selection_stability=sparse_result,
        final_manuscript_artifacts=final_result,
        artifact_paths=artifact_paths,
    )


def audit_manuscript_reproduction_outputs(
    reproduction: ManuscriptReproductionStageChainResult,
    output_root: Path | None = None,
) -> ManuscriptReproductionAuditResult:
    """Build QA tables for a completed manuscript reproduction chain.

    Parameters
    ----------
    reproduction
        Completed end-to-end manuscript reproduction chain result.
    output_root
        Optional output root used to make artifact paths repo/user portable in the manifest.

    Returns
    -------
    ManuscriptReproductionAuditResult
        Artifact manifest, metric checks, and aggregate audit summary.
    """
    artifact_manifest = _build_reproduction_artifact_manifest(
        reproduction.artifact_paths,
        output_root=output_root,
    )
    metric_checks = _build_reproduction_metric_checks(
        reproduction,
        artifact_manifest,
    )
    summary = _build_reproduction_audit_summary(artifact_manifest, metric_checks)
    return ManuscriptReproductionAuditResult(
        artifact_manifest=artifact_manifest,
        metric_checks=metric_checks,
        summary=summary,
    )


def write_manuscript_reproduction_audit(
    audit: ManuscriptReproductionAuditResult,
    output_root: Path,
) -> dict[str, Path]:
    """Write manuscript reproduction QA audit tables under ``output_root``.

    Parameters
    ----------
    audit
        Materialized reproduction audit tables.
    output_root
        Resolved manuscript output root from the runtime context.

    Returns
    -------
    dict[str, pathlib.Path]
        Paths keyed by stable audit artifact name.
    """
    audit_root = output_root / "reproduction_audit"
    audit_root.mkdir(parents=True, exist_ok=True)
    tables = {
        "artifact_manifest": audit.artifact_manifest,
        "metric_checks": audit.metric_checks,
        "audit_summary": audit.summary,
    }
    written: dict[str, Path] = {}
    for name, table in tables.items():
        path = audit_root / f"{name}.csv"
        table.to_csv(path, index=False)
        written[name] = path
    return written


def run_manuscript_reproduction_audit_stage(
    context: Any,
) -> ManuscriptReproductionAuditStageResult:
    """Run the full manuscript reproduction chain and write QA audit artifacts.

    Parameters
    ----------
    context
        ``bsm_rfm.manuscript_runtime.ManuscriptNotebookContext``. It is typed as ``Any`` here to
        avoid an import cycle between the runtime and stage modules.

    Returns
    -------
    ManuscriptReproductionAuditStageResult
        Completed stage chain, audit tables, and paths to written audit CSV files.
    """
    reproduction = run_manuscript_reproduction_stage_chain(context)
    audit = audit_manuscript_reproduction_outputs(
        reproduction,
        output_root=context.runtime.output_root,
    )
    artifact_paths = write_manuscript_reproduction_audit(
        audit,
        context.runtime.output_root,
    )
    return ManuscriptReproductionAuditStageResult(
        reproduction=reproduction,
        audit=audit,
        artifact_paths=artifact_paths,
    )


def _build_reproduction_artifact_manifest(
    artifact_paths: dict[str, dict[str, Path]],
    *,
    output_root: Path | None,
) -> pd.DataFrame:
    """Build a portable file manifest for every written reproduction artifact."""
    rows = []
    for stage_name, stage_paths in sorted(artifact_paths.items()):
        for artifact_name, raw_path in sorted(stage_paths.items()):
            path = Path(raw_path)
            exists = path.exists()
            is_file = path.is_file() if exists else False
            size_bytes = path.stat().st_size if is_file else 0
            rows.append(
                {
                    "stage": stage_name,
                    "artifact_name": artifact_name,
                    "path": _portable_artifact_path(path, output_root),
                    "suffix": path.suffix,
                    "exists": bool(exists),
                    "is_file": bool(is_file),
                    "nonempty": bool(size_bytes > 0),
                    "size_bytes": int(size_bytes),
                    "sha256": _sha256_file(path) if is_file else "",
                }
            )
    return pd.DataFrame.from_records(
        rows,
        columns=[
            "stage",
            "artifact_name",
            "path",
            "suffix",
            "exists",
            "is_file",
            "nonempty",
            "size_bytes",
            "sha256",
        ],
    )


def _build_reproduction_metric_checks(
    reproduction: ManuscriptReproductionStageChainResult,
    artifact_manifest: pd.DataFrame,
) -> pd.DataFrame:
    """Build explicit QA checks for the reproduced manuscript outputs."""
    final_artifacts = reproduction.final_manuscript_artifacts.final_artifacts
    final_summary = final_artifacts.final_ols_summary.loc[0]
    nrmse = float(final_summary["final_ols_holdout_nrmse"])
    ci_lower = float(final_summary["final_ols_holdout_nrmse_ci_lower"])
    ci_upper = float(final_summary["final_ols_holdout_nrmse_ci_upper"])
    null_nrmse = float(final_summary["null_mean_holdout_nrmse"])
    rows: list[dict[str, object]] = []

    def add_check(
        check_name: str,
        passed: bool,
        observed_value: object,
        expected: str,
        details: str,
    ) -> None:
        rows.append(
            {
                "check_name": check_name,
                "status": "pass" if passed else "fail",
                "observed_value": observed_value,
                "expected": expected,
                "details": details,
            }
        )

    add_check(
        "all_artifacts_exist",
        bool(artifact_manifest["exists"].all()),
        int(artifact_manifest["exists"].sum()),
        "every artifact path exists",
        "Catches missing handoff files after a reproduction run.",
    )
    add_check(
        "all_artifacts_nonempty",
        bool(artifact_manifest["nonempty"].all()),
        int(artifact_manifest["nonempty"].sum()),
        "every artifact file has positive size",
        "Catches empty CSV/SVG outputs before manuscript handoff.",
    )
    add_check(
        "final_ols_holdout_nrmse_positive",
        math.isfinite(nrmse) and nrmse > 0.0,
        nrmse,
        "finite nRMSE strictly greater than zero",
        "Prevents the demo holdout path from silently reporting a perfect fit.",
    )
    add_check(
        "final_ols_holdout_nrmse_ci_ordered",
        all(math.isfinite(value) for value in (ci_lower, nrmse, ci_upper))
        and ci_lower <= nrmse <= ci_upper,
        f"{ci_lower:.12g} <= {nrmse:.12g} <= {ci_upper:.12g}",
        "finite lower <= point <= upper",
        "Checks the bootstrap uncertainty summary before publication use.",
    )
    add_check(
        "null_mean_holdout_nrmse_finite_positive",
        math.isfinite(null_nrmse) and null_nrmse > 0.0,
        null_nrmse,
        "finite null baseline nRMSE strictly greater than zero",
        "Confirms the reported baseline is numerically meaningful.",
    )
    add_check(
        "final_support_nonempty",
        len(final_artifacts.final_support_features) > 0,
        int(len(final_artifacts.final_support_features)),
        "at least one final support feature",
        "Confirms the final OLS table is not generated from an empty support.",
    )
    svg_rows = artifact_manifest.loc[artifact_manifest["suffix"] == ".svg"]
    add_check(
        "registered_svg_figures_nonempty",
        len(svg_rows) == len(final_artifacts.figure_specs) and bool(svg_rows["nonempty"].all()),
        int(len(svg_rows)),
        "one nonempty SVG file per registered figure",
        "Confirms manuscript figure assets were actually written.",
    )
    workflow_stages = set(final_artifacts.workflow_stage_summary["stage"])
    add_check(
        "workflow_summary_includes_final_ols",
        "final_ols" in workflow_stages,
        ", ".join(sorted(str(stage) for stage in workflow_stages)),
        "workflow-stage summary includes final_ols",
        "Confirms the final model appears in manuscript-facing stage summaries.",
    )
    return pd.DataFrame.from_records(
        rows,
        columns=["check_name", "status", "observed_value", "expected", "details"],
    )


def _build_reproduction_audit_summary(
    artifact_manifest: pd.DataFrame,
    metric_checks: pd.DataFrame,
) -> pd.DataFrame:
    """Build the one-row reproduction audit summary table."""
    n_missing = int((~artifact_manifest["exists"]).sum())
    n_empty = int((~artifact_manifest["nonempty"]).sum())
    n_failed_checks = int((metric_checks["status"] == "fail").sum())
    qa_status = "pass" if n_missing == 0 and n_empty == 0 and n_failed_checks == 0 else "fail"
    return pd.DataFrame(
        [
            {
                "stage": "manuscript_reproduction_audit",
                "qa_status": qa_status,
                "n_artifacts": int(len(artifact_manifest)),
                "n_missing_artifacts": n_missing,
                "n_empty_artifacts": n_empty,
                "n_metric_checks": int(len(metric_checks)),
                "n_failed_metric_checks": n_failed_checks,
            }
        ]
    )


def _portable_artifact_path(path: Path, output_root: Path | None) -> str:
    """Return a path string that is portable when the output root is known."""
    if output_root is None:
        return str(path)
    try:
        return str(path.relative_to(output_root))
    except ValueError:
        return str(path)


def _sha256_file(path: Path) -> str:
    """Hash a written artifact without loading large files into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_stability_resampling_scheme(scheme: str) -> tuple[int, float, int]:
    """Parse the frozen stability-resampling scheme string."""
    pieces = scheme.split("_")
    try:
        count = int(pieces[0])
        percent = int(pieces[3])
        seed = int(pieces[-1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Unsupported stability resampling scheme: {scheme}") from exc
    if "subsamples" not in pieces or "without" not in pieces or "replacement" not in pieces:
        raise ValueError(f"Unsupported stability resampling scheme: {scheme}")
    fraction = percent / 100.0
    if count < 1 or not 0.0 < fraction <= 1.0:
        raise ValueError(f"Invalid stability resampling scheme: {scheme}")
    return count, fraction, seed


def _validate_sparse_selection_spec(spec: SparseSelectionStabilitySpec) -> None:
    """Validate sparse-selection and stability specification values."""
    if spec.model_class != "l1_penalized_linear_model_per_retained_component":
        raise ValueError(f"Unsupported sparse-selection model class: {spec.model_class}")
    expected_rule = "union_nonzero_support_across_retained_components"
    if spec.support_aggregation_rule != expected_rule:
        raise ValueError(
            f"Unsupported sparse support aggregation rule: {spec.support_aggregation_rule}"
        )
    if spec.ebic_gamma < 0.0:
        raise ValueError("ebic_gamma must be non-negative.")
    if spec.subsample_count < 1:
        raise ValueError("subsample_count must be positive.")
    if not 0.0 < spec.subsample_fraction <= 1.0:
        raise ValueError("subsample_fraction must be in the interval (0, 1].")
    if not 0.0 <= spec.jaccard_threshold <= 1.0:
        raise ValueError("jaccard_threshold must be in the interval [0, 1].")
    if not -1.0 <= spec.spearman_threshold <= 1.0:
        raise ValueError("spearman_threshold must be in the interval [-1, 1].")
    if spec.max_candidate_terms is not None and spec.max_candidate_terms < 1:
        raise ValueError("max_candidate_terms must be positive when provided.")


def _ordered_sparse_candidate_names(
    *,
    feature_catalog: pd.DataFrame,
    retained_terms: pd.DataFrame,
    retained_interaction_pairs: pd.DataFrame,
    retained_transformations: pd.DataFrame,
) -> list[str]:
    """Return sparse-selection candidates preserving catalog order plus dynamic discoveries."""
    if "feature_name" not in feature_catalog.columns:
        raise ValueError("feature_catalog must include a feature_name column.")

    ordered_from_upstream = (
        list(retained_terms["feature_name"].astype(str))
        + _ordered_retained_pair_names(retained_interaction_pairs)
        + list(retained_transformations["feature_name"].astype(str))
    )
    if not ordered_from_upstream:
        raise ValueError("Sparse selection requires at least one retained upstream term.")

    retained_names = set(ordered_from_upstream)
    ordered = [
        str(feature_name)
        for feature_name in feature_catalog["feature_name"].astype(str)
        if str(feature_name) in retained_names
    ]
    ordered_set = set(ordered)
    for feature_name in ordered_from_upstream:
        if feature_name not in ordered_set:
            ordered.append(feature_name)
            ordered_set.add(feature_name)
    if not ordered:
        raise ValueError("Sparse selection found no retained terms in feature_catalog order.")
    return ordered


def _apply_sparse_candidate_cap(
    *,
    ordered_candidates: list[str],
    retained_terms: pd.DataFrame,
    retained_interaction_pairs: pd.DataFrame,
    retained_transformations: pd.DataFrame,
    max_candidate_terms: int | None,
) -> list[str]:
    """Apply deterministic top-K sparse candidate capping from upstream stage strengths."""
    if max_candidate_terms is None or len(ordered_candidates) <= max_candidate_terms:
        return ordered_candidates

    strengths: dict[str, tuple[float, float]] = {}

    def _ingest(table: pd.DataFrame, name_column: str) -> None:
        if table.empty or name_column not in table.columns:
            return
        for _, row in table.iterrows():
            name = str(row[name_column])
            p_value = float(row.get("empirical_p_value", 1.0))
            p_value = max(p_value, 1.0e-300)
            primary = -math.log10(p_value)
            secondary = max(
                float(row.get("observed_statistic", 0.0)),
                float(row.get("observed_score", 0.0)),
                float(row.get("curvature_score", 0.0)),
            )
            candidate_strength = (primary, secondary)
            existing = strengths.get(name)
            if existing is None or candidate_strength > existing:
                strengths[name] = candidate_strength

    _ingest(retained_terms, "feature_name")
    pair_name_column = (
        "pair_name" if "pair_name" in retained_interaction_pairs.columns else "feature_name"
    )
    _ingest(retained_interaction_pairs, pair_name_column)
    _ingest(retained_transformations, "feature_name")

    indexed = list(enumerate(ordered_candidates))
    ranked = sorted(
        indexed,
        key=lambda item: (
            strengths.get(item[1], (0.0, 0.0))[0],
            strengths.get(item[1], (0.0, 0.0))[1],
            -item[0],
        ),
        reverse=True,
    )
    selected = {name for _, name in ranked[:max_candidate_terms]}
    return [name for name in ordered_candidates if name in selected]


def _retained_pair_names(retained_interaction_pairs: pd.DataFrame) -> set[str]:
    """Return retained interaction pair names from a retained-pair table."""
    if retained_interaction_pairs.empty:
        return set()
    column = "pair_name" if "pair_name" in retained_interaction_pairs.columns else "feature_name"
    if column not in retained_interaction_pairs.columns:
        raise ValueError("retained_interaction_pairs must include pair_name or feature_name.")
    return set(retained_interaction_pairs[column].astype(str))


def _ordered_retained_pair_names(retained_interaction_pairs: pd.DataFrame) -> list[str]:
    """Return retained interaction pair names preserving row order."""
    if retained_interaction_pairs.empty:
        return []
    column = "pair_name" if "pair_name" in retained_interaction_pairs.columns else "feature_name"
    if column not in retained_interaction_pairs.columns:
        raise ValueError("retained_interaction_pairs must include pair_name or feature_name.")
    return list(retained_interaction_pairs[column].astype(str))


def _feature_catalog_subset(
    feature_catalog: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    """Return rows for ``feature_names`` in order, synthesizing metadata for dynamic terms."""
    if feature_catalog["feature_name"].duplicated(keep=False).any():
        raise ValueError("feature_catalog contains duplicate feature_name values.")
    indexed = feature_catalog.set_index("feature_name", drop=False)
    rows = []
    for name in feature_names:
        if name in indexed.index:
            rows.append(indexed.loc[name].to_dict())
        else:
            rows.append(_synthesized_feature_catalog_row(feature_catalog, name))
    return pd.DataFrame(rows).reset_index(drop=True)


def _synthesized_feature_catalog_row(
    feature_catalog: pd.DataFrame,
    feature_name: str,
) -> dict[str, Any]:
    """Build a catalog-like metadata row for a dynamically discovered feature."""
    row = {column: pd.NA for column in feature_catalog.columns}
    row["feature_name"] = feature_name
    if "feature_type" in row:
        row["feature_type"] = _infer_feature_type_for_name(feature_name)
    if "origin" in row:
        row["origin"] = "dynamic_discovery"
    return row


def _infer_feature_type_for_name(feature_name: str) -> str:
    """Infer feature type from a manuscript feature expression."""
    if ":" in feature_name:
        return "interaction"
    if _parse_supported_transformation_name(feature_name) is not None:
        return "transformation"
    return "first_order"


def _fit_sparse_l1_ebic_models(
    x_scaled: np.ndarray,
    y_scaled: np.ndarray,
    *,
    feature_names: list[str],
    component_names: list[str],
    spec: SparseSelectionStabilitySpec,
    active_features: np.ndarray,
) -> dict[str, Any]:
    """Fit one EBIC-selected L1 path per response component."""
    n_features = len(feature_names)
    coefficient_matrix = np.zeros((n_features, len(component_names)), dtype=float)
    rows = []
    for component_index, component_name in enumerate(component_names):
        y = y_scaled[:, component_index]
        if np.std(y, ddof=0) <= 0.0:
            rows.append(
                _empty_sparse_component_row(
                    component_name=component_name,
                    n_rows=len(y),
                    n_features=n_features,
                    spec=spec,
                )
            )
            continue
        selected = _select_component_lasso_by_ebic(
            x_scaled,
            y,
            active_features=active_features,
            spec=spec,
        )
        coefficient_matrix[:, component_index] = selected["coefficients"]
        rows.append(
            {
                "component": component_name,
                "selected_alpha": float(selected["alpha"]),
                "selected_ebic": float(selected["ebic"]),
                "selected_rss": float(selected["rss"]),
                "selected_support_size": int(selected["support_size"]),
                "n_rows": int(len(y)),
                "n_candidate_terms": int(n_features),
                "ebic_gamma": float(spec.ebic_gamma),
            }
        )
    return {
        "coefficient_matrix": coefficient_matrix,
        "component_model_selection": pd.DataFrame.from_records(rows),
    }


def _empty_sparse_component_row(
    *,
    component_name: str,
    n_rows: int,
    n_features: int,
    spec: SparseSelectionStabilitySpec,
) -> dict[str, Any]:
    """Build model-selection diagnostics for a zero-variance response component."""
    return {
        "component": component_name,
        "selected_alpha": 0.0,
        "selected_ebic": 0.0,
        "selected_rss": 0.0,
        "selected_support_size": 0,
        "n_rows": int(n_rows),
        "n_candidate_terms": int(n_features),
        "ebic_gamma": float(spec.ebic_gamma),
    }


def _select_component_lasso_by_ebic(
    x_scaled: np.ndarray,
    y_scaled: np.ndarray,
    *,
    active_features: np.ndarray,
    spec: SparseSelectionStabilitySpec,
) -> dict[str, Any]:
    """Select one component-specific L1 penalty by EBIC."""
    n_rows, n_features = x_scaled.shape
    active_count = int(active_features.sum())
    if active_count == 0:
        return _zero_component_selection(n_features, y_scaled)
    x_active = x_scaled[:, active_features]
    xy_active = x_active.T @ y_scaled
    alpha_max = float(np.max(np.abs(xy_active)) / n_rows)
    if alpha_max <= 0.0:
        return _zero_component_selection(n_features, y_scaled)

    alphas = np.geomspace(alpha_max, max(alpha_max * 1.0e-4, 1.0e-8), num=40)
    baseline_rss = float(np.sum((y_scaled - y_scaled.mean()) ** 2))
    best = _zero_component_selection(n_features, y_scaled)
    best["ebic"] = _extended_bic(
        rss=max(baseline_rss, np.finfo(float).tiny),
        n_rows=n_rows,
        n_features=active_count,
        support_size=0,
        gamma=spec.ebic_gamma,
    )
    estimator = Lasso(
        fit_intercept=False,
        max_iter=10000,
        tol=1.0e-6,
        selection="cyclic",
        warm_start=True,
        precompute=x_active.T @ x_active,
    )
    for alpha in alphas:
        estimator.set_params(alpha=float(alpha))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            estimator.fit(x_active, y_scaled)
        coefficients = np.zeros(n_features, dtype=float)
        coefficients[active_features] = np.asarray(estimator.coef_, dtype=float)
        prediction = x_active @ np.asarray(estimator.coef_, dtype=float)
        rss = float(np.sum((y_scaled - prediction) ** 2))
        support_size = int(np.sum(np.abs(estimator.coef_) > 0.0))
        ebic = _extended_bic(
            rss=max(rss, np.finfo(float).tiny),
            n_rows=n_rows,
            n_features=active_count,
            support_size=support_size,
            gamma=spec.ebic_gamma,
        )
        if ebic < best["ebic"]:
            best = {
                "alpha": float(alpha),
                "ebic": float(ebic),
                "rss": float(rss),
                "support_size": support_size,
                "coefficients": coefficients,
            }
    return best


def _zero_component_selection(n_features: int, y_scaled: np.ndarray) -> dict[str, Any]:
    """Return a no-feature component model."""
    rss = float(np.sum((y_scaled - y_scaled.mean()) ** 2))
    return {
        "alpha": 0.0,
        "ebic": 0.0,
        "rss": rss,
        "support_size": 0,
        "coefficients": np.zeros(n_features, dtype=float),
    }


def _extended_bic(
    *,
    rss: float,
    n_rows: int,
    n_features: int,
    support_size: int,
    gamma: float,
) -> float:
    """Compute EBIC for one sparse linear model."""
    support_size = int(support_size)
    likelihood_term = n_rows * math.log(max(rss, np.finfo(float).tiny) / float(n_rows))
    bic_term = support_size * math.log(float(n_rows))
    ebic_term = 2.0 * gamma * _log_combination(max(n_features, support_size), support_size)
    return float(likelihood_term + bic_term + ebic_term)


def _log_combination(n_items: int, n_selected: int) -> float:
    """Return ``log(n choose k)`` without materializing large integers."""
    if n_selected < 0 or n_selected > n_items:
        return 0.0
    if n_selected == 0 or n_selected == n_items:
        return 0.0
    return float(
        math.lgamma(n_items + 1)
        - math.lgamma(n_selected + 1)
        - math.lgamma(n_items - n_selected + 1)
    )


def _run_stability_resamples(
    *,
    x_scaled: np.ndarray,
    y_scaled: np.ndarray,
    feature_names: list[str],
    component_names: list[str],
    full_support_mask: np.ndarray,
    full_importance: np.ndarray,
    spec: SparseSelectionStabilitySpec,
    active_features: np.ndarray,
    active_resample_indices: set[int] | None = None,
    checkpoint_dir: Path | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Run stability resamples with optional adaptive early stopping."""
    rng = np.random.default_rng(spec.random_seed)
    n_rows = len(x_scaled)
    subsample_size = max(2, int(math.floor(n_rows * spec.subsample_fraction)))
    subsample_size = min(subsample_size, n_rows)

    # Pre-generate all row-index draws so results are deterministic regardless of n_jobs.
    # First resample: uniform random selection (baseline)
    # Subsequent resamples: stratified by feature importance to improve convergence
    all_row_indices = []

    # Resample 1: Uniform random
    all_row_indices.append(np.sort(rng.choice(n_rows, size=subsample_size, replace=False)))

    # Resamples 2+: Stratified by feature importance (faster convergence)
    # Normalize full_importance across active features to get sampling weights
    active_importance = full_importance.copy()
    active_importance[~active_features] = 0.0
    importance_sum = active_importance.sum()

    if importance_sum > 0:
        # Use feature importance to weight row selection (features appear in many rows)
        # Proxy: rows with high-importance features selected more often
        row_importance = np.zeros(n_rows)
        for feat_idx in np.where(active_features)[0]:
            # Sum feature values (as proxy for importance) across rows
            row_importance += np.abs(x_scaled[:, feat_idx])

        # Normalize to probability distribution
        row_importance_sum = row_importance.sum()
        if row_importance_sum > 0:
            row_probs = row_importance / row_importance_sum
        else:
            row_probs = np.ones(n_rows) / n_rows

        for _ in range(1, spec.subsample_count):
            # Sample with replacement using importance-weighted probabilities
            indices = rng.choice(
                n_rows,
                size=subsample_size,
                replace=True,
                p=row_probs,
            )
            all_row_indices.append(np.sort(np.unique(indices)))
    else:
        # Fallback to uniform if no importance information available
        for _ in range(1, spec.subsample_count):
            all_row_indices.append(np.sort(rng.choice(n_rows, size=subsample_size, replace=False)))

    support_rows = []
    importance_rows = []
    summary_rows = []
    active_resample_set = (
        set(range(1, spec.subsample_count + 1))
        if active_resample_indices is None
        else {
            int(index)
            for index in active_resample_indices
            if 1 <= int(index) <= int(spec.subsample_count)
        }
    )
    if not active_resample_set:
        raise ValueError("active_resample_indices must include at least one valid resample id.")
    active_resample_total = max(1, len(active_resample_set))
    checkpoint_signature = {
        "subsample_count": int(spec.subsample_count),
        "subsample_fraction": float(spec.subsample_fraction),
        "random_seed": int(spec.random_seed),
        "n_rows": int(n_rows),
        "n_features": int(len(feature_names)),
        "component_names": component_names,
        "feature_names": feature_names,
        "jaccard_threshold": float(spec.jaccard_threshold),
        "spearman_threshold": float(spec.spearman_threshold),
        "active_features_sha256": hashlib.sha256(
            active_features.astype(np.uint8).tobytes()
        ).hexdigest(),
        "full_support_mask_sha256": hashlib.sha256(
            full_support_mask.astype(np.uint8).tobytes()
        ).hexdigest(),
        "full_importance_sha256": hashlib.sha256(
            np.round(full_importance.astype(float), 12).tobytes()
        ).hexdigest(),
    }
    checkpoint_run_dir = _checkpoint_run_dir(
        checkpoint_dir,
        stage="sparse_selection",
        signature_payload=checkpoint_signature,
    )

    def _resample_checkpoint_paths(resample_id: int) -> tuple[Path | None, Path | None]:
        if checkpoint_run_dir is None:
            return None, None
        stem = f"resample_{resample_id:06d}"
        return checkpoint_run_dir / f"{stem}.npz", checkpoint_run_dir / f"{stem}.json"

    cached_resamples: dict[int, tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}
    for resample_id in range(1, spec.subsample_count + 1):
        npz_path, json_path = _resample_checkpoint_paths(resample_id)
        if npz_path is None or json_path is None:
            break
        if not npz_path.exists() or not json_path.exists():
            continue
        payload = _read_json_dict(json_path)
        if payload is None:
            continue
        try:
            with np.load(npz_path, allow_pickle=False) as arrays:
                support_mask = np.asarray(arrays["support_mask"], dtype=bool)
                importance = np.asarray(arrays["importance"], dtype=float)
        except (OSError, KeyError, ValueError):
            continue
        if support_mask.shape != (len(feature_names),) or importance.shape != (len(feature_names),):
            continue
        if int(payload.get("resample_id", -1)) != resample_id:
            continue
        cached_resamples[resample_id] = (support_mask, importance, payload)

    batch_size = max(
        1,
        min(
            20,
            spec.subsample_count // 20 if spec.subsample_count > 20 else spec.subsample_count,
        ),
    )
    cached_active_count = sum(1 for idx in active_resample_set if idx in cached_resamples)
    start_detail = "starting sparse stability resampling"
    if cached_resamples:
        start_detail = (
            f"resuming sparse stability resampling ({cached_active_count}/{active_resample_total})"
        )
    _report_progress(
        stage="sparse_selection",
        completed=cached_active_count,
        total=active_resample_total,
        unit="stability_resamples",
        detail=start_detail,
    )

    actual_resample_count = spec.subsample_count
    early_stopped = False

    with Parallel(n_jobs=spec.n_jobs) as parallel:
        for start in range(0, spec.subsample_count, batch_size):
            stop = min(start + batch_size, spec.subsample_count)
            batch_ids = list(range(start + 1, stop + 1))
            jobs = [
                delayed(_run_one_stability_resample)(
                    resample_id,
                    row_indices,
                    x_scaled,
                    y_scaled,
                    feature_names,
                    component_names,
                    spec,
                    active_features,
                    full_support_mask,
                    full_importance,
                )
                for resample_id, row_indices in enumerate(
                    all_row_indices[start:stop],
                    start=start + 1,
                )
                if resample_id in active_resample_set and resample_id not in cached_resamples
            ]
            for cached_id in batch_ids:
                if cached_id in active_resample_set and cached_id in cached_resamples:
                    support_mask, importance, summary_row = cached_resamples[cached_id]
                    support_rows.append(support_mask)
                    importance_rows.append(importance)
                    summary_rows.append(summary_row)
            if jobs:
                for support_mask, importance, summary_row in parallel(jobs):
                    support_rows.append(support_mask)
                    importance_rows.append(importance)
                    summary_rows.append(summary_row)
                    resample_id = int(summary_row["resample_id"])
                    npz_path, json_path = _resample_checkpoint_paths(resample_id)
                    if npz_path is not None and json_path is not None:
                        temp_npz = npz_path.with_suffix(".tmp.npz")
                        np.savez_compressed(
                            temp_npz,
                            support_mask=np.asarray(support_mask, dtype=bool),
                            importance=np.asarray(importance, dtype=float),
                        )
                        temp_npz.replace(npz_path)
                        _write_json_atomic(json_path, summary_row)
            _report_progress(
                stage="sparse_selection",
                completed=len(summary_rows),
                total=active_resample_total,
                unit="stability_resamples",
            )

            # Check for early stopping after each batch
            if (
                spec.adaptive_early_stopping_enabled
                and active_resample_indices is None
                and len(summary_rows) >= spec.stability_convergence_window
                and not early_stopped
            ):
                # Check if last N resamples all pass stability thresholds
                recent = summary_rows[-spec.stability_convergence_window :]
                jaccard_values = [r["jaccard_with_full_support"] for r in recent]
                spearman_values = [r["spearman_with_full_importance"] for r in recent]

                all_pass_jaccard = all(j >= spec.jaccard_threshold for j in jaccard_values)
                all_pass_spearman = all(s >= spec.spearman_threshold for s in spearman_values)

                if all_pass_jaccard and all_pass_spearman and stop < spec.subsample_count:
                    # Stability has converged; fill remaining resamples synthetically
                    actual_resample_count = len(summary_rows)
                    early_stopped = True
                    _report_progress(
                        stage="sparse_selection",
                        completed=actual_resample_count,
                        total=spec.subsample_count,
                        unit="stability_resamples",
                        detail=f"early stopped at {actual_resample_count} resamples (converged)",
                    )

                    # Synthesize remaining resamples by duplicating current convergent state
                    for resample_id in range(actual_resample_count + 1, spec.subsample_count + 1):
                        # Use the last row indices repeatedly (deterministic)
                        row_indices = all_row_indices[resample_id - 1]
                        support_mask, importance, summary_row = _run_one_stability_resample(
                            resample_id,
                            row_indices,
                            x_scaled,
                            y_scaled,
                            feature_names,
                            component_names,
                            spec,
                            active_features,
                            full_support_mask,
                            full_importance,
                        )
                        support_rows.append(support_mask)
                        importance_rows.append(importance)
                        summary_rows.append(summary_row)
                        resample_id = int(summary_row["resample_id"])
                        npz_path, json_path = _resample_checkpoint_paths(resample_id)
                        if npz_path is not None and json_path is not None:
                            temp_npz = npz_path.with_suffix(".tmp.npz")
                            np.savez_compressed(
                                temp_npz,
                                support_mask=np.asarray(support_mask, dtype=bool),
                                importance=np.asarray(importance, dtype=float),
                            )
                            temp_npz.replace(npz_path)
                            _write_json_atomic(json_path, summary_row)
                    break

    # Sort by resample_id to restore deterministic order after parallel execution.
    ordered_results = sorted(
        (
            int(summary_row["resample_id"]),
            summary_row,
            support_row,
            importance_row,
        )
        for summary_row, support_row, importance_row in zip(
            summary_rows,
            support_rows,
            importance_rows,
            strict=True,
        )
    )
    summary_rows = [summary_row for _, summary_row, _, _ in ordered_results]
    support_rows = [support_row for _, _, support_row, _ in ordered_results]
    importance_rows = [importance_row for _, _, _, importance_row in ordered_results]

    return (
        pd.DataFrame.from_records(summary_rows),
        np.vstack(support_rows),
        np.vstack(importance_rows),
    )


def _run_one_stability_resample(
    resample_id: int,
    row_indices: np.ndarray,
    x_scaled: np.ndarray,
    y_scaled: np.ndarray,
    feature_names: list[str],
    component_names: list[str],
    spec: SparseSelectionStabilitySpec,
    active_features: np.ndarray,
    full_support_mask: np.ndarray,
    full_importance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Run one deterministic sparse-selection stability resample and return its diagnostics."""
    selected = _fit_sparse_l1_ebic_models(
        x_scaled[row_indices, :],
        y_scaled[row_indices, :],
        feature_names=feature_names,
        component_names=component_names,
        spec=spec,
        active_features=active_features,
    )
    coefficients = selected["coefficient_matrix"]
    support_mask = np.any(np.abs(coefficients) > 0.0, axis=1)
    importance = np.max(np.abs(coefficients), axis=1)
    summary_row = {
        "resample_id": resample_id,
        "subsample_size": int(len(row_indices)),
        "selected_support_size": int(support_mask.sum()),
        "jaccard_with_full_support": _jaccard_similarity(full_support_mask, support_mask),
        "spearman_with_full_importance": _spearman_rank_correlation(full_importance, importance),
    }
    return support_mask, importance, summary_row


def _jaccard_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Return Jaccard similarity between two boolean support masks."""
    union = np.logical_or(left, right)
    if not union.any():
        return 1.0
    intersection = np.logical_and(left, right)
    return float(intersection.sum() / union.sum())


def _spearman_rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Return Spearman correlation between two importance vectors."""
    if len(left) < 2:
        return 1.0
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    left_centered = left_ranks - left_ranks.mean()
    right_centered = right_ranks - right_ranks.mean()
    denominator = float(np.sqrt(np.sum(left_centered**2)) * np.sqrt(np.sum(right_centered**2)))
    if denominator == 0.0:
        return 1.0 if np.allclose(left_ranks, right_ranks) else 0.0
    return float(np.sum(left_centered * right_centered) / denominator)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return average ranks for a numeric vector with deterministic tie handling."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        average_rank = 0.5 * (start + stop - 1) + 1.0
        ranks[order[start:stop]] = average_rank
        start = stop
    return ranks


def _build_stability_feature_summary(
    *,
    feature_names: list[str],
    feature_active: np.ndarray,
    full_support_mask: np.ndarray,
    full_importance: np.ndarray,
    resample_supports: np.ndarray,
    resample_importances: np.ndarray,
    mean_jaccard: float,
    mean_spearman: float,
    spec: SparseSelectionStabilitySpec,
) -> pd.DataFrame:
    """Build per-feature stability diagnostics."""
    selection_frequency = resample_supports.mean(axis=0)
    mean_resample_importance = resample_importances.mean(axis=0)
    passes_global_stability = (
        mean_jaccard >= spec.jaccard_threshold and mean_spearman >= spec.spearman_threshold
    )

    # Per-feature final support based on selection frequency against the Jaccard threshold
    final_support = full_support_mask & (selection_frequency >= spec.jaccard_threshold)

    # If the global stability gates fail, fallback to a conservative top-k by full_importance
    if not passes_global_stability:
        # choose at least one feature by descending full_importance (5% rule with minimum 1)
        k = max(1, int(max(1, round(len(feature_names) * 0.05))))
        top_idx = np.argsort(-full_importance)[:k]
        fallback_support = np.zeros_like(final_support, dtype=bool)
        fallback_support[top_idx] = True
        # prefer features already in the full_support_mask but ensure non-empty final support
        final_support = np.logical_or(final_support, fallback_support)

    return pd.DataFrame(
        {
            "feature_name": feature_names,
            "nonzero_training_variance": feature_active,
            "full_support_selected": full_support_mask,
            "full_support_importance": full_importance,
            "stability_selection_frequency": selection_frequency,
            "mean_resample_importance": mean_resample_importance,
            "mean_resample_jaccard": mean_jaccard,
            "mean_resample_spearman": mean_spearman,
            "passes_jaccard_threshold": mean_jaccard >= spec.jaccard_threshold,
            "passes_spearman_threshold": mean_spearman >= spec.spearman_threshold,
            "final_stable_support": final_support,
        }
    ).sort_values(
        ["final_stable_support", "full_support_importance", "feature_name"],
        ascending=[False, False, True],
        ignore_index=True,
    )


def _build_sparse_support_candidates(
    *,
    candidate_names: list[str],
    candidate_catalog: pd.DataFrame,
    retained_terms: pd.DataFrame,
    retained_interaction_pairs: pd.DataFrame,
    retained_transformations: pd.DataFrame,
) -> pd.DataFrame:
    """Build sparse-stage candidate provenance table."""
    retained_term_names = _retained_feature_names(retained_terms)
    retained_pair_names = _retained_pair_names(retained_interaction_pairs)
    retained_transformation_names = _retained_feature_names(retained_transformations)
    rows = []
    for original_position, feature_name in enumerate(candidate_names):
        catalog_row = candidate_catalog.loc[
            candidate_catalog["feature_name"].astype(str) == feature_name
        ].iloc[0]
        rows.append(
            {
                "feature_name": feature_name,
                "original_position": int(original_position),
                "feature_type": str(catalog_row.get("feature_type", "")),
                "empirical_null_retained": feature_name in retained_term_names,
                "interaction_discovery_retained": feature_name in retained_pair_names,
                "nonlinear_discovery_retained": feature_name in retained_transformation_names,
            }
        )
    return pd.DataFrame.from_records(rows)


def _build_sparse_component_coefficients(
    *,
    candidate_names: list[str],
    component_names: list[str],
    coefficients: np.ndarray,
) -> pd.DataFrame:
    """Build long-form sparse coefficient table."""
    wide = pd.DataFrame(coefficients, columns=component_names)
    wide.insert(0, "feature_name", candidate_names)
    return wide.melt(
        id_vars="feature_name",
        var_name="component",
        value_name="standardized_sparse_coefficient",
    )


def _build_sparse_selection_summary(
    *,
    n_training_rows: int,
    n_candidate_terms: int,
    n_active_candidate_terms: int,
    n_components: int,
    n_full_support_terms: int,
    n_final_stable_support_terms: int,
    mean_jaccard: float,
    mean_spearman: float,
    spec: SparseSelectionStabilitySpec,
) -> pd.DataFrame:
    """Build the one-row sparse-selection and stability summary table."""
    return pd.DataFrame(
        [
            {
                "stage": "sparse_selection_and_stability",
                "model_class": spec.model_class,
                "support_aggregation_rule": spec.support_aggregation_rule,
                "public_implementation_method": spec.implementation_method,
                "public_implementation_status": spec.implementation_status,
                "source_workflow_reference": spec.source_workflow_reference,
                "source_workflow_equivalence_status": spec.source_workflow_equivalence_status,
                "n_training_rows": int(n_training_rows),
                "n_candidate_terms": int(n_candidate_terms),
                "n_active_candidate_terms": int(n_active_candidate_terms),
                "n_components": int(n_components),
                "ebic_gamma": float(spec.ebic_gamma),
                "n_full_support_terms": int(n_full_support_terms),
                "n_final_stable_support_terms": int(n_final_stable_support_terms),
                "stability_resampling_scheme": spec.resampling_scheme,
                "stability_subsample_count": int(spec.subsample_count),
                "stability_subsample_fraction": float(spec.subsample_fraction),
                "mean_resample_jaccard": float(mean_jaccard),
                "mean_resample_spearman": float(mean_spearman),
                "jaccard_threshold": float(spec.jaccard_threshold),
                "spearman_threshold": float(spec.spearman_threshold),
            }
        ]
    )


def _build_sparse_selection_provenance(
    *,
    n_candidate_terms: int,
    n_full_support_terms: int,
    n_final_stable_support_terms: int,
    spec: SparseSelectionStabilitySpec,
) -> pd.DataFrame:
    """Build the sparse-selection provenance and equivalence-status table."""
    return pd.DataFrame(
        [
            {
                "stage": "sparse_selection_and_stability",
                "public_implementation_method": spec.implementation_method,
                "public_implementation_status": spec.implementation_status,
                "model_class": spec.model_class,
                "source_workflow_reference": spec.source_workflow_reference,
                "source_artifact": spec.source_artifact,
                "source_workflow_equivalence_status": spec.source_workflow_equivalence_status,
                "source_selected_feature_count_reference": int(
                    spec.source_selected_feature_count_reference
                ),
                "current_candidate_term_count": int(n_candidate_terms),
                "current_full_support_term_count": int(n_full_support_terms),
                "current_final_stable_support_term_count": int(n_final_stable_support_terms),
                "equivalence_note": (
                    "Public release uses deterministic EBIC-selected L1 component models "
                    "with stability filtering; equivalence to the recovered de-biased-LASSO "
                    "notebook workflow has not yet been validated."
                ),
            }
        ]
    )


def _nonlinear_transformation_candidates(
    feature_catalog: pd.DataFrame,
) -> list[tuple[str, str, str]]:
    """Return supported nonlinear-transformation candidates from a feature catalog."""
    if "feature_name" not in feature_catalog.columns:
        raise ValueError("feature_catalog must include a feature_name column.")
    if "feature_type" in feature_catalog.columns:
        candidate_rows = feature_catalog.loc[
            feature_catalog["feature_type"].astype(str).str.lower() == "transformation"
        ]
    else:
        candidate_rows = feature_catalog

    candidates: list[tuple[str, str, str]] = []
    for feature_name in candidate_rows["feature_name"].astype(str):
        parsed = _parse_supported_transformation_name(feature_name)
        if parsed is None:
            continue
        base_feature, family = parsed
        candidates.append((feature_name, base_feature, family))
    if not candidates:
        return []
    names = [candidate[0] for candidate in candidates]
    if len(names) != len(set(names)):
        raise ValueError("feature_catalog contains duplicate nonlinear transformation names.")
    return candidates


def _gam_test_and_smooth(
    x_train: np.ndarray,
    y_train: np.ndarray,
    k: int = 3,
) -> tuple[float, float, np.ndarray | None]:
    """Fit a cubic smoothing spline (GAM surrogate) and F-test for nonlinearity.

    Returns ``(smooth_edf, p_value, smooth_values_at_x_train)``.  A linear function has
    ``smooth_edf`` ≈ 2; values above 2 indicate curvature.  ``p_value`` is from an F-test
    comparing the spline fit to a linear baseline.  ``smooth_values_at_x_train`` are the spline
    predictions at the original training-sample locations, or ``None`` if the fit failed.
    """
    from scipy.interpolate import UnivariateSpline
    from scipy.stats import f as _scipy_f

    n = len(x_train)
    if n < k + 2:
        return 2.0, 1.0, None

    order = np.argsort(x_train, kind="stable")
    xs = x_train[order]
    ys = y_train[order]

    xs_unique, inverse = np.unique(xs, return_inverse=True)
    n_unique = len(xs_unique)
    if n_unique < k + 2:
        return 2.0, 1.0, None

    counts = np.bincount(inverse)
    ys_unique = np.bincount(inverse, weights=ys) / counts

    x_lin = np.column_stack([np.ones(n_unique), xs_unique])
    beta, _, _, _ = np.linalg.lstsq(x_lin, ys_unique, rcond=None)
    rss_linear = float(np.sum((ys_unique - x_lin @ beta) ** 2))

    try:
        spl = UnivariateSpline(xs_unique, ys_unique, k=k, s=None)
        y_spl = spl(xs_unique)
        rss_spline = float(np.sum((ys_unique - y_spl) ** 2))
        n_knots = len(spl.get_knots())
        smooth_edf = float(max(n_knots + k - 1, 2))
        smooth_at_train = spl(x_train)
        df_num = max(smooth_edf - 2.0, 0.5)
        df_den = max(n_unique - smooth_edf, 1.0)
        if rss_spline <= 0.0 or rss_linear <= rss_spline:
            return smooth_edf, 1.0, smooth_at_train
        f_stat = ((rss_linear - rss_spline) / df_num) / (rss_spline / df_den)
        p_value = float(_scipy_f.sf(max(f_stat, 0.0), df_num, df_den))
        return smooth_edf, p_value, smooth_at_train
    except Exception:
        return 2.0, 1.0, None


def _score_one_nonlinear_feature(
    base_feat: str,
    base_candidates: list[tuple[str, str, str]],
    x_vals: np.ndarray,
    y_scaled: np.ndarray,
    component_names: list[str],
    active_comp_indices: list[int],
    edf_threshold: float,
    gam_p_threshold: float,
) -> tuple[str, dict, list[tuple[tuple[str, str], tuple[float, float]]]]:
    """Test one base feature for nonlinearity across all active PCA components.

    Returns ``(base_feat, feature_result, gam_cache_entries)`` where ``gam_cache_entries`` is a
    list of ``((base_feat, comp_name), (edf, p))`` pairs for reassembly into the cache.
    """
    best_edf: float = 2.0
    best_p: float = 1.0
    best_comp_idx: int = active_comp_indices[0] if active_comp_indices else 0
    best_smooth: np.ndarray | None = None
    cache_entries: list[tuple[tuple[str, str], tuple[float, float]]] = []

    for comp_idx in active_comp_indices:
        y_comp = y_scaled[:, comp_idx]
        edf, p, smooth_preds = _gam_test_and_smooth(x_vals, y_comp)
        cache_entries.append(((base_feat, component_names[comp_idx]), (edf, p)))
        if edf > best_edf or (edf > edf_threshold and p < best_p):
            best_edf = edf
            best_p = p
            best_comp_idx = comp_idx
            best_smooth = smooth_preds

    is_nonlinear = (best_edf > edf_threshold) and (best_p < gam_p_threshold)

    best_transform_name = base_candidates[0][0]
    best_transform_rmse = float("nan")

    if is_nonlinear and best_smooth is not None:
        smooth_std = _standardize_vector(best_smooth)
        best_rmse = float("inf")
        for feat_name, _, family in base_candidates:
            t_vals = _apply_transform_family(x_vals, family)
            if t_vals is not None and np.isfinite(t_vals).all():
                t_std = _standardize_vector(t_vals)
                rmse = float(np.sqrt(np.mean((t_std - smooth_std) ** 2)))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_transform_name = feat_name
        best_transform_rmse = best_rmse if best_rmse < float("inf") else float("nan")

    feature_result = {
        "nonlinear": is_nonlinear,
        "best_transform_name": best_transform_name,
        "best_edf": best_edf,
        "best_p": best_p,
        "best_comp_name": component_names[best_comp_idx],
        "best_rmse": best_transform_rmse,
    }
    return base_feat, feature_result, cache_entries


def _apply_transform_family(x_values: np.ndarray, family: str) -> np.ndarray | None:
    """Apply a named algebraic transform family to raw input values."""
    if family == "quadratic":
        return x_values**2
    if family == "logarithmic":
        return np.log1p(x_values)
    if family == "inverse":
        return 1.0 / x_values
    if family == "sqrt":
        return np.sqrt(x_values)
    if family == "exponential":
        return np.exp(x_values)
    return None


def _generate_supported_nonlinear_candidates(
    retained_first_order_features: list[str],
    input_matrix: pd.DataFrame,
) -> list[tuple[str, str, str]]:
    """Generate domain-valid nonlinear candidates from retained first-order features."""
    if "sample_id" not in input_matrix.columns:
        raise ValueError("input_matrix must include a sample_id column.")
    if not retained_first_order_features:
        return []
    candidates: list[tuple[str, str, str]] = []
    for feature_name in retained_first_order_features:
        values = _source_input_column(
            input_matrix,
            feature_name,
            feature_name,
        ).to_numpy(dtype=float)
        candidates.append((f"{feature_name}_squared", feature_name, "quadratic"))
        if np.all(values > -1.0):
            candidates.append((f"log1p_{feature_name}", feature_name, "logarithmic"))
        if np.all(values != 0.0):
            candidates.append((f"inverse_{feature_name}", feature_name, "inverse"))
        if np.all(values >= 0.0):
            candidates.append((f"sqrt_{feature_name}", feature_name, "sqrt"))
    return candidates


def _parse_supported_transformation_name(feature_name: str) -> tuple[str, str] | None:
    """Parse a supported nonlinear-transformation feature name."""
    if feature_name.endswith("_squared"):
        base = feature_name.removesuffix("_squared")
        return (base, "quadratic") if base else None
    if feature_name.startswith("log1p_"):
        base = feature_name.removeprefix("log1p_")
        return (base, "logarithmic") if base else None
    if feature_name.startswith("inverse_"):
        base = feature_name.removeprefix("inverse_")
        return (base, "inverse") if base else None
    if feature_name.startswith("sqrt_"):
        base = feature_name.removeprefix("sqrt_")
        return (base, "sqrt") if base else None
    if feature_name.startswith("exp_"):
        base = feature_name.removeprefix("exp_")
        return (base, "exponential") if base else None
    return None


def _residualized_transformation_matrix(
    input_matrix: pd.DataFrame,
    train_ids: pd.Series,
    candidates: list[tuple[str, str, str]],
) -> tuple[np.ndarray, np.ndarray]:
    """Materialize train-standardized residual nonlinear transformation terms."""
    if not candidates:
        return np.empty((len(train_ids), 0), dtype=float), np.array([], dtype=bool)
    if "sample_id" not in input_matrix.columns:
        raise ValueError("input_matrix must include a sample_id column.")
    if input_matrix["sample_id"].duplicated(keep=False).any():
        raise ValueError("input_matrix must contain unique sample_id values.")

    indexed = input_matrix.set_index("sample_id", drop=False)
    missing_ids = [sample_id for sample_id in train_ids if sample_id not in indexed.index]
    if missing_ids:
        preview = ", ".join(str(value) for value in missing_ids[:5])
        raise ValueError(f"input matrix is missing sample_id values: {preview}")
    train_rows = indexed.loc[list(train_ids)].reset_index(drop=True)

    residualized_columns: list[np.ndarray] = []
    active_columns: list[bool] = []
    for feature_name, base_feature, _ in candidates:
        base_values = _source_input_column(train_rows, base_feature, feature_name).to_numpy(
            dtype=float
        )
        transformed = _materialize_feature_column(train_rows, feature_name).to_numpy(dtype=float)
        controls = np.column_stack(
            [
                np.ones(len(transformed), dtype=float),
                _standardize_vector(base_values),
            ]
        )
        coefficients, *_ = np.linalg.lstsq(controls, transformed, rcond=None)
        residual = transformed - controls @ coefficients
        residualized = _standardize_vector(residual)
        residualized_columns.append(residualized)
        active_columns.append(bool(np.any(np.abs(residualized) > 0.0)))
    return np.column_stack(residualized_columns), np.array(active_columns, dtype=bool)


def _best_component_replacement_rmse(
    residualized_transformations: np.ndarray,
    y_scaled: np.ndarray,
    coefficients: np.ndarray,
    best_component_indices: np.ndarray,
) -> np.ndarray:
    """Return the one-term replacement RMSE for each candidate's strongest component."""
    rmse = np.zeros(residualized_transformations.shape[1], dtype=float)
    for index, component_index in enumerate(best_component_indices):
        prediction = residualized_transformations[:, index] * coefficients[index, component_index]
        residual = y_scaled[:, component_index] - prediction
        rmse[index] = float(np.sqrt(np.mean(residual**2)))
    return rmse


def _build_nonlinear_transformation_scores(
    *,
    candidates: list[tuple[str, str, str]],
    curvature_scores: np.ndarray,
    active_transformations: np.ndarray,
    retained: np.ndarray,
    retained_term_names: set[str],
    component_names: list[str],
    best_component_indices: np.ndarray,
    replacement_rmse: np.ndarray,
    spec: NonlinearDiscoverySpec,
) -> pd.DataFrame:
    """Build the transformation-level nonlinear-discovery score table."""
    rows = []
    for index, (feature_name, base_feature, family) in enumerate(candidates):
        rows.append(
            {
                "feature_name": feature_name,
                "base_feature": base_feature,
                "transformation_family": family,
                "curvature_score": float(curvature_scores[index]),
                "best_component": component_names[int(best_component_indices[index])],
                "replacement_training_rmse": float(replacement_rmse[index]),
                "nonzero_residualized_training_variance": bool(active_transformations[index]),
                "empirical_null_retained": feature_name in retained_term_names,
                "retained": bool(retained[index]),
                "curvature_rule": spec.curvature_rule,
                "replacement_selection_rule": spec.replacement_selection_rule,
            }
        )
    return pd.DataFrame.from_records(rows).sort_values(
        ["retained", "curvature_score", "feature_name"],
        ascending=[False, False, True],
        ignore_index=True,
    )


def _build_component_transformation_scores(
    *,
    candidates: list[tuple[str, str, str]],
    component_names: list[str],
    coefficients: np.ndarray,
) -> pd.DataFrame:
    """Build a long-form component-level nonlinear-coefficient table."""
    wide = pd.DataFrame(coefficients, columns=component_names)
    wide.insert(0, "feature_name", [candidate[0] for candidate in candidates])
    return wide.melt(
        id_vars="feature_name",
        var_name="component",
        value_name="standardized_residual_transformation_coefficient",
    )


def _build_nonlinear_provenance(spec: NonlinearDiscoverySpec) -> pd.DataFrame:
    """Build nonlinear-discovery provenance and manuscript-alignment status table."""
    return pd.DataFrame(
        [
            {
                "stage": "nonlinear_discovery",
                "manuscript_method": spec.method,
                "manuscript_curvature_rule": spec.curvature_rule,
                "manuscript_replacement_selection_rule": spec.replacement_selection_rule,
                "public_implementation_method": spec.implementation_method,
                "public_implementation_status": spec.implementation_status,
                "source_workflow_reference": spec.source_workflow_reference,
                "source_workflow_equivalence_status": spec.source_workflow_equivalence_status,
                "manuscript_identified_transformations_reference": int(
                    spec.identified_transformations_reference
                ),
                "manuscript_final_support_transformations_reference": int(
                    spec.final_support_transformations_reference
                ),
            }
        ]
    )


def _build_nonlinear_discovery_summary(
    *,
    n_training_rows: int,
    n_candidate_transformations: int,
    n_active_transformations: int,
    n_empirical_null_retained_transformations: int,
    n_retained_transformations: int,
    n_components: int,
    max_curvature_score: float,
    spec: NonlinearDiscoverySpec,
) -> pd.DataFrame:
    """Build the one-row nonlinear-discovery summary table."""
    return pd.DataFrame(
        [
            {
                "stage": "nonlinear_discovery",
                "method": spec.method,
                "public_implementation_method": spec.implementation_method,
                "public_implementation_status": spec.implementation_status,
                "source_workflow_equivalence_status": spec.source_workflow_equivalence_status,
                "curvature_rule": spec.curvature_rule,
                "replacement_selection_rule": spec.replacement_selection_rule,
                "n_training_rows": int(n_training_rows),
                "n_candidate_transformations": int(n_candidate_transformations),
                "n_active_transformations": int(n_active_transformations),
                "n_empirical_null_retained_transformations": int(
                    n_empirical_null_retained_transformations
                ),
                "n_components": int(n_components),
                "minimum_curvature_score": float(spec.minimum_curvature_score),
                "n_retained_transformations": int(n_retained_transformations),
                "max_curvature_score": float(max_curvature_score),
                "manuscript_identified_transformations_reference": int(
                    spec.identified_transformations_reference
                ),
                "manuscript_final_support_transformations_reference": int(
                    spec.final_support_transformations_reference
                ),
            }
        ]
    )


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
    n_train, n_outputs = matrix.shape
    k = min(spec.retained_components + 10, n_train, n_outputs)
    _, singular_values, vt = randomized_svd(matrix, n_components=k, random_state=0)
    eigenvalues = singular_values**2 / max(n_train - 1, 1)
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


def _materialize_feature_column(input_matrix: pd.DataFrame, feature_name: str) -> pd.Series:
    """Materialize one feature-catalog column from source inputs."""
    if feature_name in input_matrix.columns:
        return input_matrix[feature_name]
    if ":" in feature_name:
        factors = feature_name.split(":")
        values = pd.Series(1.0, index=input_matrix.index)
        for factor in factors:
            if factor not in input_matrix.columns:
                raise ValueError(
                    f"Interaction feature {feature_name!r} references unknown input {factor!r}."
                )
            values = values * pd.to_numeric(input_matrix[factor], errors="raise")
        return values
    if feature_name.endswith("_squared"):
        base_name = feature_name.removesuffix("_squared")
        return _source_input_column(input_matrix, base_name, feature_name) ** 2
    if feature_name.startswith("log1p_"):
        base_name = feature_name.removeprefix("log1p_")
        base_values = _source_input_column(input_matrix, base_name, feature_name)
        if (base_values <= -1.0).any():
            raise ValueError(f"Feature {feature_name!r} is undefined for values <= -1.")
        return np.log1p(base_values)
    if feature_name.startswith("inverse_"):
        base_name = feature_name.removeprefix("inverse_")
        base_values = _source_input_column(input_matrix, base_name, feature_name)
        if (base_values == 0.0).any():
            raise ValueError(f"Feature {feature_name!r} is undefined for zero values.")
        return 1.0 / base_values
    if feature_name.startswith("sqrt_"):
        base_name = feature_name.removeprefix("sqrt_")
        base_values = _source_input_column(input_matrix, base_name, feature_name)
        if (base_values < 0.0).any():
            raise ValueError(f"Feature {feature_name!r} is undefined for negative values.")
        return np.sqrt(base_values)
    if feature_name.startswith("exp_"):
        base_name = feature_name.removeprefix("exp_")
        base_values = _source_input_column(input_matrix, base_name, feature_name)
        return np.exp(base_values)
    raise ValueError(
        f"Feature {feature_name!r} is not a direct input or supported catalog expression."
    )


def _source_input_column(
    input_matrix: pd.DataFrame,
    base_name: str,
    feature_name: str,
) -> pd.Series:
    """Return a finite numeric source input column for a derived feature."""
    if base_name not in input_matrix.columns:
        raise ValueError(
            f"Derived feature {feature_name!r} references unknown input {base_name!r}."
        )
    return pd.to_numeric(input_matrix[base_name], errors="raise")


def _component_columns(pca_scores: pd.DataFrame) -> list[str]:
    """Return PCA-score component columns."""
    if "sample_id" not in pca_scores.columns:
        raise ValueError("pca_scores must include a sample_id column.")
    component_names = [str(column) for column in pca_scores.columns if column != "sample_id"]
    if not component_names:
        raise ValueError("pca_scores must include at least one component column.")
    return component_names


def _align_table_by_sample_id(
    table: pd.DataFrame,
    sample_ids: pd.Series,
    value_columns: list[str],
    table_name: str,
) -> pd.DataFrame:
    """Align a sample-id table to the requested row order."""
    if "sample_id" not in table.columns:
        raise ValueError(f"{table_name} must include a sample_id column.")
    if table["sample_id"].duplicated(keep=False).any():
        raise ValueError(f"{table_name} must contain unique sample_id values.")
    indexed = table.set_index("sample_id", drop=False)
    missing_ids = [sample_id for sample_id in sample_ids if sample_id not in indexed.index]
    if missing_ids:
        preview = ", ".join(str(value) for value in missing_ids[:5])
        raise ValueError(f"{table_name} is missing sample_id values: {preview}")
    aligned = indexed.loc[list(sample_ids), value_columns].reset_index(drop=True)
    numeric = aligned.apply(pd.to_numeric, errors="raise")
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{table_name} values must be finite numeric values.")
    return numeric


def _standardize_for_screening(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return train-standardized values and a nonzero-variance column mask."""
    values = frame.to_numpy(dtype=float)
    means = values.mean(axis=0)
    scales = values.std(axis=0, ddof=0)
    active = scales > 0.0
    standardized = np.zeros_like(values, dtype=float)
    standardized[:, active] = (values[:, active] - means[active]) / scales[active]
    return standardized, active


def _permutation_row_norm_null(
    x_scaled: np.ndarray,
    y_scaled: np.ndarray,
    *,
    n_permutations: int,
    random_seed: int,
    n_jobs: int = 1,
    active_permutation_indices: set[int] | None = None,
    checkpoint_dir: Path | None = None,
    checkpoint_signature_payload: dict[str, Any] | None = None,
) -> np.ndarray:
    """Compute featurewise coefficient-row-norm statistics under response permutations."""
    rng = np.random.default_rng(random_seed)
    seeds = [int(rng.integers(0, 2**31)) for _ in range(n_permutations)]
    n_rows = float(len(x_scaled))

    def _one_permutation(seed: int) -> np.ndarray:
        local_rng = np.random.default_rng(seed)
        permuted = y_scaled[local_rng.permutation(len(y_scaled)), :]
        coefficients = (x_scaled.T @ permuted) / n_rows
        return np.linalg.norm(coefficients, axis=1)

    n_features = int(x_scaled.shape[1])
    null_statistics = np.full((n_permutations, n_features), np.nan, dtype=float)
    active_indices = (
        set(range(n_permutations))
        if active_permutation_indices is None
        else {
            int(index)
            for index in active_permutation_indices
            if 0 <= int(index) < int(n_permutations)
        }
    )
    if not active_indices:
        raise ValueError("active_permutation_indices must include at least one valid index.")
    checkpoint_run_dir = _checkpoint_run_dir(
        checkpoint_dir,
        stage="empirical_null_screen",
        signature_payload={
            "n_permutations": int(n_permutations),
            "random_seed": int(random_seed),
            "n_rows": int(len(x_scaled)),
            "n_features": n_features,
            "n_components": int(y_scaled.shape[1]),
            **(checkpoint_signature_payload or {}),
        },
    )

    def _perm_checkpoint_path(index: int) -> Path | None:
        if checkpoint_run_dir is None:
            return None
        return checkpoint_run_dir / f"perm_{index:06d}.npy"

    pending: list[int] = []
    for perm_index in range(n_permutations):
        cache_path = _perm_checkpoint_path(perm_index)
        if cache_path is None or not cache_path.exists():
            if perm_index in active_indices:
                pending.append(perm_index)
            continue
        try:
            cached = np.load(cache_path, allow_pickle=False)
        except (OSError, ValueError):
            if perm_index in active_indices:
                pending.append(perm_index)
            continue
        cached = np.asarray(cached, dtype=float)
        if cached.shape != (n_features,):
            if perm_index in active_indices:
                pending.append(perm_index)
            continue
        null_statistics[perm_index] = cached

    batch_size = _progress_batch_size(max(1, len(active_indices)), 50)
    completed = len(active_indices) - len(pending)
    start_detail = "starting empirical null permutations"
    if completed > 0:
        start_detail = (
            f"resuming empirical null permutations ({completed}/{max(1, len(active_indices))})"
        )
    _report_progress(
        stage="empirical_null_screen",
        completed=completed,
        total=max(1, len(active_indices)),
        unit="permutations",
        detail=start_detail,
    )
    with Parallel(n_jobs=n_jobs) as parallel:
        for start in range(0, len(pending), batch_size):
            stop = min(start + batch_size, len(pending))
            batch_indices = pending[start:stop]
            jobs = [delayed(_one_permutation)(seeds[i]) for i in batch_indices]
            batch = parallel(jobs)
            for permutation_index, row in zip(batch_indices, batch, strict=True):
                null_statistics[permutation_index] = row
                cache_path = _perm_checkpoint_path(permutation_index)
                if cache_path is not None:
                    temp_path = cache_path.with_suffix(".tmp.npy")
                    np.save(temp_path, np.asarray(row, dtype=float))
                    temp_path.replace(cache_path)
            completed += len(batch_indices)
            _report_progress(
                stage="empirical_null_screen",
                completed=completed,
                total=max(1, len(active_indices)),
                unit="permutations",
            )
    return null_statistics


def _benjamini_hochberg_retention(
    p_values: np.ndarray,
    *,
    q: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return BH retention flags, ranks, and critical values."""
    n_values = len(p_values)
    order = np.argsort(p_values, kind="mergesort")
    ordered_p = p_values[order]
    ordered_ranks = np.arange(1, n_values + 1)
    ordered_critical = q * ordered_ranks / float(n_values)
    accepted = ordered_p <= ordered_critical
    retained = np.zeros(n_values, dtype=bool)
    if accepted.any():
        threshold_p = ordered_p[np.flatnonzero(accepted).max()]
        retained = p_values <= threshold_p
    ranks = np.empty(n_values, dtype=int)
    ranks[order] = ordered_ranks
    critical = np.empty(n_values, dtype=float)
    critical[order] = ordered_critical
    return retained, ranks, critical


def _build_feature_screening_statistics(
    *,
    feature_names: list[str],
    observed: np.ndarray,
    p_values: np.ndarray,
    retained: np.ndarray,
    bh_rank: np.ndarray,
    bh_critical: np.ndarray,
    feature_active: np.ndarray,
) -> pd.DataFrame:
    """Build the feature-level empirical-null screening table."""
    table = pd.DataFrame(
        {
            "feature_name": feature_names,
            "observed_statistic": observed,
            "empirical_p_value": p_values,
            "bh_rank": bh_rank,
            "bh_critical_value": bh_critical,
            "retained": retained,
            "nonzero_training_variance": feature_active,
        }
    )
    return table.sort_values(
        ["empirical_p_value", "observed_statistic", "feature_name"],
        ascending=[True, False, True],
        ignore_index=True,
    )


def _build_component_coefficients(
    *,
    feature_names: list[str],
    component_names: list[str],
    coefficients: np.ndarray,
) -> pd.DataFrame:
    """Build a long-form component-coefficient table."""
    wide = pd.DataFrame(coefficients, columns=component_names)
    wide.insert(0, "feature_name", feature_names)
    return wide.melt(
        id_vars="feature_name",
        var_name="component",
        value_name="standardized_coefficient",
    )


def _build_permutation_null_summary(
    feature_names: list[str],
    null_statistics: np.ndarray,
) -> pd.DataFrame:
    """Build feature-level summaries of the empirical null distributions."""
    return pd.DataFrame(
        {
            "feature_name": feature_names,
            "null_mean_statistic": null_statistics.mean(axis=0),
            "null_quantile_95": np.quantile(null_statistics, 0.95, axis=0),
            "null_quantile_99": np.quantile(null_statistics, 0.99, axis=0),
            "null_max_statistic": null_statistics.max(axis=0),
        }
    )


def _build_empirical_null_screening_summary(
    *,
    n_training_rows: int,
    n_candidate_terms: int,
    n_active_terms: int,
    n_components: int,
    n_retained_terms: int,
    min_p_value: float,
    spec: EmpiricalNullScreeningSpec,
) -> pd.DataFrame:
    """Build the one-row empirical-null screening summary table."""
    return pd.DataFrame(
        [
            {
                "stage": "empirical_null_screening",
                "statistic": spec.statistic,
                "n_training_rows": int(n_training_rows),
                "n_candidate_terms": int(n_candidate_terms),
                "n_active_terms": int(n_active_terms),
                "n_components": int(n_components),
                "n_permutations": int(spec.permutation_count_B),
                "bh_q_screen": float(spec.bh_q_screen),
                "max_retained_terms": (
                    int(spec.max_retained_terms) if spec.max_retained_terms is not None else np.nan
                ),
                "n_retained_terms": int(n_retained_terms),
                "min_empirical_p_value": float(min_p_value),
                "manuscript_retained_terms_reference": int(spec.retained_terms_reference),
                "implementation_method": spec.implementation_method,
                "implementation_status": spec.implementation_status,
                "source_script_equivalence_status": spec.source_script_equivalence_status,
                "random_seed": int(spec.random_seed),
            }
        ]
    )


def _build_empirical_null_screening_provenance(
    spec: EmpiricalNullScreeningSpec,
) -> pd.DataFrame:
    """Build the empirical-null provenance ledger for manuscript alignment audits."""
    return pd.DataFrame(
        [
            {
                "stage": "empirical_null_screening",
                "manuscript_contract_statistic": spec.statistic,
                "public_implementation_method": spec.implementation_method,
                "public_implementation_status": spec.implementation_status,
                "source_script_reference": spec.source_script_reference,
                "source_script_equivalence_status": spec.source_script_equivalence_status,
                "permutation_count_B": spec.permutation_count_B,
                "bh_q_screen": spec.bh_q_screen,
                "max_retained_terms": (
                    int(spec.max_retained_terms) if spec.max_retained_terms is not None else np.nan
                ),
                "retained_terms_reference": spec.retained_terms_reference,
            }
        ]
    )


def _interaction_candidate_pairs(
    feature_catalog: pd.DataFrame,
) -> list[tuple[str, str, str]]:
    """Return two-factor interaction candidates from a feature catalog."""
    if "feature_name" not in feature_catalog.columns:
        raise ValueError("feature_catalog must include a feature_name column.")
    if "feature_type" in feature_catalog.columns:
        candidate_rows = feature_catalog.loc[
            feature_catalog["feature_type"].astype(str).str.lower() == "interaction"
        ]
    else:
        candidate_rows = feature_catalog

    candidates: list[tuple[str, str, str]] = []
    for feature_name in candidate_rows["feature_name"].astype(str):
        factors = feature_name.split(":")
        if len(factors) != 2:
            continue
        left, right = factors
        if not left or not right:
            raise ValueError(f"Malformed interaction feature name: {feature_name!r}")
        candidates.append((feature_name, left, right))
    if not candidates:
        return []
    names = [candidate[0] for candidate in candidates]
    if len(names) != len(set(names)):
        raise ValueError("feature_catalog contains duplicate interaction feature names.")
    return candidates


def _residualized_interaction_matrix(
    input_matrix: pd.DataFrame,
    train_ids: pd.Series,
    candidates: list[tuple[str, str, str]],
) -> np.ndarray:
    """Materialize train-standardized residual interaction terms for candidate pairs."""
    if not candidates:
        return np.empty((len(train_ids), 0), dtype=float)
    columns = sorted({factor for _, left, right in candidates for factor in (left, right)})
    train_inputs = _align_table_by_sample_id(input_matrix, train_ids, columns, "input matrix")
    source_values = {
        column: train_inputs[column].to_numpy(dtype=float) for column in train_inputs.columns
    }
    residualized_columns = []
    for _, left, right in candidates:
        left_values = source_values[left]
        right_values = source_values[right]
        product = left_values * right_values
        controls = np.column_stack(
            [
                np.ones(len(product), dtype=float),
                _standardize_vector(left_values),
                _standardize_vector(right_values),
            ]
        )
        coefficients, *_ = np.linalg.lstsq(controls, product, rcond=None)
        residual = product - controls @ coefficients
        residualized_columns.append(_standardize_vector(residual))
    return np.column_stack(residualized_columns)


def _standardize_vector(values: np.ndarray) -> np.ndarray:
    """Return a zero-mean, unit-scale vector, or zeros for constant input."""
    mean = float(values.mean())
    scale = float(values.std(ddof=0))
    if scale <= 0.0:
        return np.zeros_like(values, dtype=float)
    return (values - mean) / scale


def _permutation_max_abs_coefficient_null(
    residualized_interactions: np.ndarray,
    y_scaled: np.ndarray,
    *,
    n_permutations: int,
    random_seed: int,
) -> np.ndarray:
    """Compute max-absolute coefficient statistics under response permutations."""
    rng = np.random.default_rng(random_seed)
    null_statistics = np.zeros(
        (n_permutations, residualized_interactions.shape[1]),
        dtype=float,
    )
    n_rows = float(len(residualized_interactions))
    for index in range(n_permutations):
        permuted = y_scaled[rng.permutation(len(y_scaled)), :]
        coefficients = (residualized_interactions.T @ permuted) / n_rows
        null_statistics[index, :] = np.max(np.abs(coefficients), axis=1)
    return null_statistics


def _retained_feature_names(retained_terms: pd.DataFrame) -> set[str]:
    """Return retained empirical-null feature names from a retained-term table."""
    if "feature_name" not in retained_terms.columns:
        raise ValueError("retained_terms must include a feature_name column.")
    return set(retained_terms["feature_name"].astype(str))


def _retained_first_order_term_names(
    retained_terms: pd.DataFrame,
    input_matrix: pd.DataFrame,
    *,
    minimum_count: int = 1,
) -> list[str]:
    """Return retained first-order terms in retained-term table order."""
    if "feature_name" not in retained_terms.columns:
        raise ValueError("retained_terms must include a feature_name column.")
    if "sample_id" not in input_matrix.columns:
        raise ValueError("input_matrix must include a sample_id column.")
    available_inputs = {
        str(column) for column in input_matrix.columns if str(column) != "sample_id"
    }
    feature_types = (
        retained_terms["feature_type"].astype(str).str.lower()
        if "feature_type" in retained_terms.columns
        else pd.Series([""] * len(retained_terms))
    )
    names: list[str] = []
    for feature_name, feature_type in zip(
        retained_terms["feature_name"].astype(str),
        feature_types,
        strict=False,
    ):
        is_first_order = feature_type in {"first_order", "numeric"} if feature_type else True
        if (
            is_first_order
            and feature_name in available_inputs
            and ":" not in feature_name
            and _parse_supported_transformation_name(feature_name) is None
        ):
            names.append(feature_name)
    deduped = list(dict.fromkeys(names))
    if len(deduped) < minimum_count:
        if minimum_count == 2:
            raise ValueError(
                "Interaction discovery requires at least two retained first-order terms."
            )
        raise ValueError("No retained first-order terms available for discovery.")
    return deduped


def _fit_tree_for_shap(
    x_train: np.ndarray,
    y_comp: np.ndarray,
    n_estimators: int,
    max_depth: int,
    random_state: int,
) -> Any:
    """Fit a gradient-boosted tree regressor for SHAP interaction scoring."""
    from sklearn.ensemble import GradientBoostingRegressor

    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )
    model.fit(x_train, y_comp)
    return model


def _shap_mean_abs_interaction_matrix(
    model: Any,
    x_train: np.ndarray,
    max_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Compute mean absolute SHAP interaction matrix averaged over training samples.

    Returns a ``(n_features, n_features)`` matrix where entry ``[i, j]`` is the mean absolute
    SHAP pairwise interaction value for the feature pair ``(i, j)`` using the upper triangle
    (``i < j``) averaged symmetrically.
    """
    import shap

    n = x_train.shape[0]
    if n > max_samples:
        indices = rng.choice(n, size=max_samples, replace=False)
        x_sample = x_train[indices]
    else:
        x_sample = x_train

    explainer = shap.TreeExplainer(model)
    interactions = explainer.shap_interaction_values(x_sample)
    mean_abs = np.abs(interactions).mean(axis=0)

    n_features = mean_abs.shape[0]
    sym = np.zeros((n_features, n_features))
    for i in range(n_features):
        for j in range(i + 1, n_features):
            score = 0.5 * (mean_abs[i, j] + mean_abs[j, i])
            sym[i, j] = score
            sym[j, i] = score
    return sym


def _score_interaction_permutation(
    y_base: np.ndarray,
    permute_response: bool,
    x_feat: np.ndarray,
    n_pairs: int,
    n_comp: int,
    active_comp_indices: list[int],
    pair_to_indices: dict[str, tuple[int, int]],
    candidates: list[tuple[str, str, str]],
    n_estimators: int,
    max_depth: int,
    max_shap_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit GBTs and score SHAP interactions for one observed/permuted response matrix.

    Returns ``(max_scores, component_scores)`` where ``max_scores`` has shape ``(n_pairs,)``
    and ``component_scores`` has shape ``(n_pairs, n_comp)``.
    """
    rng = np.random.default_rng(seed)
    if permute_response:
        y_mat = np.column_stack(
            [rng.permutation(y_base[:, comp_idx]) for comp_idx in range(y_base.shape[1])]
        )
    else:
        y_mat = y_base
    scores = np.zeros((n_pairs, n_comp))
    for comp_idx in active_comp_indices:
        y_comp = y_mat[:, comp_idx]
        model = _fit_tree_for_shap(
            x_feat,
            y_comp,
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=int(rng.integers(0, 2**31)),
        )
        shap_mat = _shap_mean_abs_interaction_matrix(
            model, x_feat, max_samples=max_shap_samples, rng=rng
        )
        for i, (_, _, _) in enumerate(candidates):
            li, ri = pair_to_indices[candidates[i][0]]
            scores[i, comp_idx] = shap_mat[li, ri]
    return np.max(scores, axis=1), scores


def _generate_pairwise_interactions(
    retained_first_order_features: list[str],
) -> list[tuple[str, str, str]]:
    """Generate all unique pairwise interaction candidates from retained features."""
    if len(retained_first_order_features) < 2:
        return []
    candidates = [
        (f"{left}:{right}", left, right)
        for left, right in combinations(retained_first_order_features, 2)
    ]
    if len({pair_name for pair_name, _, _ in candidates}) != len(candidates):
        raise ValueError("Generated duplicate interaction candidate names.")
    return candidates


def _build_interaction_pair_scores(
    *,
    candidates: list[tuple[str, str, str]],
    observed_scores: np.ndarray,
    thresholds: np.ndarray,
    p_values: np.ndarray,
    retained: np.ndarray,
    retained_term_names: set[str],
    spec: InteractionDiscoverySpec,
) -> pd.DataFrame:
    """Build the pair-level interaction-discovery score table."""
    rows = []
    for index, (pair_name, left, right) in enumerate(candidates):
        rows.append(
            {
                "pair_name": pair_name,
                "left_feature": left,
                "right_feature": right,
                "interaction_score": float(observed_scores[index]),
                "null_threshold": float(thresholds[index]),
                "empirical_p_value": float(p_values[index]),
                "retained": bool(retained[index]),
                "empirical_null_retained": pair_name in retained_term_names,
                "aggregation_rule": spec.aggregation_rule,
            }
        )
    return pd.DataFrame.from_records(rows).sort_values(
        ["retained", "interaction_score", "pair_name"],
        ascending=[False, False, True],
        ignore_index=True,
    )


def _build_component_interaction_scores(
    *,
    candidates: list[tuple[str, str, str]],
    component_names: list[str],
    coefficients: np.ndarray,
) -> pd.DataFrame:
    """Build a long-form component-level interaction-coefficient table."""
    wide = pd.DataFrame(coefficients, columns=component_names)
    wide.insert(0, "pair_name", [candidate[0] for candidate in candidates])
    return wide.melt(
        id_vars="pair_name",
        var_name="component",
        value_name="standardized_residual_interaction_coefficient",
    )


def _build_interaction_null_summary(
    candidates: list[tuple[str, str, str]],
    null_statistics: np.ndarray,
) -> pd.DataFrame:
    """Build pair-level summaries of empirical-null interaction distributions."""
    return pd.DataFrame(
        {
            "pair_name": [candidate[0] for candidate in candidates],
            "null_mean_score": null_statistics.mean(axis=0),
            "null_quantile_95": np.quantile(null_statistics, 0.95, axis=0),
            "null_quantile_99": np.quantile(null_statistics, 0.99, axis=0),
            "null_max_score": null_statistics.max(axis=0),
        }
    )


def _build_interaction_provenance(spec: InteractionDiscoverySpec) -> pd.DataFrame:
    """Build interaction-discovery provenance and manuscript-alignment status table."""
    return pd.DataFrame(
        [
            {
                "stage": "interaction_discovery",
                "manuscript_method": spec.method,
                "manuscript_aggregation_rule": spec.aggregation_rule,
                "public_implementation_method": spec.implementation_method,
                "public_implementation_status": spec.implementation_status,
                "source_workflow_reference": spec.source_workflow_reference,
                "source_workflow_equivalence_status": spec.source_workflow_equivalence_status,
                "manuscript_retained_pairs_reference": int(spec.retained_pairs_reference),
            }
        ]
    )


def _build_interaction_discovery_summary(
    *,
    n_training_rows: int,
    n_candidate_pairs: int,
    n_empirical_null_retained_pairs: int,
    n_retained_pairs: int,
    n_components: int,
    max_score: float,
    spec: InteractionDiscoverySpec,
) -> pd.DataFrame:
    """Build the one-row interaction-discovery summary table."""
    return pd.DataFrame(
        [
            {
                "stage": "interaction_discovery",
                "method": spec.method,
                "public_implementation_method": spec.implementation_method,
                "public_implementation_status": spec.implementation_status,
                "source_workflow_equivalence_status": spec.source_workflow_equivalence_status,
                "aggregation_rule": spec.aggregation_rule,
                "n_training_rows": int(n_training_rows),
                "n_candidate_pairs": int(n_candidate_pairs),
                "n_empirical_null_retained_pairs": int(n_empirical_null_retained_pairs),
                "n_components": int(n_components),
                "n_permutations": int(spec.permutation_count_B),
                "null_threshold_quantile": float(spec.null_threshold_quantile),
                "n_retained_pairs": int(n_retained_pairs),
                "max_interaction_score": float(max_score),
                "manuscript_retained_pairs_reference": int(spec.retained_pairs_reference),
                "random_seed": int(spec.random_seed),
            }
        ]
    )


def _validate_final_manuscript_artifacts_spec(spec: FinalManuscriptArtifactsSpec) -> None:
    """Validate final artifact-regeneration settings."""
    if spec.final_predictor_count_reference <= 0:
        raise ValueError("final_predictor_count_reference must be positive.")
    if spec.final_first_order_input_count_reference <= 0:
        raise ValueError("final_first_order_input_count_reference must be positive.")
    if spec.nrmse_min_range <= 0.0:
        raise ValueError("nrmse_min_range must be positive.")
    if spec.bootstrap_count < 2:
        raise ValueError("bootstrap_count must be at least two.")
    if not 0.0 < spec.bootstrap_alpha < 1.0:
        raise ValueError("bootstrap_alpha must be in the interval (0, 1).")
    if spec.nrmse_reference_matrix != "Y_train":
        raise ValueError(
            "Only Y_train nRMSE normalization is supported by the final artifact stage."
        )
    if spec.inferential_filter_interval_method != (
        "hc3_wald_95_percent_drop_if_zero_compatible_for_all_outputs"
    ):
        raise ValueError("Only the frozen 95% HC3 Wald final inferential filter is supported.")
    if not 0.0 < spec.inferential_filter_alpha < 1.0:
        raise ValueError("inferential_filter_alpha must be in the interval (0, 1).")
    valid_subset_modes = {"all", "random_fraction", "target_list", "top_variance"}
    if spec.hc3_output_subset_mode not in valid_subset_modes:
        raise ValueError(
            "hc3_output_subset_mode must be one of "
            f"{sorted(valid_subset_modes)}. Got {spec.hc3_output_subset_mode!r}."
        )
    if spec.hc3_output_fraction is not None and not 0.0 < spec.hc3_output_fraction <= 1.0:
        raise ValueError("hc3_output_fraction must be in the interval (0, 1].")
    if spec.hc3_output_max_outputs is not None and spec.hc3_output_max_outputs <= 0:
        raise ValueError("hc3_output_max_outputs must be positive when provided.")
    if spec.hc3_output_subset_metric not in {"variance"}:
        raise ValueError("Only variance-based HC3 output subsetting is supported.")
    if not 0.0 < spec.pruning_error_scale_quantile <= 1.0:
        raise ValueError("pruning_error_scale_quantile must be in the interval (0, 1].")
    if (
        spec.pruning_delta_threshold_override is not None
        and spec.pruning_delta_threshold_override < 0.0
    ):
        raise ValueError("pruning_delta_threshold_override must be nonnegative when provided.")
    if spec.pruning_remove_count_override is not None and spec.pruning_remove_count_override < 0:
        raise ValueError("pruning_remove_count_override must be nonnegative when provided.")
    if (
        spec.pruning_delta_threshold_override is not None
        and spec.pruning_remove_count_override is not None
    ):
        raise ValueError(
            "Provide at most one pruning override: delta threshold or remove-count override."
        )


def _final_support_feature_names(final_stable_support: pd.DataFrame) -> list[str]:
    """Return final support names from a stable-support table."""
    if "feature_name" not in final_stable_support.columns:
        raise ValueError("final_stable_support must include a feature_name column.")
    names = final_stable_support["feature_name"].astype(str).tolist()
    if not names:
        raise ValueError("Final manuscript artifacts require a non-empty final support.")
    if len(names) != len(set(names)):
        raise ValueError("final_stable_support contains duplicate feature names.")
    return names


def _build_final_support_features(
    *,
    final_feature_names: list[str],
    final_catalog: pd.DataFrame,
    sparse_selection: SparseSelectionStabilityResult,
) -> pd.DataFrame:
    """Join final support names to catalog and sparse-stability diagnostics."""
    catalog = final_catalog.copy()
    catalog["feature_name"] = catalog["feature_name"].astype(str)
    metadata_columns = [
        column for column in ("feature_name", "feature_type", "origin") if column in catalog.columns
    ]
    metadata = catalog.loc[:, metadata_columns].copy()
    support = sparse_selection.final_stable_support.copy()
    support["feature_name"] = support["feature_name"].astype(str)
    support_columns = [
        column
        for column in (
            "feature_name",
            "full_support_importance",
            "stability_selection_frequency",
            "stable_by_jaccard",
            "stable_by_spearman",
            "final_stable_support",
        )
        if column in support.columns
    ]
    joined = metadata.merge(
        support.loc[:, support_columns],
        on="feature_name",
        how="left",
        validate="one_to_one",
    )
    ordered = pd.Categorical(joined["feature_name"], categories=final_feature_names, ordered=True)
    joined = joined.assign(feature_order=ordered.codes)
    joined = joined.sort_values("feature_order", ignore_index=True)
    joined.insert(0, "final_support_position", range(len(joined)))
    return joined.drop(columns=["feature_order"])


def _holdout_sample_ids(holdout_assignments: pd.DataFrame) -> pd.Series:
    """Return sample IDs assigned to holdout rows."""
    if "sample_id" not in holdout_assignments.columns or "split" not in holdout_assignments.columns:
        raise ValueError("holdout_assignments must include sample_id and split columns.")
    holdout = holdout_assignments.loc[
        holdout_assignments["split"].astype(str).str.lower() == "holdout",
        "sample_id",
    ]
    if holdout.empty:
        return pd.Series(dtype=holdout_assignments["sample_id"].dtype)
    return holdout.reset_index(drop=True)


def _indexed_by_sample_id(frame: pd.DataFrame, sample_ids: pd.Series) -> pd.DataFrame:
    """Return a copy of ``frame`` indexed by aligned sample IDs."""
    indexed = frame.copy()
    indexed.index = pd.Index(sample_ids.to_numpy(), name="sample_id")
    return indexed


def _select_hc3_output_names(
    y_numeric: pd.DataFrame,
    *,
    output_subset_mode: str,
    output_fraction: float | None,
    output_names: tuple[str, ...],
    max_outputs: int | None,
    random_seed: int,
    subset_metric: str,
) -> list[str]:
    """Select output columns used for HC3 filtering."""
    all_names = [str(name) for name in y_numeric.columns]
    if not all_names:
        raise ValueError("HC3 inferential filtering requires at least one output column.")
    selected: list[str]
    if output_subset_mode == "all":
        selected = all_names
    elif output_subset_mode == "target_list":
        requested = [str(name) for name in output_names]
        selected = [name for name in requested if name in set(all_names)]
        if not selected:
            raise ValueError("HC3 target_list mode selected no valid outputs.")
    elif output_subset_mode == "random_fraction":
        if output_fraction is None:
            raise ValueError("HC3 random_fraction mode requires output_fraction.")
        n_total = len(all_names)
        n_keep = max(1, int(math.ceil(n_total * output_fraction)))
        rng = np.random.default_rng(random_seed)
        selected = sorted(rng.choice(all_names, size=n_keep, replace=False).tolist())
    elif output_subset_mode == "top_variance":
        if subset_metric != "variance":
            raise ValueError("Only variance subset metric is supported for HC3 top_variance mode.")
        variances = y_numeric.var(axis=0, ddof=1).sort_values(ascending=False)
        n_keep = max_outputs if max_outputs is not None else len(all_names)
        selected = [str(name) for name in variances.index[:n_keep]]
    else:
        raise ValueError(f"Unsupported HC3 output subset mode: {output_subset_mode!r}")
    if max_outputs is not None and len(selected) > max_outputs:
        selected = selected[:max_outputs]
    if not selected:
        raise ValueError("HC3 output selection resulted in an empty output set.")
    return selected


def _build_hc3_inferential_filter_tables(
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    *,
    alpha: float,
    interval_method: str,
    output_subset_mode: str = "all",
    output_fraction: float | None = None,
    output_names: tuple[str, ...] = (),
    max_outputs: int | None = None,
    random_seed: int = 123,
    subset_metric: str = "variance",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build per-output HC3 Wald intervals and feature-level filter decisions."""
    x_numeric = x_train.apply(pd.to_numeric, errors="raise")
    y_numeric = y_train.apply(pd.to_numeric, errors="raise")
    if x_numeric.empty or y_numeric.empty:
        raise ValueError("HC3 inferential filtering requires non-empty X and Y tables.")
    if len(x_numeric) != len(y_numeric):
        raise ValueError("HC3 inferential filtering requires aligned X and Y row counts.")
    x_values = x_numeric.to_numpy(dtype=float)
    selected_output_names = _select_hc3_output_names(
        y_numeric,
        output_subset_mode=output_subset_mode,
        output_fraction=output_fraction,
        output_names=output_names,
        max_outputs=max_outputs,
        random_seed=random_seed,
        subset_metric=subset_metric,
    )
    y_selected = y_numeric.loc[:, selected_output_names]
    y_values = y_selected.to_numpy(dtype=float)
    if not np.isfinite(x_values).all() or not np.isfinite(y_values).all():
        raise ValueError("HC3 inferential filtering requires finite numeric values.")

    design = np.column_stack([np.ones(len(x_numeric), dtype=float), x_values])
    xtx_inv = np.linalg.pinv(design.T @ design)
    beta = xtx_inv @ design.T @ y_values
    fitted = design @ beta
    residuals = y_values - fitted
    leverages = np.einsum("ij,jk,ik->i", design, xtx_inv, design)
    leverage_denominator = np.clip(1.0 - leverages, 1.0e-12, None)
    z_value = NormalDist().inv_cdf(1.0 - alpha / 2.0)

    interval_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    feature_names = list(x_numeric.columns)
    output_names = list(y_selected.columns)
    output_excludes_counts = np.zeros(len(feature_names), dtype=int)
    max_abs_t_stats = np.zeros(len(feature_names), dtype=float)
    for output_position, output_name in enumerate(output_names):
        scaled_residual = residuals[:, output_position] / leverage_denominator
        meat = design.T @ ((scaled_residual**2)[:, np.newaxis] * design)
        covariance = xtx_inv @ meat @ xtx_inv
        diag_variance = np.diag(covariance)[1:]
        std_errors = np.sqrt(np.maximum(diag_variance, 0.0))
        coefficients = beta[1:, output_position]
        lower = coefficients - z_value * std_errors
        upper = coefficients + z_value * std_errors
        zero_compatible = np.logical_and(lower <= 0.0, upper >= 0.0)
        excludes_zero = ~zero_compatible
        output_excludes_counts += excludes_zero.astype(int)
        with np.errstate(divide="ignore", invalid="ignore"):
            t_stats = coefficients / std_errors
        finite_mask = np.isfinite(t_stats)
        max_abs_t_stats = np.where(
            finite_mask,
            np.maximum(max_abs_t_stats, np.abs(t_stats)),
            np.where(coefficients != 0.0, np.inf, max_abs_t_stats),
        )
        for feature_position, feature_name in enumerate(feature_names):
            interval_rows.append(
                {
                    "feature_name": str(feature_name),
                    "output_name": str(output_name),
                    "coefficient": float(coefficients[feature_position]),
                    "hc3_standard_error": float(std_errors[feature_position]),
                    "wald_z_value": float(z_value),
                    "ci_lower": float(lower[feature_position]),
                    "ci_upper": float(upper[feature_position]),
                    "zero_compatible": bool(zero_compatible[feature_position]),
                    "excludes_zero": bool(excludes_zero[feature_position]),
                    "interval_method": interval_method,
                    "alpha": float(alpha),
                }
            )
    for feature_position, feature_name in enumerate(feature_names):
        retained = bool(output_excludes_counts[feature_position] > 0)
        feature_rows.append(
            {
                "feature_name": str(feature_name),
                "n_outputs": int(len(output_names)),
                "n_outputs_excluding_zero": int(output_excludes_counts[feature_position]),
                "hc3_retained_after_filter": bool(retained),
                "hc3_drop_reason": "retained_by_at_least_one_output"
                if retained
                else "zero_compatible_for_all_outputs",
                "max_abs_hc3_t_statistic": float(max_abs_t_stats[feature_position]),
                "interval_method": interval_method,
                "alpha": float(alpha),
            }
        )

    intervals = pd.DataFrame(interval_rows)
    summary = pd.DataFrame(feature_rows)
    if not bool(summary["hc3_retained_after_filter"].any()):
        raise ValueError(
            "The HC3 final inferential filter removed every sparse/stable feature. "
            "The final manuscript OLS stage requires at least one retained term."
        )
    return intervals, summary


def _hc3_retained_feature_names(hc3_filter_summary: pd.DataFrame) -> list[str]:
    """Return feature names retained by the final HC3 inferential filter."""
    required = {"feature_name", "hc3_retained_after_filter"}
    missing = required.difference(hc3_filter_summary.columns)
    if missing:
        raise ValueError(f"hc3_filter_summary missing required columns: {sorted(missing)}")
    retained = hc3_filter_summary.loc[
        hc3_filter_summary["hc3_retained_after_filter"].astype(bool),
        "feature_name",
    ].astype(str)
    names = retained.tolist()
    if not names:
        raise ValueError("HC3 inferential filtering retained no final features.")
    return names


def _build_final_ols_summary(
    *,
    n_training_rows: int,
    n_holdout_rows: int,
    n_prefilter_features: int,
    n_hc3_features: int,
    n_features: int,
    n_outputs: int,
    final_metric: dict[str, Any],
    null_metric: dict[str, Any],
    spec: FinalManuscriptArtifactsSpec,
) -> pd.DataFrame:
    """Build the one-row final OLS summary table."""
    return pd.DataFrame(
        [
            {
                "stage": "final_ols_and_manuscript_artifacts",
                "n_training_rows": int(n_training_rows),
                "n_holdout_rows": int(n_holdout_rows),
                "n_prefilter_features": int(n_prefilter_features),
                "n_hc3_features": int(n_hc3_features),
                "n_final_features": int(n_features),
                "n_hc3_removed_features": int(n_prefilter_features - n_hc3_features),
                "n_pruning_removed_features": int(n_hc3_features - n_features),
                "n_retained_outputs": int(n_outputs),
                "final_ols_holdout_nrmse": float(final_metric["point_estimate"]),
                "final_ols_holdout_nrmse_ci_lower": float(final_metric["ci_lower"]),
                "final_ols_holdout_nrmse_ci_upper": float(final_metric["ci_upper"]),
                "null_mean_holdout_nrmse": float(null_metric["point_estimate"]),
                "bootstrap_count": int(spec.bootstrap_count),
                "bootstrap_alpha": float(spec.bootstrap_alpha),
                "inferential_filter_interval_method": spec.inferential_filter_interval_method,
                "inferential_filter_alpha": float(spec.inferential_filter_alpha),
                "nrmse_denominator_definition": spec.nrmse_denominator_definition,
                "nrmse_reference_matrix": spec.nrmse_reference_matrix,
                "nrmse_min_range": float(spec.nrmse_min_range),
                "manuscript_final_predictor_count_reference": int(
                    spec.final_predictor_count_reference
                ),
                "manuscript_final_ols_holdout_nrmse_reference": float(
                    spec.final_ols_holdout_nrmse_reference
                ),
            }
        ]
    )


def _build_model_performance_table(
    *,
    final_metric: dict[str, Any],
    null_metric: dict[str, Any],
    spec: FinalManuscriptArtifactsSpec,
) -> pd.DataFrame:
    """Build the manuscript-facing model-performance table."""
    rows = [
        _metric_row(
            model_name="null_mean_baseline_demo",
            display_name="Null mean baseline",
            source="demo_recomputed",
            metric=null_metric,
        ),
        _metric_row(
            model_name="final_ols_demo",
            display_name="Final OLS",
            source="demo_recomputed",
            metric=final_metric,
        ),
        {
            "model_name": "intermediate_penalized_reference",
            "display_name": "Intermediate penalized model",
            "source": "manuscript_reference",
            "nrmse": float(spec.intermediate_penalized_holdout_nrmse_reference),
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "n_boot": 0,
            "bootstrap_sample_size": 0,
            "normalization_reference": spec.nrmse_reference_matrix,
        },
        {
            "model_name": "final_ols_reference",
            "display_name": "Final OLS manuscript reference",
            "source": "manuscript_reference",
            "nrmse": float(spec.final_ols_holdout_nrmse_reference),
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "n_boot": 0,
            "bootstrap_sample_size": 0,
            "normalization_reference": spec.nrmse_reference_matrix,
        },
    ]
    return pd.DataFrame(rows)


def _metric_row(
    *,
    model_name: str,
    display_name: str,
    source: str,
    metric: dict[str, Any],
) -> dict[str, Any]:
    """Build one model-performance row from a bootstrap metric payload."""
    return {
        "model_name": model_name,
        "display_name": display_name,
        "source": source,
        "nrmse": float(metric["point_estimate"]),
        "ci_lower": float(metric["ci_lower"]),
        "ci_upper": float(metric["ci_upper"]),
        "n_boot": int(metric["n_boot"]),
        "bootstrap_sample_size": int(metric["bootstrap_sample_size"]),
        "normalization_reference": str(metric["normalization_reference"]),
    }


def _build_workflow_stage_summary(
    *,
    conditioning: OutputConditioningResult,
    screening: EmpiricalNullScreeningResult,
    interactions: InteractionDiscoveryResult,
    nonlinear: NonlinearDiscoveryResult,
    sparse_selection: SparseSelectionStabilityResult,
    hc3_filter_summary: pd.DataFrame,
    final_ols_summary: pd.DataFrame,
    spec: FinalManuscriptArtifactsSpec,
) -> pd.DataFrame:
    """Build a compact manuscript-facing workflow-stage summary table."""
    return pd.DataFrame(
        [
            {
                "stage": "output_conditioning",
                "primary_quantity": "retained_scalar_outputs",
                "recomputed_value": int(len(conditioning.retained_output_names)),
                "manuscript_reference_value": np.nan,
                "artifact_family": "output_conditioning",
            },
            {
                "stage": "output_conditioning",
                "primary_quantity": "retained_pca_components",
                "recomputed_value": int(len(_component_columns(conditioning.pca_scores))),
                "manuscript_reference_value": np.nan,
                "artifact_family": "output_conditioning",
            },
            {
                "stage": "empirical_null_screening",
                "primary_quantity": "retained_terms",
                "recomputed_value": int(len(screening.retained_terms)),
                "manuscript_reference_value": _summary_reference(
                    screening.summary,
                    "manuscript_retained_terms_reference",
                ),
                "artifact_family": "empirical_null_screen",
            },
            {
                "stage": "interaction_discovery",
                "primary_quantity": "retained_pairs",
                "recomputed_value": int(len(interactions.retained_pairs)),
                "manuscript_reference_value": _summary_reference(
                    interactions.summary,
                    "manuscript_retained_pairs_reference",
                ),
                "artifact_family": "interaction_discovery",
            },
            {
                "stage": "nonlinear_discovery",
                "primary_quantity": "retained_transformations",
                "recomputed_value": int(len(nonlinear.retained_transformations)),
                "manuscript_reference_value": _summary_reference(
                    nonlinear.summary,
                    "manuscript_final_support_transformations_reference",
                ),
                "artifact_family": "nonlinear_discovery",
            },
            {
                "stage": "sparse_selection_and_stability",
                "primary_quantity": "final_stable_support_terms",
                "recomputed_value": int(len(sparse_selection.final_stable_support)),
                "manuscript_reference_value": int(spec.final_predictor_count_reference),
                "artifact_family": "sparse_selection",
            },
            {
                "stage": "final_inferential_filter",
                "primary_quantity": "hc3_retained_terms",
                "recomputed_value": int(hc3_filter_summary["hc3_retained_after_filter"].sum()),
                "manuscript_reference_value": int(spec.final_predictor_count_reference),
                "artifact_family": "final_manuscript_artifacts",
            },
            {
                "stage": "feature_pruning",
                "primary_quantity": "removed_terms_after_hc3",
                "recomputed_value": int(
                    final_ols_summary.loc[0, "n_pruning_removed_features"]
                    if "n_pruning_removed_features" in final_ols_summary.columns
                    else 0
                ),
                "manuscript_reference_value": np.nan,
                "artifact_family": "final_manuscript_artifacts",
            },
            {
                "stage": "final_ols",
                "primary_quantity": "holdout_nrmse",
                "recomputed_value": float(final_ols_summary.loc[0, "final_ols_holdout_nrmse"]),
                "manuscript_reference_value": float(spec.final_ols_holdout_nrmse_reference),
                "artifact_family": "final_manuscript_artifacts",
            },
        ]
    )


def _summary_reference(summary: pd.DataFrame, column: str) -> float:
    """Extract a numeric reference value from a one-row summary table."""
    if column not in summary.columns or summary.empty:
        return float("nan")
    return float(summary.loc[0, column])


def _build_model_performance_figure_data(model_performance: pd.DataFrame) -> pd.DataFrame:
    """Return finite model-performance rows used for the SVG bar chart."""
    figure_data = model_performance.loc[
        np.isfinite(pd.to_numeric(model_performance["nrmse"], errors="coerce"))
    ].copy()
    figure_data = figure_data.sort_values(["source", "nrmse", "model_name"], ignore_index=True)
    return figure_data


def _build_support_composition_figure_data(
    final_support_features: pd.DataFrame,
) -> pd.DataFrame:
    """Count final support terms by feature type for figure rendering."""
    if "feature_name" in final_support_features.columns:
        names = final_support_features["feature_name"].astype(str)
    else:
        names = pd.Series([""], dtype=str)
    values = names.map(_legacy_feature_type_label)
    counts = values.value_counts().rename_axis("feature_type").reset_index(name="n_features")
    return counts.sort_values(["n_features", "feature_type"], ascending=[False, True])


def _build_selected_by_module_figure_data(
    final_support_features: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize selected support features by coarse module prefix."""
    if "feature_name" in final_support_features.columns:
        names = final_support_features["feature_name"].astype(str)
    else:
        names = pd.Series(["unknown"] * len(final_support_features), dtype=str)
    modules = names.map(_legacy_module_from_factor_name)
    counts = modules.value_counts().rename_axis("module").reset_index(name="n_selected_inputs")
    total = int(counts["n_selected_inputs"].sum())
    if total <= 0:
        counts["share_selected_support"] = 0.0
    else:
        counts["share_selected_support"] = counts["n_selected_inputs"] / float(total)
    return counts.sort_values(["n_selected_inputs", "module"], ascending=[False, True]).reset_index(
        drop=True
    )


def _build_nrmse_summary_figure_data(ablation_table: pd.DataFrame) -> pd.DataFrame:
    """Build manuscript-style nRMSE summary rows with confidence intervals."""
    figure_data = ablation_table.loc[:, ["model_name", "nrmse", "ci_lower", "ci_upper"]].copy()
    figure_data["nrmse"] = pd.to_numeric(figure_data["nrmse"], errors="coerce")
    figure_data["ci_lower"] = pd.to_numeric(figure_data["ci_lower"], errors="coerce")
    figure_data["ci_upper"] = pd.to_numeric(figure_data["ci_upper"], errors="coerce")
    figure_data["ci_lower"] = figure_data["ci_lower"].fillna(figure_data["nrmse"])
    figure_data["ci_upper"] = figure_data["ci_upper"].fillna(figure_data["nrmse"])
    figure_data = figure_data.loc[np.isfinite(figure_data["nrmse"])]
    return figure_data.sort_values(["nrmse", "model_name"], ignore_index=True)


def _build_feature_pruning_diagnostics(
    *,
    final_fit: Any,
    x_holdout: pd.DataFrame,
    y_holdout: pd.DataFrame,
    y_train: pd.DataFrame,
    final_support_features: pd.DataFrame,
    spec: FinalManuscriptArtifactsSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build no-refit single-feature impact diagnostics and a tunable pruning cutoff."""
    feature_names = [str(name) for name in final_fit.feature_names]
    output_names = [str(name) for name in final_fit.output_names]
    if not feature_names:
        raise ValueError("Feature pruning diagnostics require at least one final feature.")
    x_eval = x_holdout.loc[:, feature_names].apply(pd.to_numeric, errors="raise")
    y_eval = y_holdout.loc[:, output_names].apply(pd.to_numeric, errors="raise")
    y_ref = y_train.loc[:, output_names].apply(pd.to_numeric, errors="raise")

    x_values = x_eval.to_numpy(dtype=np.float64)
    y_values = y_eval.to_numpy(dtype=np.float64)
    y_ref_values = y_ref.to_numpy(dtype=np.float64)
    pred_values = (
        x_values @ np.asarray(final_fit.coef_raw_scale, dtype=np.float64).T
        + np.asarray(final_fit.intercept_raw_scale, dtype=np.float64)[np.newaxis, :]
    )
    baseline_nrmse, _, _ = macro_nrmse_with_ref(
        y_values,
        pred_values,
        y_ref_values,
        min_range=spec.nrmse_min_range,
    )
    if not math.isfinite(baseline_nrmse):
        raise ValueError("Feature pruning diagnostics require a finite baseline macro nRMSE.")

    residual = y_values - pred_values
    n_rows = residual.shape[0]
    if n_rows <= 0:
        raise ValueError("Feature pruning diagnostics require non-empty holdout rows.")
    coefficient_matrix = np.asarray(final_fit.coef_raw_scale, dtype=np.float64)
    squared_error_baseline = np.sum(residual**2, axis=0)
    residual_projection = residual.T @ x_values
    x_squared_sum = np.sum(x_values**2, axis=0)
    squared_error_without = (
        squared_error_baseline[:, np.newaxis]
        + 2.0 * coefficient_matrix * residual_projection
        + (coefficient_matrix**2) * x_squared_sum[np.newaxis, :]
    )
    squared_error_without = np.maximum(squared_error_without, 0.0)
    rmse_without = np.sqrt(squared_error_without / float(n_rows))

    reference_range = np.nanmax(y_ref_values, axis=0) - np.nanmin(y_ref_values, axis=0)
    included_outputs_mask = reference_range >= float(spec.nrmse_min_range)
    if not bool(np.any(included_outputs_mask)):
        raise ValueError(
            "Feature pruning diagnostics require at least one output above nRMSE min-range."
        )
    usable_ranges = reference_range[included_outputs_mask]
    nrmse_without = np.mean(
        rmse_without[included_outputs_mask, :] / usable_ranges[:, np.newaxis],
        axis=0,
    )
    delta_without = nrmse_without - float(baseline_nrmse)
    delta_nonnegative = np.maximum(delta_without, 0.0)

    pruning_impact = pd.DataFrame(
        {
            "feature_name": feature_names,
            "baseline_macro_nrmse": float(baseline_nrmse),
            "noref_drop_macro_nrmse": nrmse_without.astype(float),
            "delta_nrmse_when_feature_removed": delta_without.astype(float),
            "delta_nrmse_nonnegative": delta_nonnegative.astype(float),
            "delta_nrmse_abs": np.abs(delta_without).astype(float),
        }
    )
    if "feature_name" in final_support_features.columns:
        metadata_columns = [
            column
            for column in ("feature_name", "feature_type", "origin")
            if column in final_support_features.columns
        ]
        metadata = final_support_features.loc[:, metadata_columns].copy()
        metadata["feature_name"] = metadata["feature_name"].astype(str)
        pruning_impact = pruning_impact.merge(metadata, on="feature_name", how="left")
    pruning_impact = pruning_impact.sort_values(
        ["delta_nrmse_when_feature_removed", "feature_name"],
        ascending=[True, True],
        ignore_index=True,
    )
    n_features = int(len(pruning_impact))
    pruning_impact["remove_rank"] = np.arange(1, n_features + 1, dtype=int)
    pruning_impact["retained_features"] = n_features - pruning_impact["remove_rank"]
    pruning_impact["removed_fraction"] = pruning_impact["remove_rank"] / float(n_features)
    pruning_impact["cumulative_delta_upper_bound"] = np.cumsum(
        pruning_impact["delta_nrmse_nonnegative"].to_numpy(dtype=np.float64)
    )
    pruning_impact["approx_macro_nrmse_upper_bound"] = (
        float(baseline_nrmse) + pruning_impact["cumulative_delta_upper_bound"]
    )

    cumulative = pruning_impact["cumulative_delta_upper_bound"].to_numpy(dtype=np.float64)
    error_scale = float(np.quantile(cumulative, float(spec.pruning_error_scale_quantile)))
    if not math.isfinite(error_scale) or error_scale <= 0.0:
        error_scale = max(float(cumulative[-1]), 1.0)
    error_penalty = np.clip(cumulative / error_scale, 0.0, 1.0)
    utility = pruning_impact["removed_fraction"].to_numpy(dtype=np.float64) - error_penalty
    pruning_impact["utility_score"] = utility

    max_utility = float(np.max(utility))
    auto_candidates = np.flatnonzero(np.isclose(utility, max_utility, rtol=0.0, atol=1.0e-12))
    auto_remove_count = int(auto_candidates[-1] + 1) if auto_candidates.size else 1
    override_source = "auto_robust_utility"
    effective_remove_count = auto_remove_count
    if spec.pruning_remove_count_override is not None:
        override_source = "remove_count_override"
        effective_remove_count = int(spec.pruning_remove_count_override)
    elif spec.pruning_delta_threshold_override is not None:
        override_source = "delta_threshold_override"
        threshold = float(spec.pruning_delta_threshold_override)
        eligible = np.flatnonzero(
            pruning_impact["delta_nrmse_when_feature_removed"].to_numpy(dtype=np.float64)
            <= threshold
        )
        effective_remove_count = int(eligible[-1] + 1) if eligible.size else 0
    max_remove = max(0, n_features - 1)
    auto_remove_count = min(max(auto_remove_count, 0), max_remove)
    effective_remove_count = min(max(effective_remove_count, 0), max_remove)
    pruning_impact["selected_by_auto_cutoff"] = (
        pruning_impact["remove_rank"] <= auto_remove_count
    ).astype(bool)
    pruning_impact["selected_by_effective_cutoff"] = (
        pruning_impact["remove_rank"] <= effective_remove_count
    ).astype(bool)

    def _cutoff_snapshot(remove_count: int) -> tuple[float, float, float, int]:
        if remove_count <= 0:
            return float("nan"), 0.0, float(baseline_nrmse), n_features
        row = pruning_impact.iloc[int(remove_count - 1)]
        return (
            float(row["delta_nrmse_when_feature_removed"]),
            float(row["cumulative_delta_upper_bound"]),
            float(row["approx_macro_nrmse_upper_bound"]),
            int(row["retained_features"]),
        )

    (
        auto_cutoff_delta,
        auto_cumulative_delta,
        auto_approx_nrmse,
        auto_retained_features,
    ) = _cutoff_snapshot(auto_remove_count)
    (
        effective_cutoff_delta,
        effective_cumulative_delta,
        effective_approx_nrmse,
        effective_retained_features,
    ) = _cutoff_snapshot(effective_remove_count)

    pruning_summary = pd.DataFrame(
        [
            {
                "stage": "feature_pruning_diagnostics",
                "method": "no_refit_single_feature_delta",
                "auto_cutoff_method": "robust_utility",
                "n_features_total": int(n_features),
                "n_outputs_total": int(len(output_names)),
                "n_outputs_included_in_macro": int(np.sum(included_outputs_mask)),
                "baseline_macro_nrmse": float(baseline_nrmse),
                "error_scale_quantile": float(spec.pruning_error_scale_quantile),
                "error_scale_value": float(error_scale),
                "auto_remove_count": int(auto_remove_count),
                "auto_retained_features": int(auto_retained_features),
                "auto_cutoff_delta": float(auto_cutoff_delta),
                "auto_cumulative_delta_upper_bound": float(auto_cumulative_delta),
                "auto_approx_macro_nrmse_upper_bound": float(auto_approx_nrmse),
                "override_source": override_source,
                "override_delta_threshold": (
                    float(spec.pruning_delta_threshold_override)
                    if spec.pruning_delta_threshold_override is not None
                    else np.nan
                ),
                "override_remove_count": (
                    int(spec.pruning_remove_count_override)
                    if spec.pruning_remove_count_override is not None
                    else np.nan
                ),
                "effective_remove_count": int(effective_remove_count),
                "effective_retained_features": int(effective_retained_features),
                "effective_cutoff_delta": float(effective_cutoff_delta),
                "effective_cumulative_delta_upper_bound": float(effective_cumulative_delta),
                "effective_approx_macro_nrmse_upper_bound": float(effective_approx_nrmse),
            }
        ]
    )
    curve_columns = [
        "remove_rank",
        "retained_features",
        "removed_fraction",
        "feature_name",
        "feature_type",
        "origin",
        "delta_nrmse_when_feature_removed",
        "delta_nrmse_nonnegative",
        "cumulative_delta_upper_bound",
        "approx_macro_nrmse_upper_bound",
        "utility_score",
        "selected_by_auto_cutoff",
        "selected_by_effective_cutoff",
    ]
    available_curve_columns = [
        column for column in curve_columns if column in pruning_impact.columns
    ]
    figure_curve_data = pruning_impact.loc[:, available_curve_columns].copy()
    return pruning_impact, figure_curve_data, pruning_summary


def _legacy_normalize_factor_token(name: str) -> str:
    """Normalize legacy scenario-control shorthand tokens."""
    normalized = str(name).strip()
    if normalized == "UAEORO":
        return "OI.Use AEO Reference Oil"
    if normalized == "AFSC":
        return "FM.Use Agnostic FS Conversion"
    return normalized


def _legacy_strip_nonlinear_marker(feature_name: str) -> str:
    """Remove common nonlinear prefixes/suffixes used in legacy workflows."""
    name = str(feature_name).strip()
    for prefix in ("inverse_", "log1p_", "sqrt_", "exp_"):
        if name.startswith(prefix):
            return name.removeprefix(prefix)
    for suffix in ("_quadratic", "_logarithmic", "_inverse", "_exponential", "_squared"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    parsed = _parse_supported_transformation_name(name)
    if parsed is not None:
        return parsed[0]
    return name


def _legacy_feature_type_label(feature_name: str) -> str:
    """Classify a feature into the legacy figure categories."""
    name = str(feature_name)
    lowered = name.lower()
    if "*" in name or ":" in name:
        return "Second Order"
    nonlinear_tokens = (
        "_quadratic",
        "_logarithmic",
        "_inverse",
        "_exponential",
        "_squared",
        "inverse_",
        "log1p_",
        "sqrt_",
        "exp_",
    )
    if any(token in lowered for token in nonlinear_tokens):
        return "Non-Linear"
    return "First Order"


def _legacy_module_from_factor_name(name: str) -> str:
    """Map feature/factor names to legacy manuscript module labels."""
    factor = _legacy_normalize_factor_token(_legacy_strip_nonlinear_marker(str(name)))
    lowered = factor.lower()
    if "to jet" in lowered or "atj" in lowered:
        return "Starch Ethanol to Jet"

    token = factor.split(".", 1)[0].strip() if "." in factor else factor.strip()
    token_map = {
        "AHC": "Algal Hydrocarbons",
        "CHC": "Cellulosic Hydrocarbons",
        "OHC": "Oil Hydrocarbons",
        "OI": "Oil Industry",
        "SE": "Starch Ethanol Hydrocarbons",
        "WW": "Wet Waste Hydrocarbons",
        "FM": "Cellulosic Hydrocarbons",
    }
    if token in token_map:
        return token_map[token]
    if factor in {"OI.Use AEO Reference Oil"}:
        return "Oil Industry"
    if factor in {"FM.Use Agnostic FS Conversion"}:
        return "Cellulosic Hydrocarbons"
    return canonical_module_from_factor_name(factor)


def _legacy_partner_modules_from_feature_name(feature_name: str) -> list[str]:
    """Return module labels represented by one feature term."""
    term = str(feature_name)
    if "*" in term or ":" in term:
        partners = re.split(r"[:*]", term)
        modules = {
            _legacy_module_from_factor_name(part)
            for part in [piece.strip() for piece in partners if piece.strip()]
        }
        return sorted(module for module in modules if module)
    return [_legacy_module_from_factor_name(term)]


def _build_legacy_feature_type_counts(
    final_support_features: pd.DataFrame,
) -> pd.DataFrame:
    """Build legacy feature-type summary equivalent to influential run_all output."""
    if "feature_name" in final_support_features.columns:
        names = final_support_features["feature_name"].astype(str)
    else:
        names = pd.Series([], dtype=str)
    labels = names.map(_legacy_feature_type_label)
    order = ["First Order", "Second Order", "Non-Linear"]
    counts = labels.value_counts().reindex(order).fillna(0).astype(int)
    return counts.rename_axis("feature_type").reset_index(name="count")


def _build_legacy_influential_counts_by_module(
    final_support_features: pd.DataFrame,
) -> pd.DataFrame:
    """Build legacy module-count summary from final selected support features."""
    if "feature_name" in final_support_features.columns:
        names = final_support_features["feature_name"].astype(str)
    else:
        names = pd.Series([], dtype=str)
    counts: dict[str, int] = {}
    for feature_name in names:
        modules = set(_legacy_partner_modules_from_feature_name(feature_name))
        for module in modules:
            counts[module] = counts.get(module, 0) + 1
    rows = sorted(counts.items(), key=lambda item: item[0])
    return pd.DataFrame(rows, columns=["module", "count"])


def _build_legacy_interaction_counts_by_module_pair(
    final_support_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build legacy module-pair interaction summaries from selected interaction terms."""
    if "feature_name" in final_support_features.columns:
        names = final_support_features["feature_name"].astype(str)
    else:
        names = pd.Series([], dtype=str)
    interaction_names = names[names.str.contains(r"[:*]", regex=True)]

    pair_counts: dict[tuple[str, str], int] = {}
    module_set: set[str] = set()
    for feature_name in interaction_names:
        partners = [
            piece.strip() for piece in re.split(r"[:*]", str(feature_name)) if piece.strip()
        ]
        if len(partners) < 2:
            continue
        modules_a = set(_legacy_partner_modules_from_feature_name(partners[0]))
        modules_b = set(_legacy_partner_modules_from_feature_name(partners[1]))
        pairs_for_feature = {tuple(sorted((a, b))) for a in modules_a for b in modules_b}
        for pair in pairs_for_feature:
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
            module_set.update(pair)
    if not module_set:
        module_set = {
            module
            for module in _build_legacy_influential_counts_by_module(final_support_features)[
                "module"
            ].astype(str)
        }
    ordered_modules = sorted(module_set)

    pair_rows = [
        {"module_a": module_a, "module_b": module_b, "count": count}
        for (module_a, module_b), count in sorted(pair_counts.items())
    ]
    pair_counts_df = pd.DataFrame(pair_rows, columns=["module_a", "module_b", "count"])

    matrix = pd.DataFrame(0, index=ordered_modules, columns=ordered_modules, dtype=int)
    for (module_a, module_b), count in pair_counts.items():
        matrix.loc[module_a, module_b] += int(count)
        matrix.loc[module_b, module_a] += int(count)
    return pair_counts_df, matrix


def _build_legacy_module_total_interactions(module_matrix: pd.DataFrame) -> pd.DataFrame:
    """Build legacy per-module total interaction counts from a symmetric matrix."""
    if module_matrix.empty:
        return pd.DataFrame(columns=["module", "total_interactions"])
    totals = module_matrix.sum(axis=1).rename("total_interactions").reset_index()
    totals = totals.rename(columns={totals.columns[0]: "module"})
    return totals.sort_values("module", ascending=True, ignore_index=True)


def _render_module_pair_heatmap_svg(module_matrix: pd.DataFrame) -> str:
    """Render legacy-style module-pair interaction heatmap as SVG text."""
    from html import escape

    if module_matrix.empty:
        module_matrix = pd.DataFrame([[0]], index=["no data"], columns=["no data"])
    modules = [str(v) for v in module_matrix.index]
    values = module_matrix.to_numpy(dtype=float)
    max_value = float(np.nanmax(values)) if values.size else 0.0
    if not math.isfinite(max_value) or max_value <= 0.0:
        max_value = 1.0

    n = len(modules)
    cell = 48
    left_margin = 260
    top_margin = 140
    width = left_margin + cell * n + 40
    height = top_margin + cell * n + 60

    def color_for(value: float) -> str:
        ratio = max(0.0, min(1.0, float(value) / max_value))
        # white (#ffffff) -> slate blue (#334155)
        r = int(round(255 - (255 - 51) * ratio))
        g = int(round(255 - (255 - 65) * ratio))
        b = int(round(255 - (255 - 85) * ratio))
        return f"#{r:02x}{g:02x}{b:02x}"

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="24" y="36" font-family="sans-serif" font-size="20">'
        "Interaction Density by Module Pair</text>",
    ]

    for j, module in enumerate(modules):
        x = left_margin + j * cell + (cell / 2.0)
        elements.append(
            f'<text x="{x:.2f}" y="{top_margin - 18}" font-family="sans-serif" '
            'font-size="11" text-anchor="middle" '
            f'transform="rotate(-35 {x:.2f},{top_margin - 18})">'
            f"{escape(module)}</text>"
        )
    for i, module in enumerate(modules):
        y = top_margin + i * cell + (cell / 2.0)
        elements.append(
            f'<text x="24" y="{y + 4:.2f}" font-family="sans-serif" font-size="11">'
            f"{escape(module)}</text>"
        )
        for j, _ in enumerate(modules):
            value = float(values[i, j])
            x0 = left_margin + j * cell
            y0 = top_margin + i * cell
            elements.append(
                f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{cell}" height="{cell}" '
                f'fill="{color_for(value)}" stroke="#d1d5db" stroke-width="1"/>'
            )
            elements.append(
                f'<text x="{x0 + cell / 2.0:.2f}" y="{y0 + cell / 2.0 + 4:.2f}" '
                'font-family="sans-serif" font-size="11" text-anchor="middle" fill="#111827">'
                f"{int(round(value))}</text>"
            )
    elements.append("</svg>")
    return "".join(elements)


def _build_figure_specs(
    *,
    figure_model_performance_data: pd.DataFrame,
    figure_support_composition_data: pd.DataFrame,
    figure_selected_by_module_data: pd.DataFrame,
    figure_nrmse_summary_data: pd.DataFrame,
    figure_feature_pruning_curve_data: pd.DataFrame,
    legacy_feature_type_counts: pd.DataFrame,
    legacy_influential_counts_by_module: pd.DataFrame,
    legacy_interaction_counts_by_module_pair: pd.DataFrame,
    legacy_interaction_density_module_matrix: pd.DataFrame,
    legacy_module_total_interactions: pd.DataFrame,
    svg_figures: dict[str, str],
) -> pd.DataFrame:
    """Build the generated figure registry table."""
    return pd.DataFrame(
        [
            {
                "figure_name": "figure_model_performance",
                "source_data": "figure_model_performance_data.csv",
                "asset": "figure_model_performance.svg",
                "n_source_rows": int(len(figure_model_performance_data)),
                "description": "Holdout macro nRMSE comparison for demo and reference rows.",
                "svg_bytes": len(svg_figures["figure_model_performance"].encode("utf-8")),
            },
            {
                "figure_name": "figure_support_composition",
                "source_data": "figure_support_composition_data.csv",
                "asset": "figure_support_composition.svg",
                "n_source_rows": int(len(figure_support_composition_data)),
                "description": "Final stable support count by feature type.",
                "svg_bytes": len(svg_figures["figure_support_composition"].encode("utf-8")),
            },
            {
                "figure_name": "figure_selected_by_module_count",
                "source_data": "figure_selected_by_module_data.csv",
                "asset": "figure_selected_by_module_count.svg",
                "n_source_rows": int(len(figure_selected_by_module_data)),
                "description": "Final support count by module.",
                "svg_bytes": len(svg_figures["figure_selected_by_module_count"].encode("utf-8")),
            },
            {
                "figure_name": "figure_selected_by_module_share",
                "source_data": "figure_selected_by_module_data.csv",
                "asset": "figure_selected_by_module_share.svg",
                "n_source_rows": int(len(figure_selected_by_module_data)),
                "description": "Final support share by module.",
                "svg_bytes": len(svg_figures["figure_selected_by_module_share"].encode("utf-8")),
            },
            {
                "figure_name": "figure_nrmse_bootstrap_summary",
                "source_data": "figure_nrmse_summary_data.csv",
                "asset": "figure_nrmse_bootstrap_summary.svg",
                "n_source_rows": int(len(figure_nrmse_summary_data)),
                "description": (
                    "Model macro nRMSE point estimates with bootstrap confidence intervals."
                ),
                "svg_bytes": len(svg_figures["figure_nrmse_bootstrap_summary"].encode("utf-8")),
            },
            {
                "figure_name": "fig_feature_type_distribution",
                "source_data": "feature_type_counts.csv",
                "asset": "fig_feature_type_distribution.svg",
                "n_source_rows": int(len(legacy_feature_type_counts)),
                "description": "Legacy influential feature type distribution summary.",
                "svg_bytes": len(svg_figures["fig_feature_type_distribution"].encode("utf-8")),
            },
            {
                "figure_name": "fig_influential_by_module",
                "source_data": "influential_counts_by_module.csv",
                "asset": "fig_influential_by_module.svg",
                "n_source_rows": int(len(legacy_influential_counts_by_module)),
                "description": "Legacy influential feature count summary by module.",
                "svg_bytes": len(svg_figures["fig_influential_by_module"].encode("utf-8")),
            },
            {
                "figure_name": "fig_module_pair_heatmap",
                "source_data": "interaction_density_module_matrix.csv",
                "asset": "fig_module_pair_heatmap.svg",
                "n_source_rows": int(len(legacy_interaction_counts_by_module_pair)),
                "description": "Legacy interaction-density heatmap by module pair.",
                "svg_bytes": len(svg_figures["fig_module_pair_heatmap"].encode("utf-8")),
            },
            {
                "figure_name": "fig_module_total_interactions",
                "source_data": "module_total_interactions.csv",
                "asset": "fig_module_total_interactions.svg",
                "n_source_rows": int(len(legacy_module_total_interactions)),
                "description": "Legacy total interaction counts by module.",
                "svg_bytes": len(svg_figures["fig_module_total_interactions"].encode("utf-8")),
            },
            {
                "figure_name": "figure_feature_pruning_curve",
                "source_data": "figure_feature_pruning_curve_data.csv",
                "asset": "figure_feature_pruning_curve.svg",
                "n_source_rows": int(len(figure_feature_pruning_curve_data)),
                "description": (
                    "No-refit feature-pruning frontier with automatic and effective cutoffs."
                ),
                "svg_bytes": len(svg_figures["figure_feature_pruning_curve"].encode("utf-8")),
            },
        ]
    )


def _build_final_artifact_summary(
    *,
    prefilter_support_features: pd.DataFrame,
    final_support_features: pd.DataFrame,
    hc3_filter_summary: pd.DataFrame,
    final_ols_summary: pd.DataFrame,
    workflow_stage_summary: pd.DataFrame,
    figure_specs: pd.DataFrame,
    spec: FinalManuscriptArtifactsSpec,
) -> pd.DataFrame:
    """Build the one-row summary of regenerated final manuscript artifacts."""
    n_hc3_retained = int(hc3_filter_summary["hc3_retained_after_filter"].sum())
    n_final = int(len(final_support_features))
    return pd.DataFrame(
        [
            {
                "stage": "final_manuscript_tables_and_figures",
                "n_prefilter_support_features": int(len(prefilter_support_features)),
                "n_hc3_retained_features": n_hc3_retained,
                "n_hc3_removed_features": int(len(prefilter_support_features) - n_hc3_retained),
                "n_final_support_features": n_final,
                "n_pruning_removed_features": int(n_hc3_retained - n_final),
                "n_workflow_stage_rows": int(len(workflow_stage_summary)),
                "n_figures": int(len(figure_specs)),
                "final_ols_holdout_nrmse": float(
                    final_ols_summary.loc[0, "final_ols_holdout_nrmse"]
                ),
                "manuscript_final_predictor_count_reference": int(
                    spec.final_predictor_count_reference
                ),
                "manuscript_final_first_order_input_count_reference": int(
                    spec.final_first_order_input_count_reference
                ),
                "manuscript_intermediate_penalized_holdout_nrmse_reference": float(
                    spec.intermediate_penalized_holdout_nrmse_reference
                ),
                "manuscript_final_ols_holdout_nrmse_reference": float(
                    spec.final_ols_holdout_nrmse_reference
                ),
            }
        ]
    )


def _render_feature_pruning_curve_svg(
    data: pd.DataFrame,
    *,
    auto_remove_count: int,
    effective_remove_count: int,
) -> str:
    """Render a no-refit pruning frontier SVG with auto/effective cutoff markers."""
    rows = data.loc[
        :,
        ["remove_rank", "retained_features", "approx_macro_nrmse_upper_bound"],
    ].copy()
    rows["remove_rank"] = pd.to_numeric(rows["remove_rank"], errors="coerce")
    rows["retained_features"] = pd.to_numeric(rows["retained_features"], errors="coerce")
    rows["approx_macro_nrmse_upper_bound"] = pd.to_numeric(
        rows["approx_macro_nrmse_upper_bound"],
        errors="coerce",
    )
    rows = rows.loc[
        np.isfinite(rows["remove_rank"])
        & np.isfinite(rows["retained_features"])
        & np.isfinite(rows["approx_macro_nrmse_upper_bound"])
    ]
    rows = rows.sort_values("remove_rank", ignore_index=True)
    if rows.empty:
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="220" '
            'viewBox="0 0 640 220">'
            '<rect x="0" y="0" width="640" height="220" fill="white"/>'
            '<text x="320" y="110" text-anchor="middle" font-family="sans-serif" font-size="14" '
            'fill="#374151">No feature-pruning curve data available.</text></svg>'
        )

    width = 980
    height = 560
    left = 90.0
    right = 40.0
    top = 30.0
    bottom = 70.0
    plot_w = width - left - right
    plot_h = height - top - bottom

    x_values = rows["retained_features"].to_numpy(dtype=np.float64)
    y_values = rows["approx_macro_nrmse_upper_bound"].to_numpy(dtype=np.float64)
    x_min = float(np.min(x_values))
    x_max = float(np.max(x_values))
    y_min = float(np.min(y_values))
    y_max = float(np.max(y_values))
    if math.isclose(x_min, x_max):
        x_min -= 1.0
        x_max += 1.0
    if math.isclose(y_min, y_max):
        pad = max(abs(y_min) * 0.1, 1.0e-6)
        y_min -= pad
        y_max += pad

    def sx(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_w

    def sy(value: float) -> float:
        return top + (1.0 - (value - y_min) / (y_max - y_min)) * plot_h

    points = " ".join(
        f"{sx(float(x)):.2f},{sy(float(y)):.2f}" for x, y in zip(x_values, y_values, strict=True)
    )
    baseline = float(rows["approx_macro_nrmse_upper_bound"].iloc[0])
    n_features = int(rows["retained_features"].max()) + 1

    auto_remove = int(max(auto_remove_count, 0))
    auto_remove = min(auto_remove, max(0, n_features - 1))
    effective_remove = int(max(effective_remove_count, 0))
    effective_remove = min(effective_remove, max(0, n_features - 1))
    auto_retained = n_features - auto_remove
    effective_retained = n_features - effective_remove

    x_ticks = np.linspace(x_min, x_max, num=6)
    y_ticks = np.linspace(y_min, y_max, num=6)
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>',
        f'<line x1="{left:.2f}" y1="{top:.2f}" x2="{left:.2f}" y2="{top + plot_h:.2f}" '
        'stroke="#111827" stroke-width="2"/>',
        f'<line x1="{left:.2f}" y1="{top + plot_h:.2f}" x2="{left + plot_w:.2f}" '
        f'y2="{top + plot_h:.2f}" stroke="#111827" stroke-width="2"/>',
    ]
    for tick in x_ticks:
        x = sx(float(tick))
        elements.append(
            f'<line x1="{x:.2f}" y1="{top + plot_h:.2f}" x2="{x:.2f}" y2="{top + plot_h + 6:.2f}" '
            'stroke="#374151" stroke-width="1"/>'
        )
        elements.append(
            f'<text x="{x:.2f}" y="{top + plot_h + 24:.2f}" text-anchor="middle" '
            'font-family="sans-serif" font-size="11" fill="#374151">'
            f"{int(round(tick))}</text>"
        )
    for tick in y_ticks:
        y = sy(float(tick))
        elements.append(
            f'<line x1="{left - 6:.2f}" y1="{y:.2f}" x2="{left:.2f}" y2="{y:.2f}" '
            'stroke="#374151" stroke-width="1"/>'
        )
        elements.append(
            f'<line x1="{left:.2f}" y1="{y:.2f}" x2="{left + plot_w:.2f}" y2="{y:.2f}" '
            'stroke="#e5e7eb" stroke-width="1"/>'
        )
        elements.append(
            f'<text x="{left - 10:.2f}" y="{y + 4:.2f}" text-anchor="end" '
            'font-family="sans-serif" font-size="11" fill="#374151">'
            f"{tick:.3f}</text>"
        )
    elements.extend(
        [
            f'<polyline points="{points}" fill="none" stroke="#2563eb" stroke-width="2"/>',
            f'<line x1="{left:.2f}" y1="{sy(baseline):.2f}" x2="{left + plot_w:.2f}" '
            f'y2="{sy(baseline):.2f}" stroke="#10b981" stroke-width="1.5" stroke-dasharray="5 4"/>',
        ]
    )
    if x_min <= auto_retained <= x_max:
        auto_x = sx(float(auto_retained))
        elements.append(
            f'<line x1="{auto_x:.2f}" y1="{top:.2f}" x2="{auto_x:.2f}" y2="{top + plot_h:.2f}" '
            'stroke="#dc2626" stroke-width="1.5" stroke-dasharray="6 4"/>'
        )
    if x_min <= effective_retained <= x_max:
        eff_x = sx(float(effective_retained))
        elements.append(
            f'<line x1="{eff_x:.2f}" y1="{top:.2f}" x2="{eff_x:.2f}" y2="{top + plot_h:.2f}" '
            'stroke="#7c3aed" stroke-width="1.5" stroke-dasharray="3 3"/>'
        )
    legend_x = left + 12.0
    legend_y = top + 12.0
    legend_rows = [
        ("#2563eb", "solid", "Approx. no-refit upper-bound nRMSE"),
        ("#10b981", "dash", "Baseline macro nRMSE"),
        ("#dc2626", "dash", "Auto cutoff"),
        ("#7c3aed", "dash", "Effective cutoff"),
    ]
    for idx, (color, style, label) in enumerate(legend_rows):
        y = legend_y + idx * 18.0
        dash = ' stroke-dasharray="5 4"' if style == "dash" else ""
        elements.append(
            f'<line x1="{legend_x:.2f}" y1="{y:.2f}" x2="{legend_x + 20:.2f}" y2="{y:.2f}" '
            f'stroke="{color}" stroke-width="2"{dash}/>'
        )
        elements.append(
            f'<text x="{legend_x + 26:.2f}" y="{y + 4:.2f}" font-family="sans-serif" '
            f'font-size="11" fill="#111827">{label}</text>'
        )
    elements.extend(
        [
            f'<text x="{left + plot_w / 2.0:.2f}" y="{height - 20:.2f}" text-anchor="middle" '
            'font-family="sans-serif" font-size="13" fill="#111827">Retained feature count</text>',
            f'<text x="{20:.2f}" y="{top + plot_h / 2.0:.2f}" text-anchor="middle" '
            'font-family="sans-serif" font-size="13" fill="#111827" '
            f'transform="rotate(-90 20 {top + plot_h / 2.0:.2f})">'
            "Approx. macro nRMSE upper bound</text>",
            f'<text x="{left:.2f}" y="{height - 44:.2f}" text-anchor="start" '
            'font-family="sans-serif" font-size="11" fill="#4b5563">'
            f"Auto remove={auto_remove}, effective remove={effective_remove}</text>",
            "</svg>",
        ]
    )
    return "".join(elements)


def _render_nrmse_summary_svg(data: pd.DataFrame) -> str:
    """Render a dependency-free SVG with point estimates and CI whiskers."""
    from html import escape

    rows = data.loc[:, ["model_name", "nrmse", "ci_lower", "ci_upper"]].copy()
    rows["nrmse"] = pd.to_numeric(rows["nrmse"], errors="coerce")
    rows["ci_lower"] = pd.to_numeric(rows["ci_lower"], errors="coerce")
    rows["ci_upper"] = pd.to_numeric(rows["ci_upper"], errors="coerce")
    rows = rows.loc[np.isfinite(rows["nrmse"])]
    if rows.empty:
        rows = pd.DataFrame(
            {"model_name": ["no finite data"], "nrmse": [0.0], "ci_lower": [0.0], "ci_upper": [0.0]}
        )
    rows["ci_lower"] = rows["ci_lower"].fillna(rows["nrmse"])
    rows["ci_upper"] = rows["ci_upper"].fillna(rows["nrmse"])

    min_x = float(min(rows["ci_lower"].min(), rows["nrmse"].min()))
    max_x = float(max(rows["ci_upper"].max(), rows["nrmse"].max()))
    if not math.isfinite(min_x):
        min_x = 0.0
    if not math.isfinite(max_x):
        max_x = 1.0
    if max_x <= min_x:
        max_x = min_x + 1.0

    width = 860
    row_height = 34
    top_margin = 58
    left_margin = 320
    right_margin = 110
    axis_width = width - left_margin - right_margin
    height = top_margin + row_height * len(rows) + 40

    def scale(value: float) -> float:
        return left_margin + axis_width * (value - min_x) / (max_x - min_x)

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="24" y="32" font-family="sans-serif" font-size="20">'
        "Macro nRMSE with bootstrap confidence intervals</text>",
        f'<line x1="{left_margin}" y1="{top_margin - 14}" x2="{left_margin + axis_width}" '
        f'y2="{top_margin - 14}" stroke="#9ca3af" stroke-width="1"/>',
    ]
    for row_index, (_, row) in enumerate(rows.iterrows()):
        y = top_margin + row_index * row_height
        label = escape(str(row["model_name"]))
        lower = float(row["ci_lower"])
        upper = float(row["ci_upper"])
        point = float(row["nrmse"])
        x_lower = scale(lower)
        x_upper = scale(upper)
        x_point = scale(point)
        elements.extend(
            [
                f'<text x="24" y="{y + 18}" font-family="sans-serif" font-size="13">{label}</text>',
                f'<line x1="{x_lower:.2f}" y1="{y + 12}" x2="{x_upper:.2f}" y2="{y + 12}" '
                'stroke="#4b5563" stroke-width="2"/>',
                f'<circle cx="{x_point:.2f}" cy="{y + 12}" r="4.2" fill="#111827"/>',
                f'<text x="{x_upper + 8:.2f}" y="{y + 16}" font-family="sans-serif" font-size="12">'
                f"{point:.4g}</text>",
            ]
        )
    elements.append("</svg>")
    return "".join(elements)


def _render_horizontal_bar_svg(
    data: pd.DataFrame,
    *,
    label_column: str,
    value_column: str,
    title: str,
) -> str:
    """Render a small dependency-free horizontal bar chart as SVG text."""
    from html import escape

    rows = data.loc[:, [label_column, value_column]].copy()
    rows[value_column] = pd.to_numeric(rows[value_column], errors="coerce")
    rows = rows.loc[np.isfinite(rows[value_column])]
    if rows.empty:
        rows = pd.DataFrame({label_column: ["no finite data"], value_column: [0.0]})
    width = 760
    row_height = 32
    top_margin = 54
    left_margin = 260
    right_margin = 120
    height = top_margin + row_height * len(rows) + 34
    max_value = float(rows[value_column].max())
    if max_value <= 0.0 or not math.isfinite(max_value):
        max_value = 1.0
    bar_max_width = width - left_margin - right_margin
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="24" y="32" font-family="sans-serif" font-size="20">{escape(title)}</text>',
    ]
    for row_index, (_, row) in enumerate(rows.iterrows()):
        y = top_margin + row_index * row_height
        label = escape(str(row[label_column]))
        value = float(row[value_column])
        bar_width = max(1.0, bar_max_width * value / max_value)
        elements.extend(
            [
                f'<text x="24" y="{y + 18}" font-family="sans-serif" font-size="13">{label}</text>',
                f'<rect x="{left_margin}" y="{y}" width="{bar_width:.2f}" '
                'height="20" fill="#4b5563"/>',
                f'<text x="{left_margin + bar_width + 8:.2f}" y="{y + 16}" '
                f'font-family="sans-serif" font-size="12">{value:.4g}</text>',
            ]
        )
    elements.append("</svg>")
    return "\n".join(elements) + "\n"
