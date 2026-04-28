"""Executable manuscript-reproduction stages for the BSM case study.

The functions in this module implement source-level stage computations that are shared by the
tracked manuscript notebooks and automated tests. They operate on the artifact tables resolved by
``bsm_rfm.manuscript_runtime`` and write deterministic intermediate artifacts under the resolved
manuscript output root.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso


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
    random_seed
        Deterministic random seed for the permutation sequence.
    """

    statistic: str
    permutation_count_B: int
    bh_q_screen: float
    retained_terms_reference: int
    random_seed: int = 123


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
    summary
        One-row summary of the screening stage.
    """

    feature_screening_statistics: pd.DataFrame
    component_coefficients: pd.DataFrame
    permutation_null_summary: pd.DataFrame
    retained_terms: pd.DataFrame
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
    """

    method: str
    aggregation_rule: str
    null_threshold_quantile: float
    retained_pairs_reference: int
    permutation_count_B: int
    random_seed: int = 123


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
    summary
        One-row summary of the interaction-discovery stage.
    """

    pair_scores: pd.DataFrame
    component_interaction_scores: pd.DataFrame
    interaction_null_summary: pd.DataFrame
    retained_pairs: pd.DataFrame
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
    """

    method: str
    curvature_rule: str
    replacement_selection_rule: str
    identified_transformations_reference: int
    final_support_transformations_reference: int
    minimum_curvature_score: float = 1.0e-12


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
    summary
        One-row summary of the nonlinear-discovery stage.
    """

    transformation_scores: pd.DataFrame
    component_transformation_scores: pd.DataFrame
    retained_transformations: pd.DataFrame
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
    random_seed: int = 123


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
    summary
        One-row summary of the sparse-selection and stability stage.
    """

    support_candidates: pd.DataFrame
    component_model_selection: pd.DataFrame
    component_coefficients: pd.DataFrame
    stability_resample_summary: pd.DataFrame
    stability_feature_summary: pd.DataFrame
    final_stable_support: pd.DataFrame
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
    return EmpiricalNullScreeningSpec(
        statistic=str(section["statistic"]),
        permutation_count_B=int(section["permutation_count_B"]),
        bh_q_screen=float(section["bh_q_screen"]),
        retained_terms_reference=int(section["retained_terms"]),
        random_seed=int(interface.get("holdout_random_seed", 123)),
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

    design = pd.DataFrame({"sample_id": input_matrix["sample_id"].reset_index(drop=True)})
    for feature_name in feature_catalog["feature_name"].astype(str):
        if feature_name in design.columns:
            raise ValueError(f"Duplicate feature name in feature_catalog: {feature_name}")
        values = _materialize_feature_column(input_matrix, feature_name)
        numeric = pd.to_numeric(values, errors="raise").to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise ValueError(f"Feature {feature_name!r} produced non-finite values.")
        design[feature_name] = numeric
    return design


def screen_manuscript_empirical_null_terms(
    input_matrix: pd.DataFrame,
    feature_catalog: pd.DataFrame,
    holdout_assignments: pd.DataFrame,
    pca_scores: pd.DataFrame,
    spec: EmpiricalNullScreeningSpec,
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
        Manuscript feature catalog to materialize and screen.
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

    design = build_manuscript_feature_design(input_matrix, feature_catalog)
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
    null_statistics = _permutation_row_norm_null(
        x_scaled,
        y_scaled,
        n_permutations=spec.permutation_count_B,
        random_seed=spec.random_seed,
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
    summary = _build_empirical_null_screening_summary(
        n_training_rows=len(x_train),
        n_candidate_terms=len(feature_names),
        n_active_terms=int(feature_active.sum()),
        n_components=len(component_names),
        n_retained_terms=len(retained_terms),
        min_p_value=float(p_values.min()),
        spec=spec,
    )
    return EmpiricalNullScreeningResult(
        feature_screening_statistics=feature_stats,
        component_coefficients=component_coefficients,
        permutation_null_summary=null_summary,
        retained_terms=retained_terms,
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
    return InteractionDiscoverySpec(
        method=str(interaction["method"]),
        aggregation_rule=str(interaction["aggregation_rule"]),
        null_threshold_quantile=float(interaction["null_threshold_quantile"]),
        retained_pairs_reference=int(interaction["retained_pairs"]),
        permutation_count_B=int(empirical_null["permutation_count_B"]),
        random_seed=int(interface.get("holdout_random_seed", 123)),
    )


def discover_manuscript_interactions(
    input_matrix: pd.DataFrame,
    feature_catalog: pd.DataFrame,
    holdout_assignments: pd.DataFrame,
    pca_scores: pd.DataFrame,
    retained_terms: pd.DataFrame,
    spec: InteractionDiscoverySpec,
) -> InteractionDiscoveryResult:
    """Discover candidate interaction pairs from the manuscript feature catalog.

    The public reproduction package uses the released feature catalog as the authoritative
    candidate-pair source. Each pair is scored by residualizing its product term against the
    corresponding first-order factors on training rows, then aggregating the absolute standardized
    coefficient across retained PCA components. Response permutations provide deterministic
    pair-specific null thresholds for CI/demo execution.

    Parameters
    ----------
    input_matrix
        Case-study input table with ``sample_id`` and source input columns.
    feature_catalog
        Manuscript feature catalog containing candidate colon-delimited interaction terms.
    holdout_assignments
        Table with ``sample_id`` and ``split`` columns. Only train rows are scored.
    pca_scores
        Output-conditioning PCA score table with ``sample_id`` and component columns.
    retained_terms
        Empirical-null retained-term table with at least a ``feature_name`` column. The table is
        used to annotate which candidate pairs survived the previous screening stage.
    spec
        Interaction-discovery specification.

    Returns
    -------
    InteractionDiscoveryResult
        Materialized pair-score, component-score, null-summary, retained-pair, and summary tables.
    """
    if spec.method != "tree_shap_interaction_values":
        raise ValueError(f"Unsupported interaction-discovery method: {spec.method}")
    expected_rule = "max_over_components_of_mean_absolute_shap_interaction"
    if spec.aggregation_rule != expected_rule:
        raise ValueError(f"Unsupported interaction aggregation rule: {spec.aggregation_rule}")
    if spec.permutation_count_B < 1:
        raise ValueError("permutation_count_B must be positive.")
    if not 0.0 < spec.null_threshold_quantile < 1.0:
        raise ValueError("null_threshold_quantile must be in the interval (0, 1).")

    candidates = _interaction_candidate_pairs(feature_catalog)
    component_names = _component_columns(pca_scores)
    train_ids = _train_sample_ids(holdout_assignments)
    y_train = _align_table_by_sample_id(pca_scores, train_ids, component_names, "PCA scores")
    if len(y_train) < 4:
        raise ValueError("Interaction discovery requires at least four training rows.")
    y_scaled, component_active = _standardize_for_screening(y_train)
    if not component_active.any():
        raise ValueError("All retained PCA components have zero training variance.")

    residualized = _residualized_interaction_matrix(input_matrix, train_ids, candidates)
    if residualized.shape[1] == 0:
        raise ValueError("feature_catalog does not contain any two-factor interaction candidates.")

    coefficients = (residualized.T @ y_scaled) / float(len(residualized))
    observed_scores = np.max(np.abs(coefficients), axis=1)
    null_statistics = _permutation_max_abs_coefficient_null(
        residualized,
        y_scaled,
        n_permutations=spec.permutation_count_B,
        random_seed=spec.random_seed,
    )
    thresholds = np.quantile(null_statistics, spec.null_threshold_quantile, axis=0)
    p_values = (1.0 + (null_statistics >= observed_scores[None, :]).sum(axis=0)) / (
        spec.permutation_count_B + 1.0
    )
    retained = observed_scores > thresholds
    retained_term_names = _retained_feature_names(retained_terms)
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
        coefficients=coefficients,
    )
    null_summary = _build_interaction_null_summary(candidates, null_statistics)
    retained_pairs = pair_scores.loc[pair_scores["retained"]].copy()
    retained_pairs = retained_pairs.sort_values(
        ["interaction_score", "pair_name"],
        ascending=[False, True],
        ignore_index=True,
    )
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
    )
    interaction_spec = interaction_discovery_spec_from_case_study_config(context.case_study_config)
    interactions = discover_manuscript_interactions(
        context.tables["case_study_input_matrix"],
        context.tables["manuscript_feature_catalog"],
        context.tables["fixed_holdout_assignments"],
        conditioning.pca_scores,
        screening.retained_terms,
        interaction_spec,
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
    return NonlinearDiscoverySpec(
        method=str(section["method"]),
        curvature_rule=str(section["curvature_rule"]),
        replacement_selection_rule=str(section["replacement_selection_rule"]),
        identified_transformations_reference=int(section["identified_transformations"]),
        final_support_transformations_reference=int(section["final_support_transformations"]),
    )


def discover_manuscript_nonlinear_transformations(
    input_matrix: pd.DataFrame,
    feature_catalog: pd.DataFrame,
    holdout_assignments: pd.DataFrame,
    pca_scores: pd.DataFrame,
    retained_terms: pd.DataFrame,
    spec: NonlinearDiscoverySpec,
) -> NonlinearDiscoveryResult:
    """Score catalog-defined nonlinear transformations against PCA response scores.

    The public reproduction stage implements a deterministic, dependency-light analogue of the
    frozen case-study nonlinear-discovery contract. Each candidate transformation from the released
    feature catalog is residualized against an intercept and its source first-order input on
    training rows. The residualized candidate is then scored by its maximum absolute standardized
    coefficient
    across retained PCA components. This isolates incremental nonlinear evidence rather than
    re-counting the source linear main effect.

    Parameters
    ----------
    input_matrix
        Case-study input table with ``sample_id`` and source input columns.
    feature_catalog
        Manuscript feature catalog containing candidate transformation terms.
    holdout_assignments
        Table with ``sample_id`` and ``split`` columns. Only train rows are scored.
    pca_scores
        Output-conditioning PCA score table with ``sample_id`` and component columns.
    retained_terms
        Empirical-null retained-term table with at least a ``feature_name`` column. The table is
        used to annotate which transformation candidates survived the previous screening stage.
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

    candidates = _nonlinear_transformation_candidates(feature_catalog)
    component_names = _component_columns(pca_scores)
    train_ids = _train_sample_ids(holdout_assignments)
    y_train = _align_table_by_sample_id(pca_scores, train_ids, component_names, "PCA scores")
    if len(y_train) < 3:
        raise ValueError("Nonlinear discovery requires at least three training rows.")
    y_scaled, component_active = _standardize_for_screening(y_train)
    if not component_active.any():
        raise ValueError("All retained PCA components have zero training variance.")

    residualized, active_transformations = _residualized_transformation_matrix(
        input_matrix,
        train_ids,
        candidates,
    )
    if residualized.shape[1] == 0:
        raise ValueError("feature_catalog does not contain supported nonlinear candidates.")

    coefficients = (residualized.T @ y_scaled) / float(len(residualized))
    curvature_scores = np.max(np.abs(coefficients), axis=1)
    best_component_indices = np.argmax(np.abs(coefficients), axis=1)
    replacement_rmse = _best_component_replacement_rmse(
        residualized,
        y_scaled,
        coefficients,
        best_component_indices,
    )
    retained = active_transformations & (curvature_scores > spec.minimum_curvature_score)
    retained_term_names = _retained_feature_names(retained_terms)

    transformation_scores = _build_nonlinear_transformation_scores(
        candidates=candidates,
        curvature_scores=curvature_scores,
        active_transformations=active_transformations,
        retained=retained,
        retained_term_names=retained_term_names,
        component_names=component_names,
        best_component_indices=best_component_indices,
        replacement_rmse=replacement_rmse,
        spec=spec,
    )
    component_scores = _build_component_transformation_scores(
        candidates=candidates,
        component_names=component_names,
        coefficients=coefficients,
    )
    retained_transformations = transformation_scores.loc[transformation_scores["retained"]].copy()
    retained_transformations = retained_transformations.sort_values(
        ["curvature_score", "feature_name"],
        ascending=[False, True],
        ignore_index=True,
    )
    summary = _build_nonlinear_discovery_summary(
        n_training_rows=len(y_train),
        n_candidate_transformations=len(candidates),
        n_active_transformations=int(active_transformations.sum()),
        n_empirical_null_retained_transformations=int(
            transformation_scores["empirical_null_retained"].sum()
        ),
        n_retained_transformations=len(retained_transformations),
        n_components=len(component_names),
        max_curvature_score=float(curvature_scores.max()),
        spec=spec,
    )
    return NonlinearDiscoveryResult(
        transformation_scores=transformation_scores,
        component_transformation_scores=component_scores,
        retained_transformations=retained_transformations,
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
    )
    nonlinear_spec = nonlinear_discovery_spec_from_case_study_config(context.case_study_config)
    nonlinear = discover_manuscript_nonlinear_transformations(
        context.tables["case_study_input_matrix"],
        context.tables["manuscript_feature_catalog"],
        context.tables["fixed_holdout_assignments"],
        conditioning.pca_scores,
        screening.retained_terms,
        nonlinear_spec,
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
        random_seed=seed,
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
    return SparseSelectionStabilityResult(
        support_candidates=support_candidates,
        component_model_selection=component_model_selection,
        component_coefficients=component_coefficients,
        stability_resample_summary=resample_summary,
        stability_feature_summary=stability_feature_summary,
        final_stable_support=final_stable_support,
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
        "sparse_selection_summary": result.summary,
    }
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
    )
    interaction_spec = interaction_discovery_spec_from_case_study_config(context.case_study_config)
    interactions = discover_manuscript_interactions(
        context.tables["case_study_input_matrix"],
        context.tables["manuscript_feature_catalog"],
        context.tables["fixed_holdout_assignments"],
        conditioning.pca_scores,
        screening.retained_terms,
        interaction_spec,
    )
    nonlinear_spec = nonlinear_discovery_spec_from_case_study_config(context.case_study_config)
    nonlinear = discover_manuscript_nonlinear_transformations(
        context.tables["case_study_input_matrix"],
        context.tables["manuscript_feature_catalog"],
        context.tables["fixed_holdout_assignments"],
        conditioning.pca_scores,
        screening.retained_terms,
        nonlinear_spec,
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
    )
    artifact_paths = write_sparse_selection_stability_artifacts(
        sparse_selection,
        context.runtime.output_root,
    )
    return SparseSelectionStabilityStageResult(
        sparse_selection=sparse_selection,
        artifact_paths=artifact_paths,
    )


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


def _ordered_sparse_candidate_names(
    *,
    feature_catalog: pd.DataFrame,
    retained_terms: pd.DataFrame,
    retained_interaction_pairs: pd.DataFrame,
    retained_transformations: pd.DataFrame,
) -> list[str]:
    """Return sparse-selection candidates in feature-catalog order."""
    if "feature_name" not in feature_catalog.columns:
        raise ValueError("feature_catalog must include a feature_name column.")
    retained_names = _retained_feature_names(retained_terms)
    retained_names.update(_retained_pair_names(retained_interaction_pairs))
    retained_names.update(_retained_feature_names(retained_transformations))
    if not retained_names:
        raise ValueError("Sparse selection requires at least one retained upstream term.")

    ordered = [
        str(feature_name)
        for feature_name in feature_catalog["feature_name"].astype(str)
        if str(feature_name) in retained_names
    ]
    missing = sorted(retained_names.difference(ordered))
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"Retained upstream terms are absent from feature_catalog: {preview}")
    if not ordered:
        raise ValueError("Sparse selection found no retained terms in feature_catalog order.")
    return ordered


def _retained_pair_names(retained_interaction_pairs: pd.DataFrame) -> set[str]:
    """Return retained interaction pair names from a retained-pair table."""
    if retained_interaction_pairs.empty:
        return set()
    column = "pair_name" if "pair_name" in retained_interaction_pairs.columns else "feature_name"
    if column not in retained_interaction_pairs.columns:
        raise ValueError("retained_interaction_pairs must include pair_name or feature_name.")
    return set(retained_interaction_pairs[column].astype(str))


def _feature_catalog_subset(
    feature_catalog: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    """Return feature-catalog rows for ``feature_names`` in the requested order."""
    if feature_catalog["feature_name"].duplicated(keep=False).any():
        raise ValueError("feature_catalog contains duplicate feature_name values.")
    indexed = feature_catalog.set_index("feature_name", drop=False)
    rows = [indexed.loc[name] for name in feature_names]
    return pd.DataFrame(rows).reset_index(drop=True)


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
    alpha_max = float(np.max(np.abs(x_scaled[:, active_features].T @ y_scaled)) / n_rows)
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
    for alpha in alphas:
        estimator = Lasso(
            alpha=float(alpha),
            fit_intercept=False,
            max_iter=10000,
            tol=1.0e-6,
            selection="cyclic",
        )
        estimator.fit(x_scaled, y_scaled)
        coefficients = np.asarray(estimator.coef_, dtype=float)
        coefficients[~active_features] = 0.0
        prediction = x_scaled @ coefficients
        rss = float(np.sum((y_scaled - prediction) ** 2))
        support_size = int(np.sum(np.abs(coefficients) > 0.0))
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
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Run deterministic sparse-selection stability resamples."""
    rng = np.random.default_rng(spec.random_seed)
    n_rows = len(x_scaled)
    subsample_size = max(2, int(math.floor(n_rows * spec.subsample_fraction)))
    subsample_size = min(subsample_size, n_rows)
    support_rows = []
    importance_rows = []
    summary_rows = []
    for resample_id in range(1, spec.subsample_count + 1):
        row_indices = np.sort(rng.choice(n_rows, size=subsample_size, replace=False))
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
        support_rows.append(support_mask)
        importance_rows.append(importance)
        summary_rows.append(
            {
                "resample_id": resample_id,
                "subsample_size": int(subsample_size),
                "selected_support_size": int(support_mask.sum()),
                "jaccard_with_full_support": _jaccard_similarity(
                    full_support_mask,
                    support_mask,
                ),
                "spearman_with_full_importance": _spearman_rank_correlation(
                    full_importance,
                    importance,
                ),
            }
        )
    return (
        pd.DataFrame.from_records(summary_rows),
        np.vstack(support_rows),
        np.vstack(importance_rows),
    )


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
    final_support = full_support_mask & (selection_frequency >= spec.jaccard_threshold)
    final_support = final_support & passes_global_stability
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
) -> np.ndarray:
    """Compute featurewise coefficient-row-norm statistics under response permutations."""
    rng = np.random.default_rng(random_seed)
    null_statistics = np.zeros((n_permutations, x_scaled.shape[1]), dtype=float)
    n_rows = float(len(x_scaled))
    for index in range(n_permutations):
        permuted = y_scaled[rng.permutation(len(y_scaled)), :]
        coefficients = (x_scaled.T @ permuted) / n_rows
        null_statistics[index, :] = np.linalg.norm(coefficients, axis=1)
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
                "n_retained_terms": int(n_retained_terms),
                "min_empirical_p_value": float(min_p_value),
                "manuscript_retained_terms_reference": int(spec.retained_terms_reference),
                "random_seed": int(spec.random_seed),
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
