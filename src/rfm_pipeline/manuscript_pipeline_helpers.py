"""Shared helpers for manuscript-pipeline stage loading and config translation.

This module hosts the read/load/translate helpers that were previously defined
inside ``tools/run_manuscript_pipeline.py`` and re-imported by
``rfm_pipeline.hpc_reduce``. The historic location was outside the installable
package, so the imports broke in install mode (e.g. when ``rfm-pipeline`` was
consumed as a git-pinned dependency by ``bsm-public-rf``). Hosting the helpers
under ``src/rfm_pipeline/`` makes them part of the installed wheel and removes
the need for the ``tools`` namespace at import time.

The legacy module ``tools/run_manuscript_pipeline.py`` re-exports the public
symbols here for source-tree back-compat.
"""

from __future__ import annotations

import copy
import json
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rfm_pipeline.config import OutOfCoreConfig, WorkflowConfig
from rfm_pipeline.data import SealedSplitResult, SealedTestAccessError
from rfm_pipeline.manuscript_runtime import load_manuscript_case_study_config
from rfm_pipeline.manuscript_stages import (
    EmpiricalNullScreeningResult,
    InteractionDiscoveryResult,
    NonlinearDiscoveryResult,
    OutputConditioningResult,
    SparseSelectionStabilityResult,
)
from rfm_pipeline.out_of_core import (
    ChunkedParquetReader,
    SpillToDiskBuffer,
    choose_temp_dir,
)

__all__ = [
    "resolve_repo_root",
    "_read_csv",
    "_read_json",
    "_effective_out_of_core",
    "_read_parquet_with_mode",
    "_resolve_data_root",
    "_load_tables",
    "_load_output_conditioning_result",
    "_load_empirical_null_screening_result",
    "_load_interaction_discovery_result",
    "_load_nonlinear_discovery_result",
    "_load_sparse_selection_result",
    "config_to_legacy_case_study",
    "select_threshold_on_internal_validation",
]


def resolve_repo_root() -> Path:
    """Best-effort source-tree root resolution.

    Used for fallback data/artifact paths when the workflow config does not pin
    ``dataset.path`` explicitly. The reduce CLI templates always ``cd`` to the
    rfm-pipeline source tree before invocation, so ``Path.cwd()`` reliably
    points at the repo root in production use. Callers that need a hard
    guarantee should set ``WorkflowConfig.dataset.path`` directly.
    """
    return Path(os.environ.get("RFM_PIPELINE_REPO_ROOT", Path.cwd())).resolve()


def _read_json(path: Path) -> dict[str, Any] | None:
    """Return parsed JSON or ``None`` if the file is missing.

    Raises :class:`json.JSONDecodeError` (or :class:`OSError`) for malformed
    or unreadable files so silent data corruption is not masked.
    """
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path):
    import pandas as pd

    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    return pd.read_csv(path)


def _effective_out_of_core(config: WorkflowConfig) -> OutOfCoreConfig:
    """Resolve out-of-core settings, including legacy runtime keys."""
    resolved = config.runtime.out_of_core
    legacy_map = config.runtime.chunked_io_config or {}
    if config.runtime.use_chunked_io and not resolved.enabled:
        resolved.enabled = True
    if legacy_map:
        resolved.chunk_size_mb = int(legacy_map.get("chunk_size_mb", resolved.chunk_size_mb))
        resolved.max_memory_budget_mb = int(
            legacy_map.get("max_memory_budget_mb", resolved.max_memory_budget_mb)
        )
        resolved.temp_dir = legacy_map.get("temp_dir", resolved.temp_dir)
        resolved.enable_spill_to_disk = bool(
            legacy_map.get("enable_spill_to_disk", resolved.enable_spill_to_disk)
        )
    return resolved


def _read_parquet_with_mode(
    path: Path,
    *,
    out_of_core: OutOfCoreConfig,
    columns: list[str] | None = None,
):
    import pandas as pd

    if not out_of_core.enabled:
        return pd.read_parquet(path, columns=columns)

    reader = ChunkedParquetReader(
        str(path),
        chunk_size_mb=max(1, int(out_of_core.chunk_size_mb)),
        columns=columns,
    )
    if out_of_core.enable_spill_to_disk:
        temp_root = choose_temp_dir(preferred_root=out_of_core.temp_dir)
        buffer = SpillToDiskBuffer(
            temp_dir=temp_root,
            max_memory_mb=max(64, int(out_of_core.max_memory_budget_mb)),
        )
        try:
            for chunk in reader:
                buffer.add_chunk(chunk)
            return buffer.get_final_dataframe()
        finally:
            buffer.cleanup()

    chunks = list(reader)
    if not chunks:
        return pd.read_parquet(path, columns=columns).iloc[0:0]
    return pd.concat(chunks, ignore_index=True)


def _resolve_data_root(config: WorkflowConfig) -> Path:
    """Resolve dataset root from config.dataset.{path,type}."""
    repo_root = resolve_repo_root()
    if config.dataset.path:
        candidate = Path(config.dataset.path)
        if not candidate.is_absolute():
            candidate = repo_root / candidate
        return candidate

    dataset_roots = {
        "synthetic_300_sample": repo_root / "artifacts" / "test_dataset_300",
        "synthetic_full": repo_root / "artifacts" / "test_dataset_3k",
        "synthetic_controlled_dgp": None,
    }
    try:
        root = dataset_roots[config.dataset.type]
        if root is None:
            raise ValueError(
                f"dataset.type={config.dataset.type!r} requires dataset.path to be set explicitly."
            )
        return root
    except KeyError as exc:
        raise ValueError(
            "Unsupported dataset.type for unified runner. "
            f"dataset.type={config.dataset.type!r}. "
            "Set dataset.path explicitly or use one of: "
            f"{sorted(dataset_roots.keys())}"
        ) from exc


def _load_tables(output_column_limit: int | None, config: WorkflowConfig) -> dict[str, Any]:
    """Load validation tables and normalize holdout labels."""
    repo_root = resolve_repo_root()
    data_root = _resolve_data_root(config)
    out_of_core = _effective_out_of_core(config)
    if not data_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {data_root}")
    y = _read_parquet_with_mode(data_root / "Y.parquet", out_of_core=out_of_core)
    y_cols = [c for c in y.columns if c != "sample_id"]
    if output_column_limit is not None and len(y_cols) > output_column_limit:
        y = y[["sample_id", *y_cols[:output_column_limit]]]
    holdout = _read_parquet_with_mode(
        data_root / "holdout_assignments.parquet",
        out_of_core=out_of_core,
    ).copy()
    holdout["split"] = (
        holdout["split"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"test": "holdout", "val": "holdout", "validation": "holdout"})
    )
    _catalog_candidates = [
        data_root / "actual_input_feature_catalog.parquet",
        repo_root / "artifacts" / "actual_input_feature_catalog.parquet",
    ]
    _catalog_path = next((p for p in _catalog_candidates if p.exists()), None)
    if _catalog_path is None:
        raise FileNotFoundError(
            "actual_input_feature_catalog.parquet not found. "
            f"Copy it to {data_root}/ or {repo_root / 'artifacts'}/"
        )
    return {
        "case_study_input_matrix": _read_parquet_with_mode(
            data_root / "X.parquet",
            out_of_core=out_of_core,
        ),
        "case_study_output_matrix": y,
        "fixed_holdout_assignments": holdout,
        "manuscript_feature_catalog": _read_parquet_with_mode(
            _catalog_path,
            out_of_core=out_of_core,
        ),
    }


def _load_output_conditioning_result(output_root: Path) -> OutputConditioningResult:
    stage_root = output_root / "output_conditioning"
    diagnostics = _read_csv(stage_root / "output_filter_diagnostics.csv")
    pca_scores = _read_csv(stage_root / "pca_scores.csv")
    pca_loadings = _read_csv(stage_root / "pca_loadings.csv")
    pca_explained_variance = _read_csv(stage_root / "pca_explained_variance.csv")
    summary = _read_csv(stage_root / "output_conditioning_summary.csv")
    if "retained" not in diagnostics.columns or "output_name" not in diagnostics.columns:
        raise ValueError(
            "output_filter_diagnostics.csv must include output_name and retained columns."
        )
    retained = tuple(diagnostics.loc[diagnostics["retained"], "output_name"].astype(str))
    culled = tuple(diagnostics.loc[~diagnostics["retained"], "output_name"].astype(str))
    return OutputConditioningResult(
        retained_output_names=retained,
        culled_output_names=culled,
        output_filter_diagnostics=diagnostics,
        pca_scores=pca_scores,
        pca_loadings=pca_loadings,
        pca_explained_variance=pca_explained_variance,
        summary=summary,
    )


def _load_empirical_null_screening_result(output_root: Path) -> EmpiricalNullScreeningResult:
    stage_root = output_root / "empirical_null_screen"
    return EmpiricalNullScreeningResult(
        feature_screening_statistics=_read_csv(stage_root / "feature_screening_statistics.csv"),
        component_coefficients=_read_csv(stage_root / "component_coefficients.csv"),
        permutation_null_summary=_read_csv(stage_root / "permutation_null_summary.csv"),
        retained_terms=_read_csv(stage_root / "retained_terms.csv"),
        provenance=_read_csv(stage_root / "empirical_null_provenance.csv"),
        summary=_read_csv(stage_root / "empirical_null_screen_summary.csv"),
    )


def _load_interaction_discovery_result(output_root: Path) -> InteractionDiscoveryResult:
    import pandas as pd

    stage_root = output_root / "interaction_discovery"
    canonical_pair_scores = stage_root / "interaction_pair_scores.csv"
    if canonical_pair_scores.exists():
        return InteractionDiscoveryResult(
            pair_scores=_read_csv(canonical_pair_scores),
            component_interaction_scores=_read_csv(stage_root / "component_interaction_scores.csv"),
            interaction_null_summary=_read_csv(stage_root / "interaction_null_summary.csv"),
            retained_pairs=_read_csv(stage_root / "retained_interaction_pairs.csv"),
            provenance=_read_csv(stage_root / "interaction_discovery_provenance.csv"),
            summary=_read_csv(stage_root / "interaction_discovery_summary.csv"),
        )

    merged_root = output_root / "hpc_shards_interaction_discovery" / "_merged"
    merged_pair_scores_path = merged_root / "interaction_pair_scores_merged.csv"
    merged_retained_pairs_path = merged_root / "retained_interaction_pairs_merged.csv"
    if not merged_pair_scores_path.exists() or not merged_retained_pairs_path.exists():
        raise FileNotFoundError(f"Missing required artifact: {canonical_pair_scores}")

    pair_scores = _read_csv(merged_pair_scores_path)
    retained_pairs = _read_csv(merged_retained_pairs_path)
    merged_summary = _read_json(merged_root / "interaction_discovery_merged.json") or {}
    summary_row: dict[str, Any] = {
        "stage": "interaction_discovery",
        "n_candidate_pairs": int(len(pair_scores)),
        "n_retained_pairs": int(len(retained_pairs)),
        "artifact_source": "distributed_merged_fallback",
    }
    if isinstance(merged_summary, dict):
        if "n_shards" in merged_summary:
            summary_row["n_shards"] = int(merged_summary["n_shards"])
        if "n_merged_pair_scores" in merged_summary:
            summary_row["n_merged_pair_scores"] = int(merged_summary["n_merged_pair_scores"])
        if "n_merged_retained_pairs" in merged_summary:
            summary_row["n_merged_retained_pairs"] = int(merged_summary["n_merged_retained_pairs"])

    return InteractionDiscoveryResult(
        pair_scores=pair_scores,
        component_interaction_scores=pd.DataFrame(
            columns=[
                "pair_name",
                "component",
                "standardized_residual_interaction_coefficient",
            ]
        ),
        interaction_null_summary=pd.DataFrame(
            columns=[
                "pair_name",
                "null_mean_score",
                "null_quantile_95",
                "null_quantile_99",
                "null_max_score",
            ]
        ),
        retained_pairs=retained_pairs,
        provenance=pd.DataFrame(
            [
                {
                    "stage": "interaction_discovery",
                    "public_implementation_status": "distributed_merged_fallback",
                }
            ]
        ),
        summary=pd.DataFrame([summary_row]),
    )


def _load_nonlinear_discovery_result(output_root: Path) -> NonlinearDiscoveryResult:
    stage_root = output_root / "nonlinear_discovery"
    return NonlinearDiscoveryResult(
        transformation_scores=_read_csv(stage_root / "transformation_scores.csv"),
        component_transformation_scores=_read_csv(
            stage_root / "component_transformation_scores.csv"
        ),
        retained_transformations=_read_csv(stage_root / "retained_transformations.csv"),
        provenance=_read_csv(stage_root / "nonlinear_discovery_provenance.csv"),
        summary=_read_csv(stage_root / "nonlinear_discovery_summary.csv"),
    )


def _load_sparse_selection_result(output_root: Path) -> SparseSelectionStabilityResult:
    stage_root = output_root / "sparse_selection"
    return SparseSelectionStabilityResult(
        support_candidates=_read_csv(stage_root / "support_candidates.csv"),
        component_model_selection=_read_csv(stage_root / "component_model_selection.csv"),
        component_coefficients=_read_csv(stage_root / "component_coefficients.csv"),
        stability_resample_summary=_read_csv(stage_root / "stability_resample_summary.csv"),
        stability_feature_summary=_read_csv(stage_root / "stability_feature_summary.csv"),
        final_stable_support=_read_csv(stage_root / "final_stable_support.csv"),
        provenance=_read_csv(stage_root / "sparse_selection_provenance.csv"),
        summary=_read_csv(stage_root / "sparse_selection_summary.csv"),
    )


def config_to_legacy_case_study(workflow_config: WorkflowConfig) -> dict[str, Any]:
    """Convert WorkflowConfig to legacy case_study_config format."""
    repo_root = resolve_repo_root()
    base_config = load_manuscript_case_study_config(repo_root)
    config = copy.deepcopy(base_config)

    case_study = config.setdefault("case_study", {})
    output_conditioning = case_study.setdefault("output_conditioning", {})
    empirical_null_screen = case_study.setdefault("empirical_null_screen", {})
    interaction_discovery = case_study.setdefault("interaction_discovery", {})
    nonlinear_discovery = case_study.setdefault("nonlinear_discovery", {})
    sparse_selection = case_study.setdefault("sparse_selection", {})
    stability = case_study.setdefault("stability", {})
    final_model = case_study.setdefault("final_model", {})
    runtime = case_study.setdefault("runtime", {})

    temporary_reduction = output_conditioning.setdefault("temporary_reduction", {})
    temporary_reduction["retained_variance_fraction"] = workflow_config.algorithm.variance_threshold
    if workflow_config.algorithm.retained_components is not None:
        temporary_reduction["retained_components"] = workflow_config.algorithm.retained_components

    scr = workflow_config.stages.empirical_null_screening
    empirical_null_screen["permutation_count_B"] = scr.n_permutations - 1
    empirical_null_screen["bh_q_screen"] = scr.bh_q_threshold
    if scr.max_retained_terms is not None:
        empirical_null_screen["max_retained_terms"] = int(scr.max_retained_terms)

    itr = workflow_config.stages.interaction_discovery
    interaction_discovery["p_threshold"] = itr.p_threshold
    interaction_discovery["selection_method"] = str(itr.selection_method)
    interaction_discovery["selection_alpha"] = float(itr.selection_alpha)
    interaction_discovery["minimum_selection_draws"] = int(itr.minimum_selection_draws)
    if itr.n_permutations is not None:
        interaction_discovery["permutation_count_B"] = itr.n_permutations - 1
    interaction_discovery["n_tree_estimators"] = itr.n_tree_estimators
    interaction_discovery["max_tree_depth"] = itr.max_tree_depth
    interaction_discovery["max_shap_samples"] = itr.max_shap_samples
    interaction_discovery["min_component_variance_fraction"] = itr.min_component_variance_fraction
    if itr.max_active_components is not None:
        interaction_discovery["max_active_components"] = int(itr.max_active_components)
    if itr.parallel_batch_timeout_seconds is not None:
        interaction_discovery["parallel_batch_timeout_seconds"] = int(
            itr.parallel_batch_timeout_seconds
        )
    parallelism = workflow_config.runtime.parallelism
    if parallelism.enabled:
        backend_map = {
            "joblib": "threading",
            "dask": "dask",
            "ray": "threading",
            "mpi": "threading",
        }
        interaction_discovery["parallel_backend"] = backend_map.get(
            parallelism.backend,
            "threading",
        )
        if parallelism.dask_workers is not None:
            interaction_discovery["dask_workers"] = int(parallelism.dask_workers)
        interaction_discovery["dask_cores_per_worker"] = int(parallelism.dask_cores_per_worker)
        interaction_discovery["dask_memory_per_worker"] = str(parallelism.dask_memory_per_worker)

    nlr = workflow_config.stages.nonlinear_discovery
    nonlinear_discovery["edf_threshold"] = nlr.edf_threshold

    spr = workflow_config.stages.sparse_selection
    stability["resampling_scheme"] = (
        f"{spr.n_stability_subsamples}_subsamples_of_"
        f"{int(spr.subsample_fraction * 100)}_percent_rows_without_replacement_seed_123"
    )
    if spr.max_candidate_terms is not None:
        sparse_selection["max_candidate_terms"] = int(spr.max_candidate_terms)

    fnl = workflow_config.stages.final_artifacts
    final_model["bootstrap_count"] = fnl.bootstrap_count
    final_model["bootstrap_alpha"] = fnl.bootstrap_alpha
    final_inferential_filter = case_study.setdefault("final_inferential_filter", {})
    final_inferential_filter["output_subset_mode"] = fnl.hc3_output_subset_mode
    if fnl.hc3_output_fraction is not None:
        final_inferential_filter["output_fraction"] = fnl.hc3_output_fraction
    if fnl.hc3_output_names:
        final_inferential_filter["output_names"] = list(fnl.hc3_output_names)
    if fnl.hc3_output_max_outputs is not None:
        final_inferential_filter["max_outputs"] = int(fnl.hc3_output_max_outputs)
    final_inferential_filter["random_seed"] = int(fnl.hc3_output_random_seed)
    final_inferential_filter["subset_metric"] = str(fnl.hc3_output_subset_metric)
    feature_pruning = final_inferential_filter.setdefault("feature_pruning", {})
    feature_pruning["error_scale_quantile"] = float(fnl.pruning_error_scale_quantile)
    if fnl.pruning_remove_count_override is not None:
        feature_pruning.pop("delta_threshold_override", None)
        feature_pruning["remove_count_override"] = int(fnl.pruning_remove_count_override)
    elif fnl.delta_threshold_override is not None:
        feature_pruning["delta_threshold_override"] = float(fnl.delta_threshold_override)

    runtime["n_jobs"] = workflow_config.runtime.n_jobs
    config["random_seed"] = workflow_config.output.seed
    return config


# ---------------------------------------------------------------------------
# Internal-validation threshold selection (P0-S05)
# ---------------------------------------------------------------------------


def select_threshold_on_internal_validation(
    split: SealedSplitResult,
    thresholds: Sequence[float],
    scorer: Callable[[float, pd.DataFrame, pd.DataFrame], float],
    *,
    seed: int | None = None,
) -> float:
    """Select the best threshold using only internal-validation data.

    The function reads ``split.train`` and ``split.val`` exclusively.  If the
    split has already been unsealed (i.e. the sealed test partition is
    accessible), a :class:`~rfm_pipeline.data.SealedTestAccessError` is raised
    immediately to prevent test-data leakage into the selection process.

    Parameters
    ----------
    split:
        Three-way sealed split.  Must still be sealed when this function is
        called.
    thresholds:
        Candidate threshold values to evaluate.
    scorer:
        Callable ``scorer(threshold, train_df, val_df) -> float``.  Lower
        return values indicate better performance (e.g. nRMSE).  Called with
        the training and internal-validation partitions only; the scorer must
        not access the sealed test partition.
    seed:
        Integer seed used for deterministic tie-breaking when multiple
        thresholds achieve the same best score.

    Returns
    -------
    float
        The threshold from *thresholds* that minimises the internal-validation
        score.  Ties are broken deterministically using *seed*.

    Raises
    ------
    SealedTestAccessError
        If *split* has already been unsealed, indicating that the test
        partition is exposed and could contaminate threshold selection.
    ValueError
        If *thresholds* is empty.
    """
    if not split._sealed:
        raise SealedTestAccessError(
            "select_threshold_on_internal_validation: the supplied split has "
            "already been unsealed.  Threshold selection must use only "
            "train + internal-validation data; unsealing before selection "
            "risks test-data leakage."
        )
    if not thresholds:
        raise ValueError("thresholds must be a non-empty sequence.")

    train_df = split.train
    val_df = split.val

    scores = [scorer(float(t), train_df, val_df) for t in thresholds]
    best_score = min(scores)

    # Collect all thresholds that achieve the best score.
    candidates = [float(t) for t, s in zip(thresholds, scores, strict=True) if s == best_score]

    if len(candidates) == 1:
        return candidates[0]

    # Deterministic tie-break using seed.
    rng = np.random.default_rng(seed)
    idx = int(rng.integers(len(candidates)))
    return candidates[idx]
