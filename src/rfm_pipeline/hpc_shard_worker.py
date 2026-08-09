r"""HPC shard worker — runs inside each SLURM array task.

Selects the shard at TASK_ID from the manifest, executes the designated
pipeline stage over its assigned work-item range, validates outputs, then
promotes them with an atomic _SUCCESS.json marker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from rfm_pipeline.config import apply_fast_mode_overrides, load_config
from rfm_pipeline.distributed.stage_loaders import (
    _load_empirical_null_screening_result,
    _load_interaction_discovery_result,
    _load_nonlinear_discovery_result,
    _load_output_conditioning_result,
    _load_sparse_selection_result,
    _load_tables,
    config_to_legacy_case_study,
)
from rfm_pipeline.manuscript_runtime import load_manuscript_case_study_config
from rfm_pipeline.manuscript_stages import (
    _generate_supported_nonlinear_candidates,
    condition_manuscript_outputs,
    discover_interaction_scores_only,
    discover_manuscript_nonlinear_transformations,
    empirical_null_screening_spec_from_case_study_config,
    final_manuscript_artifacts_spec_from_case_study_config,
    nonlinear_discovery_spec_from_case_study_config,
    output_conditioning_spec_from_case_study_config,
    regenerate_final_manuscript_artifacts,
    screen_manuscript_empirical_null_terms,
    select_manuscript_sparse_support,
    sparse_selection_stability_spec_from_case_study_config,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("rfm_pipeline.hpc_shard_worker")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BSM HPC shard worker")
    p.add_argument("--manifest", required=True, help="Path to JSONL manifest file")
    p.add_argument("--task-id", type=int, required=True, help="SLURM_ARRAY_TASK_ID (0-indexed)")
    p.add_argument("--output-root", required=True, help="Root dir for shard outputs")
    p.add_argument("--work-dir", required=True, help="Scratch working directory for this task")
    p.add_argument("--stage", required=True, help="Pipeline stage name")
    p.add_argument("--config", default=None, help="Optional workflow config YAML override")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and exit without computing",
    )
    return p.parse_args()


def main() -> None:
    """Entry point for the rfm-hpc-worker console script.

    Runs inside each SLURM array task. Selects the shard at TASK_ID from
    the manifest, executes the designated pipeline stage over its assigned
    work-item range, validates outputs, then promotes them with an atomic
    _SUCCESS.json marker.
    """
    args = _parse_args()

    from rfm_pipeline.distributed.manifest import load_manifest

    shards = load_manifest(args.manifest)
    if args.task_id >= len(shards):
        logger.error("task_id=%d out of range (manifest has %d shards)", args.task_id, len(shards))
        sys.exit(1)

    shard = shards[args.task_id]
    logger.info(
        "shard=%s stage=%s feature_range=%s-%s",
        shard.shard_id,
        shard.stage,
        shard.feature_start_idx,
        shard.feature_end_idx,
    )

    resolved_config = _resolve_config_path(shard, args.config)
    try:
        run_shard(
            shard=shard,
            output_root=args.output_root,
            config_path=resolved_config,
            dry_run=args.dry_run,
            manifest_path=args.manifest,
        )
    except Exception as e:
        logger.error("shard=%s FAILED: %s", shard.shard_id, e)
        sys.exit(1)


def run_shard(
    *,
    shard,
    output_root: str,
    config_path: str | None = None,
    dry_run: bool = False,
    manifest_path: str | None = None,
) -> None:
    """Run one shard with checkpointing, usable from SLURM array or MPI runner.

    Status tracking is done exclusively via per-shard checkpoint marker files
    (_RUNNING.json, _SUCCESS.json, _FAILED.json) written by CheckpointManager.
    The shared JSONL manifest is treated as read-only after generation.
    Writing back to the manifest caused Lustre/NFS-unsafe concurrent writes that
    corrupted manifest entries when many array tasks ran simultaneously.
    ``manifest_path`` is accepted but ignored to preserve call-site compatibility.
    """
    from rfm_pipeline.distributed.checkpoint import CheckpointManager

    cm = CheckpointManager(output_root, shard.shard_id)
    if cm.is_complete():
        logger.info("shard=%s already complete (_SUCCESS.json found), skipping", shard.shard_id)
        return
    if dry_run:
        logger.info("[dry-run] shard=%s validated — would run %s", shard.shard_id, shard.stage)
        return

    cm.mark_running()
    start = time.monotonic()
    try:
        args = SimpleNamespace(config=config_path)
        _run_shard_stage(shard, cm, args)
        elapsed = time.monotonic() - start
        logger.info("shard=%s COMPLETE in %.1fs", shard.shard_id, elapsed)
    except Exception as exc:
        elapsed = time.monotonic() - start
        cm.mark_failed(str(exc))
        logger.error("shard=%s FAILED after %.1fs: %s", shard.shard_id, elapsed, exc)
        raise


def _run_shard_stage(shard, cm, args) -> None:
    """Dispatch to the appropriate stage runner."""
    stage = shard.stage
    logger.info("[shard] running stage=%s for shard=%s", stage, shard.shard_id)
    cm.staging_dir.mkdir(parents=True, exist_ok=True)

    if stage == "interaction_discovery":
        _run_interaction_shard(shard, cm, config_path=args.config)
        expected_files = [
            "shard_result.json",
            "interaction_shard_scores.npz",
            "interaction_pair_names.csv",
        ]
    elif stage == "output_conditioning":
        _run_output_conditioning_shard(shard, cm, config_path=_require_config(stage, args.config))
        expected_files = ["shard_result.json"]
    elif stage == "empirical_null_screening":
        _run_empirical_null_screening_shard(
            shard,
            cm,
            config_path=_require_config(stage, args.config),
        )
        expected_files = ["shard_result.json"]
    elif stage == "nonlinear_discovery":
        _run_nonlinear_discovery_shard(shard, cm, config_path=_require_config(stage, args.config))
        expected_files = ["shard_result.json"]
    elif stage == "sparse_selection":
        _run_sparse_selection_shard(shard, cm, config_path=_require_config(stage, args.config))
        expected_files = ["shard_result.json"]
    elif stage == "final_manuscript_artifacts":
        _run_final_manuscript_artifacts_shard(
            shard,
            cm,
            config_path=_require_config(stage, args.config),
        )
        expected_files = ["shard_result.json"]
    else:
        raise ValueError(f"Unsupported shard stage: {stage}")

    cm.validate_and_promote(
        expected_files=expected_files,
        metadata={"stage": stage, "shard_id": shard.shard_id},
    )


def _resolve_config_path(shard, explicit_config: str | None) -> str | None:
    if explicit_config:
        return explicit_config
    for raw_path in getattr(shard, "input_paths", []):
        path = Path(raw_path)
        if path.suffix.lower() in {".yml", ".yaml"} and path.exists():
            return str(path)
    return None


def _require_config(stage: str, config_path: str | None) -> str:
    if config_path:
        return config_path
    raise ValueError(
        f"Stage '{stage}' requires --config (or YAML config path in manifest input_paths)."
    )


def _artifact_root_from_output_root(output_root: Path) -> Path:
    return output_root.resolve().parent


def _load_workflow_tables_and_case_config(
    config_path: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    workflow = apply_fast_mode_overrides(load_config(config_path))
    output_cap = (
        workflow.validation.fast_mode_overrides.output_cap
        if workflow.validation.fast_mode
        else None
    )
    tables = _load_tables(output_column_limit=output_cap, config=workflow)
    case_study_config = config_to_legacy_case_study(workflow)
    return tables, case_study_config


def _shard_index_subset(
    shard,
    total_items: int,
    *,
    one_based: bool = False,
) -> set[int]:
    if total_items <= 0:
        raise ValueError("total_items must be positive for shard partitioning.")
    start = int(shard.feature_start_idx or 0)
    end = int(shard.feature_end_idx or total_items)
    start = max(0, min(start, total_items))
    end = max(start, min(end, total_items))
    if start == end:
        raise ValueError(
            f"Shard {shard.shard_id} has empty work-item range for total_items={total_items}."
        )
    if one_based:
        return {idx + 1 for idx in range(start, end)}
    return set(range(start, end))


def _write_checkpoint_warmup_result(
    *,
    shard,
    cm,
    stage: str,
    config_path: str,
    work_item_start: int,
    work_item_end: int,
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "shard_id": shard.shard_id,
        "stage": stage,
        "status": "checkpoint_warmup_complete",
        "config_path": config_path,
        "work_item_start": int(work_item_start),
        "work_item_end": int(work_item_end),
    }
    if extra:
        payload.update(extra)
    (cm.staging_dir / "shard_result.json").write_text(json.dumps(payload, indent=2))


def _nonlinear_base_feature_names(
    retained_terms: pd.DataFrame,
    input_matrix: pd.DataFrame,
    *,
    transform_library=None,
) -> tuple[str, ...]:
    """Return the base-feature order used by nonlinear discovery shards."""
    retained_features = retained_terms["feature_name"].astype(str).tolist()
    candidates = _generate_supported_nonlinear_candidates(
        retained_features,
        input_matrix,
        transform_library=transform_library,
    )
    return tuple(dict.fromkeys(base_feature for _, base_feature, _ in candidates))


def _run_output_conditioning_shard(shard, cm, *, config_path: str) -> None:
    tables, case_study_config = _load_workflow_tables_and_case_config(config_path)
    spec = output_conditioning_spec_from_case_study_config(case_study_config)
    conditioning = condition_manuscript_outputs(
        tables["case_study_output_matrix"],
        tables["fixed_holdout_assignments"],
        spec,
    )
    _write_checkpoint_warmup_result(
        shard=shard,
        cm=cm,
        stage="output_conditioning",
        config_path=config_path,
        work_item_start=int(shard.feature_start_idx or 0),
        work_item_end=int(shard.feature_end_idx or 1),
        extra={
            "n_retained_outputs": int(len(conditioning.retained_output_names)),
            "n_retained_components": int(len(conditioning.pca_explained_variance)),
        },
    )


def _run_empirical_null_screening_shard(shard, cm, *, config_path: str) -> None:
    tables, case_study_config = _load_workflow_tables_and_case_config(config_path)
    artifact_root = _artifact_root_from_output_root(cm.output_root)
    conditioning = _load_output_conditioning_result(artifact_root)
    spec = empirical_null_screening_spec_from_case_study_config(case_study_config)
    active_permutation_indices = _shard_index_subset(shard, spec.permutation_count_B)
    screening = screen_manuscript_empirical_null_terms(
        tables["case_study_input_matrix"],
        tables["manuscript_feature_catalog"],
        tables["fixed_holdout_assignments"],
        conditioning.pca_scores,
        spec,
        checkpoint_dir=artifact_root / "empirical_null_screen",
        active_permutation_indices=active_permutation_indices,
    )
    _write_checkpoint_warmup_result(
        shard=shard,
        cm=cm,
        stage="empirical_null_screening",
        config_path=config_path,
        work_item_start=min(active_permutation_indices),
        work_item_end=max(active_permutation_indices) + 1,
        extra={
            "n_target_permutations": int(len(active_permutation_indices)),
            "n_retained_terms_preview": int(len(screening.retained_terms)),
        },
    )


def _run_nonlinear_discovery_shard(shard, cm, *, config_path: str) -> None:
    tables, case_study_config = _load_workflow_tables_and_case_config(config_path)
    artifact_root = _artifact_root_from_output_root(cm.output_root)
    conditioning = _load_output_conditioning_result(artifact_root)
    screening = _load_empirical_null_screening_result(artifact_root)
    spec = nonlinear_discovery_spec_from_case_study_config(case_study_config)
    base_features = _nonlinear_base_feature_names(
        screening.retained_terms,
        tables["case_study_input_matrix"],
        transform_library=spec.transform_library,
    )
    total_features = len(base_features)
    shard_start = int(shard.feature_start_idx or 0)
    if shard_start >= total_features:
        _write_checkpoint_warmup_result(
            shard=shard,
            cm=cm,
            stage="nonlinear_discovery",
            config_path=config_path,
            work_item_start=total_features,
            work_item_end=total_features,
            extra={"n_target_base_features": 0, "n_retained_transformations_preview": 0},
        )
        return
    active_feature_indices = _shard_index_subset(shard, max(1, total_features))
    nonlinear = discover_manuscript_nonlinear_transformations(
        tables["case_study_input_matrix"],
        tables["manuscript_feature_catalog"],
        tables["fixed_holdout_assignments"],
        conditioning.pca_scores,
        screening.retained_terms,
        spec,
        checkpoint_dir=artifact_root / "nonlinear_discovery",
        active_feature_indices=active_feature_indices,
    )
    _write_checkpoint_warmup_result(
        shard=shard,
        cm=cm,
        stage="nonlinear_discovery",
        config_path=config_path,
        work_item_start=min(active_feature_indices),
        work_item_end=max(active_feature_indices) + 1,
        extra={
            "n_target_base_features": int(len(active_feature_indices)),
            "n_retained_transformations_preview": int(len(nonlinear.retained_transformations)),
        },
    )


def _run_sparse_selection_shard(shard, cm, *, config_path: str) -> None:
    tables, case_study_config = _load_workflow_tables_and_case_config(config_path)
    artifact_root = _artifact_root_from_output_root(cm.output_root)
    conditioning = _load_output_conditioning_result(artifact_root)
    screening = _load_empirical_null_screening_result(artifact_root)
    interactions = _load_interaction_discovery_result(artifact_root)
    nonlinear = _load_nonlinear_discovery_result(artifact_root)
    spec = sparse_selection_stability_spec_from_case_study_config(case_study_config)
    active_resample_indices = _shard_index_subset(shard, spec.subsample_count, one_based=True)
    sparse_selection = select_manuscript_sparse_support(
        tables["case_study_input_matrix"],
        tables["manuscript_feature_catalog"],
        tables["fixed_holdout_assignments"],
        conditioning.pca_scores,
        screening.retained_terms,
        interactions.retained_pairs,
        nonlinear.retained_transformations,
        spec,
        checkpoint_dir=artifact_root / "sparse_selection",
        active_resample_indices=active_resample_indices,
    )
    _write_checkpoint_warmup_result(
        shard=shard,
        cm=cm,
        stage="sparse_selection",
        config_path=config_path,
        work_item_start=min(active_resample_indices),
        work_item_end=max(active_resample_indices) + 1,
        extra={
            "n_target_resamples": int(len(active_resample_indices)),
            "n_final_support_preview": int(len(sparse_selection.final_stable_support)),
        },
    )


def _run_final_manuscript_artifacts_shard(shard, cm, *, config_path: str) -> None:
    tables, case_study_config = _load_workflow_tables_and_case_config(config_path)
    artifact_root = _artifact_root_from_output_root(cm.output_root)
    conditioning = _load_output_conditioning_result(artifact_root)
    screening = _load_empirical_null_screening_result(artifact_root)
    interactions = _load_interaction_discovery_result(artifact_root)
    nonlinear = _load_nonlinear_discovery_result(artifact_root)
    sparse_selection = _load_sparse_selection_result(artifact_root)
    spec = final_manuscript_artifacts_spec_from_case_study_config(case_study_config)
    active_bootstrap_indices = _shard_index_subset(shard, spec.bootstrap_count)
    final_artifacts = regenerate_final_manuscript_artifacts(
        tables["case_study_input_matrix"],
        tables["case_study_output_matrix"],
        tables["manuscript_feature_catalog"],
        tables["fixed_holdout_assignments"],
        conditioning,
        screening,
        interactions,
        nonlinear,
        sparse_selection,
        spec,
        checkpoint_dir=artifact_root / "final_manuscript_artifacts",
        active_bootstrap_indices=active_bootstrap_indices,
    )
    _write_checkpoint_warmup_result(
        shard=shard,
        cm=cm,
        stage="final_manuscript_artifacts",
        config_path=config_path,
        work_item_start=min(active_bootstrap_indices),
        work_item_end=max(active_bootstrap_indices) + 1,
        extra={
            "n_target_bootstraps": int(len(active_bootstrap_indices)),
            "n_final_support_preview": int(len(final_artifacts.final_support_features)),
        },
    )


def _resolve_interaction_inputs(input_paths: list[str]) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    for raw_path in input_paths:
        path = Path(raw_path)
        name = path.name
        if name == "X.parquet":
            resolved["x"] = path
        elif name == "holdout_assignments.parquet":
            resolved["holdout"] = path
        elif name in {"actual_input_feature_catalog.parquet", "manuscript_feature_catalog.parquet"}:
            resolved["catalog"] = path
        elif name == "pca_scores.csv":
            resolved["pca_scores"] = path
        elif name == "retained_terms.csv":
            resolved["retained_terms"] = path
    required = {"x", "holdout", "catalog", "pca_scores", "retained_terms"}
    missing = sorted(required - set(resolved))
    if missing:
        raise ValueError(
            "interaction_discovery shard missing required inputs: "
            f"{missing}. input_paths={input_paths}"
        )
    return resolved


def _load_interaction_spec(config_path: str | None):
    from rfm_pipeline.manuscript_stages import interaction_discovery_spec_from_case_study_config

    if config_path:
        workflow_config = apply_fast_mode_overrides(load_config(config_path))
        case_study_config = config_to_legacy_case_study(workflow_config)
    else:
        case_study_config = load_manuscript_case_study_config(Path.cwd())
    return interaction_discovery_spec_from_case_study_config(case_study_config)


def _write_interaction_shard_outputs(
    *,
    shard,
    cm,
    shard_scores,
    candidate_family_names: list[str],
) -> None:
    """Write score-only shard outputs.  No retention decisions are made here.

    Emits:
    - ``interaction_shard_scores.npz``: observed scores, null score matrix, draw IDs.
    - ``interaction_pair_names.csv``: ordered pair names for this shard.
    - ``shard_result.json``: metadata for the reducer.

    The reducer is responsible for assembling the global null matrix and making
    the single family-wide retention decision.
    """
    npz_path = cm.staging_dir / "interaction_shard_scores.npz"
    pair_names_path = cm.staging_dir / "interaction_pair_names.csv"
    family_path = cm.staging_dir / "interaction_candidate_family.csv"

    np.savez_compressed(
        npz_path,
        observed_scores=np.asarray(shard_scores.observed_scores, dtype=np.float64),
        null_scores=np.asarray(shard_scores.null_scores, dtype=np.float64),
        draw_ids=np.asarray(shard_scores.draw_ids, dtype=np.int64),
    )
    pd.DataFrame({"pair_name": shard_scores.pair_names}).to_csv(pair_names_path, index=False)
    pd.DataFrame({"pair_name": candidate_family_names}).to_csv(family_path, index=False)

    result = {
        "shard_id": shard.shard_id,
        "stage": "interaction_discovery",
        "shard_mode": "score_only",
        "feature_start_idx": shard.feature_start_idx,
        "feature_end_idx": shard.feature_end_idx,
        "n_candidate_pairs": len(shard_scores.pair_names),
        "spec_random_seed": shard_scores.spec_random_seed,
        "spec_permutation_count_B": shard_scores.spec_permutation_count_B,
        "n_draw_ids": int(len(shard_scores.draw_ids)),
        "shard_scores_file": npz_path.name,
        "pair_names_file": pair_names_path.name,
        "candidate_family_file": family_path.name,
        "candidate_family_sha256": hashlib.sha256(
            json.dumps(candidate_family_names, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "candidate_family_count": len(candidate_family_names),
        "status": "score_only_completed",
    }
    (cm.staging_dir / "shard_result.json").write_text(json.dumps(result, indent=2))


def _run_interaction_shard(shard, cm, config_path: str | None = None) -> None:
    """Run score-only interaction scoring for the pair-index range assigned to this shard.

    Shards never make final retention decisions.  They emit observed scores and
    the complete null-score matrix keyed by shared draw IDs.  The global reducer
    assembles the full candidate-family null matrix and issues one FWER decision.
    """
    logger.info(
        "[interaction_shard:score_only] pair_range=[%s, %s) input_paths=%s",
        shard.feature_start_idx,
        shard.feature_end_idx,
        shard.input_paths[:2],
    )

    inputs = _resolve_interaction_inputs(shard.input_paths)
    x_df = pd.read_parquet(inputs["x"])
    holdout_df = pd.read_parquet(inputs["holdout"])
    feature_catalog_df = pd.read_parquet(inputs["catalog"])
    pca_scores_df = pd.read_csv(inputs["pca_scores"])
    retained_terms_df = pd.read_csv(inputs["retained_terms"])

    retained_features = (
        retained_terms_df.get("feature_name", pd.Series(dtype=str))
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    available_features = [c for c in x_df.columns if c != "sample_id"]
    retained_features = [f for f in retained_features if f in available_features]
    if len(retained_features) < 2:
        raise ValueError("interaction_discovery shard requires at least two retained features.")
    total_pairs = len(retained_features) * (len(retained_features) - 1) // 2
    candidate_family_names = [
        f"{left}:{right}" for left, right in combinations(sorted(retained_features), 2)
    ]

    start = int(shard.feature_start_idx or 0)
    end = int(shard.feature_end_idx or total_pairs)
    start = max(0, min(start, total_pairs))
    end = max(start, min(end, total_pairs))
    artifact_root = _artifact_root_from_output_root(cm.output_root)

    spec = _load_interaction_spec(config_path)
    shard_scores = discover_interaction_scores_only(
        input_matrix=x_df,
        feature_catalog=feature_catalog_df,
        holdout_assignments=holdout_df,
        pca_scores=pca_scores_df,
        retained_terms=retained_terms_df,
        spec=spec,
        checkpoint_dir=artifact_root / "interaction_discovery",
        pair_start_idx=start,
        pair_end_idx=end,
    )
    _write_interaction_shard_outputs(
        shard=shard,
        cm=cm,
        shard_scores=shard_scores,
        candidate_family_names=candidate_family_names,
    )


if __name__ == "__main__":
    main()
