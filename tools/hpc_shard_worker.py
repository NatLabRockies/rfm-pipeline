r"""HPC shard worker — runs inside each SLURM array task.

Selects the shard at TASK_ID from the manifest, executes the designated
pipeline stage over its assigned feature range, validates outputs, then
promotes them with an atomic _SUCCESS.json marker.

Called by the generated sbatch script::

    pixi run python tools/hpc_shard_worker.py \\
        --manifest manifest.jsonl \\
        --task-id $SLURM_ARRAY_TASK_ID \\
        --output-root /scratch/bsm/run/outputs \\
        --work-dir /scratch/bsm/run/task-0042 \\
        --stage interaction_discovery
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from bsm_rfm.manuscript_stages import discover_manuscript_interactions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("bsm.hpc_shard_worker")


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
    args = _parse_args()

    from bsm_rfm.distributed.checkpoint import CheckpointManager
    from bsm_rfm.distributed.manifest import load_manifest, update_shard_status

    # Load manifest and select this task's shard
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

    # Initialize checkpoint manager
    cm = CheckpointManager(args.output_root, shard.shard_id)

    # Skip if already complete (idempotent re-run)
    if cm.is_complete():
        logger.info("shard=%s already complete (_SUCCESS.json found), skipping", shard.shard_id)
        sys.exit(0)

    if args.dry_run:
        logger.info("[dry-run] shard=%s validated — would run %s", shard.shard_id, shard.stage)
        sys.exit(0)

    # Mark running and update manifest
    cm.mark_running()
    update_shard_status(args.manifest, shard.shard_id, "running")

    start = time.monotonic()
    try:
        _run_shard_stage(shard, cm, args)
        elapsed = time.monotonic() - start
        logger.info("shard=%s COMPLETE in %.1fs", shard.shard_id, elapsed)
        update_shard_status(args.manifest, shard.shard_id, "completed")
    except Exception as e:
        elapsed = time.monotonic() - start
        logger.error("shard=%s FAILED after %.1fs: %s", shard.shard_id, elapsed, e)
        cm.mark_failed(str(e))
        update_shard_status(args.manifest, shard.shard_id, "failed", error_message=str(e))
        sys.exit(1)


def _run_shard_stage(shard, cm, args) -> None:
    """Dispatch to the appropriate stage runner."""
    stage = shard.stage
    logger.info("[shard] running stage=%s for shard=%s", stage, shard.shard_id)

    # Ensure staging dir exists
    cm.staging_dir.mkdir(parents=True, exist_ok=True)

    if stage == "interaction_discovery":
        _run_interaction_shard(shard, cm, config_path=args.config)
        expected_files = [
            "shard_result.json",
            "retained_interaction_pairs.csv",
            "interaction_pair_scores.csv",
        ]
    else:
        # Generic pass-through: run the full pipeline stage for this shard's inputs
        # Stages other than interaction_discovery are not yet shard-parallelized —
        # the single-node runner handles them. Log a warning and write a placeholder.
        logger.warning(
            "[shard] stage=%s is not yet shard-parallelized; writing placeholder _SUCCESS",
            stage,
        )
        _write_placeholder(shard, cm)
        expected_files = ["shard_result.json"]

    # Validate and promote
    cm.validate_and_promote(
        expected_files=expected_files,
        metadata={"stage": stage, "shard_id": shard.shard_id},
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
    from bsm_rfm.manuscript_runtime import load_manuscript_case_study_config
    from bsm_rfm.manuscript_stages import interaction_discovery_spec_from_case_study_config

    if config_path:
        from bsm_rfm.config import load_config
        from tools.run_manuscript_pipeline import config_to_legacy_case_study

        workflow_config = load_config(config_path)
        case_study_config = config_to_legacy_case_study(workflow_config)
    else:
        case_study_config = load_manuscript_case_study_config(Path.cwd())
    return interaction_discovery_spec_from_case_study_config(case_study_config)


def _write_interaction_shard_outputs(
    *,
    shard,
    cm,
    interactions,
    selected_features: list[str],
) -> None:
    pair_scores_path = cm.staging_dir / "interaction_pair_scores.csv"
    retained_pairs_path = cm.staging_dir / "retained_interaction_pairs.csv"
    null_summary_path = cm.staging_dir / "interaction_null_summary.csv"
    component_scores_path = cm.staging_dir / "component_interaction_scores.csv"
    stage_summary_path = cm.staging_dir / "interaction_discovery_summary.csv"

    interactions.pair_scores.to_csv(pair_scores_path, index=False)
    interactions.retained_pairs.to_csv(retained_pairs_path, index=False)
    interactions.interaction_null_summary.to_csv(null_summary_path, index=False)
    interactions.component_interaction_scores.to_csv(component_scores_path, index=False)
    interactions.summary.to_csv(stage_summary_path, index=False)

    result = {
        "shard_id": shard.shard_id,
        "stage": "interaction_discovery",
        "feature_start_idx": shard.feature_start_idx,
        "feature_end_idx": shard.feature_end_idx,
        "n_feature_cols": len(selected_features),
        "selected_features": selected_features,
        "n_candidate_pairs": int(interactions.summary.loc[0, "n_candidate_pairs"]),
        "n_retained_pairs": int(interactions.summary.loc[0, "n_retained_pairs"]),
        "pair_scores_file": pair_scores_path.name,
        "retained_pairs_file": retained_pairs_path.name,
        "null_summary_file": null_summary_path.name,
        "component_scores_file": component_scores_path.name,
        "summary_file": stage_summary_path.name,
        "status": "completed",
    }
    (cm.staging_dir / "shard_result.json").write_text(json.dumps(result, indent=2))


def _run_interaction_shard(shard, cm, config_path: str | None = None) -> None:
    """Run interaction discovery scoring for the feature range assigned to this shard.

    The shard covers features [feature_start_idx, feature_end_idx) of the full
    feature set. We load X/Y from input_paths, subset to the shard's columns,
    and run the interaction scoring pipeline.

    Outputs a shard_result.json with the discovered pairs and scores.
    """
    logger.info(
        "[interaction_shard] feature_range=[%s, %s) input_paths=%s",
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

    start = int(shard.feature_start_idx or 0)
    end = int(shard.feature_end_idx or len(retained_features))
    start = max(0, min(start, len(retained_features)))
    end = max(start, min(end, len(retained_features)))
    selected_features = retained_features[start:end]
    if len(selected_features) < 2:
        selected_features = retained_features

    shard_retained_terms = retained_terms_df[
        retained_terms_df["feature_name"].astype(str).isin(selected_features)
    ].copy()
    if len(shard_retained_terms) < 2:
        raise ValueError(
            "interaction_discovery shard retained term subset has fewer than two terms."
        )

    spec = _load_interaction_spec(config_path)
    interactions = discover_manuscript_interactions(
        input_matrix=x_df,
        feature_catalog=feature_catalog_df,
        holdout_assignments=holdout_df,
        pca_scores=pca_scores_df,
        retained_terms=shard_retained_terms,
        spec=spec,
    )
    _write_interaction_shard_outputs(
        shard=shard,
        cm=cm,
        interactions=interactions,
        selected_features=selected_features,
    )


def _write_placeholder(shard, cm) -> None:
    """Write a placeholder shard_result.json for unsupported stages."""
    result = {
        "shard_id": shard.shard_id,
        "stage": shard.stage,
        "status": "placeholder",
        "note": "Stage not yet shard-parallelized; run via single-node pipeline",
    }
    result_path = cm.staging_dir / "shard_result.json"
    result_path.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
