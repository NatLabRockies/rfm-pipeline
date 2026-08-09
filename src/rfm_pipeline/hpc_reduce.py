r"""HPC reduce worker — aggregates shard outputs into stage artifacts.

Run after all SLURM array tasks complete. For interaction discovery, this merges
per-shard pair tables. For all other stages, shard tasks warm checkpoint files and
the reduce job materializes canonical stage artifacts from those checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rfm_pipeline.config import apply_fast_mode_overrides, load_config
from rfm_pipeline.manuscript_pipeline_helpers import (
    _load_empirical_null_screening_result,
    _load_interaction_discovery_result,
    _load_nonlinear_discovery_result,
    _load_output_conditioning_result,
    _load_sparse_selection_result,
    _load_tables,
    config_to_legacy_case_study,
)
from rfm_pipeline.manuscript_stages import (
    InteractionScoresShard,
    condition_manuscript_outputs,
    discover_manuscript_nonlinear_transformations,
    empirical_null_screening_spec_from_case_study_config,
    final_manuscript_artifacts_spec_from_case_study_config,
    interaction_discovery_spec_from_case_study_config,
    nonlinear_discovery_spec_from_case_study_config,
    output_conditioning_spec_from_case_study_config,
    reduce_interaction_family_decision,
    regenerate_final_manuscript_artifacts,
    screen_manuscript_empirical_null_terms,
    select_manuscript_sparse_support,
    sparse_selection_stability_spec_from_case_study_config,
    write_empirical_null_screening_artifacts,
    write_final_manuscript_artifacts,
    write_nonlinear_discovery_artifacts,
    write_output_conditioning_artifacts,
    write_sparse_selection_stability_artifacts,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("bsm.hpc_reduce")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BSM HPC reduce worker")
    p.add_argument("--manifest", required=True, help="Path to JSONL manifest file")
    p.add_argument("--output-root", required=True, help="Root dir for shard outputs")
    p.add_argument("--stage", required=True, help="Pipeline stage to reduce")
    p.add_argument("--config", default=None, help="Optional workflow config YAML override")
    p.add_argument(
        "--final-output",
        default=None,
        help="Path for final merged artifact (default: <output-root>/_merged/)",
    )
    return p.parse_args()


def main() -> None:
    """Reduce per-shard outputs into a consolidated artifact bundle (CLI entry point)."""
    args = _parse_args()

    from rfm_pipeline.distributed.checkpoint import CheckpointManager
    from rfm_pipeline.distributed.manifest import load_manifest

    shards = load_manifest(args.manifest)
    shard_ids = [s.shard_id for s in shards]
    output_root = Path(args.output_root)

    logger.info("[reduce] stage=%s shards=%d output_root=%s", args.stage, len(shards), output_root)

    summary = CheckpointManager.completion_summary(str(output_root), shard_ids)
    logger.info("[reduce] shard status: %s", summary)

    failed = summary.get("failed", 0)
    pending = summary.get("pending", 0)
    if failed > 0 or pending > 0:
        logger.error(
            "[reduce] Cannot reduce: %d failed, %d pending shards. Fix and rerun.",
            failed,
            pending,
        )
        sys.exit(1)

    shard_results = []
    for shard_id in shard_ids:
        result_path = output_root / shard_id / "shard_result.json"
        if not result_path.exists():
            logger.error("[reduce] Missing shard_result.json for shard=%s", shard_id)
            sys.exit(1)
        shard_results.append(json.loads(result_path.read_text()))

    logger.info("[reduce] loaded %d shard results", len(shard_results))
    final_output = Path(args.final_output) if args.final_output else output_root / "_merged"
    final_output.mkdir(parents=True, exist_ok=True)

    if args.stage == "interaction_discovery":
        config_path = args.config
        _reduce_interaction_discovery(
            shard_results, final_output, output_root, config_path=config_path
        )
    else:
        config_path = _resolve_config_path(args.config, shard_results)
        _reduce_checkpoint_warmed_stage(
            stage=args.stage,
            shard_results=shard_results,
            output_root=output_root,
            output_dir=final_output,
            config_path=config_path,
        )

    logger.info("[reduce] COMPLETE — merged artifacts at %s", final_output)


def _resolve_config_path(explicit_config_path: str | None, shard_results: list[dict]) -> str:
    if explicit_config_path:
        return explicit_config_path
    for shard_result in shard_results:
        candidate = shard_result.get("config_path")
        if isinstance(candidate, str) and candidate:
            return candidate
    raise ValueError(
        "Reduce stage requires --config (or shard_result.json must include config_path)."
    )


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


def _reduce_checkpoint_warmed_stage(
    *,
    stage: str,
    shard_results: list[dict],
    output_root: Path,
    output_dir: Path,
    config_path: str,
) -> None:
    """Materialize full stage outputs from checkpoint files warmed by shard jobs."""
    artifact_root = output_root.resolve().parent
    tables, case_study_config = _load_workflow_tables_and_case_config(config_path)

    if stage == "output_conditioning":
        spec = output_conditioning_spec_from_case_study_config(case_study_config)
        conditioning = condition_manuscript_outputs(
            tables["case_study_output_matrix"],
            tables["fixed_holdout_assignments"],
            spec,
        )
        artifact_paths = write_output_conditioning_artifacts(conditioning, artifact_root)
        stage_summary = conditioning.summary.iloc[0].to_dict()
    elif stage == "empirical_null_screening":
        conditioning = _load_output_conditioning_result(artifact_root)
        spec = empirical_null_screening_spec_from_case_study_config(case_study_config)
        screening = screen_manuscript_empirical_null_terms(
            tables["case_study_input_matrix"],
            tables["manuscript_feature_catalog"],
            tables["fixed_holdout_assignments"],
            conditioning.pca_scores,
            spec,
            checkpoint_dir=artifact_root / "empirical_null_screen",
        )
        artifact_paths = write_empirical_null_screening_artifacts(screening, artifact_root)
        stage_summary = screening.summary.iloc[0].to_dict()
    elif stage == "nonlinear_discovery":
        conditioning = _load_output_conditioning_result(artifact_root)
        screening = _load_empirical_null_screening_result(artifact_root)
        spec = nonlinear_discovery_spec_from_case_study_config(case_study_config)
        nonlinear = discover_manuscript_nonlinear_transformations(
            tables["case_study_input_matrix"],
            tables["manuscript_feature_catalog"],
            tables["fixed_holdout_assignments"],
            conditioning.pca_scores,
            screening.retained_terms,
            spec,
            checkpoint_dir=artifact_root / "nonlinear_discovery",
        )
        artifact_paths = write_nonlinear_discovery_artifacts(nonlinear, artifact_root)
        stage_summary = nonlinear.summary.iloc[0].to_dict()
    elif stage == "sparse_selection":
        conditioning = _load_output_conditioning_result(artifact_root)
        screening = _load_empirical_null_screening_result(artifact_root)
        interactions = _load_interaction_discovery_result(artifact_root)
        nonlinear = _load_nonlinear_discovery_result(artifact_root)
        spec = sparse_selection_stability_spec_from_case_study_config(case_study_config)
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
        )
        artifact_paths = write_sparse_selection_stability_artifacts(sparse_selection, artifact_root)
        stage_summary = sparse_selection.summary.iloc[0].to_dict()
    elif stage == "final_manuscript_artifacts":
        conditioning = _load_output_conditioning_result(artifact_root)
        screening = _load_empirical_null_screening_result(artifact_root)
        interactions = _load_interaction_discovery_result(artifact_root)
        nonlinear = _load_nonlinear_discovery_result(artifact_root)
        sparse_selection = _load_sparse_selection_result(artifact_root)
        spec = final_manuscript_artifacts_spec_from_case_study_config(case_study_config)
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
        )
        artifact_paths = write_final_manuscript_artifacts(final_artifacts, artifact_root)
        stage_summary = final_artifacts.summary.iloc[0].to_dict()
    else:
        raise ValueError(f"Unsupported reduce stage: {stage}")

    summary = {
        "stage": stage,
        "status": "materialized_from_checkpoints",
        "config_path": str(Path(config_path)),
        "n_shards": len(shard_results),
        "artifact_root": str(artifact_root),
        "written_artifacts": {name: str(path) for name, path in artifact_paths.items()},
        "stage_summary": stage_summary,
        "shards": [
            {"shard_id": r.get("shard_id"), "status": r.get("status")} for r in shard_results
        ],
    }
    (output_dir / f"{stage}_merged.json").write_text(json.dumps(summary, indent=2))
    logger.info("[reduce:%s] checkpoint-backed materialization complete", stage)


def _reduce_interaction_discovery(
    shard_results: list[dict],
    output_dir: Path,
    output_root: Path,
    *,
    config_path: str | None = None,
) -> None:
    """Assemble global candidate-family null matrix and make one FWER decision.

    All shards must have been produced in score-only mode (``shard_mode=score_only``).
    The reducer:
    1. Loads each shard's ``interaction_shard_scores.npz`` and ``interaction_pair_names.csv``.
    2. Verifies that all shards share the same draw IDs (same random seed / permutation count).
    3. Validates that pair names are disjoint and cover the full candidate family.
    4. Assembles the global null matrix (B × P_total) by column-concatenating shard null matrices.
    5. Calls ``reduce_interaction_family_decision`` to apply one global FWER decision.
    6. Writes ``retained_interaction_pairs_merged.csv``, ``interaction_pair_scores_merged.csv``,
       and ``interaction_discovery_merged.json``.
    """
    # Spec is mandatory — no default fallback allowed.
    if not config_path:
        raise ValueError(
            "_reduce_interaction_discovery requires --config (or config_path). "
            "A canonical config is mandatory; no default-spec fallback is permitted. "
            "Pass the exact production config used to generate the interaction shards."
        )
    workflow_config = apply_fast_mode_overrides(load_config(config_path))
    case_study_config = config_to_legacy_case_study(workflow_config)
    spec = interaction_discovery_spec_from_case_study_config(case_study_config)

    # Load score-only shard NPZ files.
    loaded_shards: list[InteractionScoresShard] = []
    expected_pair_names: list[str] | None = None
    expected_family_hash: str | None = None
    for shard_meta in shard_results:
        shard_id = str(shard_meta.get("shard_id", ""))
        if shard_meta.get("shard_mode") != "score_only":
            raise ValueError(
                f"Shard {shard_id!r} is not score-only; global interaction reduction "
                "requires every shard to omit retention decisions."
            )
        shard_dir = output_root / shard_id
        npz_name = str(shard_meta.get("shard_scores_file", "interaction_shard_scores.npz"))
        pair_names_file = str(shard_meta.get("pair_names_file", "interaction_pair_names.csv"))
        family_file = str(
            shard_meta.get("candidate_family_file", "interaction_candidate_family.csv")
        )
        npz_path = shard_dir / npz_name
        pair_names_path = shard_dir / pair_names_file
        family_path = shard_dir / family_file

        if not npz_path.exists() or not pair_names_path.exists() or not family_path.exists():
            raise FileNotFoundError(
                f"Shard {shard_id!r} is missing score-only outputs. "
                f"Expected {npz_path}, {pair_names_path}, and {family_path}. "
                "Ensure all shards were run in score_only mode before reducing."
            )
        with np.load(npz_path, allow_pickle=False) as data:
            observed_scores = np.asarray(data["observed_scores"], dtype=np.float64)
            null_scores = np.asarray(data["null_scores"], dtype=np.float64)
            draw_ids = np.asarray(data["draw_ids"], dtype=np.int64)
        pair_names_df = pd.read_csv(pair_names_path)
        if "pair_name" not in pair_names_df.columns:
            raise ValueError(f"Shard {shard_id!r} pair-name file lacks 'pair_name'.")
        pair_names = pair_names_df["pair_name"].astype(str).tolist()
        family_df = pd.read_csv(family_path)
        if "pair_name" not in family_df.columns:
            raise ValueError(f"Shard {shard_id!r} candidate-family file lacks 'pair_name'.")
        family_names = family_df["pair_name"].astype(str).tolist()
        family_hash = hashlib.sha256(
            json.dumps(family_names, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if expected_pair_names is None:
            expected_pair_names = family_names
            expected_family_hash = family_hash
        elif family_hash != expected_family_hash or family_names != expected_pair_names:
            raise ValueError(
                f"Shard {shard_id!r} candidate family differs from the canonical family."
            )
        if observed_scores.ndim != 1 or len(observed_scores) != len(pair_names):
            raise ValueError(
                f"Shard {shard_id!r} observed score count does not match pair-name count."
            )
        if null_scores.ndim != 2 or null_scores.shape[1] != len(pair_names):
            raise ValueError(f"Shard {shard_id!r} null score columns do not match pair-name count.")
        if draw_ids.ndim != 1 or null_scores.shape[0] != len(draw_ids):
            raise ValueError(f"Shard {shard_id!r} draw IDs do not match null-score rows.")
        loaded_shards.append(
            InteractionScoresShard(
                pair_names=pair_names,
                observed_scores=observed_scores,
                null_scores=null_scores,
                draw_ids=draw_ids,
                n_training_rows=0,
                spec_random_seed=int(shard_meta.get("spec_random_seed", spec.random_seed)),
                spec_permutation_count_B=int(
                    shard_meta.get("spec_permutation_count_B", spec.permutation_count_B)
                ),
            )
        )

    # One global family decision.
    result = reduce_interaction_family_decision(
        loaded_shards,
        spec,
        expected_pair_names=expected_pair_names,
    )

    retained_out = output_dir / "retained_interaction_pairs_merged.csv"
    pair_scores_out = output_dir / "interaction_pair_scores_merged.csv"
    result.retained_pairs.to_csv(retained_out, index=False)
    result.pair_scores.to_csv(pair_scores_out, index=False)

    # Canonical hashes for provenance and verification.
    # family_hash: ordered candidate pair names (canonical schedule).
    _family_names_ordered = (
        expected_pair_names
        if expected_pair_names is not None
        else [shard.pair_names for shard in loaded_shards]
    )
    if isinstance(_family_names_ordered[0], list):
        _family_flat = [n for sub in _family_names_ordered for n in sub]
    else:
        _family_flat = list(_family_names_ordered)
    family_hash = hashlib.sha256(
        json.dumps(_family_flat, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    # null_matrix_hash: assembled global null matrix (B × m).
    _global_null = np.concatenate([s.null_scores for s in loaded_shards], axis=1)
    _null_contiguous = np.ascontiguousarray(_global_null, dtype=np.float64)
    null_matrix_hash = hashlib.sha256(
        repr(_null_contiguous.shape).encode("utf-8") + _null_contiguous.tobytes()
    ).hexdigest()

    # p_values_hash: empirical p-values sorted by pair_name.
    _pscores_sorted = result.pair_scores.sort_values("pair_name")
    _pvals = np.ascontiguousarray(_pscores_sorted["empirical_p_value"].to_numpy(dtype=np.float64))
    p_values_hash = hashlib.sha256(
        repr(_pvals.shape).encode("utf-8") + _pvals.tobytes()
    ).hexdigest()

    # threshold_hash: per-pair null quantile thresholds sorted by pair_name.
    _thresh = np.ascontiguousarray(_pscores_sorted["null_threshold"].to_numpy(dtype=np.float64))
    threshold_hash = hashlib.sha256(
        repr(_thresh.shape).encode("utf-8") + _thresh.tobytes()
    ).hexdigest()

    # retained_set_hash: sorted retained pair names.
    _retained_names = sorted(result.retained_pairs["pair_name"].astype(str).tolist())
    retained_set_hash = hashlib.sha256(
        json.dumps(_retained_names, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    summary = {
        "stage": "interaction_discovery",
        "reducer": "global_family_decision",
        "n_shards": len(shard_results),
        "n_candidate_pairs": len(result.pair_scores),
        "n_retained_pairs": int(result.summary.loc[0, "n_retained_pairs"])
        if not result.summary.empty
        else len(result.retained_pairs),
        "merged_retained_pairs_file": retained_out.name,
        "merged_pair_scores_file": pair_scores_out.name,
        "canonical_hashes": {
            "family_sha256": family_hash,
            "null_matrix_sha256": null_matrix_hash,
            "p_values_sha256": p_values_hash,
            "threshold_sha256": threshold_hash,
            "retained_set_sha256": retained_set_hash,
        },
        "shards": [
            {
                "shard_id": r["shard_id"],
                "feature_start_idx": r.get("feature_start_idx"),
                "feature_end_idx": r.get("feature_end_idx"),
                "n_candidate_pairs": r.get("n_candidate_pairs"),
                "candidate_family_sha256": r.get("candidate_family_sha256"),
                "status": r.get("status"),
            }
            for r in shard_results
        ],
    }
    (output_dir / "interaction_discovery_merged.json").write_text(json.dumps(summary, indent=2))
    logger.info(
        "[reduce:interaction_discovery] global family decision: %d/%d pairs retained → %s",
        len(result.retained_pairs),
        len(result.pair_scores),
        output_dir,
    )


if __name__ == "__main__":
    main()
