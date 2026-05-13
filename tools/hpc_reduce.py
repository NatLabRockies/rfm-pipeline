r"""HPC reduce worker — aggregates shard outputs into a single stage artifact.

Run after all SLURM array tasks complete (typically via --dependency=afterok).
Reads shard_result.json from each shard output directory, validates all shards
completed successfully, and merges results into a combined stage artifact.

Usage::

    pixi run python tools/hpc_reduce.py \\
        --manifest manifest.jsonl \\
        --output-root /scratch/bsm/run/outputs \\
        --stage interaction_discovery
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

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
    p.add_argument(
        "--final-output",
        default=None,
        help="Path for final merged artifact (default: <output-root>/_merged/)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    from bsm_rfm.distributed.checkpoint import CheckpointManager
    from bsm_rfm.distributed.manifest import load_manifest

    shards = load_manifest(args.manifest)
    shard_ids = [s.shard_id for s in shards]
    output_root = Path(args.output_root)

    logger.info("[reduce] stage=%s shards=%d output_root=%s", args.stage, len(shards), output_root)

    # Verify all shards completed
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

    # Load all shard results
    shard_results = []
    for shard_id in shard_ids:
        result_path = output_root / shard_id / "shard_result.json"
        if not result_path.exists():
            logger.error("[reduce] Missing shard_result.json for shard=%s", shard_id)
            sys.exit(1)
        shard_results.append(json.loads(result_path.read_text()))

    logger.info("[reduce] loaded %d shard results", len(shard_results))

    # Merge results by stage
    final_output = Path(args.final_output) if args.final_output else output_root / "_merged"
    final_output.mkdir(parents=True, exist_ok=True)

    if args.stage == "interaction_discovery":
        _reduce_interaction_discovery(shard_results, final_output)
    else:
        _reduce_generic(shard_results, final_output, args.stage)

    logger.info("[reduce] COMPLETE — merged artifacts at %s", final_output)


def _reduce_interaction_discovery(shard_results: list[dict], output_dir: Path) -> None:
    """Merge interaction discovery shard results into a combined artifact.

    Currently writes a summary JSON. Full merge (concatenating parquet results)
    will be added as shard execution is fully implemented.
    """
    total_cols = sum(
        (r.get("feature_end_idx", 0) or 0) - (r.get("feature_start_idx", 0) or 0)
        for r in shard_results
    )
    summary = {
        "stage": "interaction_discovery",
        "n_shards": len(shard_results),
        "total_feature_columns_covered": total_cols,
        "shards": [
            {
                "shard_id": r["shard_id"],
                "feature_start_idx": r.get("feature_start_idx"),
                "feature_end_idx": r.get("feature_end_idx"),
                "n_feature_cols": r.get("n_feature_cols"),
                "n_rows": r.get("n_rows"),
                "status": r.get("status"),
            }
            for r in shard_results
        ],
    }
    (output_dir / "interaction_discovery_merged.json").write_text(json.dumps(summary, indent=2))
    logger.info(
        "[reduce:interaction_discovery] %d shards merged → %s", len(shard_results), output_dir
    )


def _reduce_generic(shard_results: list[dict], output_dir: Path, stage: str) -> None:
    """Write a generic merged summary for unsupported/placeholder stages."""
    summary = {
        "stage": stage,
        "n_shards": len(shard_results),
        "note": "Stage not yet shard-parallelized; placeholder reduce",
        "shards": [{"shard_id": r["shard_id"], "status": r.get("status")} for r in shard_results],
    }
    (output_dir / f"{stage}_merged.json").write_text(json.dumps(summary, indent=2))
    logger.info("[reduce:%s] placeholder merge complete", stage)


if __name__ == "__main__":
    main()
