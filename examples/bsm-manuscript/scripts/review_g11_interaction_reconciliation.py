#!/usr/bin/env python3
"""Independent formula-level review of a reconciled G11 interaction reducer."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def independent_max_t(
    observed: np.ndarray, null: np.ndarray, *, alpha: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute finite-permutation single-step maxT without pipeline imports."""
    observed = np.asarray(observed, dtype=np.float64)
    null = np.asarray(null, dtype=np.float64)
    if observed.ndim != 1 or null.ndim != 2 or null.shape[1] != observed.size:
        raise ValueError("independent maxT inputs have incompatible shapes")
    if not 0.0 < alpha < 1.0 or null.shape[0] < 1:
        raise ValueError("independent maxT requires draws and alpha in (0, 1)")
    row_max = null.max(axis=1)
    adjusted = (
        1.0 + (row_max[:, None] >= observed[None, :]).sum(axis=0)
    ) / (null.shape[0] + 1.0)
    selected = adjusted <= alpha
    k = int(math.floor(alpha * (null.shape[0] + 1)))
    if k == 0:
        raise ValueError("finite permutation schedule cannot resolve alpha")
    threshold = float(np.partition(row_max, null.shape[0] - k)[null.shape[0] - k])
    return selected, adjusted, threshold


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def review(args: argparse.Namespace) -> int:
    results_root = Path(args.results_root)
    reducer_root = Path(args.reducer_root)
    success_path = reducer_root / "_SUCCESS.json"
    reduced_result_path = reducer_root / "reduced_result.json"
    reconciliation_path = reducer_root / "runtime_identity_reconciliation.json"
    success = json.loads(success_path.read_text(encoding="utf-8"))
    reduced = json.loads(reduced_result_path.read_text(encoding="utf-8"))
    reconciliation = json.loads(reconciliation_path.read_text(encoding="utf-8"))
    if (
        success.get("stage") != "applied_interaction"
        or success.get("status") != "completed"
        or success.get("artifact_sha256") != _sha256_file(reduced_result_path)
        or reduced.get("status") != "completed"
        or reduced.get("records") != args.expected_tasks
        or reconciliation.get("status") != "PASS"
        or reconciliation.get("scientific_scores_modified") is not False
        or reconciliation.get("original_worker_artifacts_modified") is not False
    ):
        raise ValueError("reconciled reducer identity is invalid")

    inventory_by_shard = {
        row["shard_id"]: row for row in reconciliation["worker_inventory"]
    }
    if set(inventory_by_shard) != {
        f"task-{index:04d}" for index in range(args.expected_tasks)
    }:
        raise ValueError("review worker inventory lacks exact task coverage")
    pair_names: tuple[str, ...] | None = None
    observed: np.ndarray | None = None
    null_blocks: list[np.ndarray] = []
    expected_start = 0
    raw_score_hashes = []
    for index in range(args.expected_tasks):
        shard_id = f"task-{index:04d}"
        shard_root = results_root / shard_id
        result_path = shard_root / "result.json"
        marker_path = shard_root / "_SUCCESS.json"
        metadata_path = shard_root / "interaction_block/score_only_interaction.json"
        npz_path = shard_root / "interaction_block/score_only_interaction.npz"
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        inventory = inventory_by_shard[shard_id]
        if (
            marker.get("artifact_sha256") != _sha256_file(result_path)
            or inventory["result_sha256"] != _sha256_file(result_path)
            or metadata.get("npz_sha256") != _sha256_file(npz_path)
            or inventory["score_payload_sha256"] != metadata.get("payload_sha256")
        ):
            raise ValueError(f"review hash mismatch for {shard_id}")
        start = int(metadata["draw_range_start"])
        end = int(metadata["draw_range_end"])
        if start != expected_start or end <= start:
            raise ValueError("review found noncontiguous interaction draw coverage")
        names = tuple(str(name) for name in metadata["pair_names"])
        if pair_names is None:
            pair_names = names
        elif names != pair_names:
            raise ValueError("review found interaction candidate-order drift")
        with np.load(npz_path, allow_pickle=False) as payload:
            block_observed = np.asarray(payload["observed_scores"], dtype=np.float64)
            block_null = np.asarray(payload["null_scores"], dtype=np.float64)
            draw_ids = np.asarray(payload["draw_ids"], dtype=np.int64)
        if not np.array_equal(draw_ids, np.arange(start, end, dtype=np.int64)):
            raise ValueError("review found noncanonical draw IDs")
        if block_null.shape != (end - start, len(names)):
            raise ValueError("review found an invalid null-score block shape")
        if start == 0:
            observed = block_observed
        elif block_observed.size:
            raise ValueError("review found duplicate observed interaction scores")
        if not np.isfinite(block_null).all():
            raise ValueError("review found non-finite null scores")
        null_blocks.append(block_null)
        raw_score_hashes.append(_sha256_file(npz_path))
        expected_start = end
    if pair_names is None or observed is None or expected_start != args.expected_draws:
        raise ValueError("review found incomplete interaction-score coverage")
    if observed.shape != (len(pair_names),) or not np.isfinite(observed).all():
        raise ValueError("review found invalid observed interaction scores")
    null = np.concatenate(null_blocks, axis=0)
    selected, adjusted, threshold = independent_max_t(
        observed, null, alpha=args.alpha
    )

    score_rows = _load_csv(
        reducer_root / "interaction_discovery/interaction_pair_scores.csv"
    )
    rows_by_name = {row["pair_name"]: row for row in score_rows}
    if set(rows_by_name) != set(pair_names) or len(score_rows) != len(pair_names):
        raise ValueError("review output candidate family differs from raw scores")
    for position, name in enumerate(pair_names):
        row = rows_by_name[name]
        if (
            (row["retained"] == "True") != bool(selected[position])
            or not np.isclose(
                float(row["interaction_score"]),
                observed[position],
                rtol=0.0,
                atol=1e-15,
            )
            or not np.isclose(
                float(row["empirical_p_value"]),
                adjusted[position],
                rtol=0.0,
                atol=1e-15,
            )
            or not np.isclose(
                float(row["selection_threshold"]),
                threshold,
                rtol=0.0,
                atol=1e-15,
            )
        ):
            raise ValueError(f"independent maxT decision differs for {name}")
    retained_rows = _load_csv(
        reducer_root / "interaction_discovery/retained_interaction_pairs.csv"
    )
    retained_names = {row["pair_name"] for row in retained_rows}
    independently_retained = {
        name for name, keep in zip(pair_names, selected, strict=True) if keep
    }
    if retained_names != independently_retained:
        raise ValueError("review retained-pair table differs from independent maxT")

    reference_path = Path(args.reference_diagnostic)
    candidate_path = Path(args.candidate_diagnostic)
    with np.load(reference_path, allow_pickle=False) as reference, np.load(
        candidate_path, allow_pickle=False
    ) as candidate:
        ids_equal = np.array_equal(
            reference["training_sample_ids"].astype(str),
            candidate["training_sample_ids"].astype(str),
        )
        features_equal = np.array_equal(
            reference["feature_matrix"], candidate["feature_matrix"]
        )
        response_equal = np.array_equal(
            reference["response_matrix"], candidate["response_matrix"]
        )
    if not (ids_equal and features_equal and response_equal):
        raise ValueError("independent diagnostic matrix equality failed")

    report = {
        "schema_version": 1,
        "stage": "applied_interaction",
        "status": "PASS",
        "review_kind": "INDEPENDENT_FORMULA_AND_BYTE_REVIEW",
        "expected_tasks": args.expected_tasks,
        "draws": args.expected_draws,
        "candidate_pairs": len(pair_names),
        "retained_pairs": int(selected.sum()),
        "selection_alpha": args.alpha,
        "max_t_threshold": threshold,
        "training_sample_ids_exact": ids_equal,
        "feature_matrix_exact": features_equal,
        "response_matrix_exact": response_equal,
        "raw_score_npz_sha256": raw_score_hashes,
        "reduced_result_sha256": _sha256_file(reduced_result_path),
        "reconciliation_sha256": _sha256_file(reconciliation_path),
        "success_marker_sha256": _sha256_file(success_path),
    }
    _write_json(Path(args.output), report)
    print(json.dumps(report, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--reducer-root", required=True)
    parser.add_argument("--reference-diagnostic", required=True)
    parser.add_argument("--candidate-diagnostic", required=True)
    parser.add_argument("--expected-tasks", type=int, default=20)
    parser.add_argument("--expected-draws", type=int, default=999)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output", required=True)
    return review(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
