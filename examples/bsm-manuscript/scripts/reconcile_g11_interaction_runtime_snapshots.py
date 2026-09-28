#!/usr/bin/env python3
"""Reconcile non-scientific cross-host identity drift in G11 interaction blocks.

This control utility never rescales, edits, or recomputes interaction scores.  It
first proves that complete, self-verifying score blocks differ only in runtime
identity fields, compares independently reconstructed pre-score matrices, and
then substitutes one verified reference snapshot in memory for reduction.  The
original worker artifacts remain immutable and are inventoried in the evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from itertools import combinations
from pathlib import Path
from typing import Any, Iterator

import numpy as np


ALLOWED_SNAPSHOT_RUNTIME_DRIFT = frozenset(
    {
        "contract_sha256",
        "training_sample_ids_sha256",
        "response_matrix_sha256",
    }
)
RESPONSE_RTOL = 1e-12
RESPONSE_ATOL = 1e-12


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def validate_snapshot_runtime_drift(
    reference: dict[str, Any], candidate: dict[str, Any]
) -> set[str]:
    """Return the allowed runtime-only snapshot differences or fail closed."""
    if set(reference) != set(candidate):
        missing = sorted(set(reference) - set(candidate))
        extra = sorted(set(candidate) - set(reference))
        raise ValueError(
            f"snapshot fields differ; missing={missing}, extra={extra}"
        )
    differences = {key for key in reference if reference[key] != candidate[key]}
    unexpected = differences - ALLOWED_SNAPSHOT_RUNTIME_DRIFT
    if unexpected:
        raise ValueError(
            "scientific snapshot identity differs in disallowed fields: "
            + ", ".join(sorted(unexpected))
        )
    if not differences:
        raise ValueError("runtime reconciliation requires two distinct snapshots")
    return differences


def validate_contract_runtime_drift(
    reference: dict[str, Any], candidate: dict[str, Any]
) -> tuple[int, int]:
    """Require canonical interaction controls to differ only in worker count."""
    if set(reference) != set(candidate):
        raise ValueError("canonical interaction contract fields differ")
    differences = {key for key in reference if reference[key] != candidate[key]}
    if differences != {"n_jobs"}:
        rendered = ", ".join(sorted(differences)) or "none"
        raise ValueError(
            "canonical interaction contracts must differ only in n_jobs; "
            f"observed: {rendered}"
        )
    return int(reference["n_jobs"]), int(candidate["n_jobs"])


def classify_snapshot_runtime(
    snapshot: dict[str, Any],
    *,
    reference: dict[str, Any],
    candidate: dict[str, Any],
) -> str:
    """Bind one worker snapshot to a diagnosed runtime despite pointer hashing."""
    matches: list[str] = []
    for label, diagnostic in (("reference", reference), ("candidate", candidate)):
        if set(snapshot) != set(diagnostic):
            continue
        stable_fields_match = all(
            snapshot[key] == diagnostic[key]
            for key in snapshot
            if key not in ALLOWED_SNAPSHOT_RUNTIME_DRIFT
        )
        # The legacy response hash is process-sensitive because upstream
        # conditioning uses floating-point linear algebra.  The separately
        # persisted diagnostic matrices establish strict numerical equality;
        # the canonical contract checksum identifies the runtime group here.
        runtime_group_matches = (
            snapshot["contract_sha256"] == diagnostic["contract_sha256"]
        )
        if stable_fields_match and runtime_group_matches:
            matches.append(label)
    if len(matches) != 1:
        raise ValueError(
            "worker snapshot does not bind exactly one diagnosed runtime group"
        )
    return matches[0]


def compare_diagnostic_bundles(
    reference_path: Path, candidate_path: Path
) -> dict[str, Any]:
    """Compare independently reconstructed pre-score matrices fail-closed."""
    with np.load(reference_path, allow_pickle=False) as reference, np.load(
        candidate_path, allow_pickle=False
    ) as candidate:
        required = {"training_sample_ids", "feature_matrix", "response_matrix"}
        if set(reference.files) != required or set(candidate.files) != required:
            raise ValueError("diagnostic bundles have unexpected array members")
        reference_ids = reference["training_sample_ids"].astype(str)
        candidate_ids = candidate["training_sample_ids"].astype(str)
        reference_features = np.asarray(reference["feature_matrix"], dtype=np.float64)
        candidate_features = np.asarray(candidate["feature_matrix"], dtype=np.float64)
        reference_response = np.asarray(reference["response_matrix"], dtype=np.float64)
        candidate_response = np.asarray(candidate["response_matrix"], dtype=np.float64)

    ids_equal = np.array_equal(reference_ids, candidate_ids)
    features_equal = np.array_equal(reference_features, candidate_features)
    response_shape_equal = reference_response.shape == candidate_response.shape
    max_abs_diff = (
        float(np.max(np.abs(reference_response - candidate_response)))
        if response_shape_equal and reference_response.size
        else float("inf")
    )
    response_allclose = response_shape_equal and bool(
        np.allclose(
            reference_response,
            candidate_response,
            rtol=RESPONSE_RTOL,
            atol=RESPONSE_ATOL,
            equal_nan=False,
        )
    )
    if not ids_equal:
        raise ValueError("diagnostic training sample IDs differ")
    if not features_equal:
        raise ValueError("diagnostic feature matrices differ")
    if not response_allclose:
        raise ValueError(
            "diagnostic response matrices differ beyond the strict numerical "
            f"tolerance; max_abs_diff={max_abs_diff!r}"
        )
    return {
        "training_sample_ids_equal": ids_equal,
        "feature_matrix_equal": features_equal,
        "response_matrix_allclose": response_allclose,
        "response_matrix_max_abs_diff": max_abs_diff,
        "response_matrix_rtol": RESPONSE_RTOL,
        "response_matrix_atol": RESPONSE_ATOL,
        "training_row_count": int(reference_ids.shape[0]),
        "feature_matrix_shape": list(reference_features.shape),
        "response_matrix_shape": list(reference_response.shape),
    }


@contextmanager
def _temporary_n_jobs(n_jobs: int) -> Iterator[None]:
    previous = os.environ.get("SLURM_CPUS_PER_TASK")
    os.environ["SLURM_CPUS_PER_TASK"] = str(n_jobs)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("SLURM_CPUS_PER_TASK", None)
        else:
            os.environ["SLURM_CPUS_PER_TASK"] = previous


def _load_manifest_record(manifest_path: Path, task_id: int) -> dict[str, Any]:
    records = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    matches = [
        record
        for record in records
        if int(str(record["shard_id"]).split("-")[-1]) == task_id
    ]
    if len(matches) != 1:
        raise ValueError(f"manifest does not contain exactly one task {task_id}")
    return matches[0]


def _build_pre_score_diagnostic(
    record: dict[str, Any], *, n_jobs: int
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    from rfm_pipeline.campaign_contract import load_contract
    from rfm_pipeline.interaction_contract import (
        build_control_snapshot,
        canonical_execution_contract_from_specs,
    )
    from rfm_pipeline.manuscript_stages import (
        _align_table_by_sample_id,
        _component_columns,
        _condition_pca_scores_on_main_effects,
        _generate_pairwise_interactions,
        _interaction_pair_detectors,
        _retained_first_order_term_names,
        _source_input_column,
        _standardize_for_screening,
        _train_sample_ids,
    )
    from scripts import g11_campaign_adapter as adapter

    campaign_contract, contract_hash = load_contract(
        Path(str(record["contract_config_path"]))
    )
    if contract_hash != record["config_hash"]:
        raise ValueError("diagnostic contract hash differs from the manifest")
    inputs, assignments, _catalog = adapter._load_applied_train_tables(
        record, names=("inputs", "assignments", "catalog")
    )
    conditioning = adapter._load_conditioning(record)
    screening = adapter._load_screening(record)
    with _temporary_n_jobs(n_jobs):
        _, _, interaction, *_ = adapter._applied_specifications(
            record, campaign_contract
        )
    pca_scores = conditioning.pca_scores
    if interaction.condition_main_effects:
        pca_scores = _condition_pca_scores_on_main_effects(
            inputs,
            assignments,
            pca_scores,
            screening.retained_terms,
            degree=interaction.main_effect_conditioning_degree,
        )
    retained_first_order = _retained_first_order_term_names(
        screening.retained_terms, inputs, minimum_count=0
    )
    candidates = _generate_pairwise_interactions(retained_first_order)
    component_names = _component_columns(pca_scores)
    train_ids = _train_sample_ids(assignments)
    y_train = _align_table_by_sample_id(
        pca_scores, train_ids, component_names, "PCA scores"
    )
    y_scaled, _ = _standardize_for_screening(y_train)
    feature_names = sorted(
        {name for _, left, right in candidates for name in (left, right)}
    )
    indexed = inputs.set_index("sample_id", drop=False)
    train_rows = indexed.loc[list(train_ids)].reset_index(drop=True)
    x_full = np.column_stack(
        [
            _source_input_column(train_rows, name, name).to_numpy(dtype=float)
            for name in feature_names
        ]
    )
    pair_to_indices = {
        pair_name: (feature_names.index(left), feature_names.index(right))
        for pair_name, left, right in candidates
    }
    pair_detectors = _interaction_pair_detectors(
        x_full, candidates, pair_to_indices, method=interaction.method
    )
    canonical = canonical_execution_contract_from_specs(interaction)
    snapshot = build_control_snapshot(
        canonical,
        candidate_pair_names=tuple(pair_name for pair_name, _, _ in candidates),
        candidate_pair_detectors=pair_detectors,
        training_sample_ids=train_ids.to_numpy(),
        feature_matrix=x_full,
        response_matrix=y_scaled,
        component_names=tuple(component_names),
    )
    metadata = {
        "schema_version": 1,
        "stage": "applied_interaction",
        "task_id": int(str(record["shard_id"]).split("-")[-1]),
        "n_jobs": n_jobs,
        "config_hash": contract_hash,
        "parent_hash": record["parent_hash"],
        "control_snapshot": snapshot.to_dict(),
        "canonical_contract": canonical.to_dict(),
    }
    return (
        metadata,
        np.asarray(train_ids.astype(str).tolist(), dtype=np.str_),
        np.asarray(x_full, dtype=np.float64),
        np.asarray(y_scaled, dtype=np.float64),
    )


def diagnose(args: argparse.Namespace) -> int:
    record = _load_manifest_record(Path(args.manifest), args.task_id)
    metadata, train_ids, feature_matrix, response_matrix = _build_pre_score_diagnostic(
        record, n_jobs=args.n_jobs
    )
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    bundle_path = output_root / "pre_score_arrays.npz"
    np.savez_compressed(
        bundle_path,
        training_sample_ids=train_ids,
        feature_matrix=feature_matrix,
        response_matrix=response_matrix,
    )
    metadata["bundle_sha256"] = _sha256_file(bundle_path)
    metadata_path = output_root / "diagnostic.json"
    _write_json(metadata_path, metadata)
    print(json.dumps({"status": "PASS", "diagnostic": str(metadata_path)}))
    return 0


def _load_diagnostic(path: Path) -> tuple[dict[str, Any], Path]:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    bundle_path = path.parent / "pre_score_arrays.npz"
    if not bundle_path.is_file() or metadata.get("bundle_sha256") != _sha256_file(
        bundle_path
    ):
        raise ValueError("diagnostic bundle hash mismatch")
    return metadata, bundle_path


def reconcile(args: argparse.Namespace) -> int:
    from rfm_pipeline.campaign_contract import load_contract
    from rfm_pipeline.hpc_campaign_package import _stable_hash
    from rfm_pipeline.interaction_contract import (
        ScoreOnlyInteractionArtifact,
        canonical_execution_contract_from_specs,
    )
    from rfm_pipeline.manuscript_stages import (
        reduce_score_only_interaction_artifacts,
        write_interaction_discovery_artifacts,
    )
    from scripts import g11_campaign_adapter as adapter

    manifest_path = Path(args.manifest)
    records = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) != args.expected_tasks:
        raise ValueError("reconciliation manifest has incomplete task coverage")
    if [record["shard_id"] for record in records] != [
        f"task-{index:04d}" for index in range(args.expected_tasks)
    ]:
        raise ValueError("reconciliation manifest task order is not canonical")
    if {record["parent_hash"] for record in records} != {args.expected_parent_hash}:
        raise ValueError("reconciliation parent hash differs from the manifest")
    if {record["config_hash"] for record in records} != {args.contract_hash}:
        raise ValueError("reconciliation contract hash differs from the manifest")

    audit_success = Path(args.audit_success)
    audit_result = audit_success.parent / "audit_result.json"
    audit_marker = json.loads(audit_success.read_text(encoding="utf-8"))
    if (
        audit_marker.get("stage") != "applied_interaction"
        or audit_marker.get("status") != "completed"
        or audit_marker.get("artifact_sha256") != _sha256_file(audit_result)
    ):
        raise ValueError("reconciliation requires valid exact-coverage audit evidence")

    results_root = Path(args.results_root)
    artifacts = []
    artifact_hashes = []
    worker_inventory = []
    for record in records:
        shard_dir = results_root / record["shard_id"]
        result_path = shard_dir / "result.json"
        marker_path = shard_dir / "_SUCCESS.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        result_sha256 = _sha256_file(result_path)
        if (
            result.get("status") != "completed"
            or result.get("stage") != "applied_interaction"
            or marker.get("status") != "completed"
            or marker.get("stage") != "applied_interaction"
            or marker.get("shard_id") != record["shard_id"]
            or marker.get("artifact_sha256") != result_sha256
            or marker.get("output_hash") != record["output_hash"]
            or marker.get("parent_hash") != record["parent_hash"]
        ):
            raise ValueError(f"invalid worker identity: {record['shard_id']}")
        for item in result.get("scientific_artifacts", []):
            artifact_path = shard_dir / str(item["relative_path"])
            if (
                not artifact_path.is_file()
                or artifact_path.stat().st_size != int(item["bytes"])
                or _sha256_file(artifact_path) != item["sha256"]
            ):
                raise ValueError(
                    f"scientific artifact hash mismatch: {record['shard_id']}"
                )
        artifact = ScoreOnlyInteractionArtifact.read_from(
            shard_dir / "interaction_block"
        )
        artifacts.append(artifact)
        artifact_hashes.append(result_sha256)
        worker_inventory.append(
            {
                "shard_id": record["shard_id"],
                "result_sha256": result_sha256,
                "score_payload_sha256": artifact.payload_sha256,
                "control_snapshot_sha256": artifact.control_snapshot.checksum,
                "draw_range": [artifact.draw_range_start, artifact.draw_range_end],
            }
        )

    reference_metadata, reference_bundle = _load_diagnostic(
        Path(args.reference_diagnostic)
    )
    candidate_metadata, candidate_bundle = _load_diagnostic(
        Path(args.candidate_diagnostic)
    )
    bundle_report = compare_diagnostic_bundles(reference_bundle, candidate_bundle)
    reference_snapshot = reference_metadata["control_snapshot"]
    candidate_snapshot = candidate_metadata["control_snapshot"]
    snapshot_differences = validate_snapshot_runtime_drift(
        reference_snapshot, candidate_snapshot
    )
    reference_n_jobs, candidate_n_jobs = validate_contract_runtime_drift(
        reference_metadata["canonical_contract"]["interaction_controls"],
        candidate_metadata["canonical_contract"]["interaction_controls"],
    )
    runtime_groups = [
        classify_snapshot_runtime(
            artifact.control_snapshot.to_dict(),
            reference=reference_snapshot,
            candidate=candidate_snapshot,
        )
        for artifact in artifacts
    ]
    if set(runtime_groups) != {"reference", "candidate"}:
        raise ValueError("worker artifacts do not contain both diagnosed runtime groups")
    if {
        artifact.control_snapshot.contract_sha256 for artifact in artifacts
    } != {
        reference_snapshot["contract_sha256"],
        candidate_snapshot["contract_sha256"],
    }:
        raise ValueError("diagnostic contracts do not bind both worker groups")

    campaign_contract, loaded_hash = load_contract(Path(args.contract_config))
    if loaded_hash != args.contract_hash:
        raise ValueError("reconciliation contract bytes differ from the requested hash")
    with _temporary_n_jobs(reference_n_jobs):
        _, _, spec, *_ = adapter._applied_specifications(records[0], campaign_contract)
    canonical = canonical_execution_contract_from_specs(spec)
    if canonical.checksum != reference_snapshot["contract_sha256"]:
        raise ValueError("reference diagnostic is not the canonical reduction contract")

    reference_object = next(
        artifact.control_snapshot
        for artifact in artifacts
        if artifact.control_snapshot.contract_sha256 == canonical.checksum
    )
    normalized = [
        artifact
        if artifact.control_snapshot.checksum == reference_object.checksum
        else replace(artifact, control_snapshot=reference_object)
        for artifact in artifacts
    ]
    screening = adapter._load_screening(records[0])
    retained_names = screening.retained_terms["feature_name"].astype(str).tolist()
    expected_pairs = tuple(
        f"{left}:{right}" for left, right in combinations(retained_names, 2)
    )
    result = reduce_score_only_interaction_artifacts(
        normalized,
        spec=spec,
        contract=canonical,
        expected_pair_names=expected_pairs,
    )

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    written = write_interaction_discovery_artifacts(result, output_root)
    reconciliation = {
        "schema_version": 1,
        "status": "PASS",
        "stage": "applied_interaction",
        "decision": "CONTROL_ONLY_RUNTIME_IDENTITY_RECONCILIATION",
        "scientific_scores_modified": False,
        "original_worker_artifacts_modified": False,
        "expected_task_count": args.expected_tasks,
        "worker_inventory": worker_inventory,
        "allowed_snapshot_runtime_drift": sorted(ALLOWED_SNAPSHOT_RUNTIME_DRIFT),
        "observed_snapshot_runtime_drift": sorted(snapshot_differences),
        "canonical_n_jobs": reference_n_jobs,
        "alternate_runtime_n_jobs": candidate_n_jobs,
        "runtime_group_counts": {
            "reference": runtime_groups.count("reference"),
            "candidate": runtime_groups.count("candidate"),
        },
        "distinct_legacy_training_id_hashes": len(
            {
                artifact.control_snapshot.training_sample_ids_sha256
                for artifact in artifacts
            }
        ),
        "distinct_legacy_response_matrix_hashes": len(
            {
                artifact.control_snapshot.response_matrix_sha256
                for artifact in artifacts
            }
        ),
        "diagnostic_bundle_comparison": bundle_report,
        "reference_diagnostic_sha256": _sha256_file(
            Path(args.reference_diagnostic)
        ),
        "candidate_diagnostic_sha256": _sha256_file(
            Path(args.candidate_diagnostic)
        ),
        "audit_success_sha256": _sha256_file(audit_success),
        "audit_result_sha256": _sha256_file(audit_result),
        "written_artifacts": {
            name: {"path": str(path), "sha256": _sha256_file(path)}
            for name, path in written.items()
        },
    }
    reconciliation_path = output_root / "runtime_identity_reconciliation.json"
    _write_json(reconciliation_path, reconciliation)

    scientific_reduction = {
        "operation": "applied_interaction",
        "status": "completed",
        "candidate_pair_count": len(result.pair_scores),
        "retained_pair_count": len(result.retained_pairs),
        "draw_count": campaign_contract.B_interaction,
        "scientific_artifacts": [
            {
                "relative_path": str(path.relative_to(output_root)),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for path in sorted(output_root.rglob("*"))
            if path.is_file()
        ],
        "runtime_identity_reconciliation_sha256": _sha256_file(
            reconciliation_path
        ),
    }
    reduced_result = {
        "schema_version": 2,
        "stage": "applied_interaction",
        "records": len(records),
        "contract_hash": args.contract_hash,
        "parent_hash": args.expected_parent_hash,
        "artifact_hashes": artifact_hashes,
        "status": "completed",
        "scientific_reduction": scientific_reduction,
    }
    reduced_result_path = output_root / "reduced_result.json"
    _write_json(reduced_result_path, reduced_result)
    reducer_output_hash = _stable_hash(
        {
            "stage": "applied_interaction",
            "job_count": len(records),
            "parent_hash": args.expected_parent_hash,
            "source_hash": records[0]["source_hash"],
            "config_hash": records[0]["config_hash"],
            "lock_hash": records[0]["lock_hash"],
        }
    )
    marker = {
        "schema_version": 2,
        "stage": "applied_interaction",
        "status": "completed",
        "output_hash": reducer_output_hash,
        "artifact_sha256": _sha256_file(reduced_result_path),
    }
    _write_json(output_root / "_SUCCESS.json", marker)
    print(json.dumps(marker, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    diagnostic = subparsers.add_parser("diagnose")
    diagnostic.add_argument("--manifest", required=True)
    diagnostic.add_argument("--task-id", type=int, required=True)
    diagnostic.add_argument("--n-jobs", type=int, required=True)
    diagnostic.add_argument("--output-root", required=True)
    diagnostic.set_defaults(func=diagnose)

    reduction = subparsers.add_parser("reconcile")
    reduction.add_argument("--manifest", required=True)
    reduction.add_argument("--results-root", required=True)
    reduction.add_argument("--audit-success", required=True)
    reduction.add_argument("--contract-config", required=True)
    reduction.add_argument("--contract-hash", required=True)
    reduction.add_argument("--expected-parent-hash", required=True)
    reduction.add_argument("--expected-tasks", type=int, required=True)
    reduction.add_argument("--reference-diagnostic", required=True)
    reduction.add_argument("--candidate-diagnostic", required=True)
    reduction.add_argument("--output-root", required=True)
    reduction.set_defaults(func=reconcile)
    return parser


def main() -> int:
    args = _parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
