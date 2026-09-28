#!/usr/bin/env python3
"""Re-evaluate immutable fixed-family scores with the frozen detector partitions."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_adapter(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("_g11_partition_adapter", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load adapter from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    """Run the immutable-artifact diagnostic and write one hashed result."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--contract-config", type=Path, required=True)
    parser.add_argument("--source-inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tree-alpha", type=float)
    parser.add_argument("--binary-binary-alpha", type=float)
    parser.add_argument(
        "--analysis-status",
        choices=("wiring_diagnostic", "development_only"),
        default="wiring_diagnostic",
    )
    args = parser.parse_args()

    from rfm_pipeline.campaign_contract import load_contract, wilson_upper_bound

    adapter = _load_adapter(args.adapter)
    contract, contract_hash = load_contract(args.contract_config)
    tree_alpha = (
        contract.tree_family_alpha if args.tree_alpha is None else float(args.tree_alpha)
    )
    binary_binary_alpha = (
        contract.binary_binary_family_alpha
        if args.binary_binary_alpha is None
        else float(args.binary_binary_alpha)
    )
    uses_contract_alphas = (
        tree_alpha == contract.tree_family_alpha
        and binary_binary_alpha == contract.binary_binary_family_alpha
    )
    if not uses_contract_alphas and args.analysis_status != "development_only":
        raise ValueError("alpha overrides are permitted only for development_only analysis")
    reference_pairs: tuple[str, ...] | None = None
    reference_detectors: tuple[str, ...] | None = None
    selection_rows: list[dict[str, Any]] = []
    artifact_hashes: list[dict[str, Any]] = []
    for index in range(contract.fixed_family_replicates):
        task_dir = args.result_root / f"task-{index:04d}"
        success_path = task_dir / "_SUCCESS.json"
        terminal_path = task_dir / "terminal_record.json"
        artifact_dir = task_dir / "interaction_score_blocks"
        metadata_path = artifact_dir / "score_only_interaction.json"
        npz_path = artifact_dir / "score_only_interaction.npz"
        for required in (success_path, terminal_path, metadata_path, npz_path):
            if not required.is_file():
                raise ValueError(f"missing immutable fixed-family artifact: {required}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        npz_sha256 = _sha256(npz_path)
        if metadata.get("npz_sha256") != npz_sha256:
            raise ValueError(f"task {index:04d} score NPZ hash mismatch")
        pair_order = tuple(str(value) for value in metadata["pair_names"])
        detectors = tuple(
            str(value)
            for value in metadata["control_snapshot"]["candidate_pair_detectors"]
        )
        if reference_pairs is None:
            reference_pairs = pair_order
            reference_detectors = detectors
        elif pair_order != reference_pairs or detectors != reference_detectors:
            raise ValueError(f"task {index:04d} candidate family identity drift")
        with np.load(npz_path, allow_pickle=False) as arrays:
            selected = adapter._select_fixed_family_pairs(
                observed_scores=arrays["observed_scores"],
                null_scores=arrays["null_scores"],
                pair_order=pair_order,
                pair_detectors=detectors,
                family_sizes=contract.fixed_family_sizes,
                tree_family_alpha=tree_alpha,
                binary_binary_family_alpha=binary_binary_alpha,
            )
        selection_rows.append(
            {
                "global_null_replicate_index": index,
                "selected_pairs": selected,
            }
        )
        artifact_hashes.append(
            {
                "global_null_replicate_index": index,
                "metadata_sha256": _sha256(metadata_path),
                "npz_sha256": npz_sha256,
                "success_sha256": _sha256(success_path),
                "terminal_sha256": _sha256(terminal_path),
            }
        )

    summaries: dict[str, dict[str, Any]] = {}
    passed = True
    for family_size in contract.fixed_family_sizes:
        key = str(family_size)
        event_rows = [row for row in selection_rows if row["selected_pairs"][key]]
        events = len(event_rows)
        upper = wilson_upper_bound(
            events,
            contract.fixed_family_replicates,
            contract.calibration_confidence,
        )
        family_passed = upper <= contract.gate_value
        passed &= family_passed
        summaries[key] = {
            "denominator": contract.fixed_family_replicates,
            "false_selection_events": events,
            "event_replicate_indices": [
                row["global_null_replicate_index"] for row in event_rows
            ],
            "selected_pairs_by_event": [row["selected_pairs"][key] for row in event_rows],
            "fwer": events / contract.fixed_family_replicates,
            "one_sided_wilson_upper": upper,
            "gate_value": contract.gate_value,
            "passes_calibration": family_passed,
        }

    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS" if passed else "FAIL",
        "analysis_status": args.analysis_status,
        "confirmatory_evidence": False,
        "operation": "fixed_family_partition_diagnostic",
        "contract_hash": contract_hash,
        "contract_config_sha256": _sha256(args.contract_config),
        "source_inventory_sha256": _sha256(args.source_inventory),
        "adapter_sha256": _sha256(args.adapter),
        "family_partition_method": contract.family_partition_method,
        "contract_tree_family_alpha": contract.tree_family_alpha,
        "contract_binary_binary_family_alpha": contract.binary_binary_family_alpha,
        "evaluated_tree_family_alpha": tree_alpha,
        "evaluated_binary_binary_family_alpha": binary_binary_alpha,
        "candidate_pair_count": len(reference_pairs or ()),
        "candidate_pair_detectors": list(reference_detectors or ()),
        "terminal_record_count": len(selection_rows),
        "families": summaries,
        "artifact_hashes": artifact_hashes,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["payload_sha256"] = hashlib.sha256(canonical).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
