#!/usr/bin/env python3
"""Development-only strong-signal power check for amended detector alphas."""

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
    spec = importlib.util.spec_from_file_location("_g11_power_adapter", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load adapter from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_complete_score(
    task_dir: Path,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], tuple[str, ...]]:
    block_dirs = sorted((task_dir / "pipeline" / "interaction_score_blocks").glob("block-*"))
    if not block_dirs:
        raise ValueError(f"missing interaction score blocks: {task_dir}")
    observed: np.ndarray | None = None
    null_blocks: list[np.ndarray] = []
    expected_start = 0
    pair_order: tuple[str, ...] | None = None
    detectors: tuple[str, ...] | None = None
    for block_dir in block_dirs:
        metadata_path = block_dir / "score_only_interaction.json"
        npz_path = block_dir / "score_only_interaction.npz"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("npz_sha256") != _sha256(npz_path):
            raise ValueError(f"score block hash mismatch: {npz_path}")
        start = int(metadata["draw_range_start"])
        end = int(metadata["draw_range_end"])
        if start != expected_start or end <= start:
            raise ValueError(f"noncontiguous interaction score blocks: {task_dir}")
        block_pairs = tuple(str(value) for value in metadata["pair_names"])
        block_detectors = tuple(
            str(value)
            for value in metadata["control_snapshot"]["candidate_pair_detectors"]
        )
        if pair_order is None:
            pair_order = block_pairs
            detectors = block_detectors
        elif block_pairs != pair_order or block_detectors != detectors:
            raise ValueError(f"interaction score identity drift: {task_dir}")
        with np.load(npz_path, allow_pickle=False) as arrays:
            if start == 0:
                observed = np.asarray(arrays["observed_scores"], dtype=float)
            elif arrays["observed_scores"].size:
                raise ValueError(
                    "only the first score block may contain observed scores: "
                    f"{task_dir}"
                )
            null_blocks.append(np.asarray(arrays["null_scores"], dtype=float))
        expected_start = end
    if observed is None or pair_order is None or detectors is None:
        raise ValueError(f"incomplete interaction scores: {task_dir}")
    return observed, np.concatenate(null_blocks, axis=0), pair_order, detectors


def main() -> int:
    """Evaluate strong-signal power at development-only detector alphas."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--tree-alpha", type=float, required=True)
    parser.add_argument("--binary-binary-alpha", type=float, required=True)
    parser.add_argument("--power-lower-bound", type=float, required=True)
    parser.add_argument("--source-completion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    adapter = _load_adapter(args.adapter)
    scenarios = ("strong_cc", "strong_bc", "strong_bb")
    rows: dict[str, list[dict[str, Any]]] = {scenario: [] for scenario in scenarios}
    for terminal_path in sorted(args.result_root.glob("task-*/terminal_record.json")):
        terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
        scenario = str(terminal.get("scenario", ""))
        if scenario not in rows:
            continue
        observed, null, pair_order, detectors = _load_complete_score(terminal_path.parent)
        selected = adapter._select_fixed_family_pairs(
            observed_scores=observed,
            null_scores=null,
            pair_order=pair_order,
            pair_detectors=detectors,
            family_sizes=(len(pair_order),),
            tree_family_alpha=args.tree_alpha,
            binary_binary_family_alpha=args.binary_binary_alpha,
        )[str(len(pair_order))]
        truth = tuple(str(value) for value in terminal["truth_interaction_ids"])
        canonical_truth = adapter._canonical_support(
            truth,
            field="truth_interaction_ids",
        )
        canonical_selected = adapter._canonical_support(
            selected,
            field="selected_interaction_ids",
        )
        rows[scenario].append(
            {
                "replicate_index": int(terminal["replicate_index"]),
                "truth_interaction_ids": sorted(canonical_truth),
                "selected_interaction_ids": sorted(canonical_selected),
                "planted_interaction_discovered": canonical_truth <= canonical_selected,
            }
        )

    summaries: dict[str, dict[str, Any]] = {}
    passed = True
    for scenario in scenarios:
        ordered = sorted(rows[scenario], key=lambda row: row["replicate_index"])
        if [row["replicate_index"] for row in ordered] != list(range(200)):
            raise ValueError(f"{scenario} lacks exact 200-replicate coverage")
        events = sum(bool(row["planted_interaction_discovered"]) for row in ordered)
        lower = adapter._wilson_lower_bound(events, len(ordered), 0.95)
        scenario_passed = lower >= args.power_lower_bound
        passed &= scenario_passed
        summaries[scenario] = {
            "denominator": len(ordered),
            "discovery_events": events,
            "power": events / len(ordered),
            "one_sided_wilson_lower": lower,
            "power_gate_lower_bound": args.power_lower_bound,
            "passes_power": scenario_passed,
            "failure_replicate_indices": [
                row["replicate_index"]
                for row in ordered
                if not row["planted_interaction_discovered"]
            ],
        }

    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS" if passed else "FAIL",
        "analysis_status": "development_only",
        "confirmatory_evidence": False,
        "operation": "gate_b_amended_alpha_power_development",
        "evaluated_tree_family_alpha": args.tree_alpha,
        "evaluated_binary_binary_family_alpha": args.binary_binary_alpha,
        "source_completion_sha256": _sha256(args.source_completion),
        "adapter_sha256": _sha256(args.adapter),
        "scenarios": summaries,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["payload_sha256"] = hashlib.sha256(canonical).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
