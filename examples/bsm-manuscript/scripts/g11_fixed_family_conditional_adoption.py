"""Seal a disclosed fixed-family deviation without relabeling the frozen gate PASS."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


_PRESPECIFIED_GATE = 0.09
_REPORTING_CEILING = 0.10
_CALIBRATION_CONFIDENCE = 0.95
_EXPECTED_REPLICATES = 300
_EXPECTED_FAMILIES = (10, 100, 1000)
_REDUCER_FAILURE = "fixed-family supplement failed its prespecified calibration rule"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_hash(payload: Any) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"required adoption input is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"adoption JSON input is not an object: {path}")
    return payload


def _inventory_hash(rows: list[dict[str, str]]) -> str:
    return _stable_hash(sorted(rows, key=lambda row: row["shard_id"]))


def build_fixed_family_conditional_adoption(
    *,
    contract_path: str | Path,
    manifest_path: str | Path,
    audit_log_path: str | Path,
    reducer_log_path: str | Path,
    user_direction: str,
) -> dict[str, Any]:
    """Recompute the completed supplement and return a fail-preserving adoption."""
    from rfm_pipeline.campaign_contract import load_contract, wilson_upper_bound

    contract_file = Path(contract_path).resolve()
    manifest_file = Path(manifest_path).resolve()
    audit_log = Path(audit_log_path).resolve()
    reducer_log = Path(reducer_log_path).resolve()
    contract, contract_hash = load_contract(contract_file)
    if (
        contract.fixed_family_replicates != _EXPECTED_REPLICATES
        or tuple(contract.fixed_family_sizes) != _EXPECTED_FAMILIES
        or not math.isclose(
            float(contract.calibration_confidence),
            _CALIBRATION_CONFIDENCE,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or not math.isclose(
            float(contract.gate_value), _PRESPECIFIED_GATE, rel_tol=0.0, abs_tol=0.0
        )
    ):
        raise ValueError("fixed-family adoption requires the exact generation-12 contract")
    if not isinstance(user_direction, str) or not user_direction.strip():
        raise ValueError("fixed-family adoption requires nonempty user direction")
    if not manifest_file.is_file():
        raise ValueError("fixed-family stage manifest is missing")
    manifest = [
        json.loads(line)
        for line in manifest_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(manifest) != _EXPECTED_REPLICATES:
        raise ValueError("fixed-family manifest does not contain exactly 300 records")
    expected_shards = {f"task-{index:04d}" for index in range(_EXPECTED_REPLICATES)}
    by_shard = {str(row.get("shard_id", "")): row for row in manifest}
    if set(by_shard) != expected_shards or any(
        row.get("stage") != "fixed_family_supplement"
        or row.get("config_hash") != contract_hash
        for row in manifest
    ):
        raise ValueError("fixed-family manifest identity or shard coverage differs")

    payloads: list[dict[str, Any]] = []
    result_inventory: list[dict[str, str]] = []
    success_inventory: list[dict[str, str]] = []
    for shard_id in sorted(expected_shards):
        output_dir = Path(str(by_shard[shard_id].get("output_dir", ""))).resolve()
        result_path = output_dir / "result.json"
        marker_path = output_dir / "_SUCCESS.json"
        if not result_path.is_file() or not marker_path.is_file():
            raise ValueError("fixed-family adoption requires exactly 300 completed results")
        result = _read_object(result_path)
        marker = _read_object(marker_path)
        result_sha = _sha256(result_path)
        if (
            marker.get("stage") != "fixed_family_supplement"
            or marker.get("shard_id") != shard_id
            or marker.get("status") != "completed"
            or marker.get("artifact_sha256") != result_sha
            or result.get("operation") != "fixed_family_supplement"
            or result.get("status") != "completed"
            or result.get("config_hash") != contract_hash
            or set(map(str, result.get("selected_pairs", {})))
            != {str(value) for value in _EXPECTED_FAMILIES}
        ):
            raise ValueError(f"fixed-family result or success marker differs for {shard_id}")
        payloads.append(result)
        result_inventory.append({"shard_id": shard_id, "sha256": result_sha})
        success_inventory.append({"shard_id": shard_id, "sha256": _sha256(marker_path)})
    indices = [int(row.get("global_null_replicate_index", -1)) for row in payloads]
    if sorted(indices) != list(range(_EXPECTED_REPLICATES)):
        raise ValueError("fixed-family replicate coverage differs")

    families: dict[str, dict[str, Any]] = {}
    for family_size in _EXPECTED_FAMILIES:
        family = str(family_size)
        events = sum(bool(row["selected_pairs"][family]) for row in payloads)
        upper = wilson_upper_bound(events, _EXPECTED_REPLICATES, _CALIBRATION_CONFIDENCE)
        families[family] = {
            "denominator": _EXPECTED_REPLICATES,
            "false_selection_events": events,
            "fwer": events / _EXPECTED_REPLICATES,
            "one_sided_wilson_upper": upper,
            "passes_prespecified_calibration": upper <= _PRESPECIFIED_GATE,
            "passes_reporting_ceiling": upper <= _REPORTING_CEILING,
        }
    if all(row["passes_prespecified_calibration"] for row in families.values()):
        raise ValueError("fixed-family adoption cannot replace a passing prespecified reducer")
    if not all(row["passes_reporting_ceiling"] for row in families.values()):
        raise ValueError("fixed-family result exceeds the disclosed reporting ceiling")

    audit = _read_object(audit_log)
    if (
        audit.get("stage") != "fixed_family_supplement"
        or audit.get("status") != "completed"
        or re.fullmatch(r"[0-9a-f]{64}", str(audit.get("artifact_sha256", ""))) is None
    ):
        raise ValueError("fixed-family audit did not complete with a valid artifact")
    reducer_text = reducer_log.read_text(encoding="utf-8")
    if _REDUCER_FAILURE not in reducer_text:
        raise ValueError("fixed-family reducer log does not contain the prespecified failure")

    identity: dict[str, Any] = {
        "schema_version": 1,
        "operation": "fixed_family_supplement",
        "status": "completed",
        "decision": "CONDITIONALLY_ADOPTED",
        "contract_hash": contract_hash,
        "terminal_record_count": _EXPECTED_REPLICATES,
        "prespecified_decision": "FAIL",
        "prespecified_gate_value": _PRESPECIFIED_GATE,
        "reporting_ceiling": _REPORTING_CEILING,
        "calibration_confidence": _CALIBRATION_CONFIDENCE,
        "families": families,
        "downstream_execution_permitted": True,
        "decision_authority": "user",
        "user_direction": user_direction.strip(),
        "deviation_disclosure_required": True,
        "publication_language": (
            "The prespecified 0.09 fixed-family confidence-bound criterion failed; "
            "the completed supplement was conditionally adopted under an explicitly "
            "post-hoc 0.10 reporting ceiling, with the deviation and exact estimates disclosed."
        ),
        "source_stage_manifest_sha256": _sha256(manifest_file),
        "source_result_inventory_sha256": _inventory_hash(result_inventory),
        "source_success_inventory_sha256": _inventory_hash(success_inventory),
        "source_audit_artifact_sha256": str(audit["artifact_sha256"]),
        "source_audit_log_sha256": _sha256(audit_log),
        "source_reducer_failure_sha256": _sha256(reducer_log),
    }
    return {**identity, "adoption_sha256": _stable_hash(identity)}


def write_fixed_family_conditional_adoption(
    *,
    contract_path: str | Path,
    manifest_path: str | Path,
    audit_log_path: str | Path,
    reducer_log_path: str | Path,
    output_path: str | Path,
    user_direction: str,
) -> dict[str, Any]:
    """Write one new immutable conditional-adoption record."""
    output = Path(output_path).resolve()
    if output.exists():
        raise ValueError("fixed-family conditional-adoption output already exists")
    adoption = build_fixed_family_conditional_adoption(
        contract_path=contract_path,
        manifest_path=manifest_path,
        audit_log_path=audit_log_path,
        reducer_log_path=reducer_log_path,
        user_direction=user_direction,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(adoption, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return adoption


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--audit-log", required=True)
    parser.add_argument("--reducer-log", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--user-direction", required=True)
    args = parser.parse_args(argv)
    adoption = write_fixed_family_conditional_adoption(
        contract_path=args.contract,
        manifest_path=args.manifest,
        audit_log_path=args.audit_log,
        reducer_log_path=args.reducer_log,
        output_path=args.output,
        user_direction=args.user_direction,
    )
    print(json.dumps(adoption, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
