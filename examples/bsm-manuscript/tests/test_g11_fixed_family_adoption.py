"""Generation-12 fixed-family conditional-adoption evidence."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = _ROOT / "scripts" / "g11_fixed_family_conditional_adoption.py"
    spec = importlib.util.spec_from_file_location("g11_fixed_family_conditional_adoption", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    from rfm_pipeline.campaign_contract import load_contract

    contract_path = _ROOT / "configs" / "g11_campaign_contract.toml"
    _, contract_hash = load_contract(contract_path)
    results = tmp_path / "results"
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_rows = []
    event_sets = {
        "10": set(range(9)),
        "100": set(range(13)),
        "1000": set(range(20)),
    }
    for index in range(300):
        shard = results / f"task-{index:04d}"
        shard.mkdir(parents=True)
        result = {
            "operation": "fixed_family_supplement",
            "status": "completed",
            "global_null_replicate_index": index,
            "config_hash": contract_hash,
            "selected_pairs": {
                family: ([f"x{index}:y{index}"] if index in events else [])
                for family, events in event_sets.items()
            },
        }
        result_path = shard / "result.json"
        result_path.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
        marker = {
            "schema_version": 2,
            "stage": "fixed_family_supplement",
            "shard_id": shard.name,
            "artifact_sha256": _sha256(result_path),
            "status": "completed",
        }
        (shard / "_SUCCESS.json").write_text(
            json.dumps(marker, sort_keys=True), encoding="utf-8"
        )
        manifest_rows.append(
            {
                "stage": "fixed_family_supplement",
                "shard_id": shard.name,
                "output_dir": str(shard),
                "config_hash": contract_hash,
            }
        )
    manifest_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )
    audit_log = tmp_path / "audit.out"
    audit_log.write_text(
        json.dumps(
            {
                "artifact_sha256": "6a2d78f99569cd29c9b1af4ffc74c321b36d64a406874fa2df1d72eed459107f",
                "stage": "fixed_family_supplement",
                "status": "completed",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    reducer_log = tmp_path / "reducer.out"
    reducer_log.write_text(
        "ValueError: fixed-family supplement failed its prespecified calibration rule\n",
        encoding="utf-8",
    )
    return contract_path, manifest_path, audit_log, reducer_log


def test_conditional_adoption_preserves_fail_and_recomputes_exact_metrics(
    tmp_path: Path,
) -> None:
    module = _load_module()
    contract, manifest, audit, reducer = _fixture(tmp_path)
    output = tmp_path / "fixed_family_conditional_adoption.json"

    adoption = module.write_fixed_family_conditional_adoption(
        contract_path=contract,
        manifest_path=manifest,
        audit_log_path=audit,
        reducer_log_path=reducer,
        output_path=output,
        user_direction=(
            "Validate the completed stage without further HPC work and proceed with disclosure."
        ),
    )

    assert adoption["decision"] == "CONDITIONALLY_ADOPTED"
    assert adoption["prespecified_decision"] == "FAIL"
    assert adoption["prespecified_gate_value"] == pytest.approx(0.09)
    assert adoption["reporting_ceiling"] == pytest.approx(0.10)
    assert adoption["terminal_record_count"] == 300
    assert {
        family: row["false_selection_events"]
        for family, row in adoption["families"].items()
    } == {"10": 9, "100": 13, "1000": 20}
    assert adoption["families"]["1000"]["one_sided_wilson_upper"] == pytest.approx(
        0.094438175549846
    )
    assert adoption["families"]["1000"]["passes_prespecified_calibration"] is False
    assert adoption["families"]["1000"]["passes_reporting_ceiling"] is True
    identity = {key: value for key, value in adoption.items() if key != "adoption_sha256"}
    expected = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert adoption["adoption_sha256"] == expected
    assert json.loads(output.read_text(encoding="utf-8")) == adoption


def test_conditional_adoption_rejects_missing_success(tmp_path: Path) -> None:
    module = _load_module()
    contract, manifest, audit, reducer = _fixture(tmp_path)
    marker = tmp_path / "results" / "task-0299" / "_SUCCESS.json"
    marker.unlink()

    with pytest.raises(ValueError, match="exactly 300 completed results"):
        module.write_fixed_family_conditional_adoption(
            contract_path=contract,
            manifest_path=manifest,
            audit_log_path=audit,
            reducer_log_path=reducer,
            output_path=tmp_path / "adoption.json",
            user_direction="Proceed without additional HPC work.",
        )


def test_conditional_adoption_refuses_a_result_above_reporting_ceiling(
    tmp_path: Path,
) -> None:
    module = _load_module()
    contract, manifest, audit, reducer = _fixture(tmp_path)
    for index in range(20, 24):
        result_path = tmp_path / "results" / f"task-{index:04d}" / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["selected_pairs"]["1000"] = [f"x{index}:y{index}"]
        result_path.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
        marker_path = result_path.with_name("_SUCCESS.json")
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker["artifact_sha256"] = _sha256(result_path)
        marker_path.write_text(json.dumps(marker, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="reporting ceiling"):
        module.write_fixed_family_conditional_adoption(
            contract_path=contract,
            manifest_path=manifest,
            audit_log_path=audit,
            reducer_log_path=reducer,
            output_path=tmp_path / "adoption.json",
            user_direction="Proceed without additional HPC work.",
        )
