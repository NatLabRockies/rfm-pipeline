"""Tests for HPC bundle manifest creation and zip analysis utilities."""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path

from tools.hpc_bundle_manifest import analyze_zip, create_run_manifest


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_create_run_manifest_writes_status_rows(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    logs_root = tmp_path / "logs"
    suite_root = artifacts_root / "kestrel_cpu_scaling_suite"
    cpu2_run = artifacts_root / "kestrel_cpu_scale_2_run"
    cpu2_shard = cpu2_run / "hpc_shards" / "task-0000"
    cpu2_merged = cpu2_run / "hpc_shards" / "_merged"
    cpu2_scripts = cpu2_run / "hpc_scripts"

    cpu2_scripts.mkdir(parents=True, exist_ok=True)
    (cpu2_scripts / "manifest.jsonl").write_text('{"shard_id":"task-0000"}\n', encoding="utf-8")
    cpu2_shard.mkdir(parents=True, exist_ok=True)
    (cpu2_shard / "retained_interaction_pairs.csv").write_text(
        "pair,score\na,1\n",
        encoding="utf-8",
    )
    (cpu2_shard / "shard_result.json").write_text(
        json.dumps({"status": "completed"}) + "\n", encoding="utf-8"
    )
    cpu2_merged.mkdir(parents=True, exist_ok=True)
    (cpu2_merged / "interaction_discovery_merged.json").write_text("{}", encoding="utf-8")
    (cpu2_merged / "retained_interaction_pairs_merged.csv").write_text(
        "pair,score\na,1\n", encoding="utf-8"
    )
    (cpu2_merged / "interaction_pair_scores_merged.csv").write_text(
        "pair,score\na,1\n", encoding="utf-8"
    )
    cpu2_log_dir = logs_root / "bsm_kestrel_cpu_scale_2" / "logs"
    cpu2_log_dir.mkdir(parents=True, exist_ok=True)
    (cpu2_log_dir / "bsm_interaction_discovery_123_0.out").write_text("ok\n", encoding="utf-8")

    out_json = tmp_path / "manifest" / "hpc_run_manifest.json"
    out_csv = tmp_path / "manifest" / "run_summary.csv"
    args = argparse.Namespace(
        hpc_repo_root="/projects/bsm/bsm-public-rf",
        artifacts_root=str(artifacts_root),
        logs_root=str(logs_root),
        suite_root=str(suite_root),
        pullback_mode="reporting_bundle",
        output_json=str(out_json),
        output_csv=str(out_csv),
    )
    create_run_manifest(args)

    rows = list(csv.DictReader(out_csv.read_text(encoding="utf-8").splitlines()))
    cpu2_row = next(row for row in rows if row["target"] == "cpu_2")
    assert cpu2_row["status"] == "complete"
    assert cpu2_row["health"] == "LOOKS_ACTIVE"
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["pullback_mode"] == "reporting_bundle"
    assert payload["aggregate"]["n_targets"] == 4


def test_analyze_zip_reads_manifest_summary(tmp_path: Path) -> None:
    zip_path = tmp_path / "bundle.zip"
    csv_content = (
        "target,run_dir,manifest_path,manifest_shards,shard_results,completed_shards,failed_shards,"
        "merged_retained_pairs,merged_pair_scores,status,health,log_dir,latest_array_log,latest_reduce_log\n"
        "cpu_10,/a,/a/manifest.jsonl,10,8,8,0,100,100,running_or_waiting_reduce,LOOKS_ACTIVE,/logs,a.out,r.out\n"
    )
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest/run_summary.csv", csv_content)
    out_csv = tmp_path / "analysis.csv"
    out_txt = tmp_path / "analysis.txt"
    args = argparse.Namespace(zip=str(zip_path), out_csv=str(out_csv), out_txt=str(out_txt))
    analyze_zip(args)
    assert "cpu_10" in out_csv.read_text(encoding="utf-8")
    text = out_txt.read_text(encoding="utf-8")
    assert "HPC artifact analysis (manifest-driven)" in text
    assert "LOOKS_ACTIVE" in text
