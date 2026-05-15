"""Tests for HPC bundle manifest creation and zip analysis utilities."""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path

from tools.hpc_bundle_manifest import analyze_zip, create_run_manifest, write_study_metadata


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


def test_write_study_metadata_writes_manifest_and_inventory(tmp_path: Path) -> None:
    bundle_root = tmp_path / "bundle"
    sample_file = (
        bundle_root
        / "runs"
        / "cpu_2"
        / "run_artifacts"
        / "final_manuscript_artifacts"
        / "figures"
        / "a.svg"
    )
    sample_file.parent.mkdir(parents=True, exist_ok=True)
    sample_file.write_text("<svg/>", encoding="utf-8")
    retained_terms = (
        bundle_root
        / "runs"
        / "cpu_2"
        / "run_artifacts"
        / "empirical_null_screen"
        / "retained_terms.csv"
    )
    retained_terms.parent.mkdir(parents=True, exist_ok=True)
    retained_terms.write_text("feature_name\nf1\nf2\n", encoding="utf-8")
    submit_script = (
        bundle_root / "runs" / "cpu_2" / "hpc_scripts" / "submit_interaction_discovery_array.sh"
    )
    submit_script.parent.mkdir(parents=True, exist_ok=True)
    submit_script.write_text(
        "\n".join(
            [
                "#!/bin/bash",
                "#SBATCH --account=bsm",
                "pixi run python tools/hpc_shard_worker.py \\",
                "  --manifest manifest.jsonl \\",
                "  --task-id 0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    run_started = bundle_root / "runs" / "cpu_2" / "run_artifacts" / "run_started.json"
    run_started.parent.mkdir(parents=True, exist_ok=True)
    run_started.write_text(
        json.dumps(
            {"start_stage": "output_conditioning", "stop_stage": "final_manuscript_artifacts"}
        )
        + "\n",
        encoding="utf-8",
    )
    env_dir = bundle_root / "manifest" / "environment"
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / "pixi.lock").write_text("lock", encoding="utf-8")

    target_specs = {
        "targets": [
            {
                "target": "cpu_2",
                "run_dir": (
                    "/scratch/alice/bsm/bsm-public-rf/artifacts/kestrel_cpu_scale_2_smoke_run"
                ),
                "log_dir": "/scratch/alice/bsm/bsm_kestrel_cpu_scale_2_smoke/logs",
                "suite_manifest_path": "",
                "gpu_mode": False,
                "config_path": (
                    "/home/alice/src/bsm-public-rf/configs/hpc/kestrel_cpu_scale_2_smoke.yml"
                ),
            }
        ]
    }
    target_specs_path = tmp_path / "target_specs.json"
    target_specs_path.write_text(json.dumps(target_specs), encoding="utf-8")

    out_json = tmp_path / "manifest" / "study_metadata_manifest.json"
    out_csv = tmp_path / "manifest" / "study_file_inventory.csv"
    out_commands = tmp_path / "manifest" / "commands.json"
    out_recipe = tmp_path / "manifest" / "reproduction_recipe.md"
    args = argparse.Namespace(
        bundle_root=str(bundle_root),
        hpc_repo_root=str(tmp_path),
        target_specs_json=str(target_specs_path),
        pullback_mode="study_package",
        output_json=str(out_json),
        output_csv=str(out_csv),
        commands_json=str(out_commands),
        reproduction_recipe_md=str(out_recipe),
    )
    write_study_metadata(args)

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["pullback_mode"] == "study_package"
    assert payload["inventory"]["n_files"] == 5
    assert payload["figure_assets"]
    assert payload["stage_metrics"]["cpu_2"]["retained_counts"]["n_retained_first_order_terms"] == 2
    assert payload["execution_trace"]["n_targets"] == 1
    assert payload["execution_trace"]["n_traced_commands"] == 1
    csv_text = out_csv.read_text(encoding="utf-8")
    assert "relative_path,size_bytes,sha256" in csv_text
    commands_payload = json.loads(out_commands.read_text(encoding="utf-8"))
    target_trace = commands_payload["targets"]["cpu_2"]
    assert target_trace["submission_scripts"]
    assert target_trace["run_started"]["start_stage"] == "output_conditioning"
    assert (
        "pixi run python tools/hpc_shard_worker.py --manifest manifest.jsonl --task-id 0"
        in (commands_payload["all_commands"])
    )
    recipe_text = out_recipe.read_text(encoding="utf-8")
    assert "manifest/environment/pixi.lock" in recipe_text
    assert "Target `cpu_2`" in recipe_text
