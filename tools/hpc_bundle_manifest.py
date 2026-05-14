#!/usr/bin/env python
"""Build and consume HPC run manifests used by artifact pullback workflows."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class TargetSummary:
    """Per-target summary row for CPU/GPU HPC runs."""

    target: str
    run_dir: str
    manifest_path: str
    manifest_shards: int
    shard_results: int
    completed_shards: int
    failed_shards: int
    merged_retained_pairs: int
    merged_pair_scores: int
    status: str
    health: str
    log_dir: str
    latest_array_log: str
    latest_reduce_log: str


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _classify(status: str, failed_shards: int) -> str:
    if failed_shards > 0 or status == "failed":
        return "ATTENTION"
    if status in {"complete", "running_or_waiting_reduce", "queued_or_pending"}:
        return "LOOKS_ACTIVE"
    return "NO_SIGNAL"


def _count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8", errors="replace") as handle:
        lines = [line for line in handle.read().splitlines() if line.strip()]
    return max(0, len(lines) - 1)


def _latest_path(pattern: str) -> str:
    matches = sorted(Path(p) for p in Path().glob(pattern))
    if not matches:
        return ""
    matches.sort(key=lambda p: p.stat().st_mtime)
    return str(matches[-1])


def _latest_glob(pattern: str) -> str:
    import glob

    matches = glob.glob(pattern)
    if not matches:
        return ""
    matches.sort(key=lambda p: os.path.getmtime(p))
    return matches[-1]


def _shard_status_counts(shards_root: Path) -> tuple[int, int, int, int]:
    shard_result_paths = list(shards_root.glob("*/shard_result.json"))
    shard_results = len(shard_result_paths)
    completed = 0
    failed = 0
    for path in shard_result_paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        status = str(payload.get("status", "")).lower().strip()
        if status == "completed":
            completed += 1
        elif status == "failed":
            failed += 1
    retained = len(list(shards_root.glob("*/retained_interaction_pairs.csv")))
    if retained > completed:
        completed = retained
    if retained > shard_results:
        shard_results = retained
    return shard_results, completed, failed, retained


def _resolve_manifest_path(run_dir: Path, suite_fallback: Path | None) -> Path:
    primary = run_dir / "hpc_scripts" / "manifest.jsonl"
    if primary.exists():
        return primary
    if suite_fallback and suite_fallback.exists():
        return suite_fallback
    return primary


def _summarize_target(
    *,
    target: str,
    run_dir: Path,
    log_dir: Path,
    suite_manifest_fallback: Path | None = None,
    gpu_mode: bool = False,
) -> TargetSummary:
    manifest_path = _resolve_manifest_path(run_dir, suite_manifest_fallback)
    manifest_shards = 0
    if manifest_path.exists():
        manifest_lines = manifest_path.read_text(encoding="utf-8").splitlines()
        manifest_shards = len([line for line in manifest_lines if line.strip()])
    shards_root = run_dir / "hpc_shards"
    shard_results = 0
    completed = 0
    failed = 0
    if shards_root.exists():
        shard_results, completed, failed, _ = _shard_status_counts(shards_root)
    merged_root = shards_root / "_merged"
    merged_json = merged_root / "interaction_discovery_merged.json"
    merged_retained = merged_root / "retained_interaction_pairs_merged.csv"
    merged_scores = merged_root / "interaction_pair_scores_merged.csv"
    merged_retained_rows = _count_csv_rows(merged_retained)
    merged_score_rows = _count_csv_rows(merged_scores)

    if gpu_mode:
        array_log = _latest_glob(str(log_dir / "bsm_gpu_interaction_discovery_*.out"))
        if not array_log:
            array_log = _latest_glob(str(log_dir / "bsm_interaction_discovery_*.out"))
    else:
        array_log = _latest_glob(str(log_dir / "bsm_interaction_discovery_*.out"))
    reduce_log = _latest_glob(str(log_dir / "bsm_reduce_interaction_discovery_*.out"))

    if manifest_shards == 0 and shard_results == 0:
        status = "scripts_missing"
    elif failed > 0:
        status = "failed"
    elif merged_json.exists():
        status = "complete"
    elif shard_results > 0:
        status = "running_or_waiting_reduce"
    else:
        status = "queued_or_pending"

    return TargetSummary(
        target=target,
        run_dir=str(run_dir),
        manifest_path=str(manifest_path),
        manifest_shards=manifest_shards,
        shard_results=shard_results,
        completed_shards=completed,
        failed_shards=failed,
        merged_retained_pairs=merged_retained_rows,
        merged_pair_scores=merged_score_rows,
        status=status,
        health=_classify(status, failed),
        log_dir=str(log_dir),
        latest_array_log=array_log,
        latest_reduce_log=reduce_log,
    )


def create_run_manifest(args: argparse.Namespace) -> int:
    """Create manifest JSON + CSV from live HPC run directories."""
    artifacts_root = Path(args.artifacts_root)
    logs_root = Path(args.logs_root)
    suite_root = Path(args.suite_root)

    rows: list[TargetSummary] = []
    for tier in (2, 10, 1000):
        rows.append(
            _summarize_target(
                target=f"cpu_{tier}",
                run_dir=artifacts_root / f"kestrel_cpu_scale_{tier}_run",
                suite_manifest_fallback=(
                    suite_root / f"cpu_nodes_{tier}" / "hpc_scripts" / "manifest.jsonl"
                ),
                log_dir=logs_root / f"bsm_kestrel_cpu_scale_{tier}" / "logs",
                gpu_mode=False,
            )
        )
    rows.append(
        _summarize_target(
            target="gpu_h100",
            run_dir=artifacts_root / "kestrel_gpu_h100_run",
            log_dir=logs_root / "bsm_kestrel_gpu_h100" / "logs",
            gpu_mode=True,
        )
    )

    payload = {
        "manifest_version": "1",
        "generated_at_utc": _utc_now(),
        "hpc_repo_root": args.hpc_repo_root,
        "artifacts_root": str(artifacts_root),
        "logs_root": str(logs_root),
        "suite_root": str(suite_root),
        "pullback_mode": args.pullback_mode,
        "targets": [asdict(row) for row in rows],
        "aggregate": {
            "n_targets": len(rows),
            "n_failed_targets": sum(1 for row in rows if row.status == "failed"),
            "n_complete_targets": sum(1 for row in rows if row.status == "complete"),
            "n_failed_shards": sum(row.failed_shards for row in rows),
            "n_completed_shards": sum(row.completed_shards for row in rows),
            "n_manifest_shards": sum(row.manifest_shards for row in rows),
        },
    }

    out_json = Path(args.output_json)
    out_csv = Path(args.output_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = (
            list(asdict(rows[0]).keys()) if rows else list(TargetSummary.__annotations__.keys())
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    return 0


def _summarize_rows_to_text(rows: list[dict[str, str]]) -> str:
    lines = ["HPC artifact analysis (manifest-driven)", ""]
    for row in rows:
        lines.append(
            f"{row['target']}: {row['health']} "
            f"(status={row['status']}, completed={row['completed_shards']}, "
            f"failed={row['failed_shards']}, merged_pairs={row['merged_retained_pairs']})"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "- LOOKS_ACTIVE => no failed shards detected and progress/queued signal present",
            "- ATTENTION => failed shard(s) detected",
            "- NO_SIGNAL => scripts/artifacts not visible in pulled bundle",
        ]
    )
    return "\n".join(lines) + "\n"


def analyze_zip(args: argparse.Namespace) -> int:
    """Extract run summary rows from a pulled snapshot zip."""
    zip_path = Path(args.zip)
    out_csv = Path(args.out_csv)
    out_txt = Path(args.out_txt)

    with zipfile.ZipFile(zip_path, "r") as archive:
        names = set(archive.namelist())
        if "manifest/run_summary.csv" not in names:
            raise FileNotFoundError(
                "manifest/run_summary.csv missing from zip; "
                "regenerate bundle with updated pull script."
            )
        rows: list[dict[str, str]] = []
        with archive.open("manifest/run_summary.csv") as handle:
            decoded = handle.read().decode("utf-8", errors="replace").splitlines()
            reader = csv.DictReader(decoded)
            for row in reader:
                rows.append(dict(row))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with out_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        out_csv.write_text("", encoding="utf-8")
    out_txt.write_text(_summarize_rows_to_text(rows), encoding="utf-8")
    print(out_txt.read_text(encoding="utf-8"), end="")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser("create-run-manifest")
    create_parser.add_argument("--hpc-repo-root", required=True)
    create_parser.add_argument("--artifacts-root", required=True)
    create_parser.add_argument("--logs-root", required=True)
    create_parser.add_argument("--suite-root", required=True)
    create_parser.add_argument("--pullback-mode", required=True)
    create_parser.add_argument("--output-json", required=True)
    create_parser.add_argument("--output-csv", required=True)
    create_parser.set_defaults(func=create_run_manifest)

    analyze_parser = subparsers.add_parser("analyze-zip")
    analyze_parser.add_argument("--zip", required=True)
    analyze_parser.add_argument("--out-csv", required=True)
    analyze_parser.add_argument("--out-txt", required=True)
    analyze_parser.set_defaults(func=analyze_zip)
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
