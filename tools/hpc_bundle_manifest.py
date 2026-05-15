#!/usr/bin/env python
"""Build and consume HPC run manifests used by artifact pullback workflows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
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


@dataclass
class TargetSpec:
    """Config-resolved target metadata used for summaries and bundle assembly."""

    target: str
    run_dir: str
    log_dir: str
    suite_manifest_path: str = ""
    gpu_mode: bool = False
    config_path: str = ""


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


def _safe_run(cmd: list[str], cwd: Path | None = None) -> str:
    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            cwd=str(cwd) if cwd else None,
        )
    except Exception:
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _collect_code_provenance(hpc_repo_root: Path) -> dict[str, str | bool]:
    return {
        "git_commit": _safe_run(["git", "rev-parse", "HEAD"], cwd=hpc_repo_root),
        "git_commit_short": _safe_run(["git", "rev-parse", "--short", "HEAD"], cwd=hpc_repo_root),
        "git_branch": _safe_run(["git", "branch", "--show-current"], cwd=hpc_repo_root),
        "git_describe": _safe_run(["git", "describe", "--tags", "--always"], cwd=hpc_repo_root),
        "git_remote_origin": _safe_run(
            ["git", "config", "--get", "remote.origin.url"], cwd=hpc_repo_root
        ),
        "git_is_dirty": bool(_safe_run(["git", "status", "--porcelain"], cwd=hpc_repo_root)),
    }


def _collect_package_versions() -> dict[str, str]:
    packages = (
        "numpy",
        "pandas",
        "scikit-learn",
        "xgboost",
        "shap",
        "pyarrow",
        "scipy",
    )
    versions: dict[str, str] = {}
    for pkg in packages:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = ""
    return versions


def _load_target_specs(path: Path) -> list[TargetSpec]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    specs: list[TargetSpec] = []
    for entry in raw.get("targets", []):
        specs.append(
            TargetSpec(
                target=str(entry.get("target", "")),
                run_dir=str(entry.get("run_dir", "")),
                log_dir=str(entry.get("log_dir", "")),
                suite_manifest_path=str(entry.get("suite_manifest_path", "")),
                gpu_mode=bool(entry.get("gpu_mode", False)),
                config_path=str(entry.get("config_path", "")),
            )
        )
    return specs


def _read_csv_first_row(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            return {str(k): str(v) for k, v in row.items()}
    return {}


def _collect_stage_metrics(bundle_root: Path, target: str) -> dict[str, object]:
    run_root = bundle_root / "runs" / target / "run_artifacts"
    if not run_root.exists():
        return {}

    stage_summary_files = {
        "output_conditioning": run_root / "output_conditioning" / "output_conditioning_summary.csv",
        "empirical_null_screen": (
            run_root / "empirical_null_screen" / "empirical_null_screen_summary.csv"
        ),
        "interaction_discovery": (
            run_root / "interaction_discovery" / "interaction_discovery_summary.csv"
        ),
        "nonlinear_discovery": run_root / "nonlinear_discovery" / "nonlinear_discovery_summary.csv",
        "sparse_selection": run_root / "sparse_selection" / "sparse_selection_summary.csv",
        "final_manuscript_artifacts": (
            run_root / "final_manuscript_artifacts" / "final_artifact_summary.csv"
        ),
    }
    retained_files = {
        "n_retained_first_order_terms": (run_root / "empirical_null_screen" / "retained_terms.csv"),
        "n_retained_interactions": (
            run_root / "interaction_discovery" / "retained_interaction_pairs.csv"
        ),
        "n_retained_nonlinear_terms": (
            run_root / "nonlinear_discovery" / "retained_transformations.csv"
        ),
        "n_final_stable_support_terms": (
            run_root / "sparse_selection" / "final_stable_support.csv"
        ),
    }

    stage_summaries = {}
    for stage, path in stage_summary_files.items():
        if path.exists():
            stage_summaries[stage] = _read_csv_first_row(path)
    retained_counts = {name: _count_csv_rows(path) for name, path in retained_files.items()}
    return {
        "stage_summaries": stage_summaries,
        "retained_counts": retained_counts,
    }


def _read_json_file(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if isinstance(payload, dict):
        return payload
    return {}


def _extract_shell_commands(script_path: Path) -> list[str]:
    commands: list[str] = []
    pending_parts: list[str] = []
    command_start = re.compile(r"^(pixi|python|python3|sbatch|bash)\b")

    for raw in script_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if pending_parts:
            pending_parts.append(line.rstrip("\\").strip())
            if not line.endswith("\\"):
                commands.append(" ".join(part for part in pending_parts if part))
                pending_parts = []
            continue
        if not command_start.match(line):
            continue
        pending_parts = [line.rstrip("\\").strip()]
        if not line.endswith("\\"):
            commands.append(" ".join(part for part in pending_parts if part))
            pending_parts = []
    if pending_parts:
        commands.append(" ".join(part for part in pending_parts if part))
    return commands


def _collect_target_script_trace(bundle_root: Path, target: str) -> list[dict[str, object]]:
    script_dir = bundle_root / "runs" / target / "hpc_scripts"
    if not script_dir.exists():
        return []
    traces: list[dict[str, object]] = []
    for script_path in sorted(script_dir.glob("*.sh")):
        directives = []
        for raw in script_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if line.startswith("#SBATCH"):
                directives.append(line)
        traces.append(
            {
                "relative_path": script_path.relative_to(bundle_root).as_posix(),
                "sha256": _sha256_file(script_path),
                "sbatch_directives": directives,
                "shell_commands": _extract_shell_commands(script_path),
            }
        )
    return traces


def _collect_execution_trace(
    *,
    bundle_root: Path,
    target_specs: list[TargetSpec],
) -> dict[str, object]:
    targets: dict[str, dict[str, object]] = {}
    all_commands: list[str] = []
    for spec in target_specs:
        target_root = bundle_root / "runs" / spec.target
        run_root = target_root / "run_artifacts"
        run_started_path = run_root / "run_started.json"
        run_complete_path = run_root / "run_complete.json"
        config_snapshot = bundle_root / "manifest" / "configs" / f"{spec.target}.yml"
        suite_manifest = target_root / "hpc_scripts" / "suite_manifest.jsonl"
        scripts = _collect_target_script_trace(bundle_root, spec.target)
        for script in scripts:
            for command in script.get("shell_commands", []):
                all_commands.append(str(command))
        targets[spec.target] = {
            "target": spec.target,
            "config_path_hpc": spec.config_path,
            "config_snapshot": (
                config_snapshot.relative_to(bundle_root).as_posix()
                if config_snapshot.exists()
                else ""
            ),
            "suite_manifest": (
                suite_manifest.relative_to(bundle_root).as_posix()
                if suite_manifest.exists()
                else ""
            ),
            "run_started": _read_json_file(run_started_path),
            "run_complete": _read_json_file(run_complete_path),
            "submission_scripts": scripts,
        }
    unique_commands = sorted({command for command in all_commands if command.strip()})
    return {
        "generated_at_utc": _utc_now(),
        "targets": targets,
        "all_commands": unique_commands,
    }


def _build_reproduction_recipe(
    *,
    metadata: dict[str, object],
    execution_trace: dict[str, object],
    bundle_root: Path,
) -> str:
    provenance = metadata.get("provenance", {})
    code = provenance.get("code", {}) if isinstance(provenance, dict) else {}
    environment = provenance.get("environment", {}) if isinstance(provenance, dict) else {}
    targets = execution_trace.get("targets", {})

    lines = [
        "# Study reproduction recipe",
        "",
        (
            "This bundle captures run outputs, generated submit scripts, "
            "and provenance for replay/audit."
        ),
        "",
        "## Code provenance",
        f"- git_commit: `{code.get('git_commit', '')}`",
        f"- git_branch: `{code.get('git_branch', '')}`",
        f"- git_describe: `{code.get('git_describe', '')}`",
        f"- git_remote_origin: `{code.get('git_remote_origin', '')}`",
        "",
        "## Environment provenance",
        f"- python_version: `{environment.get('python_version', '')}`",
        f"- platform: `{environment.get('platform', '')}`",
        f"- pixi_version: `{environment.get('pixi_version', '')}`",
    ]

    lockfiles = [
        "manifest/environment/pixi.lock",
        "manifest/environment/pixi.toml",
        "manifest/environment/pyproject.toml",
    ]
    existing_lockfiles = [path for path in lockfiles if (bundle_root / path).exists()]
    if existing_lockfiles:
        lines.extend(
            [
                "",
                "## Environment lock/config snapshots",
                *[f"- `{path}`" for path in existing_lockfiles],
            ]
        )

    commands = execution_trace.get("all_commands", [])
    if isinstance(commands, list) and commands:
        lines.extend(
            [
                "",
                "## Command trace",
                *[f"- `{str(command)}`" for command in commands],
            ]
        )

    if isinstance(targets, dict):
        for target, target_trace in sorted(targets.items()):
            if not isinstance(target_trace, dict):
                continue
            lines.extend(["", f"## Target `{target}`"])
            config_snapshot = str(target_trace.get("config_snapshot", ""))
            if config_snapshot:
                lines.append(f"- config_snapshot: `{config_snapshot}`")
            suite_manifest = str(target_trace.get("suite_manifest", ""))
            if suite_manifest:
                lines.append(f"- suite_manifest: `{suite_manifest}`")
            run_started = target_trace.get("run_started", {})
            if isinstance(run_started, dict):
                start_stage = run_started.get("start_stage", "")
                stop_stage = run_started.get("stop_stage", "")
                if start_stage or stop_stage:
                    lines.append(f"- stage_window: `{start_stage}` -> `{stop_stage}`")
            scripts = target_trace.get("submission_scripts", [])
            if isinstance(scripts, list) and scripts:
                lines.append("- generated_submit_scripts:")
                for script in scripts:
                    if not isinstance(script, dict):
                        continue
                    rel = str(script.get("relative_path", ""))
                    sha = str(script.get("sha256", ""))
                    if rel:
                        lines.append(f"  - `{rel}` (sha256: `{sha}`)")

    lines.append("")
    return "\n".join(lines)


def _path_for_metadata(path: Path | None, *, bundle_root: Path) -> str:
    if path is None:
        return ""
    try:
        return path.resolve().relative_to(bundle_root).as_posix()
    except ValueError:
        return str(path)


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
    hpc_repo_root = Path(args.hpc_repo_root)

    rows: list[TargetSummary] = []
    target_specs_json = getattr(args, "target_specs_json", None)
    if target_specs_json:
        target_specs = _load_target_specs(Path(target_specs_json))
        for spec in target_specs:
            rows.append(
                _summarize_target(
                    target=spec.target,
                    run_dir=Path(spec.run_dir),
                    suite_manifest_fallback=(
                        Path(spec.suite_manifest_path) if spec.suite_manifest_path else None
                    ),
                    log_dir=Path(spec.log_dir),
                    gpu_mode=spec.gpu_mode,
                )
            )
    else:
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
        "provenance": {
            "code": _collect_code_provenance(hpc_repo_root),
            "environment": {
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "pixi_version": _safe_run(["pixi", "--version"]) if shutil.which("pixi") else "",
                "package_versions": _collect_package_versions(),
            },
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


def write_study_metadata(args: argparse.Namespace) -> int:
    """Write reproducibility metadata + file inventory for a collected study bundle."""
    bundle_root = Path(args.bundle_root).resolve()
    hpc_repo_root = Path(args.hpc_repo_root).resolve()
    target_specs = _load_target_specs(Path(args.target_specs_json))
    out_json = Path(args.output_json)
    out_csv = Path(args.output_csv)
    commands_json_path = Path(args.commands_json) if getattr(args, "commands_json", "") else None
    recipe_path = (
        Path(args.reproduction_recipe_md) if getattr(args, "reproduction_recipe_md", "") else None
    )

    inventory_rows: list[dict[str, str | int]] = []
    for path in sorted(bundle_root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(bundle_root).as_posix()
        inventory_rows.append(
            {
                "relative_path": rel,
                "size_bytes": int(path.stat().st_size),
                "sha256": _sha256_file(path),
            }
        )

    execution_trace = _collect_execution_trace(bundle_root=bundle_root, target_specs=target_specs)

    metadata = {
        "manifest_version": "1",
        "generated_at_utc": _utc_now(),
        "pullback_mode": args.pullback_mode,
        "bundle_root": str(bundle_root),
        "hpc_repo_root": str(hpc_repo_root),
        "provenance": {
            "code": _collect_code_provenance(hpc_repo_root),
            "environment": {
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "pixi_version": _safe_run(["pixi", "--version"]) if shutil.which("pixi") else "",
                "package_versions": _collect_package_versions(),
            },
        },
        "targets": [asdict(spec) for spec in target_specs],
        "stage_metrics": {
            spec.target: _collect_stage_metrics(bundle_root, spec.target) for spec in target_specs
        },
        "execution_trace": {
            "n_targets": len(execution_trace.get("targets", {}))
            if isinstance(execution_trace.get("targets", {}), dict)
            else 0,
            "n_traced_commands": len(execution_trace.get("all_commands", []))
            if isinstance(execution_trace.get("all_commands", []), list)
            else 0,
            "commands_json": _path_for_metadata(commands_json_path, bundle_root=bundle_root),
            "reproduction_recipe_md": _path_for_metadata(recipe_path, bundle_root=bundle_root),
        },
        "inventory": {
            "n_files": len(inventory_rows),
            "total_bytes": int(sum(int(row["size_bytes"]) for row in inventory_rows)),
        },
        "final_artifacts": sorted(
            [
                row["relative_path"]
                for row in inventory_rows
                if str(row["relative_path"]).startswith("runs/")
                and "/final_manuscript_artifacts/" in str(row["relative_path"])
            ]
        ),
        "figure_assets": sorted(
            [
                row["relative_path"]
                for row in inventory_rows
                if str(row["relative_path"]).startswith("runs/")
                and (
                    str(row["relative_path"]).endswith(".svg")
                    or "/figures/" in str(row["relative_path"])
                )
            ]
        ),
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["relative_path", "size_bytes", "sha256"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(inventory_rows)

    if commands_json_path:
        commands_json_path.parent.mkdir(parents=True, exist_ok=True)
        commands_json_path.write_text(
            json.dumps(execution_trace, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if recipe_path:
        recipe_path.parent.mkdir(parents=True, exist_ok=True)
        recipe_path.write_text(
            _build_reproduction_recipe(
                metadata=metadata,
                execution_trace=execution_trace,
                bundle_root=bundle_root,
            ),
            encoding="utf-8",
        )
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
    create_parser.add_argument("--target-specs-json", required=False)
    create_parser.add_argument("--output-json", required=True)
    create_parser.add_argument("--output-csv", required=True)
    create_parser.set_defaults(func=create_run_manifest)

    analyze_parser = subparsers.add_parser("analyze-zip")
    analyze_parser.add_argument("--zip", required=True)
    analyze_parser.add_argument("--out-csv", required=True)
    analyze_parser.add_argument("--out-txt", required=True)
    analyze_parser.set_defaults(func=analyze_zip)

    metadata_parser = subparsers.add_parser("write-study-metadata")
    metadata_parser.add_argument("--bundle-root", required=True)
    metadata_parser.add_argument("--hpc-repo-root", required=True)
    metadata_parser.add_argument("--target-specs-json", required=True)
    metadata_parser.add_argument("--pullback-mode", required=True)
    metadata_parser.add_argument("--output-json", required=True)
    metadata_parser.add_argument("--output-csv", required=True)
    metadata_parser.add_argument("--commands-json", required=False, default="")
    metadata_parser.add_argument("--reproduction-recipe-md", required=False, default="")
    metadata_parser.set_defaults(func=write_study_metadata)
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
