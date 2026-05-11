#!/usr/bin/env python
"""Runtime investigation workflow: generate ladder configs, run, and project full runtime."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_PROFILES = ("small", "medium", "large")

PROFILE_OVERRIDES: dict[str, dict[str, Any]] = {
    "small": {
        "stages": {
            "empirical_null_screening": {"n_permutations": 21, "max_retained_terms": 40},
            "interaction_discovery": {
                "n_permutations": 11,
                "n_tree_estimators": 50,
                "max_tree_depth": 3,
            },
            "sparse_selection": {"n_stability_subsamples": 4, "max_candidate_terms": 40},
            "final_artifacts": {"bootstrap_count": 5},
        }
    },
    "medium": {
        "stages": {
            "empirical_null_screening": {"n_permutations": 101, "max_retained_terms": 63},
            "interaction_discovery": {
                "n_permutations": 21,
                "n_tree_estimators": 100,
                "max_tree_depth": 3,
            },
            "sparse_selection": {"n_stability_subsamples": 8, "max_candidate_terms": 120},
            "final_artifacts": {"bootstrap_count": 10},
        }
    },
    "large": {
        "stages": {
            "empirical_null_screening": {"n_permutations": 201, "max_retained_terms": 80},
            "interaction_discovery": {
                "n_permutations": 41,
                "n_tree_estimators": 100,  # Reduced from 120 (Phase 2 optimization)
                "max_tree_depth": 3,  # Reduced from 4 (Phase 2 optimization)
            },
            "sparse_selection": {"n_stability_subsamples": 12, "max_candidate_terms": 250},
            "final_artifacts": {"bootstrap_count": 20},
        }
    },
}


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _nested_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _nested_update(base[key], value)
        else:
            base[key] = value
    return base


def _as_int(value: Any, default: int) -> int:
    if value is None:
        return default
    return int(value)


def _json_field(path: Path, key: str) -> Any | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload.get(key)


def materialize_ladder_configs(
    *,
    base_config: dict[str, Any],
    output_root: Path,
    profiles: tuple[str, ...],
    dataset_path: Path | None,
) -> dict[str, Path]:
    """Write scaled investigation configs and return profile->config path."""
    configs_dir = output_root / "configs"
    runs_dir = output_root / "runs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    generated: dict[str, Path] = {}
    for profile in profiles:
        if profile not in PROFILE_OVERRIDES:
            raise ValueError(f"Unknown profile: {profile}")

        cfg = copy.deepcopy(base_config)
        _nested_update(cfg, PROFILE_OVERRIDES[profile])

        if dataset_path is not None:
            cfg.setdefault("dataset", {})
            cfg["dataset"]["path"] = str(dataset_path)
            cfg["dataset"]["type"] = "custom_dataset"

        cfg.setdefault("output", {})
        cfg["output"]["artifact_dir"] = str((runs_dir / profile).resolve())
        cfg["output"]["verbose"] = True
        cfg_path = configs_dir / f"{profile}.yml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        generated[profile] = cfg_path

    return generated


def _status_from_markers(run_dir: Path) -> str:
    if (run_dir / "run_complete.json").exists():
        return "complete"
    if (run_dir / "run_failed.json").exists():
        return "failed"
    if (run_dir / "run_abandoned.json").exists():
        return "abandoned"
    if (run_dir / "run_interrupted.json").exists():
        return "interrupted"
    if (run_dir / "run_started.json").exists():
        return "running_or_stale"
    return "not_started"


def collect_run_metrics(
    run_dir: Path,
    profile: str,
    profile_config: dict[str, Any],
) -> dict[str, Any]:
    """Read profile run markers and diagnostics into one metrics dictionary."""
    status = _status_from_markers(run_dir)
    elapsed = _json_field(run_dir / "run_complete.json", "elapsed_seconds")
    runtime_csv = run_dir / "runtime_diagnostics" / "stage_runtime_summary.csv"
    final_summary_csv = run_dir / "final_manuscript_artifacts" / "final_artifact_summary.csv"

    sparse_seconds: float | None = None
    final_seconds: float | None = None
    if runtime_csv.exists():
        with runtime_csv.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                stage = row.get("stage")
                seconds = row.get("elapsed_seconds")
                if not seconds:
                    continue
                if stage == "sparse_selection_and_stability":
                    sparse_seconds = float(seconds)
                elif stage == "final_manuscript_tables_and_figures":
                    final_seconds = float(seconds)

    n_prefilter_support: float | None = None
    n_final_support: float | None = None
    if final_summary_csv.exists():
        with final_summary_csv.open(newline="", encoding="utf-8") as handle:
            first = next(csv.DictReader(handle), None)
        if first:
            if first.get("n_prefilter_support_features"):
                n_prefilter_support = float(first["n_prefilter_support_features"])
            if first.get("n_final_support_features"):
                n_final_support = float(first["n_final_support_features"])

    sparse_cfg = profile_config.get("stages", {}).get("sparse_selection", {})
    final_cfg = profile_config.get("stages", {}).get("final_artifacts", {})
    max_candidate_terms = _as_int(sparse_cfg.get("max_candidate_terms"), 0)
    n_stability_subsamples = _as_int(sparse_cfg.get("n_stability_subsamples"), 0)
    bootstrap_count = _as_int(final_cfg.get("bootstrap_count"), 0)

    sparse_units = float(max_candidate_terms * n_stability_subsamples)
    support_for_units = (
        n_prefilter_support if n_prefilter_support is not None else float(max_candidate_terms)
    )
    final_units = float(bootstrap_count * support_for_units)

    return {
        "profile": profile,
        "status": status,
        "elapsed_seconds": float(elapsed) if elapsed is not None else None,
        "sparse_seconds": sparse_seconds,
        "final_seconds": final_seconds,
        "max_candidate_terms": max_candidate_terms,
        "n_stability_subsamples": n_stability_subsamples,
        "bootstrap_count": bootstrap_count,
        "n_prefilter_support": n_prefilter_support,
        "n_final_support": n_final_support,
        "sparse_units": sparse_units,
        "final_units": final_units,
        "run_dir": str(run_dir),
    }


def compute_runtime_projection(
    rows: list[dict[str, Any]],
    target: dict[str, Any],
) -> dict[str, Any]:
    """Project target runtime from completed profile unit-rate medians."""
    complete = [row for row in rows if row.get("status") == "complete"]
    sparse_rates: list[float] = []
    final_rates: list[float] = []
    for row in complete:
        sparse_seconds = row.get("sparse_seconds")
        final_seconds = row.get("final_seconds")
        sparse_units = row.get("sparse_units") or 0.0
        final_units = row.get("final_units") or 0.0
        if sparse_seconds is not None and sparse_units > 0:
            sparse_rates.append(float(sparse_seconds) / float(sparse_units))
        if final_seconds is not None and final_units > 0:
            final_rates.append(float(final_seconds) / float(final_units))

    if not sparse_rates or not final_rates:
        return {
            "status": "insufficient_data",
            "completed_profiles": len(complete),
            "reason": "Need at least one complete row with sparse/final runtime + unit counts",
            "target_profile_name": target.get("target_profile_name"),
        }

    sparse_rate = statistics.median(sparse_rates)
    final_rate = statistics.median(final_rates)
    target_sparse_units = float(target["target_sparse_units"])
    target_final_units = float(target["target_final_units"])
    projected_sparse = round(sparse_rate * target_sparse_units, 3)
    projected_final = round(final_rate * target_final_units, 3)
    projected_total = round(projected_sparse + projected_final, 3)
    return {
        "status": "ok",
        "target_profile_name": target.get("target_profile_name"),
        "completed_profiles": len(complete),
        "sparse_rate_seconds_per_unit": round(sparse_rate, 6),
        "final_rate_seconds_per_unit": round(final_rate, 6),
        "target_sparse_units": target_sparse_units,
        "target_final_units": target_final_units,
        "projected_sparse_seconds": projected_sparse,
        "projected_final_seconds": projected_final,
        "projected_total_seconds": projected_total,
        "projected_total_minutes": round(projected_total / 60.0, 2),
    }


def _target_units(base_config: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    sparse_cfg = base_config.get("stages", {}).get("sparse_selection", {})
    final_cfg = base_config.get("stages", {}).get("final_artifacts", {})
    max_candidate_terms = sparse_cfg.get("max_candidate_terms")
    if max_candidate_terms is None:
        observed = [
            row.get("n_prefilter_support") for row in rows if row.get("n_prefilter_support")
        ]
        max_candidate_terms = int(max(observed)) if observed else 0
    max_candidate_terms = _as_int(max_candidate_terms, 0)
    n_stability_subsamples = _as_int(sparse_cfg.get("n_stability_subsamples"), 0)
    bootstrap_count = _as_int(final_cfg.get("bootstrap_count"), 0)
    return {
        "target_profile_name": "base_config_target",
        "target_sparse_units": float(max_candidate_terms * n_stability_subsamples),
        "target_final_units": float(max_candidate_terms * bootstrap_count),
    }


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "profile",
        "status",
        "elapsed_seconds",
        "sparse_seconds",
        "final_seconds",
        "max_candidate_terms",
        "n_stability_subsamples",
        "bootstrap_count",
        "n_prefilter_support",
        "n_final_support",
        "sparse_units",
        "final_units",
        "run_dir",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def _write_summary_markdown(
    *,
    path: Path,
    rows: list[dict[str, Any]],
    projection: dict[str, Any],
    output_root: Path,
    monitor_interval_seconds: int,
) -> None:
    lines = [
        "# Runtime Investigation Summary",
        "",
        f"- Generated: {datetime.now(timezone.utc).isoformat()}",
        f"- Output root: `{output_root}`",
        "",
        "## Profile runs",
        "",
        "| profile | status | elapsed_s | sparse_s | final_s | n_final_support |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {profile} | {status} | {elapsed} | {sparse} | {final} | {n_final} |".format(
                profile=row.get("profile", ""),
                status=row.get("status", ""),
                elapsed=row.get("elapsed_seconds", ""),
                sparse=row.get("sparse_seconds", ""),
                final=row.get("final_seconds", ""),
                n_final=row.get("n_final_support", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Projected base-config runtime",
            "",
            "```json",
            json.dumps(projection, indent=2, sort_keys=True),
            "```",
            "",
            "## Monitoring",
            "",
            "```bash",
            f"scripts/watch_final_cost_ladder.sh {output_root / 'runs'} {monitor_interval_seconds}",
            "```",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_profile(
    *,
    config_path: Path,
    output_dir: Path,
    progress_batch_size: int,
    start_stage: str | None,
    stop_stage: str | None,
) -> int:
    cmd = [
        sys.executable,
        "tools/run_manuscript_pipeline.py",
        str(config_path),
        "--output-dir",
        str(output_dir),
    ]
    if start_stage:
        cmd.extend(["--start-stage", start_stage])
    if stop_stage:
        cmd.extend(["--stop-stage", stop_stage])

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "1")
    env["OPENBLAS_NUM_THREADS"] = env.get("OPENBLAS_NUM_THREADS", "1")
    env["MKL_NUM_THREADS"] = env.get("MKL_NUM_THREADS", "1")
    env["NUMEXPR_NUM_THREADS"] = env.get("NUMEXPR_NUM_THREADS", "1")
    env["VECLIB_MAXIMUM_THREADS"] = env.get("VECLIB_MAXIMUM_THREADS", "1")
    env["BSM_PROGRESS_BATCH_SIZE"] = str(progress_batch_size)

    completed = subprocess.run(cmd, cwd=REPO_ROOT, check=False, env=env)  # noqa: S603
    return int(completed.returncode)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-config",
        required=True,
        help="Base config YAML to scale up from.",
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        help=(
            "Optional dataset root override (contains X.parquet/Y.parquet/feature_catalog/holdout)."
        ),
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/runtime_investigation",
        help="Root directory for generated configs/runs/reports.",
    )
    parser.add_argument(
        "--label",
        default="runtime-ladder",
        help="Label for this investigation run.",
    )
    parser.add_argument(
        "--profiles",
        default="small,medium,large",
        help="Comma-separated ladder profiles. Supported: small,medium,large",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Generate configs/reports without running.",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Rerun profiles even if run_complete exists.",
    )
    parser.add_argument("--progress-batch-size", type=int, default=1)
    parser.add_argument("--monitor-interval-seconds", type=int, default=600)
    stage_choices = [
        "output_conditioning",
        "empirical_null_screen",
        "interaction_discovery",
        "nonlinear_discovery",
        "sparse_selection",
        "final_manuscript_artifacts",
    ]
    parser.add_argument("--start-stage", choices=stage_choices, default=None)
    parser.add_argument("--stop-stage", choices=stage_choices, default=None)
    return parser.parse_args()


def main() -> int:
    """Execute runtime investigation and write generated artifacts/reports."""
    args = parse_args()
    base_config_path = Path(args.base_config).resolve()
    base_config = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    profiles = tuple(p.strip() for p in args.profiles.split(",") if p.strip())
    dataset_path = Path(args.dataset_path).resolve() if args.dataset_path else None

    run_root = Path(args.output_root).resolve() / f"{_utc_stamp()}-{args.label}"
    run_root.mkdir(parents=True, exist_ok=True)
    generated = materialize_ladder_configs(
        base_config=base_config,
        output_root=run_root,
        profiles=profiles,
        dataset_path=dataset_path,
    )

    rows: list[dict[str, Any]] = []
    for profile in profiles:
        config_path = generated[profile]
        profile_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        output_dir = run_root / "runs" / profile
        if not args.generate_only:
            if args.force_rerun or not (output_dir / "run_complete.json").exists():
                print(f">>> running profile={profile} config={config_path}")
                return_code = _run_profile(
                    config_path=config_path,
                    output_dir=output_dir,
                    progress_batch_size=max(1, args.progress_batch_size),
                    start_stage=args.start_stage,
                    stop_stage=args.stop_stage,
                )
                print(f">>> profile={profile} return_code={return_code}")
        rows.append(collect_run_metrics(output_dir, profile, profile_cfg))

    target = _target_units(base_config, rows)
    projection = compute_runtime_projection(rows, target)

    report_dir = run_root / "report"
    summary_csv = report_dir / "runtime_investigation_summary.csv"
    projection_json = report_dir / "runtime_projection.json"
    summary_md = report_dir / "runtime_investigation_summary.md"
    monitor_txt = report_dir / "monitor_command.txt"
    _write_summary_csv(summary_csv, rows)
    projection_json.write_text(
        json.dumps(projection, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_summary_markdown(
        path=summary_md,
        rows=rows,
        projection=projection,
        output_root=run_root,
        monitor_interval_seconds=max(1, args.monitor_interval_seconds),
    )
    monitor_cmd = "scripts/watch_final_cost_ladder.sh {} {}\n".format(
        run_root / "runs",
        max(1, args.monitor_interval_seconds),
    )
    monitor_txt.write_text(monitor_cmd, encoding="utf-8")

    print(f"Generated configs: {run_root / 'configs'}")
    print(f"Runs root: {run_root / 'runs'}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Projection JSON: {projection_json}")
    print(f"Summary Markdown: {summary_md}")
    print(f"Monitor command: {monitor_cmd.strip()}")

    if projection.get("status") == "ok":
        return 0
    return 1 if not args.generate_only else 0


if __name__ == "__main__":
    raise SystemExit(main())
