#!/usr/bin/env python
"""Run manuscript workflow with durable run tracking and timestamps."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.bsm_rfm.config import apply_fast_mode_overrides, load_config  # noqa: E402

STAGE_DIRS = [
    "output_conditioning",
    "empirical_null_screen",
    "interaction_discovery",
    "nonlinear_discovery",
    "sparse_selection",
    "final_manuscript_artifacts",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", text).strip("-").lower() or "run"


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _status_from_markers(artifact_dir: Path) -> str:
    if (artifact_dir / "run_complete.json").exists():
        return "complete"
    if (artifact_dir / "run_failed.json").exists():
        return "failed"
    if (artifact_dir / "run_interrupted.json").exists():
        return "interrupted"
    return "unknown"


def parse_args() -> argparse.Namespace:
    """Parse CLI args for tracked workflow execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="Workflow config path (e.g., configs/validation_300_sample_no_caps.yml)",
    )
    parser.add_argument(
        "--run-label",
        default=None,
        help="Optional run label used in run id naming",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional artifact directory override",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=15,
        help="Progress poll interval while command is running",
    )
    parser.add_argument(
        "--start-stage",
        choices=STAGE_DIRS,
        default=None,
        help="Optional stage to start from (resume from existing artifacts).",
    )
    parser.add_argument(
        "--stop-stage",
        choices=STAGE_DIRS,
        default=None,
        help="Optional stage to stop after for partial/debug runs.",
    )
    return parser.parse_args()


def main() -> int:
    """Run config-driven pipeline with timestamped tracking artifacts."""
    args = parse_args()
    config_path = Path(args.config).resolve()
    workflow_config = load_config(str(config_path))
    workflow_config = apply_fast_mode_overrides(workflow_config)

    artifact_dir = Path(args.output_dir or workflow_config.output.artifact_dir).resolve()
    tracking_root = REPO_ROOT / "artifacts" / "workflow_runs"
    tracking_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label = args.run_label or config_path.stem
    run_id = f"{timestamp}-{_slug(label)}"
    run_dir = tracking_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "pipeline.log"
    meta_path = run_dir / "run_metadata.json"
    summary_path = run_dir / "run_summary.json"
    history_path = tracking_root / "run_history.jsonl"

    cmd = [sys.executable, "tools/run_manuscript_pipeline.py", str(config_path)]
    if args.output_dir:
        cmd.extend(["--output-dir", args.output_dir])
    if args.start_stage:
        cmd.extend(["--start-stage", args.start_stage])
    if args.stop_stage:
        cmd.extend(["--stop-stage", args.stop_stage])

    metadata = {
        "run_id": run_id,
        "status": "running",
        "started_at_utc": _utc_now(),
        "config_path": str(config_path),
        "artifact_dir": str(artifact_dir),
        "command": cmd,
        "run_log": str(log_path),
    }
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _append_jsonl(history_path, metadata)

    print(f"Run ID: {run_id}")
    print(f"Config: {config_path}")
    print(f"Artifacts: {artifact_dir}")
    print(f"Tracking: {run_dir}")

    stage_seen: dict[str, str] = {}
    t0 = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_handle:
        proc = subprocess.Popen(  # noqa: S603
            cmd,
            cwd=str(REPO_ROOT),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            while proc.poll() is None:
                for stage in STAGE_DIRS:
                    stage_dir = artifact_dir / stage
                    if stage not in stage_seen and stage_dir.exists():
                        stage_seen[stage] = _utc_now()
                        print(f"[{stage_seen[stage]}] stage-start: {stage}")
                time.sleep(max(args.poll_seconds, 1))
        except KeyboardInterrupt:
            proc.terminate()
            proc.wait(timeout=30)
            status = "interrupted"
            elapsed = time.perf_counter() - t0
            summary = {
                "run_id": run_id,
                "status": status,
                "started_at_utc": metadata["started_at_utc"],
                "ended_at_utc": _utc_now(),
                "elapsed_seconds": round(elapsed, 3),
                "config_path": str(config_path),
                "artifact_dir": str(artifact_dir),
                "run_log": str(log_path),
                "tracking_dir": str(run_dir),
                "stages_seen": stage_seen,
            }
            summary_path.write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            _append_jsonl(history_path, summary)
            print("Interrupted by user.")
            return 130

    return_code = int(proc.returncode or 0)
    elapsed = time.perf_counter() - t0
    marker_status = _status_from_markers(artifact_dir)
    if return_code == 0 and marker_status == "complete":
        status = "complete"
    elif marker_status in {"failed", "interrupted"}:
        status = marker_status
    elif return_code == 0:
        status = "unknown-success-no-marker"
    else:
        status = "failed"

    summary = {
        "run_id": run_id,
        "status": status,
        "return_code": return_code,
        "started_at_utc": metadata["started_at_utc"],
        "ended_at_utc": _utc_now(),
        "elapsed_seconds": round(elapsed, 3),
        "config_path": str(config_path),
        "artifact_dir": str(artifact_dir),
        "run_log": str(log_path),
        "tracking_dir": str(run_dir),
        "stages_seen": stage_seen,
        "marker_files": {
            "run_started": str(artifact_dir / "run_started.json"),
            "run_complete": str(artifact_dir / "run_complete.json"),
            "run_failed": str(artifact_dir / "run_failed.json"),
            "run_interrupted": str(artifact_dir / "run_interrupted.json"),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _append_jsonl(history_path, summary)

    print(f"Status: {status}")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Run summary: {summary_path}")
    return 0 if status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
