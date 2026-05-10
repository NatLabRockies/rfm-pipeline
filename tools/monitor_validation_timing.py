#!/usr/bin/env python
"""Track and report timing for manuscript pipeline stages.

Monitors output artifacts directory and logs completion times for each stage.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

ARTIFACT_DIR = Path("./artifacts/validation_300_sample_no_caps")
LOG_FILE = ARTIFACT_DIR / "timing_log.json"
STAGES = [
    "output_conditioning",
    "empirical_null_screening",
    "interaction_discovery",
    "nonlinear_discovery",
    "sparse_selection",
    "final_manuscript_artifacts",
]


def check_stage_complete(stage: str) -> bool:
    """Check if a stage is complete by looking for its output directory."""
    stage_dir = ARTIFACT_DIR / stage
    if not stage_dir.exists():
        return False
    # Check for key output files
    if stage == "output_conditioning":
        return (stage_dir / "data.parquet").exists()
    elif stage == "empirical_null_screening":
        return (stage_dir / "screened_features.json").exists()
    elif stage == "interaction_discovery":
        return (stage_dir / "interaction_pairs.json").exists()
    elif stage == "nonlinear_discovery":
        return (stage_dir / "nonlinear_transforms.json").exists()
    elif stage == "sparse_selection":
        return (stage_dir / "sparse_selection_results.json").exists()
    elif stage == "final_manuscript_artifacts":
        return (stage_dir / "final_ols_summary.csv").exists()
    return False


def log_timing(stage: str, elapsed: float) -> None:
    """Log stage completion time."""
    if not LOG_FILE.exists():
        log_data = {}
    else:
        with open(LOG_FILE) as f:
            log_data = json.load(f)

    log_data[stage] = {
        "completed_at": datetime.now().isoformat(),
        "elapsed_seconds": elapsed,
    }

    with open(LOG_FILE, "w") as f:
        json.dump(log_data, f, indent=2)

    print(f"✓ {stage}: {elapsed:.1f}s ({elapsed / 60:.1f}m)")


def monitor_stages() -> None:
    """Monitor stages and log completion times."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    stage_start_times = {}
    stage_completed = set()

    print(f"Monitoring {ARTIFACT_DIR}")
    print(f"Logging to {LOG_FILE}")
    print(f"Stages: {', '.join(STAGES)}")
    print()

    start_time = time.time()
    last_report = start_time

    while len(stage_completed) < len(STAGES):
        current_time = time.time()
        elapsed_total = current_time - start_time

        # Check each stage
        for stage in STAGES:
            if stage not in stage_completed:
                if stage not in stage_start_times:
                    # Start timing this stage if output exists
                    if (ARTIFACT_DIR / stage).exists():
                        stage_start_times[stage] = current_time

                elif check_stage_complete(stage):
                    # Stage is complete
                    stage_elapsed = current_time - stage_start_times[stage]
                    log_timing(stage, stage_elapsed)
                    stage_completed.add(stage)

        # Print progress every 30 seconds
        if current_time - last_report >= 30:
            remaining = len(STAGES) - len(stage_completed)
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] "
                f"Elapsed: {elapsed_total / 60:.1f}m, "
                f"Stages complete: {len(stage_completed)}/{len(STAGES)}, "
                f"Remaining: {remaining}"
            )
            last_report = current_time

        time.sleep(5)

    # Final summary
    total_elapsed = time.time() - start_time
    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    with open(LOG_FILE) as f:
        log_data = json.load(f)

    for stage in STAGES:
        if stage in log_data:
            elapsed = log_data[stage]["elapsed_seconds"]
            print(f"{stage:30s}: {elapsed:7.1f}s ({elapsed / 60:6.1f}m)")

    print("-" * 70)
    print(f"{'TOTAL':30s}: {total_elapsed:7.1f}s ({total_elapsed / 60:6.1f}m)")
    print("=" * 70)


if __name__ == "__main__":
    monitor_stages()
