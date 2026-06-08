#!/usr/bin/env python
"""Unified manuscript pipeline runner with config-driven parameterization."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rfm_pipeline.config import (
    WorkflowConfig,
    apply_fast_mode_overrides,
    load_config,
)
from rfm_pipeline.manuscript_pipeline_helpers import (
    _load_empirical_null_screening_result,
    _load_interaction_discovery_result,
    _load_nonlinear_discovery_result,
    _load_output_conditioning_result,
    _load_sparse_selection_result,
    _load_tables,
    _read_json,
    _resolve_data_root,
    config_to_legacy_case_study,
)
from rfm_pipeline.manuscript_stages import (
    EmpiricalNullScreeningResult,
    InteractionDiscoveryResult,
    NonlinearDiscoveryResult,
    OutputConditioningResult,
    SparseSelectionStabilityResult,
    condition_manuscript_outputs,
    configure_progress_telemetry,
    discover_manuscript_interactions,
    discover_manuscript_nonlinear_transformations,
    empirical_null_screening_spec_from_case_study_config,
    final_manuscript_artifacts_spec_from_case_study_config,
    interaction_discovery_spec_from_case_study_config,
    nonlinear_discovery_spec_from_case_study_config,
    output_conditioning_spec_from_case_study_config,
    regenerate_final_manuscript_artifacts,
    screen_manuscript_empirical_null_terms,
    select_manuscript_sparse_support,
    sparse_selection_stability_spec_from_case_study_config,
    write_empirical_null_screening_artifacts,
    write_final_manuscript_artifacts,
    write_interaction_discovery_artifacts,
    write_nonlinear_discovery_artifacts,
    write_output_conditioning_artifacts,
    write_sparse_selection_stability_artifacts,
)

STAGES = (
    "output_conditioning",
    "empirical_null_screen",
    "interaction_discovery",
    "nonlinear_discovery",
    "sparse_selection",
    "final_manuscript_artifacts",
)


@dataclass
class _FakeRuntime:
    """Fake runtime object for compatibility."""

    output_root: Path


@dataclass
class _FakeContext:
    """Fake context object for compatibility with manuscript stages."""

    case_study_config: dict[str, Any]
    tables: dict[str, Any]
    runtime: _FakeRuntime


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    else:
        return True


def _mark_abandoned_run_if_needed(
    *,
    output_root: Path,
    started_path: Path,
    complete_path: Path,
    failed_path: Path,
    interrupted_path: Path,
    abandoned_path: Path,
    progress_path: Path,
) -> None:
    """Write an abandoned-run marker when a prior run has no terminal marker and dead PID."""
    if not started_path.exists():
        return
    if (
        complete_path.exists()
        or failed_path.exists()
        or interrupted_path.exists()
        or abandoned_path.exists()
    ):
        return
    started = _read_json(started_path) or {}
    prior_pid = int(started.get("pid", -1))
    if _pid_is_alive(prior_pid):
        return
    payload: dict[str, Any] = {
        "status": "abandoned",
        "abandoned_at_utc": _utc_now(),
        "output_root": str(output_root.resolve()),
        "reason": "prior run has run_started marker but no terminal marker and no live PID",
        "prior_run_started": started,
    }
    progress = _read_json(progress_path)
    if progress is not None:
        payload["last_progress"] = progress
    _write_json(abandoned_path, payload)


def _tables_memory_mb(tables: dict[str, Any]) -> float:
    total_bytes = 0
    for table in tables.values():
        total_bytes += int(table.memory_usage(index=True, deep=True).sum())
    return total_bytes / (1024.0 * 1024.0)


def _enforce_oom_policy(
    *,
    config: WorkflowConfig,
    output_column_limit: int | None,
) -> tuple[dict[str, Any], float, int | None, bool, Path]:
    """Load tables and apply OOM fallback policy when configured."""
    data_root = _resolve_data_root(config)
    tables = _load_tables(output_column_limit=output_column_limit, config=config)
    memory_mb = _tables_memory_mb(tables)
    applied_oom_cap = False
    limit_mb = config.runtime.max_loaded_table_mb
    if limit_mb is None or memory_mb <= limit_mb:
        return tables, memory_mb, output_column_limit, applied_oom_cap, data_root

    fallback_cap = config.runtime.oom_output_cap
    should_reload = fallback_cap is not None and (
        output_column_limit is None or fallback_cap < output_column_limit
    )
    if should_reload:
        tables = _load_tables(output_column_limit=fallback_cap, config=config)
        memory_mb = _tables_memory_mb(tables)
        output_column_limit = fallback_cap
        applied_oom_cap = True

    if memory_mb > limit_mb:
        raise MemoryError(
            "Loaded table memory exceeds runtime.max_loaded_table_mb. "
            f"limit_mb={limit_mb:.1f}, observed_mb={memory_mb:.1f}. "
            "Increase runtime.max_loaded_table_mb or set runtime.oom_output_cap."
        )
    return tables, memory_mb, output_column_limit, applied_oom_cap, data_root


def _stage_indices(start_stage: str, stop_stage: str) -> tuple[int, int]:
    start = STAGES.index(start_stage)
    stop = STAGES.index(stop_stage)
    if start > stop:
        raise ValueError(
            "start-stage must be before or equal to stop-stage. "
            f"start={start_stage}, stop={stop_stage}"
        )
    return start, stop


def _summary_to_record(stage: str, elapsed_seconds: float, summary: Any) -> dict[str, Any]:
    row = {
        "stage": stage,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "recorded_at_utc": _utc_now(),
    }
    if getattr(summary, "empty", True):
        return row
    first = summary.iloc[0].to_dict()
    for key, value in first.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            row[str(key)] = value
        else:
            row[str(key)] = str(value)
    return row


def _write_runtime_diagnostics(output_root: Path, records: list[dict[str, Any]]) -> None:
    import pandas as pd

    runtime_root = output_root / "runtime_diagnostics"
    runtime_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(records).to_csv(
        runtime_root / "stage_runtime_summary.csv", index=False
    )


def _configure_parallel_runtime(output_root: Path) -> None:
    """Stabilize joblib/loky behavior across long runs and large artifacts."""
    tmp_dir = output_root / ".joblib_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("JOBLIB_TEMP_FOLDER", str(tmp_dir.resolve()))
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))


def main() -> int:
    """Parse args, run stage-windowed workflow, and persist run markers."""
    parser = argparse.ArgumentParser(
        description="Run manuscript pipeline with config file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("config", type=str, help="Path to config YAML file")
    parser.add_argument("--fast", action="store_true", help="Enable fast-mode overrides")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Override output artifact directory"
    )
    parser.add_argument(
        "--start-stage",
        type=str,
        default=STAGES[0],
        choices=STAGES,
        help="Stage to start from (resume from existing artifacts).",
    )
    parser.add_argument(
        "--stop-stage",
        type=str,
        default=STAGES[-1],
        choices=STAGES,
        help="Stage to stop at (for partial/debug runs).",
    )
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"✗ {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"✗ Failed to load config: {e}", file=sys.stderr)
        return 1

    if args.fast:
        config.validation.fast_mode = True
    config = apply_fast_mode_overrides(config)
    if args.seed is not None:
        config.output.seed = args.seed
    if args.output_dir is not None:
        config.output.artifact_dir = args.output_dir

    try:
        start_idx, stop_idx = _stage_indices(args.start_stage, args.stop_stage)
    except ValueError as e:
        print(f"✗ {e}", file=sys.stderr)
        return 1

    output_root = Path(config.output.artifact_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    _configure_parallel_runtime(output_root)
    progress_path = output_root / "runtime_diagnostics" / "stage_progress.json"
    configure_progress_telemetry(progress_path)
    started_path = output_root / "run_started.json"
    complete_path = output_root / "run_complete.json"
    failed_path = output_root / "run_failed.json"
    interrupted_path = output_root / "run_interrupted.json"
    abandoned_path = output_root / "run_abandoned.json"
    _mark_abandoned_run_if_needed(
        output_root=output_root,
        started_path=started_path,
        complete_path=complete_path,
        failed_path=failed_path,
        interrupted_path=interrupted_path,
        abandoned_path=abandoned_path,
        progress_path=progress_path,
    )
    for stale in (complete_path, failed_path, interrupted_path):
        if stale.exists():
            stale.unlink()

    print(f"\n{'─' * 70}")
    print("  Unified Manuscript Pipeline Runner")
    print(f"{'─' * 70}")
    print(f"  Config: {args.config}")
    print(f"  Dataset: {config.dataset.type}")
    print(f"  Runtime n_jobs: {config.runtime.n_jobs}")
    print(f"  Output: {output_root}")
    print(f"  Progress telemetry: {progress_path}")
    print(f"  Fast mode: {config.validation.fast_mode}")
    print(f"  Stage window: {args.start_stage} → {args.stop_stage}")
    print("  Stages: 6 (conditioning → screening → interaction → nonlinear → sparse → final)")

    output_column_limit = None
    if config.validation.fast_mode:
        output_column_limit = config.validation.fast_mode_overrides.output_cap

    try:
        (
            tables,
            loaded_memory_mb,
            output_column_limit,
            oom_cap_applied,
            data_root,
        ) = _enforce_oom_policy(config=config, output_column_limit=output_column_limit)
    except Exception as e:
        print(f"✗ Failed to load data: {e}", file=sys.stderr)
        return 1

    legacy_config = config_to_legacy_case_study(config)
    ctx = _FakeContext(
        case_study_config=legacy_config,
        tables=tables,
        runtime=_FakeRuntime(output_root=output_root),
    )

    print("\n  Running pipeline...")
    sys.stdout.flush()

    t0 = time.perf_counter()
    _write_json(
        started_path,
        {
            "status": "running",
            "started_at_utc": _utc_now(),
            "pid": os.getpid(),
            "config_path": str(Path(args.config).resolve()),
            "output_root": str(output_root.resolve()),
            "runtime_n_jobs": config.runtime.n_jobs,
            "start_stage": args.start_stage,
            "stop_stage": args.stop_stage,
            "loaded_table_memory_mb": round(loaded_memory_mb, 3),
            "dataset_root": str(data_root.resolve()),
            "dataset_rows": int(len(tables["case_study_input_matrix"])),
            "dataset_outputs": int(max(0, len(tables["case_study_output_matrix"].columns) - 1)),
            "output_column_limit": output_column_limit,
            "oom_output_cap_applied": bool(oom_cap_applied),
            "progress_telemetry_path": str(progress_path.resolve()),
        },
    )

    def _handle_signal(signum: int, _frame: object) -> None:
        signal_name = signal.Signals(signum).name
        _write_json(
            interrupted_path,
            {
                "status": "interrupted",
                "interrupted_at_utc": _utc_now(),
                "pid": os.getpid(),
                "signal": signal_name,
                "elapsed_seconds": round(time.perf_counter() - t0, 3),
            },
        )
        print(
            f"\n✗ Pipeline interrupted by {signal_name}. Details: {interrupted_path}",
            file=sys.stderr,
        )
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, _handle_signal)
    if hasattr(signal, "SIGQUIT"):
        signal.signal(signal.SIGQUIT, _handle_signal)

    stage_records: list[dict[str, Any]] = []
    conditioning: OutputConditioningResult | None = None
    screening: EmpiricalNullScreeningResult | None = None
    interactions: InteractionDiscoveryResult | None = None
    nonlinear: NonlinearDiscoveryResult | None = None
    sparse_selection: SparseSelectionStabilityResult | None = None

    try:
        if start_idx > 0:
            conditioning = _load_output_conditioning_result(output_root)
        if start_idx > 1:
            screening = _load_empirical_null_screening_result(output_root)
        if start_idx > 2:
            interactions = _load_interaction_discovery_result(output_root)
        if start_idx > 3:
            nonlinear = _load_nonlinear_discovery_result(output_root)
        if start_idx > 4:
            sparse_selection = _load_sparse_selection_result(output_root)

        for stage_idx in range(start_idx, stop_idx + 1):
            stage = STAGES[stage_idx]
            ts = time.perf_counter()

            if stage == "output_conditioning":
                spec = output_conditioning_spec_from_case_study_config(ctx.case_study_config)
                conditioning = condition_manuscript_outputs(
                    ctx.tables["case_study_output_matrix"],
                    ctx.tables["fixed_holdout_assignments"],
                    spec,
                )
                write_output_conditioning_artifacts(conditioning, output_root)
                summary = conditioning.summary

            elif stage == "empirical_null_screen":
                if conditioning is None:
                    raise RuntimeError("Missing conditioning result before empirical null stage.")
                spec = empirical_null_screening_spec_from_case_study_config(ctx.case_study_config)
                screening = screen_manuscript_empirical_null_terms(
                    ctx.tables["case_study_input_matrix"],
                    ctx.tables["manuscript_feature_catalog"],
                    ctx.tables["fixed_holdout_assignments"],
                    conditioning.pca_scores,
                    spec,
                    checkpoint_dir=output_root / "empirical_null_screen",
                )
                write_empirical_null_screening_artifacts(screening, output_root)
                summary = screening.summary

            elif stage == "interaction_discovery":
                if conditioning is None or screening is None:
                    raise RuntimeError("Missing prerequisites before interaction stage.")
                spec = interaction_discovery_spec_from_case_study_config(ctx.case_study_config)
                interactions = discover_manuscript_interactions(
                    ctx.tables["case_study_input_matrix"],
                    ctx.tables["manuscript_feature_catalog"],
                    ctx.tables["fixed_holdout_assignments"],
                    conditioning.pca_scores,
                    screening.retained_terms,
                    spec,
                    checkpoint_dir=output_root / "interaction_discovery" / "_batch_checkpoints",
                )
                write_interaction_discovery_artifacts(interactions, output_root)
                summary = interactions.summary

            elif stage == "nonlinear_discovery":
                if conditioning is None or screening is None:
                    raise RuntimeError("Missing prerequisites before nonlinear stage.")
                spec = nonlinear_discovery_spec_from_case_study_config(ctx.case_study_config)
                nonlinear = discover_manuscript_nonlinear_transformations(
                    ctx.tables["case_study_input_matrix"],
                    ctx.tables["manuscript_feature_catalog"],
                    ctx.tables["fixed_holdout_assignments"],
                    conditioning.pca_scores,
                    screening.retained_terms,
                    spec,
                    checkpoint_dir=output_root / "nonlinear_discovery",
                )
                write_nonlinear_discovery_artifacts(nonlinear, output_root)
                summary = nonlinear.summary

            elif stage == "sparse_selection":
                if (
                    conditioning is None
                    or screening is None
                    or interactions is None
                    or nonlinear is None
                ):
                    raise RuntimeError("Missing prerequisites before sparse stage.")
                spec = sparse_selection_stability_spec_from_case_study_config(ctx.case_study_config)
                sparse_selection = select_manuscript_sparse_support(
                    ctx.tables["case_study_input_matrix"],
                    ctx.tables["manuscript_feature_catalog"],
                    ctx.tables["fixed_holdout_assignments"],
                    conditioning.pca_scores,
                    screening.retained_terms,
                    interactions.retained_pairs,
                    nonlinear.retained_transformations,
                    spec,
                    checkpoint_dir=output_root / "sparse_selection",
                )
                write_sparse_selection_stability_artifacts(sparse_selection, output_root)
                summary = sparse_selection.summary

            else:
                if (
                    conditioning is None
                    or screening is None
                    or interactions is None
                    or nonlinear is None
                    or sparse_selection is None
                ):
                    raise RuntimeError("Missing prerequisites before final stage.")
                spec = final_manuscript_artifacts_spec_from_case_study_config(ctx.case_study_config)
                final_artifacts = regenerate_final_manuscript_artifacts(
                    ctx.tables["case_study_input_matrix"],
                    ctx.tables["case_study_output_matrix"],
                    ctx.tables["manuscript_feature_catalog"],
                    ctx.tables["fixed_holdout_assignments"],
                    conditioning,
                    screening,
                    interactions,
                    nonlinear,
                    sparse_selection,
                    spec,
                    checkpoint_dir=output_root / "final_manuscript_artifacts",
                )
                write_final_manuscript_artifacts(final_artifacts, output_root)
                summary = final_artifacts.summary

            elapsed_stage = time.perf_counter() - ts
            stage_records.append(_summary_to_record(stage, elapsed_stage, summary))
            _write_runtime_diagnostics(output_root, stage_records)

        elapsed = time.perf_counter() - t0
        _write_json(
            complete_path,
            {
                "status": "complete",
                "completed_at_utc": _utc_now(),
                "pid": os.getpid(),
                "elapsed_seconds": round(elapsed, 3),
                "start_stage": args.start_stage,
                "stop_stage": args.stop_stage,
                "partial_run": bool(stop_idx < len(STAGES) - 1),
                "loaded_table_memory_mb": round(loaded_memory_mb, 3),
                "output_column_limit": output_column_limit,
                "progress_telemetry_path": str(progress_path.resolve()),
            },
        )
        print(f"\n✓ Pipeline complete ({elapsed:.0f}s)")
        configure_progress_telemetry(None)
        return 0
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else 1
    except Exception as e:
        elapsed = time.perf_counter() - t0
        _write_json(
            failed_path,
            {
                "status": "failed",
                "failed_at_utc": _utc_now(),
                "pid": os.getpid(),
                "elapsed_seconds": round(elapsed, 3),
                "error_type": type(e).__name__,
                "error_message": str(e),
                "start_stage": args.start_stage,
                "stop_stage": args.stop_stage,
                "progress_telemetry_path": str(progress_path.resolve()),
            },
        )
        print(f"\n✗ Pipeline failed: {e}", file=sys.stderr)
        traceback.print_exc()
        print(f"Failure details: {failed_path}", file=sys.stderr)
        configure_progress_telemetry(None)
        return 1


if __name__ == "__main__":
    sys.exit(main())
