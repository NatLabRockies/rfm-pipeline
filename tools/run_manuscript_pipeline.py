#!/usr/bin/env python
"""Unified manuscript pipeline runner with config-driven parameterization."""

from __future__ import annotations

import argparse
import copy
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

# Ensure repo root is in sys.path for src imports
REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.bsm_rfm.config import WorkflowConfig, apply_fast_mode_overrides, load_config  # noqa: E402
from src.bsm_rfm.manuscript_runtime import load_manuscript_case_study_config  # noqa: E402
from src.bsm_rfm.manuscript_stages import (  # noqa: E402
    EmpiricalNullScreeningResult,
    InteractionDiscoveryResult,
    NonlinearDiscoveryResult,
    OutputConditioningResult,
    SparseSelectionStabilityResult,
    condition_manuscript_outputs,
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


def _read_csv(path: Path):
    import pandas as pd

    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    return pd.read_csv(path)


def _load_tables(output_column_limit: int | None) -> dict[str, Any]:
    """Load validation tables and normalize holdout labels."""
    import pandas as pd

    data_root = REPO_ROOT / "artifacts" / "test_dataset_300"
    y = pd.read_parquet(data_root / "Y.parquet")
    y_cols = [c for c in y.columns if c != "sample_id"]
    if output_column_limit is not None and len(y_cols) > output_column_limit:
        y = y[["sample_id", *y_cols[:output_column_limit]]]
    holdout = pd.read_parquet(data_root / "holdout_assignments.parquet").copy()
    holdout["split"] = (
        holdout["split"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"test": "holdout", "val": "holdout", "validation": "holdout"})
    )
    return {
        "case_study_input_matrix": pd.read_parquet(data_root / "X.parquet"),
        "case_study_output_matrix": y,
        "fixed_holdout_assignments": holdout,
        "manuscript_feature_catalog": pd.read_parquet(
            REPO_ROOT / "artifacts" / "actual_input_feature_catalog.parquet"
        ),
    }


def _tables_memory_mb(tables: dict[str, Any]) -> float:
    total_bytes = 0
    for table in tables.values():
        total_bytes += int(table.memory_usage(index=True, deep=True).sum())
    return total_bytes / (1024.0 * 1024.0)


def _enforce_oom_policy(
    *,
    config: WorkflowConfig,
    output_column_limit: int | None,
) -> tuple[dict[str, Any], float, int | None, bool]:
    """Load tables and apply OOM fallback policy when configured."""
    tables = _load_tables(output_column_limit=output_column_limit)
    memory_mb = _tables_memory_mb(tables)
    applied_oom_cap = False
    limit_mb = config.runtime.max_loaded_table_mb
    if limit_mb is None or memory_mb <= limit_mb:
        return tables, memory_mb, output_column_limit, applied_oom_cap

    fallback_cap = config.runtime.oom_output_cap
    should_reload = fallback_cap is not None and (
        output_column_limit is None or fallback_cap < output_column_limit
    )
    if should_reload:
        tables = _load_tables(output_column_limit=fallback_cap)
        memory_mb = _tables_memory_mb(tables)
        output_column_limit = fallback_cap
        applied_oom_cap = True

    if memory_mb > limit_mb:
        raise MemoryError(
            "Loaded table memory exceeds runtime.max_loaded_table_mb. "
            f"limit_mb={limit_mb:.1f}, observed_mb={memory_mb:.1f}. "
            "Increase runtime.max_loaded_table_mb or set runtime.oom_output_cap."
        )
    return tables, memory_mb, output_column_limit, applied_oom_cap


def _stage_indices(start_stage: str, stop_stage: str) -> tuple[int, int]:
    start = STAGES.index(start_stage)
    stop = STAGES.index(stop_stage)
    if start > stop:
        raise ValueError(
            "start-stage must be before or equal to stop-stage. "
            f"start={start_stage}, stop={stop_stage}"
        )
    return start, stop


def _load_output_conditioning_result(output_root: Path) -> OutputConditioningResult:
    stage_root = output_root / "output_conditioning"
    diagnostics = _read_csv(stage_root / "output_filter_diagnostics.csv")
    pca_scores = _read_csv(stage_root / "pca_scores.csv")
    pca_loadings = _read_csv(stage_root / "pca_loadings.csv")
    pca_explained_variance = _read_csv(stage_root / "pca_explained_variance.csv")
    summary = _read_csv(stage_root / "output_conditioning_summary.csv")
    if "retained" not in diagnostics.columns or "output_name" not in diagnostics.columns:
        raise ValueError(
            "output_filter_diagnostics.csv must include output_name and retained columns."
        )
    retained = tuple(diagnostics.loc[diagnostics["retained"], "output_name"].astype(str))
    culled = tuple(diagnostics.loc[~diagnostics["retained"], "output_name"].astype(str))
    return OutputConditioningResult(
        retained_output_names=retained,
        culled_output_names=culled,
        output_filter_diagnostics=diagnostics,
        pca_scores=pca_scores,
        pca_loadings=pca_loadings,
        pca_explained_variance=pca_explained_variance,
        summary=summary,
    )


def _load_empirical_null_screening_result(output_root: Path) -> EmpiricalNullScreeningResult:
    stage_root = output_root / "empirical_null_screen"
    return EmpiricalNullScreeningResult(
        feature_screening_statistics=_read_csv(stage_root / "feature_screening_statistics.csv"),
        component_coefficients=_read_csv(stage_root / "component_coefficients.csv"),
        permutation_null_summary=_read_csv(stage_root / "permutation_null_summary.csv"),
        retained_terms=_read_csv(stage_root / "retained_terms.csv"),
        provenance=_read_csv(stage_root / "empirical_null_provenance.csv"),
        summary=_read_csv(stage_root / "empirical_null_screen_summary.csv"),
    )


def _load_interaction_discovery_result(output_root: Path) -> InteractionDiscoveryResult:
    stage_root = output_root / "interaction_discovery"
    return InteractionDiscoveryResult(
        pair_scores=_read_csv(stage_root / "interaction_pair_scores.csv"),
        component_interaction_scores=_read_csv(stage_root / "component_interaction_scores.csv"),
        interaction_null_summary=_read_csv(stage_root / "interaction_null_summary.csv"),
        retained_pairs=_read_csv(stage_root / "retained_interaction_pairs.csv"),
        provenance=_read_csv(stage_root / "interaction_discovery_provenance.csv"),
        summary=_read_csv(stage_root / "interaction_discovery_summary.csv"),
    )


def _load_nonlinear_discovery_result(output_root: Path) -> NonlinearDiscoveryResult:
    stage_root = output_root / "nonlinear_discovery"
    return NonlinearDiscoveryResult(
        transformation_scores=_read_csv(stage_root / "transformation_scores.csv"),
        component_transformation_scores=_read_csv(
            stage_root / "component_transformation_scores.csv"
        ),
        retained_transformations=_read_csv(stage_root / "retained_transformations.csv"),
        provenance=_read_csv(stage_root / "nonlinear_discovery_provenance.csv"),
        summary=_read_csv(stage_root / "nonlinear_discovery_summary.csv"),
    )


def _load_sparse_selection_result(output_root: Path) -> SparseSelectionStabilityResult:
    stage_root = output_root / "sparse_selection"
    return SparseSelectionStabilityResult(
        support_candidates=_read_csv(stage_root / "support_candidates.csv"),
        component_model_selection=_read_csv(stage_root / "component_model_selection.csv"),
        component_coefficients=_read_csv(stage_root / "component_coefficients.csv"),
        stability_resample_summary=_read_csv(stage_root / "stability_resample_summary.csv"),
        stability_feature_summary=_read_csv(stage_root / "stability_feature_summary.csv"),
        final_stable_support=_read_csv(stage_root / "final_stable_support.csv"),
        provenance=_read_csv(stage_root / "sparse_selection_provenance.csv"),
        summary=_read_csv(stage_root / "sparse_selection_summary.csv"),
    )


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


def config_to_legacy_case_study(workflow_config: WorkflowConfig) -> dict[str, Any]:
    """Convert WorkflowConfig to legacy case_study_config format."""
    base_config = load_manuscript_case_study_config(REPO_ROOT)
    config = copy.deepcopy(base_config)

    case_study = config.setdefault("case_study", {})
    output_conditioning = case_study.setdefault("output_conditioning", {})
    empirical_null_screen = case_study.setdefault("empirical_null_screen", {})
    interaction_discovery = case_study.setdefault("interaction_discovery", {})
    nonlinear_discovery = case_study.setdefault("nonlinear_discovery", {})
    sparse_selection = case_study.setdefault("sparse_selection", {})
    stability = case_study.setdefault("stability", {})
    final_model = case_study.setdefault("final_model", {})
    runtime = case_study.setdefault("runtime", {})

    if workflow_config.algorithm.retained_components is not None:
        output_conditioning["retained_components"] = workflow_config.algorithm.retained_components
    else:
        output_conditioning["variance_explained_threshold"] = (
            workflow_config.algorithm.variance_threshold
        )

    scr = workflow_config.stages.empirical_null_screening
    empirical_null_screen["permutation_count_B"] = scr.n_permutations - 1
    empirical_null_screen["bh_q_screen"] = scr.bh_q_threshold

    itr = workflow_config.stages.interaction_discovery
    interaction_discovery["p_threshold"] = itr.p_threshold
    interaction_discovery["n_tree_estimators"] = itr.n_tree_estimators
    interaction_discovery["max_tree_depth"] = itr.max_tree_depth

    nlr = workflow_config.stages.nonlinear_discovery
    nonlinear_discovery["edf_threshold"] = nlr.edf_threshold

    spr = workflow_config.stages.sparse_selection
    stability["resampling_scheme"] = (
        f"{spr.n_stability_subsamples}_subsamples_of_"
        f"{int(spr.subsample_fraction * 100)}_percent_rows_without_replacement_seed_123"
    )
    if spr.max_candidate_terms is not None:
        sparse_selection["max_candidate_terms"] = int(spr.max_candidate_terms)

    fnl = workflow_config.stages.final_artifacts
    final_model["bootstrap_count"] = fnl.bootstrap_count
    final_model["bootstrap_alpha"] = fnl.bootstrap_alpha

    runtime["n_jobs"] = workflow_config.runtime.n_jobs
    config["random_seed"] = workflow_config.output.seed
    return config


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
    started_path = output_root / "run_started.json"
    complete_path = output_root / "run_complete.json"
    failed_path = output_root / "run_failed.json"
    interrupted_path = output_root / "run_interrupted.json"
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
    print(f"  Fast mode: {config.validation.fast_mode}")
    print(f"  Stage window: {args.start_stage} → {args.stop_stage}")
    print("  Stages: 6 (conditioning → screening → interaction → nonlinear → sparse → final)")

    output_column_limit = None
    if config.validation.fast_mode:
        output_column_limit = config.validation.fast_mode_overrides.output_cap

    try:
        tables, loaded_memory_mb, output_column_limit, oom_cap_applied = _enforce_oom_policy(
            config=config,
            output_column_limit=output_column_limit,
        )
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
            "output_column_limit": output_column_limit,
            "oom_output_cap_applied": bool(oom_cap_applied),
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
            },
        )
        print(f"\n✓ Pipeline complete ({elapsed:.0f}s)")
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
            },
        )
        print(f"\n✗ Pipeline failed: {e}", file=sys.stderr)
        traceback.print_exc()
        print(f"Failure details: {failed_path}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
