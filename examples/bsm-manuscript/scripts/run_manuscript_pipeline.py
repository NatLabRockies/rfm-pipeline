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
import yaml
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure repo root is in sys.path for local wrapper imports.
REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rfm_pipeline.config import (  # noqa: E402
    OutOfCoreConfig,
    WorkflowConfig,
    apply_fast_mode_overrides,
    load_config,
)
from rfm_pipeline.manuscript_runtime import load_manuscript_case_study_config  # noqa: E402
from rfm_pipeline.manuscript_stages import (  # noqa: E402
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
from rfm_pipeline.out_of_core import (  # noqa: E402
    ChunkedParquetReader,
    SpillToDiskBuffer,
    choose_temp_dir,
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


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


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


def _read_csv(path: Path):
    import pandas as pd

    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    return pd.read_csv(path)


def _effective_out_of_core(config: WorkflowConfig) -> OutOfCoreConfig:
    """Resolve out-of-core settings, including legacy runtime keys."""
    resolved = config.runtime.out_of_core
    legacy_map = config.runtime.chunked_io_config or {}
    if config.runtime.use_chunked_io and not resolved.enabled:
        resolved.enabled = True
    if legacy_map:
        resolved.chunk_size_mb = int(legacy_map.get("chunk_size_mb", resolved.chunk_size_mb))
        resolved.max_memory_budget_mb = int(
            legacy_map.get("max_memory_budget_mb", resolved.max_memory_budget_mb)
        )
        resolved.temp_dir = legacy_map.get("temp_dir", resolved.temp_dir)
        resolved.enable_spill_to_disk = bool(
            legacy_map.get("enable_spill_to_disk", resolved.enable_spill_to_disk)
        )
    return resolved


def _read_parquet_with_mode(
    path: Path,
    *,
    out_of_core: OutOfCoreConfig,
    columns: list[str] | None = None,
):
    import pandas as pd

    if not out_of_core.enabled:
        return pd.read_parquet(path, columns=columns)

    reader = ChunkedParquetReader(
        str(path),
        chunk_size_mb=max(1, int(out_of_core.chunk_size_mb)),
        columns=columns,
    )
    if out_of_core.enable_spill_to_disk:
        temp_root = choose_temp_dir(preferred_root=out_of_core.temp_dir)
        buffer = SpillToDiskBuffer(
            temp_dir=temp_root,
            max_memory_mb=max(64, int(out_of_core.max_memory_budget_mb)),
        )
        try:
            for chunk in reader:
                buffer.add_chunk(chunk)
            return buffer.get_final_dataframe()
        finally:
            buffer.cleanup()

    chunks = list(reader)
    if not chunks:
        return pd.read_parquet(path, columns=columns).iloc[0:0]
    return pd.concat(chunks, ignore_index=True)


def _resolve_data_root(config: WorkflowConfig) -> Path:
    """Resolve dataset root from config.dataset.{path,type}.

    Resolution order:

    1. Explicit ``config.dataset.path`` (highest priority).
    2. ``configs/manuscript_paths.yml`` overrides for the BSM ``real_full_dataset``
       dataset type — the parent directory of ``case_study_input_matrix``
       becomes the data root. This is the user-facing override path
       advertised in the README (``cp manuscript_paths_template.yml
       manuscript_paths.yml``).
    3. Hardcoded per-``dataset.type`` fallback rooted under ``REPO_ROOT``.
    """
    if config.dataset.path:
        candidate = Path(config.dataset.path)
        if not candidate.is_absolute():
            candidate = REPO_ROOT / candidate
        return candidate

    dataset_roots = {
        "synthetic_300_sample": REPO_ROOT / "artifacts" / "test_dataset_300",
        "synthetic_full": REPO_ROOT / "artifacts" / "test_dataset_3k",
        "real_full_dataset": REPO_ROOT / "artifacts" / "preprocessed_real_data_30k",
        # Sensitivity study: caller must provide dataset.path
        "synthetic_controlled_dgp": None,
    }

    # For the BSM publication dataset, honor configs/manuscript_paths.yml when
    # present so that operators can point at scratch-resident inputs without
    # editing a committed config. The runner consumes the parent directory
    # because the rest of the pipeline assumes X/Y/holdout/feature_catalog
    # live alongside each other.
    if config.dataset.type == "real_full_dataset":
        paths_override = REPO_ROOT / "configs" / "manuscript_paths.yml"
        if paths_override.exists():
            try:
                with open(paths_override, encoding="utf-8") as f:
                    overrides = yaml.safe_load(f) or {}
            except (OSError, yaml.YAMLError) as exc:
                raise RuntimeError(
                    f"Failed to read manuscript paths override {paths_override}: {exc}. "
                    "Fix the file (or delete it to fall back to the default data root)."
                ) from exc
            input_matrix = overrides.get("case_study_input_matrix")
            if input_matrix:
                return Path(input_matrix).expanduser().resolve().parent

    try:
        root = dataset_roots[config.dataset.type]
        if root is None:
            raise ValueError(
                f"dataset.type={config.dataset.type!r} requires dataset.path to be set explicitly."
            )
        return root
    except KeyError as exc:
        raise ValueError(
            "Unsupported dataset.type for unified runner. "
            f"dataset.type={config.dataset.type!r}. "
            "Set dataset.path explicitly or use one of: "
            f"{sorted(dataset_roots.keys())}"
        ) from exc


def _load_tables(output_column_limit: int | None, config: WorkflowConfig) -> dict[str, Any]:
    """Load validation tables and normalize holdout labels."""
    data_root = _resolve_data_root(config)
    out_of_core = _effective_out_of_core(config)
    if not data_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {data_root}")
    y = _read_parquet_with_mode(data_root / "Y.parquet", out_of_core=out_of_core)
    y_cols = [c for c in y.columns if c != "sample_id"]
    if output_column_limit is not None and len(y_cols) > output_column_limit:
        y = y[["sample_id", *y_cols[:output_column_limit]]]
    holdout = _read_parquet_with_mode(
        data_root / "fixed_holdout_assignments.parquet",
        out_of_core=out_of_core,
    ).copy()
    holdout["split"] = (
        holdout["split"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"test": "holdout", "val": "holdout", "validation": "holdout"})
    )
    # Resolve feature catalog: prefer dataset dir, fall back to repo artifacts/
    _catalog_candidates = [
        data_root / "manuscript_feature_catalog.parquet",
        REPO_ROOT / "artifacts" / "manuscript_feature_catalog.parquet",
    ]
    _catalog_path = next((p for p in _catalog_candidates if p.exists()), None)
    if _catalog_path is None:
        raise FileNotFoundError(
            "manuscript_feature_catalog.parquet not found. "
            f"Copy it to {data_root}/ or {REPO_ROOT / 'artifacts'}/"
        )
    return {
        "case_study_input_matrix": _read_parquet_with_mode(
            data_root / "X.parquet",
            out_of_core=out_of_core,
        ),
        "case_study_output_matrix": y,
        "fixed_holdout_assignments": holdout,
        "manuscript_feature_catalog": _read_parquet_with_mode(
            _catalog_path,
            out_of_core=out_of_core,
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
    canonical_pair_scores = stage_root / "interaction_pair_scores.csv"
    if canonical_pair_scores.exists():
        return InteractionDiscoveryResult(
            pair_scores=_read_csv(canonical_pair_scores),
            component_interaction_scores=_read_csv(stage_root / "component_interaction_scores.csv"),
            interaction_null_summary=_read_csv(stage_root / "interaction_null_summary.csv"),
            retained_pairs=_read_csv(stage_root / "retained_interaction_pairs.csv"),
            provenance=_read_csv(stage_root / "interaction_discovery_provenance.csv"),
            summary=_read_csv(stage_root / "interaction_discovery_summary.csv"),
        )

    # Distributed fallback: interaction reduce currently writes merged pair tables under
    # hpc_shards_interaction_discovery/_merged rather than canonical stage artifacts.
    merged_root = output_root / "hpc_shards_interaction_discovery" / "_merged"
    merged_pair_scores_path = merged_root / "interaction_pair_scores_merged.csv"
    merged_retained_pairs_path = merged_root / "retained_interaction_pairs_merged.csv"
    if not merged_pair_scores_path.exists() or not merged_retained_pairs_path.exists():
        raise FileNotFoundError(f"Missing required artifact: {canonical_pair_scores}")

    import pandas as pd

    pair_scores = _read_csv(merged_pair_scores_path)
    retained_pairs = _read_csv(merged_retained_pairs_path)
    merged_summary = _read_json(merged_root / "interaction_discovery_merged.json") or {}
    summary_row: dict[str, Any] = {
        "stage": "interaction_discovery",
        "n_candidate_pairs": int(len(pair_scores)),
        "n_retained_pairs": int(len(retained_pairs)),
        "artifact_source": "distributed_merged_fallback",
    }
    if isinstance(merged_summary, dict):
        if "n_shards" in merged_summary:
            summary_row["n_shards"] = int(merged_summary["n_shards"])
        if "n_merged_pair_scores" in merged_summary:
            summary_row["n_merged_pair_scores"] = int(merged_summary["n_merged_pair_scores"])
        if "n_merged_retained_pairs" in merged_summary:
            summary_row["n_merged_retained_pairs"] = int(merged_summary["n_merged_retained_pairs"])

    return InteractionDiscoveryResult(
        pair_scores=pair_scores,
        component_interaction_scores=pd.DataFrame(
            columns=[
                "pair_name",
                "component",
                "standardized_residual_interaction_coefficient",
            ]
        ),
        interaction_null_summary=pd.DataFrame(
            columns=[
                "pair_name",
                "null_mean_score",
                "null_quantile_95",
                "null_quantile_99",
                "null_max_score",
            ]
        ),
        retained_pairs=retained_pairs,
        provenance=pd.DataFrame(
            [
                {
                    "stage": "interaction_discovery",
                    "public_implementation_status": "distributed_merged_fallback",
                }
            ]
        ),
        summary=pd.DataFrame([summary_row]),
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

    temporary_reduction = output_conditioning.setdefault("temporary_reduction", {})
    temporary_reduction["retained_variance_fraction"] = workflow_config.algorithm.variance_threshold
    if workflow_config.algorithm.retained_components is not None:
        temporary_reduction["retained_components"] = workflow_config.algorithm.retained_components

    scr = workflow_config.stages.empirical_null_screening
    empirical_null_screen["permutation_count_B"] = scr.n_permutations - 1
    empirical_null_screen["bh_q_screen"] = scr.bh_q_threshold
    if scr.max_retained_terms is not None:
        empirical_null_screen["max_retained_terms"] = int(scr.max_retained_terms)

    itr = workflow_config.stages.interaction_discovery
    interaction_discovery["p_threshold"] = itr.p_threshold
    if itr.n_permutations is not None:
        interaction_discovery["permutation_count_B"] = itr.n_permutations - 1
    interaction_discovery["n_tree_estimators"] = itr.n_tree_estimators
    interaction_discovery["max_tree_depth"] = itr.max_tree_depth
    interaction_discovery["max_shap_samples"] = itr.max_shap_samples
    interaction_discovery["min_component_variance_fraction"] = itr.min_component_variance_fraction
    if itr.max_active_components is not None:
        interaction_discovery["max_active_components"] = int(itr.max_active_components)
    if itr.parallel_batch_timeout_seconds is not None:
        interaction_discovery["parallel_batch_timeout_seconds"] = int(
            itr.parallel_batch_timeout_seconds
        )
    parallelism = workflow_config.runtime.parallelism
    if parallelism.enabled:
        backend_map = {
            "joblib": "threading",
            "dask": "dask",
            "ray": "threading",
            "mpi": "threading",
        }
        interaction_discovery["parallel_backend"] = backend_map.get(
            parallelism.backend,
            "threading",
        )
        if parallelism.dask_workers is not None:
            interaction_discovery["dask_workers"] = int(parallelism.dask_workers)
        interaction_discovery["dask_cores_per_worker"] = int(parallelism.dask_cores_per_worker)
        interaction_discovery["dask_memory_per_worker"] = str(parallelism.dask_memory_per_worker)

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
    final_inferential_filter = case_study.setdefault("final_inferential_filter", {})
    final_inferential_filter["output_subset_mode"] = fnl.hc3_output_subset_mode
    if fnl.hc3_output_fraction is not None:
        final_inferential_filter["output_fraction"] = fnl.hc3_output_fraction
    if fnl.hc3_output_names:
        final_inferential_filter["output_names"] = list(fnl.hc3_output_names)
    if fnl.hc3_output_max_outputs is not None:
        final_inferential_filter["max_outputs"] = int(fnl.hc3_output_max_outputs)
    final_inferential_filter["random_seed"] = int(fnl.hc3_output_random_seed)
    final_inferential_filter["subset_metric"] = str(fnl.hc3_output_subset_metric)
    if fnl.delta_threshold_override is not None:
        feature_pruning = case_study.setdefault("feature_pruning", {})
        feature_pruning["delta_threshold_override"] = float(fnl.delta_threshold_override)

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
