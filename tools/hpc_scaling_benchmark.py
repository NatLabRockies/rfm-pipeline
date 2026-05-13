#!/usr/bin/env python
"""HPC single-node scaling benchmark for the BSM manuscript pipeline.

Runs a systematic grid of interaction_discovery configurations and records
per-cell wall-clock time. Output feeds into hpc_compute_calculator.py to
fit a runtime model and build the user-facing compute estimator.

Usage::

    # Run on an HPC node (or workstation) — uses all available cores by default
    pixi run hpc-scaling-benchmark

    # Specify output directory and max cores
    pixi run hpc-scaling-benchmark -- \\
        --output-root /scratch/my_user/bsm_benchmark \\
        --max-cores 64 \\
        --dataset-path /projects/bsm/bsm-public-rf/artifacts/test_dataset_3k

    # Quick smoke test (3-cell mini grid)
    pixi run hpc-scaling-benchmark -- --quick

    # Dry run: print the config matrix without running anything
    pixi run hpc-scaling-benchmark -- --dry-run

Design
------
The bottleneck stage is interaction_discovery. The benchmark runs that stage
alone via ``--start-stage interaction_discovery --stop-stage interaction_discovery``
so other stages (output_conditioning, empirical_null_screen) do not add noise.

Because interaction_discovery reads retained terms from the empirical_null_screen
artifacts, the benchmark pre-populates a minimal "stub" artifact directory so
the stage can start immediately without running the full preceding pipeline.

Configuration axes
------------------
The four knobs most strongly affecting runtime:

1. n_jobs          — parallelism (cores used)
2. max_retained_terms — feature count → controls O(N²) pair expansion
3. n_permutations  — permutation trials per pair
4. n_tree_estimators — boosting rounds per model fit

Additionally, n_samples affects runtime. The benchmark uses the available
dataset; specify a different --dataset-path to test with a larger sample set.

Output
------
<output_root>/scaling_results.csv  — one row per benchmark cell
<output_root>/benchmark_manifest.json — full run metadata
<output_root>/benchmark_log.txt — per-cell stdout/stderr
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import subprocess
import sys
import textwrap
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_TOOL = REPO_ROOT / "tools" / "run_manuscript_pipeline.py"

logger = logging.getLogger("hpc_scaling_benchmark")

# ---------------------------------------------------------------------------
# Benchmark grid definition
# ---------------------------------------------------------------------------

# "Core scaling" sweep: holds all settings moderate, varies only n_jobs.
# Purpose: measure parallel efficiency (Amdahl / Gustafson law parameters).
CORE_SCALING_CELLS = [
    {"n_jobs": 1},
    {"n_jobs": 2},
    {"n_jobs": 4},
    {"n_jobs": 8},
    {"n_jobs": 16},
    {"n_jobs": 32},
    {"n_jobs": 64},
    {"n_jobs": -1},  # all cores
]

# "Feature scaling" sweep: varies max_retained_terms → O(N²) pair growth.
# n_jobs fixed at "all available"; all other settings moderate.
FEATURE_SCALING_CELLS = [
    {"max_retained_terms": 20},
    {"max_retained_terms": 40},
    {"max_retained_terms": 75},
    {"max_retained_terms": 100},
    {"max_retained_terms": 150},
    {"max_retained_terms": 200},
    {"max_retained_terms": 300},
    {"max_retained_terms": None},  # uncapped — all retained terms
]

# "Permutation scaling" sweep: varies n_permutations (linear in time).
PERM_SCALING_CELLS = [
    {"n_permutations": 5},
    {"n_permutations": 11},
    {"n_permutations": 21},
    {"n_permutations": 41},
    {"n_permutations": 101},
]

# "Tree scaling" sweep: varies n_tree_estimators.
TREE_SCALING_CELLS = [
    {"n_tree_estimators": 25},
    {"n_tree_estimators": 50},
    {"n_tree_estimators": 100},
    {"n_tree_estimators": 200},
]

# "Cross sweep" — selected (n_jobs, max_retained_terms) pairs for 2D interaction model.
# Keeps everything else at moderate defaults.
CROSS_SWEEP_CELLS = [
    {"n_jobs": 1, "max_retained_terms": 50},
    {"n_jobs": 4, "max_retained_terms": 50},
    {"n_jobs": 16, "max_retained_terms": 50},
    {"n_jobs": -1, "max_retained_terms": 50},
    {"n_jobs": 1, "max_retained_terms": 150},
    {"n_jobs": 4, "max_retained_terms": 150},
    {"n_jobs": 16, "max_retained_terms": 150},
    {"n_jobs": -1, "max_retained_terms": 150},
]

# Quick smoke grid (3 cells only, for --quick mode)
QUICK_CELLS = [
    {"n_jobs": 1, "max_retained_terms": 30, "n_permutations": 5, "n_tree_estimators": 25},
    {"n_jobs": 4, "max_retained_terms": 30, "n_permutations": 5, "n_tree_estimators": 25},
    {"n_jobs": -1, "max_retained_terms": 50, "n_permutations": 11, "n_tree_estimators": 50},
]

# Moderate defaults applied to every cell (overridden per axis)
MODERATE_DEFAULTS: dict[str, Any] = {
    "n_jobs": -1,  # all cores (overridden in core-scaling sweep)
    "max_retained_terms": 100,
    "n_permutations": 21,
    "n_tree_estimators": 100,
    "max_tree_depth": 3,
    "p_threshold": 0.05,
    "variance_threshold": 0.90,
    "seed": 0,
}


def _build_full_grid(max_cores: int) -> list[dict[str, Any]]:
    """Assemble all benchmark cells with metadata."""
    cells = []

    def _add(sweep_name: str, overrides: dict[str, Any]) -> None:
        cell = dict(MODERATE_DEFAULTS)
        cell.update(overrides)
        cell["_sweep"] = sweep_name
        # Resolve -1 to actual max_cores
        if cell["n_jobs"] == -1:
            cell["n_jobs"] = max_cores
        cells.append(cell)

    for c in CORE_SCALING_CELLS:
        _add("core_scaling", c)
    for c in FEATURE_SCALING_CELLS:
        _add("feature_scaling", c)
    for c in PERM_SCALING_CELLS:
        _add("perm_scaling", c)
    for c in TREE_SCALING_CELLS:
        _add("tree_scaling", c)
    for c in CROSS_SWEEP_CELLS:
        _add("cross_sweep", c)

    # Deduplicate by canonical key (same params should run once)
    seen = set()
    unique = []
    for c in cells:
        key = _cell_key(c)
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def _cell_key(cell: dict[str, Any]) -> str:
    return json.dumps(
        {k: v for k, v in sorted(cell.items()) if not k.startswith("_")},
        sort_keys=True,
    )


# ---------------------------------------------------------------------------
# Stub artifact builder
# ---------------------------------------------------------------------------


def _build_stub_artifacts(
    dataset_path: Path,
    output_root: Path,
    retained_terms: int | None,
    variance_threshold: float,
    seed: int,
) -> Path:
    """Build a minimal stub artifact directory for interaction_discovery.

    Runs output_conditioning + empirical_null_screen with fast/cheap settings
    so that the retained_terms catalogue is available for interaction_discovery
    without adding noise to the timing measurement.

    Returns the artifact output root path.
    """
    stub_dir = output_root / "_stub"
    stub_dir.mkdir(parents=True, exist_ok=True)

    # Write a minimal stub config
    max_rt = retained_terms if retained_terms is not None else None
    stub_cfg = {
        "dataset": {"type": "custom", "path": str(dataset_path)},
        "algorithm": {"variance_threshold": variance_threshold},
        "runtime": {"n_jobs": 1, "use_chunked_io": False},
        "stages": {
            "empirical_null_screening": {
                # Stub purpose: seed retained_terms for timing, not filter rigorously.
                # bh_q_threshold=1.0 retains everything up to max_retained_terms,
                # guaranteeing >=2 terms regardless of permutation count.
                "n_permutations": 21,
                "bh_q_threshold": 1.0,
                **({"max_retained_terms": max_rt} if max_rt is not None else {}),
            },
            "interaction_discovery": {
                "n_permutations": 5,
                "n_tree_estimators": 10,
                "max_tree_depth": 3,
                "p_threshold": 0.05,
            },
            "nonlinear_discovery": {"edf_threshold": 2.5},
            "sparse_selection": {"n_stability_subsamples": 4},
            "final_artifacts": {"bootstrap_count": 3},
        },
        "output": {
            "artifact_dir": str(stub_dir),
            "seed": seed,
            "verbose": False,
        },
        "validation": {"fast_mode": False},
    }

    stub_cfg_path = output_root / "_stub_config.yml"
    stub_cfg_path.write_text(yaml.dump(stub_cfg))

    # Run only through empirical_null_screen
    complete_marker = stub_dir / "empirical_null_screen" / "retained_terms.parquet"
    if complete_marker.exists():
        logger.info("[stub] reusing existing stub artifacts in %s", stub_dir)
        return stub_dir

    logger.info("[stub] building stub artifacts (output_conditioning + empirical_null_screen)...")
    cmd = [
        sys.executable,
        str(PIPELINE_TOOL),
        str(stub_cfg_path),
        "--start-stage",
        "output_conditioning",
        "--stop-stage",
        "empirical_null_screen",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"Stub artifact build failed:\n{result.stderr[-2000:]}")

    # Validate that we have enough retained terms for interaction_discovery
    retained_csv = stub_dir / "empirical_null_screen" / "retained_terms.csv"
    if retained_csv.exists():
        import csv as _csv

        with retained_csv.open() as _f:
            n_retained = sum(1 for _ in _csv.reader(_f)) - 1  # minus header
        if n_retained < 2:
            raise RuntimeError(
                f"Stub produced only {n_retained} retained term(s); "
                "interaction_discovery requires at least 2. "
                "Try lowering --variance-threshold or check your feature catalog."
            )
        logger.info("[stub] stub artifacts ready (%d retained terms)", n_retained)
    else:
        logger.info("[stub] stub artifacts ready")
    return stub_dir


# ---------------------------------------------------------------------------
# Per-cell config writer
# ---------------------------------------------------------------------------


def _write_cell_config(
    cell: dict[str, Any],
    stub_dir: Path,
    dataset_path: Path,
    cell_output_dir: Path,
    config_path: Path,
) -> None:
    """Write a YAML config for a single benchmark cell."""
    n_jobs = int(cell["n_jobs"])
    max_retained = cell.get("max_retained_terms")

    cfg = {
        "dataset": {"type": "custom", "path": str(dataset_path)},
        "algorithm": {"variance_threshold": float(cell["variance_threshold"])},
        "runtime": {
            "n_jobs": n_jobs,
            "use_chunked_io": False,  # disable chunked I/O for clean timing
        },
        "stages": {
            "empirical_null_screening": {
                "n_permutations": 5,
                "bh_q_threshold": 0.10,
                **({"max_retained_terms": int(max_retained)} if max_retained is not None else {}),
            },
            "interaction_discovery": {
                "n_permutations": int(cell["n_permutations"]),
                "n_tree_estimators": int(cell["n_tree_estimators"]),
                "max_tree_depth": int(cell["max_tree_depth"]),
                "p_threshold": float(cell["p_threshold"]),
            },
            "nonlinear_discovery": {"edf_threshold": 2.5},
            "sparse_selection": {"n_stability_subsamples": 4},
            "final_artifacts": {"bootstrap_count": 3},
        },
        "output": {
            "artifact_dir": str(cell_output_dir),
            "seed": int(cell["seed"]),
            "verbose": False,
        },
        "validation": {"fast_mode": False},
    }
    config_path.write_text(yaml.dump(cfg))


# ---------------------------------------------------------------------------
# Cell runner
# ---------------------------------------------------------------------------


@dataclass
class CellResult:
    """Result for one benchmark cell."""

    cell_id: str
    sweep: str
    n_jobs: int
    max_retained_terms: int | None
    n_permutations: int
    n_tree_estimators: int
    max_tree_depth: int
    n_samples: int
    n_features: int
    n_retained_terms: int | None
    n_candidate_pairs: int | None
    elapsed_seconds: float | None
    status: str  # "ok" | "timeout" | "error"
    error_message: str | None = None
    recorded_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    hostname: str = field(default_factory=lambda: os.uname().nodename)


def _run_cell(
    cell: dict[str, Any],
    cell_id: str,
    stub_dir: Path,
    dataset_path: Path,
    output_root: Path,
    log_fh,
    timeout: int = 3600 * 4,  # 4-hour safety timeout per cell
) -> CellResult:
    """Run one benchmark cell and return timing result."""
    cell_dir = output_root / cell_id
    cell_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_root / f"{cell_id}_config.yml"

    _write_cell_config(cell, stub_dir, dataset_path, cell_dir, config_path)

    # Copy stub artifacts so interaction_discovery can find retained_terms
    for sub in ("output_conditioning", "empirical_null_screen"):
        src = stub_dir / sub
        dst = cell_dir / sub
        if src.exists() and not dst.exists():
            shutil.copytree(src, dst)

    cmd = [
        sys.executable,
        str(PIPELINE_TOOL),
        str(config_path),
        "--start-stage",
        "interaction_discovery",
        "--stop-stage",
        "interaction_discovery",
    ]

    log_fh.write(f"\n{'=' * 60}\n[{cell_id}] Starting\n")
    log_fh.write(f"  cmd: {' '.join(cmd)}\n")
    log_fh.flush()

    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = time.perf_counter() - t0
        log_fh.write(result.stdout[-4000:] if result.stdout else "")
        log_fh.write(result.stderr[-2000:] if result.stderr else "")
        log_fh.flush()

        if result.returncode != 0:
            return CellResult(
                cell_id=cell_id,
                sweep=cell.get("_sweep", "unknown"),
                n_jobs=cell["n_jobs"],
                max_retained_terms=cell.get("max_retained_terms"),
                n_permutations=cell["n_permutations"],
                n_tree_estimators=cell["n_tree_estimators"],
                max_tree_depth=cell["max_tree_depth"],
                n_samples=0,
                n_features=0,
                n_retained_terms=None,
                n_candidate_pairs=None,
                elapsed_seconds=elapsed,
                status="error",
                error_message=result.stderr[-500:],
            )

        # Extract timing from stage_runtime_summary.csv
        timing_csv = cell_dir / "runtime_diagnostics" / "stage_runtime_summary.csv"
        n_samples, n_retained, n_pairs = 0, None, None
        cell_elapsed = elapsed  # fallback to wall clock
        if timing_csv.exists():
            with timing_csv.open() as f:
                for row in csv.DictReader(f):
                    if row["stage"] == "interaction_discovery":
                        cell_elapsed = float(row["elapsed_seconds"])
                        n_samples = int(float(row.get("n_training_rows", 0) or 0))
                        n_retained = _int_or_none(row.get("n_retained_terms"))
                        n_pairs = _int_or_none(row.get("n_candidate_pairs"))
                        break

        return CellResult(
            cell_id=cell_id,
            sweep=cell.get("_sweep", "unknown"),
            n_jobs=cell["n_jobs"],
            max_retained_terms=cell.get("max_retained_terms"),
            n_permutations=cell["n_permutations"],
            n_tree_estimators=cell["n_tree_estimators"],
            max_tree_depth=cell["max_tree_depth"],
            n_samples=n_samples,
            n_features=0,
            n_retained_terms=n_retained,
            n_candidate_pairs=n_pairs,
            elapsed_seconds=cell_elapsed,
            status="ok",
        )

    except subprocess.TimeoutExpired:
        return CellResult(
            cell_id=cell_id,
            sweep=cell.get("_sweep", "unknown"),
            n_jobs=cell["n_jobs"],
            max_retained_terms=cell.get("max_retained_terms"),
            n_permutations=cell["n_permutations"],
            n_tree_estimators=cell["n_tree_estimators"],
            max_tree_depth=cell["max_tree_depth"],
            n_samples=0,
            n_features=0,
            n_retained_terms=None,
            n_candidate_pairs=None,
            elapsed_seconds=None,
            status="timeout",
            error_message=f"Cell exceeded {timeout}s timeout",
        )


def _int_or_none(value: Any) -> int | None:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Results writer
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "cell_id",
    "sweep",
    "n_jobs",
    "max_retained_terms",
    "n_permutations",
    "n_tree_estimators",
    "max_tree_depth",
    "n_samples",
    "n_features",
    "n_retained_terms",
    "n_candidate_pairs",
    "elapsed_seconds",
    "status",
    "error_message",
    "recorded_at_utc",
    "hostname",
]


def _write_results(results: list[CellResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        w.writeheader()
        for r in results:
            w.writerow(asdict(r))


def _print_summary(results: list[CellResult], max_cores: int) -> None:
    ok = [r for r in results if r.status == "ok"]
    err = [r for r in results if r.status != "ok"]

    print(f"\n{'=' * 60}")
    print(f"BENCHMARK COMPLETE: {len(ok)}/{len(results)} cells succeeded")
    print(f"{'=' * 60}")

    if ok:
        print(
            f"\n{'Sweep':<18} {'n_jobs':>7} {'max_ret':>8} {'n_perm':>7} "
            f"{'n_trees':>8} {'elapsed(s)':>11} {'pairs':>8}"
        )
        print("-" * 75)
        for r in sorted(ok, key=lambda x: (x.sweep, x.n_jobs, x.max_retained_terms or 0)):
            print(
                f"{r.sweep:<18} {r.n_jobs:>7} "
                f"{str(r.max_retained_terms or 'uncapped'):>8} "
                f"{r.n_permutations:>7} {r.n_tree_estimators:>8} "
                f"{r.elapsed_seconds:>11.1f} "
                f"{str(r.n_candidate_pairs or '?'):>8}"
            )

    if err:
        print(f"\n{'=' * 60}")
        print(f"FAILED / TIMED OUT: {len(err)} cells")
        for r in err:
            print(f"  [{r.cell_id}] {r.status}: {r.error_message or ''}")


# ---------------------------------------------------------------------------
# Detect available dataset
# ---------------------------------------------------------------------------


def _detect_dataset(user_path: str | None) -> Path:
    """Resolve dataset path: user-specified > 3k > 300 > error."""
    if user_path:
        p = Path(user_path)
        if not (p / "X.parquet").exists():
            raise FileNotFoundError(f"No X.parquet found in {p}")
        return p

    candidates = [
        REPO_ROOT / "artifacts" / "test_dataset_3k",
        REPO_ROOT / "artifacts" / "test_dataset_300",
    ]
    for c in candidates:
        if (c / "X.parquet").exists():
            return c
    raise FileNotFoundError(
        "No dataset found. Specify --dataset-path or ensure "
        "artifacts/test_dataset_3k/ or artifacts/test_dataset_300/ exists."
    )


def _detect_dataset_shape(dataset_path: Path) -> tuple[int, int]:
    """Return (n_samples, n_features) from X.parquet."""
    try:
        import pandas as pd

        df = pd.read_parquet(dataset_path / "X.parquet")
        return df.shape
    except Exception:
        return 0, 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="BSM HPC single-node scaling benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          pixi run hpc-scaling-benchmark
          pixi run hpc-scaling-benchmark -- --quick --max-cores 16
          pixi run hpc-scaling-benchmark -- --output-root /scratch/$USER/bsm_bench
          pixi run hpc-scaling-benchmark -- --dry-run
        """),
    )
    parser.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "artifacts" / "hpc_scaling_benchmark"),
        help="Directory to write results into (default: artifacts/hpc_scaling_benchmark/)",
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Path to dataset directory containing X.parquet (auto-detected if omitted)",
    )
    parser.add_argument(
        "--max-cores",
        type=int,
        default=None,
        help="Override detected CPU count (default: os.cpu_count())",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a 3-cell smoke grid only (fast, ~5 minutes)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the cell matrix without executing anything",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=14400,  # 4 hours
        help="Per-cell timeout in seconds (default: 14400 = 4 hours)",
    )
    parser.add_argument(
        "--sweeps",
        nargs="+",
        choices=["core_scaling", "feature_scaling", "perm_scaling", "tree_scaling", "cross_sweep"],
        default=None,
        help="Run only specific sweeps (default: all sweeps)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    max_cores = args.max_cores or os.cpu_count() or 1
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    dataset_path = _detect_dataset(args.dataset_path)
    n_samples, n_features = _detect_dataset_shape(dataset_path)

    print("\nBSM HPC Scaling Benchmark")
    print(f"  Dataset  : {dataset_path} ({n_samples} samples, {n_features} features)")
    print(f"  Max cores: {max_cores}")
    print(f"  Output   : {output_root}")

    # Build grid
    if args.quick:
        cells = []
        for c in QUICK_CELLS:
            cell = dict(MODERATE_DEFAULTS)
            cell.update(c)
            cell["_sweep"] = "quick"
            if cell["n_jobs"] == -1:
                cell["n_jobs"] = max_cores
            cells.append(cell)
    else:
        cells = _build_full_grid(max_cores)
        if args.sweeps:
            cells = [c for c in cells if c.get("_sweep") in args.sweeps]

    # Skip cells that would exceed hardware
    cells = [c for c in cells if int(c["n_jobs"]) <= max_cores]

    print(f"  Cells    : {len(cells)}\n")

    if args.dry_run:
        print(f"{'#':>4}  {'sweep':<18} {'n_jobs':>7} {'max_ret':>8} {'n_perm':>7} {'n_trees':>8}")
        print("-" * 60)
        for i, c in enumerate(cells):
            print(
                f"{i + 1:>4}  {c.get('_sweep', '?'):<18} {c['n_jobs']:>7} "
                f"{str(c.get('max_retained_terms') or 'uncap'):>8} "
                f"{c['n_permutations']:>7} {c['n_tree_estimators']:>8}"
            )
        print(f"\n{len(cells)} cells total (dry run, not executed)")
        return 0

    # Build stub artifacts (shared across all cells)
    try:
        stub_dir = _build_stub_artifacts(
            dataset_path=dataset_path,
            output_root=output_root,
            retained_terms=MODERATE_DEFAULTS["max_retained_terms"],
            variance_threshold=MODERATE_DEFAULTS["variance_threshold"],
            seed=MODERATE_DEFAULTS["seed"],
        )
    except Exception as e:
        logger.error("Failed to build stub artifacts: %s", e)
        return 1

    results_csv = output_root / "scaling_results.csv"
    log_path = output_root / "benchmark_log.txt"

    manifest = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_path),
        "n_samples": n_samples,
        "n_features": n_features,
        "max_cores": max_cores,
        "n_cells": len(cells),
        "hostname": os.uname().nodename,
        "python": sys.executable,
        "mode": "quick" if args.quick else "full",
    }
    (output_root / "benchmark_manifest.json").write_text(json.dumps(manifest, indent=2))

    results: list[CellResult] = []

    with log_path.open("w") as log_fh:
        log_fh.write(f"BSM HPC Scaling Benchmark — {manifest['started_at_utc']}\n")
        log_fh.write(f"dataset: {dataset_path}\n")
        log_fh.write(f"max_cores: {max_cores}\n\n")

        for i, cell in enumerate(cells):
            cell_id = f"cell_{i + 1:03d}_{cell.get('_sweep', 'unknown')}"
            sweep = cell.get("_sweep", "?")
            print(
                f"[{i + 1:2d}/{len(cells)}] {sweep:<18} "
                f"n_jobs={cell['n_jobs']:>3}  "
                f"max_ret={str(cell.get('max_retained_terms') or 'uncap'):>6}  "
                f"n_perm={cell['n_permutations']:>4}  "
                f"n_trees={cell['n_tree_estimators']:>4}  ...",
                end=" ",
                flush=True,
            )

            r = _run_cell(
                cell=cell,
                cell_id=cell_id,
                stub_dir=stub_dir,
                dataset_path=dataset_path,
                output_root=output_root,
                log_fh=log_fh,
                timeout=args.timeout,
            )
            results.append(r)
            _write_results(results, results_csv)  # checkpoint after every cell

            if r.status == "ok":
                print(f"✓ {r.elapsed_seconds:.1f}s  ({r.n_candidate_pairs or '?'} pairs)")
            else:
                print(f"✗ {r.status}: {r.error_message or ''}")

    # Finalize
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["n_ok"] = sum(1 for r in results if r.status == "ok")
    manifest["n_error"] = sum(1 for r in results if r.status != "ok")
    (output_root / "benchmark_manifest.json").write_text(json.dumps(manifest, indent=2))

    _print_summary(results, max_cores)
    print(f"\nResults : {results_csv}")
    print(f"Log     : {log_path}")
    print(f"\nNext: pixi run hpc-compute-calculator -- --results {results_csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
