#!/usr/bin/env python
"""BSM HPC Compute Calculator — runtime estimator from benchmark data.

Reads the CSV output of hpc_scaling_benchmark.py, fits a log-linear scaling
model, then provides a command-line interface for estimating runtimes for
arbitrary configurations.

Model
-----
The dominant cost is interaction_discovery. The wall-clock time scales as:

    T = alpha * C(N, 2) * B * E / n_jobs^gamma * (S / S_ref)^delta

Where:
    N       = max_retained_terms (feature count after screening)
    C(N,2)  = N*(N-1)/2  (pair count)
    B       = n_permutations
    E       = n_tree_estimators
    n_jobs  = parallel workers
    S       = n_samples
    S_ref   = reference sample count (from benchmark data)
    alpha   = fitted intercept (seconds per pair·perm·tree on 1 core)
    gamma   = parallelism efficiency exponent  (ideal=1.0, Amdahl ~0.8–0.95)
    delta   = sample scaling exponent (usually ~1.0–1.5 for tree methods)

Fitting is done in log space via least-squares regression, so all exponents
are estimated simultaneously from the benchmark data.

Usage::

    # Fit model and show goodness-of-fit
    pixi run hpc-compute-calculator -- --results artifacts/hpc_scaling_benchmark/scaling_results.csv

    # Estimate a specific config
    pixi run hpc-compute-calculator -- \\
        --results artifacts/hpc_scaling_benchmark/scaling_results.csv \\
        --n-jobs 32 \\
        --max-retained-terms 300 \\
        --n-permutations 21 \\
        --n-tree-estimators 100 \\
        --n-samples 3000

    # Sweep over core counts for a fixed config (decision table)
    pixi run hpc-compute-calculator -- \\
        --results artifacts/hpc_scaling_benchmark/scaling_results.csv \\
        --sweep-cores 1 2 4 8 16 32 64 104 \\
        --max-retained-terms 350 \\
        --n-permutations 21 \\
        --n-tree-estimators 100 \\
        --n-samples 30000

    # Save model + report
    pixi run hpc-compute-calculator -- \\
        --results artifacts/hpc_scaling_benchmark/scaling_results.csv \\
        --save-model artifacts/hpc_scaling_benchmark/compute_model.json \\
        --report artifacts/hpc_scaling_benchmark/compute_report.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import textwrap
from pathlib import Path
from typing import Any

# Optional: scipy for curve fitting; numpy for fallback OLS
try:
    import numpy as np

    _NUMPY = True
except ImportError:
    _NUMPY = False

try:
    from scipy import stats as scipy_stats  # noqa: F401

    _SCIPY = True
except ImportError:
    _SCIPY = False


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_results(csv_path: str) -> list[dict[str, Any]]:
    """Load benchmark results CSV; return only successful cells."""
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row.get("status") != "ok":
                continue
            elapsed = _float_or_none(row.get("elapsed_seconds"))
            n_retained = _int_or_none(row.get("n_retained_terms"))
            n_pairs = _int_or_none(row.get("n_candidate_pairs"))
            n_jobs = _int_or_none(row.get("n_jobs"))
            n_perm = _int_or_none(row.get("n_permutations"))
            n_trees = _int_or_none(row.get("n_tree_estimators"))
            n_samples = _int_or_none(row.get("n_samples"))

            if None in (elapsed, n_jobs, n_perm, n_trees) or elapsed <= 0:
                continue
            # Infer n_pairs from n_retained if not directly recorded
            if n_pairs is None and n_retained is not None:
                n_pairs = n_retained * (n_retained - 1) // 2

            rows.append(
                {
                    "sweep": row.get("sweep", "unknown"),
                    "n_jobs": n_jobs,
                    "max_retained_terms": _int_or_none(row.get("max_retained_terms")),
                    "n_permutations": n_perm,
                    "n_tree_estimators": n_trees,
                    "n_samples": n_samples or 0,
                    "n_retained_terms": n_retained,
                    "n_candidate_pairs": n_pairs,
                    "elapsed_seconds": elapsed,
                }
            )
    return rows


def _float_or_none(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _int_or_none(v: Any) -> int | None:
    try:
        f = float(v)
        return int(f) if not math.isnan(f) else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------


def _n_pairs(n_features: int) -> int:
    return max(1, n_features * (n_features - 1) // 2)


def fit_model(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Fit log-linear scaling model to benchmark data.

    Returns a model dict with:
      log_alpha, gamma (parallelism exp), delta (sample exp),
      beta_pairs (pair-count exponent, should be ~1)
      beta_perm  (permutation exponent, should be ~1)
      beta_trees (tree exponent, should be ~1)
      r_squared, n_observations, reference_n_samples
    """
    if not _NUMPY:
        raise ImportError("numpy is required for model fitting. Run: pip install numpy")

    # Filter rows with all required fields
    valid = [
        r
        for r in rows
        if all(
            [
                r["n_jobs"],
                r["n_candidate_pairs"],
                r["n_permutations"],
                r["n_tree_estimators"],
                r["elapsed_seconds"] > 0,
            ]
        )
    ]

    if len(valid) < 3:
        raise ValueError(
            f"Need at least 3 successful benchmark cells to fit model, "
            f"got {len(valid)}. Run hpc_scaling_benchmark.py first."
        )

    ref_samples = _median([r["n_samples"] for r in valid if r["n_samples"] > 0]) or 1

    # Build design matrix: log(T) = log(alpha) + b_pairs*log(pairs)
    #   + b_perm*log(perms) + b_trees*log(trees) - gamma*log(jobs)
    #   + delta*log(samples/ref)
    log_T = []
    X_rows = []
    for r in valid:
        samples = r["n_samples"] or ref_samples
        log_T.append(math.log(r["elapsed_seconds"]))
        X_rows.append(
            [
                1.0,  # intercept → log(alpha)
                math.log(r["n_candidate_pairs"]),  # b_pairs
                math.log(r["n_permutations"]),  # b_perm
                math.log(r["n_tree_estimators"]),  # b_trees
                -math.log(r["n_jobs"]),  # gamma (sign flipped)
                math.log(max(samples, 1) / max(ref_samples, 1)),  # delta
            ]
        )

    X = np.array(X_rows)
    y = np.array(log_T)

    # OLS via normal equations (or lstsq for stability)
    coef, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ coef
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    log_alpha, b_pairs, b_perm, b_trees, gamma, delta = coef

    model = {
        "log_alpha": float(log_alpha),
        "alpha_seconds_per_unit": float(math.exp(log_alpha)),
        "b_pairs": float(b_pairs),
        "b_perm": float(b_perm),
        "b_trees": float(b_trees),
        "gamma": float(gamma),  # parallelism exponent
        "delta": float(delta),  # sample scaling exponent
        "r_squared": float(r_squared),
        "n_observations": len(valid),
        "reference_n_samples": int(ref_samples),
        "notes": (
            "b_pairs/b_perm/b_trees should be ~1.0 for linear scaling. "
            "gamma should be 0.8–1.0 for good parallelism. "
            "r_squared > 0.95 indicates a reliable model."
        ),
    }
    return model


def _median(values: list[float | int]) -> float | None:
    vals = sorted(v for v in values if v is not None and v > 0)
    if not vals:
        return None
    mid = len(vals) // 2
    if len(vals) % 2 == 0:
        return (vals[mid - 1] + vals[mid]) / 2.0
    return float(vals[mid])


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def predict_seconds(
    model: dict[str, Any],
    n_retained_terms: int,
    n_permutations: int,
    n_tree_estimators: int,
    n_jobs: int,
    n_samples: int,
) -> float:
    """Predict interaction_discovery wall-clock seconds from model.

    Parameters
    ----------
    model
        Fitted model dict from fit_model().
    n_retained_terms
        Number of retained features after empirical null screening.
        Pairs = C(n_retained_terms, 2).
    n_permutations
        Permutation trials per pair.
    n_tree_estimators
        Number of gradient boosting trees.
    n_jobs
        Parallel workers.
    n_samples
        Training sample count.

    Returns
    -------
    float
        Estimated seconds.
    """
    pairs = _n_pairs(n_retained_terms)
    ref = model["reference_n_samples"]
    log_t = (
        model["log_alpha"]
        + model["b_pairs"] * math.log(pairs)
        + model["b_perm"] * math.log(n_permutations)
        + model["b_trees"] * math.log(n_tree_estimators)
        - model["gamma"] * math.log(max(n_jobs, 1))
        + model["delta"] * math.log(max(n_samples, 1) / max(ref, 1))
    )
    return math.exp(log_t)


def _fmt_duration(seconds: float) -> str:
    """Human-readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    if seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------


def _model_summary(model: dict[str, Any]) -> str:
    lines = [
        "## Fitted Scaling Model",
        "",
        "```",
        "T(N, B, E, n_jobs, S) = α × C(N,2)^β₁ × B^β₂ × E^β₃ / n_jobs^γ × (S/S_ref)^δ",
        "```",
        "",
        f"  α (seconds/unit)  = {model['alpha_seconds_per_unit']:.3e}",
        f"  β₁ (pair exp)     = {model['b_pairs']:.3f}  (ideal = 1.0)",
        f"  β₂ (perm exp)     = {model['b_perm']:.3f}  (ideal = 1.0)",
        f"  β₃ (tree exp)     = {model['b_trees']:.3f}  (ideal = 1.0)",
        f"  γ  (parallel exp) = {model['gamma']:.3f}  (ideal = 1.0, Amdahl < 1.0)",
        f"  δ  (sample exp)   = {model['delta']:.3f}  "
        f"(reference S = {model['reference_n_samples']})",
        f"  R²                = {model['r_squared']:.4f}  (>0.95 = reliable)",
        f"  n observations    = {model['n_observations']}",
    ]
    return "\n".join(lines)


def generate_decision_table(
    model: dict[str, Any],
    n_retained_terms: int,
    n_permutations: int,
    n_tree_estimators: int,
    n_samples: int,
    core_counts: list[int],
) -> str:
    """Generate a core-count decision table for fixed config."""
    lines = [
        f"## Decision Table: n_retained={n_retained_terms}  "
        f"n_perm={n_permutations}  n_trees={n_tree_estimators}  "
        f"n_samples={n_samples}",
        "",
        f"  {'Cores':>6}  {'Pairs':>8}  {'Estimated time':>16}  {'Speedup':>8}",
        "  " + "-" * 45,
    ]
    pairs = _n_pairs(n_retained_terms)
    t1 = predict_seconds(model, n_retained_terms, n_permutations, n_tree_estimators, 1, n_samples)
    for n_jobs in core_counts:
        t = predict_seconds(
            model, n_retained_terms, n_permutations, n_tree_estimators, n_jobs, n_samples
        )
        speedup = t1 / t
        lines.append(f"  {n_jobs:>6}  {pairs:>8,d}  {_fmt_duration(t):>16}  {speedup:>7.1f}×")
    return "\n".join(lines)


def generate_knob_table(
    model: dict[str, Any],
    n_jobs: int,
    n_samples: int,
    n_permutations: int,
    n_tree_estimators: int,
) -> str:
    """Generate a feature-count → runtime table for a fixed n_jobs."""
    feature_counts = [20, 50, 75, 100, 125, 150, 200, 250, 300, 350, 400, 500]
    lines = [
        f"## Feature Count → Runtime (n_jobs={n_jobs}  "
        f"n_perm={n_permutations}  n_trees={n_tree_estimators}  "
        f"n_samples={n_samples})",
        "",
        f"  {'Features':>9}  {'Pairs':>8}  {'Estimated time':>16}",
        "  " + "-" * 38,
    ]
    for n_feat in feature_counts:
        t = predict_seconds(model, n_feat, n_permutations, n_tree_estimators, n_jobs, n_samples)
        lines.append(f"  {n_feat:>9}  {_n_pairs(n_feat):>8,d}  {_fmt_duration(t):>16}")
    return "\n".join(lines)


def build_report(
    model: dict[str, Any],
    rows: list[dict[str, Any]],
    query: dict[str, Any] | None,
    core_counts: list[int],
    n_samples: int,
) -> str:
    """Build a full markdown report."""
    n_retained = query["n_retained_terms"] if query else 100
    n_perms = query["n_permutations"] if query else 21
    n_trees = query["n_tree_estimators"] if query else 100
    n_jobs = query["n_jobs"] if query else max(core_counts)

    parts = [
        "# BSM Compute Calculator Report",
        "",
        _model_summary(model),
        "",
    ]

    if query:
        t = predict_seconds(model, n_retained, n_perms, n_trees, n_jobs, n_samples)
        parts += [
            "## Point Estimate",
            "",
            f"  Config   : n_jobs={n_jobs}, max_retained={n_retained}, "
            f"n_perm={n_perms}, n_trees={n_trees}, n_samples={n_samples}",
            f"  Pairs    : {_n_pairs(n_retained):,}",
            f"  Estimated: **{_fmt_duration(t)}** ({t:.0f}s)",
            "",
        ]

    parts += [
        generate_decision_table(model, n_retained, n_perms, n_trees, n_samples, core_counts),
        "",
        generate_knob_table(model, n_jobs, n_samples, n_perms, n_trees),
        "",
        "## Benchmark Observations Used for Fit",
        "",
        f"  {'sweep':<18} {'n_jobs':>7} {'max_ret':>8} {'n_perm':>7} "
        f"{'n_trees':>8} {'elapsed(s)':>11} {'pairs':>8}",
        "  " + "-" * 75,
    ]
    for r in sorted(rows, key=lambda x: (x["sweep"], x["n_jobs"], x["max_retained_terms"] or 0)):
        parts.append(
            f"  {r['sweep']:<18} {r['n_jobs']:>7} "
            f"{str(r['max_retained_terms'] or 'uncap'):>8} "
            f"{r['n_permutations']:>7} {r['n_tree_estimators']:>8} "
            f"{r['elapsed_seconds']:>11.1f} "
            f"{str(r['n_candidate_pairs'] or '?'):>8}"
        )
    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="BSM HPC compute calculator — estimates runtime from benchmark data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          # Show model fit
          pixi run hpc-compute-calculator -- \\
              --results artifacts/hpc_scaling_benchmark/scaling_results.csv

          # Point estimate
          pixi run hpc-compute-calculator -- \\
              --results artifacts/hpc_scaling_benchmark/scaling_results.csv \\
              --n-jobs 64 --max-retained-terms 350 \\
              --n-permutations 21 --n-tree-estimators 100 --n-samples 30000

          # Core sweep decision table
          pixi run hpc-compute-calculator -- \\
              --results artifacts/hpc_scaling_benchmark/scaling_results.csv \\
              --sweep-cores 1 2 4 8 16 32 64 104 \\
              --max-retained-terms 350 --n-samples 30000

          # Save report
          pixi run hpc-compute-calculator -- \\
              --results artifacts/hpc_scaling_benchmark/scaling_results.csv \\
              --report artifacts/hpc_scaling_benchmark/compute_report.md
        """),
    )
    parser.add_argument("--results", required=True, help="Path to scaling_results.csv")
    parser.add_argument("--n-jobs", type=int, default=None, help="Cores for point estimate")
    parser.add_argument(
        "--max-retained-terms",
        type=int,
        default=None,
        help="Feature count (retained after screening)",
    )
    parser.add_argument("--n-permutations", type=int, default=21)
    parser.add_argument("--n-tree-estimators", type=int, default=100)
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Sample count for estimate (default: benchmark reference)",
    )
    parser.add_argument(
        "--sweep-cores",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 104],
        help="Core counts for decision table",
    )
    parser.add_argument("--save-model", default=None, help="Save fitted model as JSON")
    parser.add_argument("--report", default=None, help="Save markdown report to path")
    args = parser.parse_args(argv)

    if not _NUMPY:
        print("ERROR: numpy is required. Run: pixi install", file=sys.stderr)
        return 1

    # Load and fit
    rows = load_results(args.results)
    print(f"Loaded {len(rows)} valid benchmark observations from {args.results}")

    if len(rows) < 3:
        print(
            "ERROR: need ≥3 successful benchmark cells to fit a model. "
            "Run hpc_scaling_benchmark.py first.",
            file=sys.stderr,
        )
        return 1

    try:
        model = fit_model(rows)
    except Exception as e:
        print(f"ERROR fitting model: {e}", file=sys.stderr)
        return 1

    # Print model summary
    print()
    print(_model_summary(model))

    # Resolve query params
    n_retained = args.max_retained_terms or 100
    n_perms = args.n_permutations
    n_trees = args.n_tree_estimators
    n_samples = args.n_samples or model["reference_n_samples"]
    n_jobs = args.n_jobs or max(args.sweep_cores)

    query = (
        {
            "n_jobs": n_jobs,
            "n_retained_terms": n_retained,
            "n_permutations": n_perms,
            "n_tree_estimators": n_trees,
            "n_samples": n_samples,
        }
        if args.n_jobs or args.max_retained_terms
        else None
    )

    if query:
        t = predict_seconds(model, n_retained, n_perms, n_trees, n_jobs, n_samples)
        print("\n--- Point Estimate ---")
        print(
            f"  n_jobs={n_jobs}, max_retained={n_retained}, "
            f"n_perm={n_perms}, n_trees={n_trees}, n_samples={n_samples}"
        )
        print(f"  Pairs    : {_n_pairs(n_retained):,}")
        print(f"  Estimated: {_fmt_duration(t)} ({t:.0f}s)")

    print()
    print(generate_decision_table(model, n_retained, n_perms, n_trees, n_samples, args.sweep_cores))
    print()
    print(generate_knob_table(model, n_jobs, n_samples, n_perms, n_trees))

    if args.save_model:
        Path(args.save_model).write_text(json.dumps(model, indent=2))
        print(f"\nModel saved: {args.save_model}")

    if args.report:
        report = build_report(model, rows, query, args.sweep_cores, n_samples)
        Path(args.report).write_text(report)
        print(f"Report saved: {args.report}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
