#!/usr/bin/env python3
# Copyright (c) 2026 Dylan Hettinger
"""Small-scale prespecified recovery study driver.

Runs each of the 7 prespecified scenarios at a documented reduced scale,
producing released aggregate tables, figure data, and a reproduction log.
The empirical interaction-FWER is measured on the actual interaction-discovery
+ ``max_t`` selection stage.

Quick mode (``--quick``) runs at a tiny scale for smoke testing; the reduced
scale is documented in the reproduction log as a deliberate reduction that
preserves correlated inputs, binaries, multivariate responses, PCA reduction,
structured discovery, and train-only selection.

Usage
-----
    pixi run python scripts/run_recovery_study.py [--quick] [--output-dir DIR] [--seed N]
    pixi run python scripts/run_recovery_study.py --help
"""

from __future__ import annotations

import argparse
import datetime
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from rfm_pipeline import multiplicity_controlled_interaction_selection
from rfm_pipeline.baselines import (
    ElasticNetBaseline,
    GBTBaseline,
    OracleOLSBaseline,
    compare_baselines,
)
from rfm_pipeline.recovery_study import (
    RecoveryScenario,
    empirical_interaction_fwer,
    prespecified_recovery_scenarios,
    recovery_estimands,
)
from rfm_pipeline.synthetic_dgp import (
    DGPTrueSupport,
    SyntheticDGPSpec,
    generate_calibrated_structure_synthetic,
    generate_pure_synthetic,
)

# ---------------------------------------------------------------------------
# Scale configuration
# ---------------------------------------------------------------------------

_QUICK_DESCRIPTION = (
    "quick-smoke: n_inputs=12, n_outputs=5, n_runs=200, B=19, B_screen=19, "
    "fwer_reps=10 — deliberately reduced from full scale. "
    "Preserves correlated inputs (AR(1) rho), binary interaction structure, "
    "multivariate responses (multi-output Y), PCA reduction of Y, "
    "structured interaction discovery (permutation null scoring), and "
    "train-only selection (holdout withheld during all tuning). "
    "Candidate-family logic unchanged; scale chosen for sub-minute smoke runs."
)

_FULL_DESCRIPTION = (
    "reduced-local: n_inputs=30, n_outputs=40, n_runs=2000, B=199, "
    "B_screen=199, fwer_reps=100, alt_reps=20 — deliberate reduction from HPC scale. "
    "Preserves correlated inputs (AR(1) rho), binary interaction structure, "
    "multivariate responses (multi-output Y), PCA reduction of Y, "
    "structured interaction discovery (permutation null scoring), and "
    "train-only selection (holdout withheld during all tuning). "
    "Candidate-family logic (BH screening, max_t selection) unchanged; "
    "scale chosen so the full study completes in minutes on a laptop."
)


@dataclass(frozen=True)
class StudyScale:
    """Scale parameters for one recovery study run."""

    n_inputs: int
    n_outputs: int
    n_runs: int
    B: int
    B_screen: int
    fwer_reps: int
    alt_reps: int
    alpha: float
    description: str


_QUICK_SCALE = StudyScale(
    n_inputs=12,
    n_outputs=5,
    n_runs=200,
    B=19,
    B_screen=19,
    fwer_reps=10,
    alt_reps=1,
    alpha=0.2,
    description=_QUICK_DESCRIPTION,
)

_FULL_SCALE = StudyScale(
    n_inputs=30,
    n_outputs=40,
    n_runs=2000,
    B=199,
    B_screen=199,
    fwer_reps=100,
    alt_reps=20,
    alpha=0.1,
    description=_FULL_DESCRIPTION,
)

# ---------------------------------------------------------------------------
# DGP helpers
# ---------------------------------------------------------------------------


def _override_spec(scenario: RecoveryScenario, scale: StudyScale, seed: int) -> SyntheticDGPSpec:
    """Return a SyntheticDGPSpec from *scenario* with scale overrides applied."""
    s = scenario.spec
    n_inputs = min(scale.n_inputs, s.n_inputs)
    n_outputs = min(scale.n_outputs, s.n_outputs)
    # factor_model_rank must be ≥ 1 and ≤ n_inputs
    factor_rank = max(1, min(s.factor_model_rank, n_inputs))
    return SyntheticDGPSpec(
        n_inputs=n_inputs,
        n_runs=scale.n_runs,
        n_outputs=n_outputs,
        sparsity=s.sparsity,
        interaction_density=s.interaction_density,
        nonlinearity_strength=s.nonlinearity_strength,
        noise_snr=s.noise_snr,
        holdout_fraction=s.holdout_fraction,
        dgp_family=s.dgp_family,
        seed=seed,
        factor_model_rank=factor_rank,
        input_correlation_strength=s.input_correlation_strength,
        per_output_snr_heterogeneity=s.per_output_snr_heterogeneity,
    )


def _generate_dataset(spec: SyntheticDGPSpec):
    """Generate dataset of the right family."""
    if spec.dgp_family == "pure_synthetic":
        return generate_pure_synthetic(spec)
    return generate_calibrated_structure_synthetic(spec)


# ---------------------------------------------------------------------------
# Pipeline utilities (screening + interaction scoring + FWER selection)
# ---------------------------------------------------------------------------


def _extract_train_eval(
    dataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Extract train/eval X and Y arrays.

    Returns X_train, Y_train, X_eval, Y_eval, feature_names.
    """
    input_cols = [c for c in dataset.input_matrix.columns if c != "sample_id"]
    output_cols = [c for c in dataset.output_matrix.columns if c != "sample_id"]
    merged = dataset.holdout_assignments.merge(dataset.input_matrix, on="sample_id").merge(
        dataset.output_matrix, on="sample_id"
    )
    train_mask = merged["split"] == "train"
    X_train = merged.loc[train_mask, input_cols].to_numpy(dtype=np.float64)
    Y_train = merged.loc[train_mask, output_cols].to_numpy(dtype=np.float64)
    X_eval = merged.loc[~train_mask, input_cols].to_numpy(dtype=np.float64)
    Y_eval = merged.loc[~train_mask, output_cols].to_numpy(dtype=np.float64)
    return X_train, Y_train, X_eval, Y_eval, list(input_cols)


def _pca_reduce(Y: np.ndarray, n_components: int) -> np.ndarray:
    """PCA-reduce Y to at most n_components."""
    n, p = Y.shape
    k = min(n_components, p, n - 1)
    if k <= 0:
        return Y
    return PCA(n_components=k, random_state=0).fit_transform(Y)


def _benjamini_hochberg(p_values: np.ndarray, q: float) -> np.ndarray:
    """Return boolean retention mask via BH(q) correction."""
    m = len(p_values)
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p_values)
    sorted_p = p_values[order]
    thresholds = (np.arange(1, m + 1) * q) / m
    # Largest rank passing: all ranks up to and including it are retained.
    passing = np.where(sorted_p <= thresholds)[0]
    retained_sorted = np.zeros(m, dtype=bool)
    if passing.size > 0:
        retained_sorted[: passing[-1] + 1] = True
    retained = np.empty(m, dtype=bool)
    retained[order] = retained_sorted
    return retained


def _screen_inputs(
    X_train: np.ndarray,
    Y_pca: np.ndarray,
    feature_names: list[str],
    B: int,
    rng: np.random.Generator,
    bh_q: float = 0.20,
) -> tuple[list[str], pd.DataFrame]:
    """Screen inputs with a permutation-based BH test (L2-norm of corr vector).

    Returns (retained_names, screening_stats_df).
    """
    n, p_in = X_train.shape
    X_std = (X_train - X_train.mean(0)) / np.where(
        X_train.std(0, ddof=1) > 1e-12, X_train.std(0, ddof=1), 1.0
    )
    Y_std = (Y_pca - Y_pca.mean(0)) / np.where(
        Y_pca.std(0, ddof=1) > 1e-12, Y_pca.std(0, ddof=1), 1.0
    )
    # Observed
    corr = X_std.T @ Y_std / n  # (p_in, n_comps)
    observed = np.linalg.norm(corr, axis=1)  # (p_in,)
    # Null
    null_stats = np.zeros((B, p_in), dtype=np.float64)
    for b in range(B):
        perm = rng.permutation(n)
        null_stats[b] = np.linalg.norm(X_std.T @ Y_std[perm] / n, axis=1)
    p_values = (1.0 + (null_stats >= observed[None, :]).sum(0)) / (B + 1.0)
    retained_mask = _benjamini_hochberg(p_values, bh_q)
    retained_names = [feature_names[i] for i in range(p_in) if retained_mask[i]]
    stats_df = pd.DataFrame(
        {
            "feature_name": feature_names,
            "observed_statistic": observed,
            "empirical_p_value": p_values,
            "retained": retained_mask,
        }
    )
    return retained_names, stats_df


def _residualize_on(design: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return the residual of *target* after least-squares projection onto *design*.

    An intercept column is appended to *design* so the projection removes both
    the mean and the linear main-effect component.  Used to make interaction
    scoring *hierarchical*: interaction signal is measured only as the value it
    adds over the retained main effects, so a scenario with main effects but no
    true interactions (e.g. ``global_null``/``interaction_null``) does not leak
    main-effect signal into the interaction scores.
    """
    n = target.shape[0]
    M = np.column_stack([np.ones(n), design]) if design.size else np.ones((n, 1))
    coef, *_ = np.linalg.lstsq(M, target, rcond=None)
    return target - M @ coef


def _score_interaction_pairs(
    X_main: np.ndarray,
    Y_pca: np.ndarray,
    inter_columns: list[np.ndarray],
    B: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute observed interaction scores and permutation null.

    The interaction features and the PCA responses are first residualized on the
    retained main-effect design *X_main* (hierarchical control), so the score
    reflects only the interaction's added value over main effects.  Score per
    pair = max_k |corr(resid(xi*xj), resid(pc_k))| over PCA components; the null
    permutes the response residuals (shared-response permutations).

    Returns (observed, null_stats) shapes (n_pairs,) and (B, n_pairs).
    """
    n_pairs = len(inter_columns)
    if n_pairs == 0:
        return np.zeros(0, dtype=np.float64), np.zeros((B, 0), dtype=np.float64)
    n = Y_pca.shape[0]
    X_inter = np.column_stack(inter_columns)  # (n, n_pairs)
    # Hierarchical residualization on the retained main-effect design.
    X_inter_res = _residualize_on(X_main, X_inter)
    Y_res = _residualize_on(X_main, Y_pca)
    X_inter_std = (X_inter_res - X_inter_res.mean(0)) / np.where(
        X_inter_res.std(0, ddof=1) > 1e-12, X_inter_res.std(0, ddof=1), 1.0
    )
    Y_std = (Y_res - Y_res.mean(0)) / np.where(
        Y_res.std(0, ddof=1) > 1e-12, Y_res.std(0, ddof=1), 1.0
    )
    corr = X_inter_std.T @ Y_std / n  # (n_pairs, n_comps)
    observed = np.abs(corr).max(axis=1)
    null_stats = np.zeros((B, n_pairs), dtype=np.float64)
    for b in range(B):
        perm = rng.permutation(n)
        cn = X_inter_std.T @ Y_std[perm] / n
        null_stats[b] = np.abs(cn).max(axis=1)
    return observed, null_stats


def _run_pipeline(
    dataset,
    scale: StudyScale,
    rng: np.random.Generator,
    true_support_override: DGPTrueSupport | None = None,
) -> dict:
    """Run the simplified screening + interaction discovery + FWER selection pipeline.

    Returns a result dict with keys:
        selected_support, stage_retention, comparator_df,
        pair_names, observed_scores, null_stats.
    """
    X_train, Y_train, X_eval, Y_eval, feature_names = _extract_train_eval(dataset)
    true_support = true_support_override or dataset.true_support

    # PCA-reduce Y to at most 8 components (train-only)
    n_pca = min(8, Y_train.shape[1])
    Y_pca = _pca_reduce(Y_train, n_pca)

    # Stage 1: screen inputs
    retained_names, screen_stats = _screen_inputs(
        X_train, Y_pca, feature_names, scale.B_screen, rng, bh_q=0.20
    )
    stage_retention: dict[str, dict[str, int]] = {
        "screening": {
            "n_candidates": len(feature_names),
            "n_retained": len(retained_names),
        }
    }

    # Stage 2: candidate pairs from retained inputs
    retained_indices = [feature_names.index(n) for n in retained_names]
    pair_names_list: list[str] = []
    pair_indices_list: list[tuple[int, int]] = []
    for a, b in combinations(range(len(retained_indices)), 2):
        li, lj = retained_indices[a], retained_indices[b]
        fname_i, fname_j = feature_names[li], feature_names[lj]
        pair_names_list.append(f"{fname_i}:{fname_j}")
        pair_indices_list.append((li, lj))

    selected_interactions: frozenset[tuple[str, str]] = frozenset()
    observed_scores = np.zeros(0, dtype=np.float64)
    null_stats_arr = np.zeros((scale.B, 0), dtype=np.float64)

    # Stage 3: interaction scoring + exact max_t selection
    if len(pair_names_list) >= 1 and scale.B >= 1:
        X_ret = X_train[:, retained_indices]
        # Hierarchical main-effect design: retained mains plus their nonlinear
        # (quadratic) transforms, mirroring the workflow's main-effect +
        # transformation support.  Interactions are scored only for the value
        # they add over this basis, so nonlinear main-effect signal does not
        # leak into interaction scores.
        X_main = np.column_stack([X_ret, X_ret**2]) if X_ret.size else X_ret
        inter_columns = [X_train[:, li] * X_train[:, lj] for li, lj in pair_indices_list]
        observed_scores, null_stats_arr = _score_interaction_pairs(
            X_main, Y_pca, inter_columns, scale.B, rng
        )
        selected_bool, _p_adj, _threshold = multiplicity_controlled_interaction_selection(
            observed_scores, null_stats_arr, scale.alpha, method="max_t"
        )
        selected_interactions = frozenset(
            tuple(sorted(pair_names_list[k].split(":", 1)))
            for k in range(len(pair_names_list))
            if selected_bool[k]
        )

    stage_retention["interaction_selection"] = {
        "n_candidates": len(pair_names_list),
        "n_retained": len(selected_interactions),
    }

    selected_support = DGPTrueSupport(
        true_active_inputs=frozenset(retained_names),
        true_active_interactions=selected_interactions,
        true_active_nonlinear=frozenset(),
    )

    # Comparators (oracle-OLS, GBT, elastic-net)
    oracle = OracleOLSBaseline(
        true_active_inputs=true_support.true_active_inputs,
        feature_names=feature_names,
    )
    gbt = GBTBaseline(n_estimators=20, max_depth=2)
    en = ElasticNetBaseline(alpha=0.01)
    if X_eval.shape[0] >= 2 and Y_eval.shape[0] >= 2:
        comparator_df = compare_baselines([oracle, gbt, en], X_train, Y_train, X_eval, Y_eval)
    else:
        comparator_df = pd.DataFrame(
            columns=[
                "name",
                "rmse",
                "r2",
                "model_size_bytes",
                "fit_time_s",
                "eval_time_s",
                "peak_memory_mb",
            ]
        )

    return {
        "selected_support": selected_support,
        "stage_retention": stage_retention,
        "pair_names": pair_names_list,
        "observed_scores": observed_scores,
        "null_stats": null_stats_arr,
        "comparator_df": comparator_df,
    }


# ---------------------------------------------------------------------------
# FWER replication loop (null scenarios only)
# ---------------------------------------------------------------------------

# Scenarios where every selected interaction is a false positive
_NULL_INTERACTION_SCENARIOS = {"global_null", "interaction_null"}


def _run_fwer_replicates(
    scenario: RecoveryScenario,
    scale: StudyScale,
    master_rng: np.random.Generator,
) -> list[int]:
    """Run *scale.fwer_reps* global-null replicates; return per-replicate false pair counts."""
    false_pair_flags: list[int] = []
    for _ in range(scale.fwer_reps):
        rep_seed = int(master_rng.integers(2**31))
        spec = _override_spec(scenario, scale, seed=rep_seed)
        ds = _generate_dataset(spec)
        rep_rng = np.random.default_rng(rep_seed)
        result = _run_pipeline(ds, scale, rep_rng)
        n_false = len(result["selected_support"].true_active_interactions)
        false_pair_flags.append(n_false)
    return false_pair_flags


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _build_fwer_row(
    scenario_name: str,
    alpha: float,
    flags: list[int],
) -> dict:
    stats = empirical_interaction_fwer(flags)
    return {
        "scenario": scenario_name,
        "alpha": alpha,
        "fwer_proportion": stats["fwer_proportion"],
        "wilson_ci_lower": stats["wilson_ci_lower"],
        "wilson_ci_upper": stats["wilson_ci_upper"],
        "n_replicates": stats["n_replicates"],
        "n_false_pair_replicates": stats["n_false_pair_replicates"],
        "mean_false_pair_count": stats["mean_false_pair_count"],
    }


def _build_estimand_rows(
    scenario_name: str, true_support: DGPTrueSupport, result: dict
) -> list[dict]:
    est = recovery_estimands(true_support, result["selected_support"])
    rows = []
    for family, metrics in est.items():
        rows.append(
            {
                "scenario": scenario_name,
                "family": family,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "fdp": metrics["fdp"],
                "exact_support_recovery": metrics["exact_support_recovery"],
                "selected_size": metrics["selected_size"],
            }
        )
    return rows


def _build_retention_rows(scenario_name: str, stage_retention: dict) -> list[dict]:
    rows = []
    for stage, counts in stage_retention.items():
        rows.append(
            {
                "scenario": scenario_name,
                "stage": stage,
                "n_candidates": counts["n_candidates"],
                "n_retained": counts["n_retained"],
            }
        )
    return rows


def _build_comparator_rows(scenario_name: str, comparator_df: pd.DataFrame) -> list[dict]:
    rows = []
    for _, row in comparator_df.iterrows():
        rows.append(
            {
                "scenario": scenario_name,
                "name": row["name"],
                "rmse": row["rmse"],
                "r2": row["r2"],
                "model_size_bytes": row.get("model_size_bytes", 0),
                "fit_time_s": row.get("fit_time_s", 0.0),
                "eval_time_s": row.get("eval_time_s", 0.0),
                "peak_memory_mb": row.get("peak_memory_mb", 0.0),
            }
        )
    return rows


def _write_reproduction_log(
    output_dir: Path,
    scale: StudyScale,
    scenarios: list[RecoveryScenario],
    master_seed: int,
    success_criteria: dict,
    start_ts: datetime.datetime,
    end_ts: datetime.datetime,
    rfm_version: str,
) -> Path:
    """Write reproduction_log.md documenting the run manifest."""
    path = output_dir / "reproduction_log.md"
    lines = [
        "# Recovery Study Reproduction Log",
        "",
        f"Generated: {end_ts.isoformat()}",
        f"Master seed: {master_seed}",
        f"rfm-pipeline version: {rfm_version}",
        f"Run duration: {(end_ts - start_ts).total_seconds():.1f}s",
        "",
        "## Scale (deliberate reduction from HPC scale)",
        "",
        scale.description,
        "",
        "## Scale parameters",
        "",
        f"- n_inputs: {scale.n_inputs} (per-scenario cap; original scenarios specify 20–30)",
        f"- n_outputs: {scale.n_outputs} (per-scenario cap; original scenarios specify 5–10)",
        f"- n_runs: {scale.n_runs} (original HPC runs: ~50 000+)",
        f"- B (interaction null permutations): {scale.B}",
        f"- B_screen (screening null permutations): {scale.B_screen}",
        f"- fwer_reps (global-null FWER replicates): {scale.fwer_reps}",
        f"- alt_reps: {scale.alt_reps}",
        f"- alpha: {scale.alpha}",
        "",
        "## Preserved properties",
        "",
        "The reduced scale preserves:",
        "- Correlated inputs (AR(1) rho ≥ 0.8 in correlated_redundant scenario)",
        "- Binary/multivariate responses (multi-output Y with n_outputs ≥ 5)",
        "- PCA reduction of Y before stage computations",
        "- Structured interaction discovery (permutation null scoring per pair)",
        "- Train-only selection (holdout withheld during all BH screening and FWER selection)",
        "- Candidate-family logic unchanged (BH screening → max_t selection)",
        "",
        "## Scenario manifest",
        "",
    ]
    for sc in scenarios:
        lines += [
            f"### {sc.name}",
            f"- seed: {sc.seed}",
            f"- DGP family: {sc.spec.dgp_family}",
            f"- interaction_density: {sc.spec.interaction_density}",
            f"- nonlinearity_strength: {sc.spec.nonlinearity_strength}",
            f"- sparsity: {sc.spec.sparsity}",
            f"- noise_snr: {sc.spec.noise_snr}",
            f"- planted_interactions: {len(sc.true_support.true_active_interactions)}",
            f"- planted_main_effects: {len(sc.true_support.true_active_inputs)}",
            "",
        ]
    lines += [
        "## Success criteria",
        "",
    ]
    for criterion, value in success_criteria.items():
        lines += [f"- {criterion}: {value}"]
    lines += [""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_recovery_study.py",
        description=(
            "Small-scale prespecified recovery study. Runs all 7 scenarios at a "
            "documented reduced scale and writes released artifacts to outputs/recovery_study/."
        ),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        default=False,
        help="Tiny smoke-test scale (n_inputs=12, n_runs=200, B=19, fwer_reps=10).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory root. Defaults to outputs/recovery_study/ relative to the "
            "repository root (parent of scripts/)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Master random seed for the study (default: 42).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the recovery study end-to-end and write artifacts; return exit code."""
    args = _parse_args(argv)
    scale = _QUICK_SCALE if args.quick else _FULL_SCALE
    master_seed = args.seed

    # Output directory: default to <repo_root>/outputs/recovery_study/
    if args.output_dir is None:
        repo_root = Path(__file__).parent.parent
        output_dir = repo_root / "outputs" / "recovery_study"
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figure_data").mkdir(parents=True, exist_ok=True)

    # Deterministic master RNG
    master_rng = np.random.default_rng(master_seed)

    try:
        from rfm_pipeline import __version__ as rfm_version
    except Exception:
        rfm_version = "unknown"

    start_ts = datetime.datetime.now(tz=datetime.timezone.utc)
    print(f"[recovery_study] scale={scale.description[:60]}...", flush=True)
    print(f"[recovery_study] output_dir={output_dir}", flush=True)

    scenarios = prespecified_recovery_scenarios()

    fwer_rows: list[dict] = []
    estimand_rows: list[dict] = []
    retention_rows: list[dict] = []
    comparator_rows: list[dict] = []

    for sc in scenarios:
        print(f"[recovery_study]  scenario={sc.name}", flush=True)

        # Override the spec to reduced scale but keep seed for the primary run
        spec = _override_spec(sc, scale, seed=sc.seed)
        ds = _generate_dataset(spec)

        # True support for the overridden scenario (may differ from full spec)
        # Use the dataset's own true_support (regenerated at scale-overridden spec)
        # For pure-interaction scenario, override to declare empty main effects
        if sc.name == "pure_interaction":
            true_support = DGPTrueSupport(
                true_active_inputs=frozenset(),
                true_active_interactions=ds.true_support.true_active_interactions,
                true_active_nonlinear=frozenset(),
            )
        else:
            true_support = ds.true_support

        sc_rng = np.random.default_rng(int(master_rng.integers(2**31)))
        result = _run_pipeline(ds, scale, sc_rng, true_support_override=true_support)

        # Recovery estimands
        estimand_rows.extend(_build_estimand_rows(sc.name, true_support, result))

        # Stage retention
        retention_rows.extend(_build_retention_rows(sc.name, result["stage_retention"]))

        # Comparators
        comparator_rows.extend(_build_comparator_rows(sc.name, result["comparator_df"]))

        # FWER replication (null-interaction scenarios only)
        if sc.name in _NULL_INTERACTION_SCENARIOS:
            fwer_rng = np.random.default_rng(int(master_rng.integers(2**31)))
            flags = _run_fwer_replicates(sc, scale, fwer_rng)
            fwer_rows.append(_build_fwer_row(sc.name, scale.alpha, flags))
            print(
                f"[recovery_study]    FWER({sc.name}): "
                f"{fwer_rows[-1]['fwer_proportion']:.3f} "
                f"[{fwer_rows[-1]['wilson_ci_lower']:.3f}, "
                f"{fwer_rows[-1]['wilson_ci_upper']:.3f}] "
                f"n={fwer_rows[-1]['n_replicates']}",
                flush=True,
            )

    end_ts = datetime.datetime.now(tz=datetime.timezone.utc)

    # Write artifacts
    _write_csv(pd.DataFrame(fwer_rows), output_dir / "fwer_calibration.csv")
    _write_csv(pd.DataFrame(estimand_rows), output_dir / "recovery_estimands.csv")
    _write_csv(pd.DataFrame(retention_rows), output_dir / "stage_retention.csv")
    _write_csv(pd.DataFrame(comparator_rows), output_dir / "comparator_metrics.csv")

    # Figure data
    fig_fwer = pd.DataFrame(
        [
            {
                "scenario": r["scenario"],
                "alpha": r["alpha"],
                "fwer_proportion": r["fwer_proportion"],
                "wilson_ci_lower": r["wilson_ci_lower"],
                "wilson_ci_upper": r["wilson_ci_upper"],
                "n_replicates": r["n_replicates"],
            }
            for r in fwer_rows
        ]
    )
    _write_csv(fig_fwer, output_dir / "figure_data" / "fwer_by_scenario.csv")

    recall_rows = [
        {
            "scenario": r["scenario"],
            "family": r["family"],
            "recall": r["recall"],
            "precision": r["precision"],
            "fdp": r["fdp"],
            "selected_size": r["selected_size"],
        }
        for r in estimand_rows
    ]
    _write_csv(pd.DataFrame(recall_rows), output_dir / "figure_data" / "recall_by_scenario.csv")

    # Success criteria for reproduction log
    global_null_fwer = next((r for r in fwer_rows if r["scenario"] == "global_null"), None)
    success_criteria: dict = {
        "fwer_calibration_global_null_fwer": (
            f"{global_null_fwer['fwer_proportion']:.4f} "
            f"CI [{global_null_fwer['wilson_ci_lower']:.4f}, "
            f"{global_null_fwer['wilson_ci_upper']:.4f}]"
            if global_null_fwer
            else "not_run"
        ),
        "alpha": scale.alpha,
        "fwer_control_criterion": f"fwer_proportion <= alpha + 3*SE (alpha={scale.alpha})",
    }

    _write_reproduction_log(
        output_dir,
        scale,
        scenarios,
        master_seed,
        success_criteria,
        start_ts,
        end_ts,
        rfm_version,
    )

    print(f"[recovery_study] Done. Artifacts written to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
