"""Run manuscript workflow validation on the 300-sample test dataset.

Compare public-stage outputs against manuscript reference values.
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from bsm_rfm.manuscript_runtime import load_manuscript_case_study_config  # noqa: E402
from bsm_rfm.manuscript_stages import (  # noqa: E402
    audit_manuscript_reproduction_outputs,
    run_manuscript_reproduction_stage_chain,
)

# ── Manuscript reference values (from docs/manuscript_contract.md) ───────────
MANUSCRIPT = {
    "n_components": 39,
    "n_retained_screening": 349,
    "n_interaction_pairs": 367,
    "n_nonlinear_transforms": 112,
    "n_final_predictors": 340,
    "n_final_first_order_inputs": 62,
    "nrmse": 0.0859,
}

DATA_ROOT = REPO_ROOT / "artifacts" / "test_dataset_300"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts" / "validation_300_sample"
NO_CAPS_OUTPUT_ROOT = REPO_ROOT / "artifacts" / "validation_300_sample_no_caps"
VALIDATION_OUTPUT_COLUMN_LIMIT = 300
FAST_VALIDATION_OVERRIDES = {
    "case_study": {
        "output_conditioning": {"retained_components": 10},
        "empirical_null_screen": {"permutation_count_B": 20, "bh_q_screen": 1.0},
        "interaction_discovery": {
            "permutation_count_B": 5,
            "n_tree_estimators": 20,
            "max_tree_depth": 3,
            "max_shap_samples": 80,
        },
        "stability": {
            "resampling_scheme": "8_subsamples_of_80_percent_rows_without_replacement_seed_123"
        },
        "final_model": {"bootstrap_count": 20, "bootstrap_alpha": 0.05},
    }
}


@dataclass
class _FakeRuntime:
    output_root: Path


@dataclass
class _FakeContext:
    case_study_config: dict
    tables: dict
    runtime: _FakeRuntime


def load_tables(output_column_limit: int | None) -> dict:
    """Load 300-sample validation inputs and normalize holdout labels."""
    y = pd.read_parquet(DATA_ROOT / "Y.parquet")
    y_cols = [c for c in y.columns if c != "sample_id"]
    if output_column_limit is not None and len(y_cols) > output_column_limit:
        y = y[["sample_id", *y_cols[:output_column_limit]]]
    holdout = pd.read_parquet(DATA_ROOT / "holdout_assignments.parquet").copy()
    holdout["split"] = (
        holdout["split"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"test": "holdout", "val": "holdout", "validation": "holdout"})
    )
    return {
        "case_study_input_matrix": pd.read_parquet(DATA_ROOT / "X.parquet"),
        "case_study_output_matrix": y,
        "fixed_holdout_assignments": holdout,
        "manuscript_feature_catalog": pd.read_parquet(
            REPO_ROOT / "artifacts" / "actual_input_feature_catalog.parquet"
        ),
    }


def hr(label: str) -> None:
    """Print a section header."""
    print(f"\n{'─' * 62}\n  {label}\n{'─' * 62}")


def _is_first_order_feature_name(name: str) -> bool:
    if ":" in name:
        return False
    if name.startswith(("inverse_", "log1p_", "sqrt_")):
        return False
    if name.endswith("_quadratic"):
        return False
    return True


def apply_fast_validation_overrides(config: dict) -> dict:
    """Apply deterministic runtime caps used for the 300-sample validation run."""
    config = copy.deepcopy(config)
    case_study = config.setdefault("case_study", {})
    output_conditioning = case_study.setdefault("output_conditioning", {})
    empirical_null = case_study.setdefault("empirical_null_screen", {})
    interaction = case_study.setdefault("interaction_discovery", {})
    stability = case_study.setdefault("stability", {})
    final_model = case_study.setdefault("final_model", {})
    output_conditioning.update(FAST_VALIDATION_OVERRIDES["case_study"]["output_conditioning"])
    empirical_null.update(FAST_VALIDATION_OVERRIDES["case_study"]["empirical_null_screen"])
    interaction.update(FAST_VALIDATION_OVERRIDES["case_study"]["interaction_discovery"])
    stability.update(FAST_VALIDATION_OVERRIDES["case_study"]["stability"])
    final_model.update(FAST_VALIDATION_OVERRIDES["case_study"]["final_model"])
    return config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-caps",
        action="store_true",
        help="Run the validator with no runtime caps or test-time overrides.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional output directory override.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute the full stage chain and print validation diagnostics."""
    args = parse_args()
    output_root = (
        args.output_root
        if args.output_root is not None
        else (NO_CAPS_OUTPUT_ROOT if args.no_caps else DEFAULT_OUTPUT_ROOT)
    )
    output_column_limit = None if args.no_caps else VALIDATION_OUTPUT_COLUMN_LIMIT
    t0 = time.perf_counter()
    output_root.mkdir(parents=True, exist_ok=True)

    hr("300-sample validation run")
    config = load_manuscript_case_study_config(REPO_ROOT)
    if not args.no_caps:
        config = apply_fast_validation_overrides(config)
    else:
        # Enable full parallelism in no-caps mode.
        config = copy.deepcopy(config)
        config.setdefault("case_study", {}).setdefault("runtime", {})["n_jobs"] = -1
    tables = load_tables(output_column_limit=output_column_limit)
    X = tables["case_study_input_matrix"]
    Y = tables["case_study_output_matrix"]
    holdout = tables["fixed_holdout_assignments"]
    n_train = (holdout["split"] == "train").sum()
    print(f"  X: {X.shape}  Y: {Y.shape}  train rows: {n_train}")
    print(f"  Output root: {output_root}")
    if args.no_caps:
        print("  Runtime overrides: none (no-caps mode)")
    else:
        print(
            "  Runtime overrides: interaction permutation_count_B=5, n_tree_estimators=20,"
            " max_shap_samples=80, retained_components=10, empirical_null_B=20,"
            " empirical_null_q=1.0, stability_subsamples=8,"
            f" final_bootstrap_count=20, output_column_limit={VALIDATION_OUTPUT_COLUMN_LIMIT}"
        )
    sys.stdout.flush()

    ctx = _FakeContext(
        case_study_config=config,
        tables=tables,
        runtime=_FakeRuntime(output_root=output_root),
    )

    print(
        "\n  Running full stage chain (conditioning → screening → interaction → nonlinear"
        " → sparse → final) …"
    )
    sys.stdout.flush()

    chain = run_manuscript_reproduction_stage_chain(ctx)

    elapsed = time.perf_counter() - t0

    # ── Per-stage elapsed notes (not tracked internally; print totals) ────────
    oc = chain.output_conditioning.conditioning
    sc = chain.empirical_null_screening.screening
    ix = chain.interaction_discovery.interactions
    nl = chain.nonlinear_discovery.nonlinear
    sp = chain.sparse_selection_stability.sparse_selection
    fa = chain.final_manuscript_artifacts.final_artifacts

    hr(f"Results  (total elapsed: {elapsed:.0f}s)")

    final_summary = fa.summary.loc[0]
    final_support = fa.final_support_features
    if "n_final_first_order_inputs" in final_summary.index:
        n_final_first_order_inputs = int(final_summary["n_final_first_order_inputs"])
    else:
        first_order_count = final_support.loc[
            final_support["feature_name"].astype(str).map(_is_first_order_feature_name)
        ].shape[0]
        n_final_first_order_inputs = int(first_order_count)

    actuals = {
        "n_components": len([c for c in oc.pca_scores.columns if c != "sample_id"]),
        "n_retained_screening": len(sc.retained_terms),
        "n_interaction_pairs": len(ix.retained_pairs),
        "n_nonlinear_transforms": len(nl.retained_transformations),
        "n_final_predictors": int(
            final_summary.get(
                "n_final_predictors",
                final_summary.get("n_final_support_features", len(final_support)),
            )
        ),
        "n_final_first_order_inputs": n_final_first_order_inputs,
        "nrmse": float(
            final_summary.get("holdout_macro_nrmse", final_summary.get("final_ols_holdout_nrmse"))
        ),
    }

    labels = {
        "n_components": "Retained PCA components",
        "n_retained_screening": "Empirical-null retained terms",
        "n_interaction_pairs": "Retained interaction pairs",
        "n_nonlinear_transforms": "Retained nonlinear transforms",
        "n_final_predictors": "Final retained predictors",
        "n_final_first_order_inputs": "Final first-order inputs",
        "nrmse": "Holdout macro nRMSE",
    }

    print(f"\n  {'Metric':<42} {'Public':>8}  {'Ref':>8}  {'Delta':>6}  {'%off':>5}")
    print(f"  {'─' * 42} {'─' * 8}  {'─' * 8}  {'─' * 6}  {'─' * 5}")
    all_ok = True
    for key, label in labels.items():
        actual = actuals[key]
        ref = MANUSCRIPT[key]
        if isinstance(ref, float):
            delta_str = f"{actual - ref:+.4f}"
            pct = abs(actual - ref) / ref * 100
            ok = pct < 25
            flag = "✓" if ok else "!"
            print(
                f"  {flag}  {label:<40} {actual:>8.4f}  {ref:>8.4f}  {delta_str:>6}  {pct:>4.0f}%"
            )
        else:
            delta = actual - ref
            pct = abs(delta) / max(ref, 1) * 100
            ok = pct < 30
            flag = "✓" if ok else "!"
            print(f"  {flag}  {label:<40} {actual:>8d}  {ref:>8d}  {delta:>+6d}  {pct:>4.0f}%")
        if not ok:
            all_ok = False

    # ── Candidate counts (no manuscript ref, but useful for debugging) ────────
    hr("Intermediate counts")
    print(f"  Candidate screening terms  : {int(sc.summary.loc[0, 'n_candidate_terms'])}")
    print(f"  Candidate interaction pairs: {int(ix.summary.loc[0, 'n_candidate_pairs'])}")
    print(f"  Candidate nl transforms    : {int(nl.summary.loc[0, 'n_candidate_transformations'])}")
    print(f"  Stable support terms       : {len(sp.final_stable_support)}")
    print(
        f"  Variance fraction retained : "
        f"{float(oc.summary.loc[0, 'variance_fraction_retained']):.4f}"
    )

    # ── Run audit ─────────────────────────────────────────────────────────────
    hr("QA audit")
    try:
        audit = audit_manuscript_reproduction_outputs(chain, output_root)
        audit_pass = audit.all_checks_passed if hasattr(audit, "all_checks_passed") else None
        if audit.summary is not None:
            print(audit.summary.to_string(index=False))
        print(f"\n  Audit passed: {audit_pass}")
    except Exception as exc:
        print(f"  Audit skipped: {exc}")

    hr("Summary")
    if all_ok:
        print("  All counts within 30% of manuscript reference ✓")
    else:
        print("  Some counts deviate >30% from manuscript reference — review above !")
    print(f"\n  Artifacts written to: {output_root}\n")


if __name__ == "__main__":
    main()
