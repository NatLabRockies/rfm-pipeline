"""Fast 6-stage validation with 20 resamples for pipeline verification."""

from __future__ import annotations

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
    run_manuscript_reproduction_stage_chain,
)

DATA_ROOT = REPO_ROOT / "artifacts" / "test_dataset_300"
FAST_OUTPUT_ROOT = REPO_ROOT / "artifacts" / "validation_300_sample_fast_sparse"


@dataclass
class _FakeRuntime:
    output_root: Path


@dataclass
class _FakeContext:
    case_study_config: dict
    tables: dict
    runtime: _FakeRuntime


def load_tables() -> dict:
    """Load 300-sample validation inputs."""
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
        "case_study_output_matrix": pd.read_parquet(DATA_ROOT / "Y.parquet"),
        "fixed_holdout_assignments": holdout,
        "manuscript_feature_catalog": pd.read_parquet(
            REPO_ROOT / "artifacts" / "actual_input_feature_catalog.parquet"
        ),
    }


def main() -> None:
    """Execute fast 6-stage validation with 20 resamples."""
    t0 = time.perf_counter()
    FAST_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("\n" + "─" * 62)
    print("  Fast 6-stage validation (20 resamples for sparse testing)")
    print("─" * 62)

    config = load_manuscript_case_study_config(REPO_ROOT)
    # Reduce resamples
    config_fast = copy.deepcopy(config)
    config_fast.setdefault("case_study", {}).setdefault("stability", {})["resampling_scheme"] = (
        "20_subsamples_of_80_percent_rows_without_replacement_seed_123"
    )
    # Full parallelism
    config_fast.setdefault("case_study", {}).setdefault("runtime", {})["n_jobs"] = -1

    tables = load_tables()
    X = tables["case_study_input_matrix"]
    Y = tables["case_study_output_matrix"]
    holdout = tables["fixed_holdout_assignments"]
    n_train = (holdout["split"] == "train").sum()

    print(f"  X: {X.shape}  Y: {Y.shape}  train rows: {n_train}")
    print(f"  Output root: {FAST_OUTPUT_ROOT}")
    print("  Sparse resamples: 20 (reduced for fast validation)")
    print("  Parallelism: n_jobs=-1 (all cores)")
    print("\n  Running 6-stage chain (fast sparse mode)...")
    sys.stdout.flush()

    ctx = _FakeContext(
        case_study_config=config_fast,
        tables=tables,
        runtime=_FakeRuntime(output_root=FAST_OUTPUT_ROOT),
    )

    chain = run_manuscript_reproduction_stage_chain(ctx)
    elapsed = time.perf_counter() - t0

    # Extract results
    oc = chain.output_conditioning.conditioning
    sc = chain.empirical_null_screening.screening
    ix = chain.interaction_discovery.interactions
    nl = chain.nonlinear_discovery.nonlinear
    sp = chain.sparse_selection_stability.sparse_selection

    print(f"\n{'─' * 62}\n  Results (elapsed: {elapsed:.0f}s)\n{'─' * 62}")

    n_components = len([c for c in oc.pca_scores.columns if c != "sample_id"])
    n_screening = len(sc.retained_terms)
    n_pairs = len(ix.retained_pairs)
    n_nonlinear = len(nl.retained_transformations)
    n_stable = len(sp.final_stable_support)

    print(f"  ✓ PCA components: {n_components}")
    print(f"  ✓ Screening retained terms: {n_screening}")
    print(f"  ✓ Interaction pairs: {n_pairs}")
    print(f"  ✓ Nonlinear transforms: {n_nonlinear}")
    print(f"  ✓ Stable support terms: {n_stable}")
    print(f"\n  Artifacts written to: {FAST_OUTPUT_ROOT}\n")


if __name__ == "__main__":
    main()
