#!/usr/bin/env python
"""Unified manuscript pipeline runner with config-driven parameterization.

Replaces dataset-specific scripts (run_300_sample_validation.py, etc.) with a
single entry point that accepts a config file defining all pipeline parameters.

Usage:
    pixi run python tools/run_manuscript_pipeline.py \
        configs/validation_300_sample_no_caps.yml
    pixi run python tools/run_manuscript_pipeline.py \
        configs/validation_300_sample_fast.yml --fast
    pixi run python tools/run_manuscript_pipeline.py \
        configs/full_manuscript.yml --output-dir /tmp/custom
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

# Ensure repo root is in sys.path for src imports
REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src.bsm_rfm.config import (  # noqa: E402
    WorkflowConfig,
    apply_fast_mode_overrides,
    load_config,
)
from src.bsm_rfm.manuscript_runtime import (  # noqa: E402
    load_manuscript_case_study_config,
)
from src.bsm_rfm.manuscript_stages import (  # noqa: E402
    run_manuscript_reproduction_stage_chain,
)


@dataclass
class _FakeRuntime:
    """Fake runtime object for compatibility."""

    output_root: Path


@dataclass
class _FakeContext:
    """Fake context object for compatibility with manuscript_stages."""

    case_study_config: dict
    tables: dict
    runtime: _FakeRuntime


def config_to_legacy_case_study(
    workflow_config: WorkflowConfig,
) -> dict:
    """Convert WorkflowConfig to legacy case_study_config format.

    Maps:
    - algorithm → output_conditioning
    - stages → per-stage configs
    - runtime.n_jobs → case_study.runtime.n_jobs
    """
    base_config = load_manuscript_case_study_config(REPO_ROOT)
    config = copy.deepcopy(base_config)

    # Ensure nested dicts exist
    case_study = config.setdefault("case_study", {})
    output_conditioning = case_study.setdefault("output_conditioning", {})
    empirical_null_screen = case_study.setdefault("empirical_null_screen", {})
    interaction_discovery = case_study.setdefault("interaction_discovery", {})
    nonlinear_discovery = case_study.setdefault("nonlinear_discovery", {})
    stability = case_study.setdefault("stability", {})
    final_model = case_study.setdefault("final_model", {})
    runtime = case_study.setdefault("runtime", {})

    # Map algorithm config
    if workflow_config.algorithm.retained_components is not None:
        output_conditioning["retained_components"] = workflow_config.algorithm.retained_components
    else:
        output_conditioning["variance_explained_threshold"] = (
            workflow_config.algorithm.variance_threshold
        )

    # Map stage configs
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
        f"{int(spr.subsample_fraction * 100)}_percent_rows_"
        "without_replacement_seed_123"
    )

    fnl = workflow_config.stages.final_artifacts
    final_model["bootstrap_count"] = fnl.bootstrap_count
    final_model["bootstrap_alpha"] = fnl.bootstrap_alpha

    # Map runtime config
    runtime["n_jobs"] = workflow_config.runtime.n_jobs

    # Map random seed
    config["random_seed"] = workflow_config.output.seed

    return config


def main() -> int:
    """Parse args, load config, run pipeline, and return exit code."""
    parser = argparse.ArgumentParser(
        description="Run manuscript pipeline with config file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to config YAML file (e.g., configs/validation_300_sample_no_caps.yml)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Enable fast-mode overrides (CI/quick testing)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output artifact directory",
    )
    args = parser.parse_args()

    # Load and validate config
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"✗ {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"✗ Failed to load config: {e}", file=sys.stderr)
        return 1

    # Apply fast-mode overrides if requested
    if args.fast:
        config.validation.fast_mode = True
        config = apply_fast_mode_overrides(config)

    # Override with CLI args if provided
    if args.seed is not None:
        config.output.seed = args.seed
    if args.output_dir is not None:
        config.output.artifact_dir = args.output_dir

    # Convert to legacy format and run pipeline
    output_root = Path(config.output.artifact_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"\n{'─' * 70}")
    print("  Unified Manuscript Pipeline Runner")
    print(f"{'─' * 70}")
    print(f"  Config: {args.config}")
    print(f"  Dataset: {config.dataset.type}")
    print(f"  Runtime n_jobs: {config.runtime.n_jobs}")
    print(f"  Output: {output_root}")
    print(f"  Fast mode: {config.validation.fast_mode}")
    print("  Stages: 6 (conditioning → screening → interaction → nonlinear → sparse → final)")

    # Load tables (assuming 300-sample dataset for now)
    # TODO: Generalize dataset loading based on config.dataset.type
    try:
        import pandas as pd

        data_root = REPO_ROOT / "artifacts" / "test_dataset_300"
        tables = {
            "case_study_input_matrix": pd.read_parquet(data_root / "X.parquet"),
            "case_study_output_matrix": pd.read_parquet(data_root / "Y.parquet"),
            "fixed_holdout_assignments": pd.read_parquet(data_root / "holdout_assignments.parquet"),
            "manuscript_feature_catalog": pd.read_parquet(
                REPO_ROOT / "artifacts" / "actual_input_feature_catalog.parquet"
            ),
        }
    except Exception as e:
        print(f"✗ Failed to load data: {e}", file=sys.stderr)
        return 1

    # Convert config to legacy format
    legacy_config = config_to_legacy_case_study(config)

    # Create context
    ctx = _FakeContext(
        case_study_config=legacy_config,
        tables=tables,
        runtime=_FakeRuntime(output_root=output_root),
    )

    print("\n  Running pipeline...")
    sys.stdout.flush()

    t0 = time.perf_counter()
    try:
        run_manuscript_reproduction_stage_chain(ctx)
        elapsed = time.perf_counter() - t0
        print(f"\n✓ Pipeline complete ({elapsed:.0f}s)")
        return 0
    except Exception as e:
        print(f"\n✗ Pipeline failed: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
