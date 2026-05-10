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
import sys
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bsm_rfm.config import apply_fast_mode_overrides, load_config


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

    # TODO: Integrate with manuscript_stages.run_manuscript_reproduction_stage_chain
    # For now, placeholder that demonstrates config loading works
    print(f"✓ Config loaded: {args.config}")
    print(f"  Dataset: {config.dataset.type}")
    print(f"  Runtime: n_jobs={config.runtime.n_jobs}")
    print(f"  Output: {config.output.artifact_dir}")
    print(f"  Fast mode: {config.validation.fast_mode}")
    print()
    print("Pipeline execution not yet implemented (pending manuscript_stages integration)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
