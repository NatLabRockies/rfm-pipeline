"""Run a deterministic reduced-form modeling example."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from rfm_pipeline import (
    load_postfit_bundle,
    run_canonical_workflow,
    write_postfit_bundle,
)

DATASET_TAG = "toy-reproducibility-example"


def make_example_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build a tiny deterministic train/holdout split for the public workflow example."""
    x_train = pd.DataFrame(
        {
            "x1": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "x2": [1.0, 0.5, 2.0, 1.5, 3.0, 2.5, 3.5, 4.0],
            "x3": [0.2, -1.1, 0.5, -0.7, 1.4, -0.3, 0.8, -1.5],
        }
    )
    y_train = pd.DataFrame(
        {
            "y1": 1.0 + 2.0 * x_train["x1"] - 1.0 * x_train["x2"],
            "y2": -0.5 + 0.75 * x_train["x1"] + 0.5 * x_train["x2"],
        }
    )
    x_holdout = pd.DataFrame(
        {
            "x1": [8.0, 9.0, 10.0],
            "x2": [4.5, 5.0, 5.5],
            "x3": [1.2, -0.4, 0.1],
        }
    )
    holdout_residual = pd.DataFrame(
        {
            "y1": [0.20, -0.15, 0.10],
            "y2": [-0.08, 0.12, -0.06],
        }
    )
    y_holdout = (
        pd.DataFrame(
            {
                "y1": 1.0 + 2.0 * x_holdout["x1"] - 1.0 * x_holdout["x2"],
                "y2": -0.5 + 0.75 * x_holdout["x1"] + 0.5 * x_holdout["x2"],
            }
        )
        + holdout_residual
    )
    return x_train, y_train, x_holdout, y_holdout


def run_reproducibility_example(output_dir: Path | str) -> dict[str, object]:
    """Execute the canonical workflow, write the bundle, and reload it from disk."""
    x_train, y_train, x_holdout, y_holdout = make_example_frames()
    bundle_root = Path(output_dir)
    run = run_canonical_workflow(
        x_train,
        y_train,
        x_holdout,
        y_holdout,
        dataset_tag=DATASET_TAG,
        screening_cv=3,
        screening_l1_ratio=(0.9, 1.0),
        screening_alphas=50,
        screening_random_state=17,
        n_boot=25,
        bootstrap_random_state=19,
        bootstrap_sample_size=len(x_holdout),
        artifact_format="csv",
        upstream_provenance={"source": "examples/end_to_end_reproducibility.py"},
    )
    written = write_postfit_bundle(run.artifacts, bundle_root)
    loaded = load_postfit_bundle(bundle_root)
    manifest = json.loads((bundle_root / "manifest.json").read_text(encoding="utf-8"))
    return {
        "bundle_root": bundle_root,
        "written": written,
        "loaded": loaded,
        "manifest": manifest,
        "holdout_summary": run.holdout_summary.copy(),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run deterministic toy workflow examples, write canonical artifacts, "
            "and reload or validate the generated outputs."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts") / DATASET_TAG,
        help="Directory where the canonical workflow example bundle should be written.",
    )
    return parser


def main() -> None:
    """Run the example script from the command line."""
    args = _build_parser().parse_args()
    result = run_reproducibility_example(args.output_dir)
    manifest = result["manifest"]
    holdout_summary = result["holdout_summary"]
    print(f"Wrote bundle to: {result['bundle_root']}")
    print(f"Artifact tables: {', '.join(sorted(manifest['files']))}")
    print(f"Holdout macro nRMSE point estimate: {holdout_summary.loc[0, 'point_estimate']:.6f}")


if __name__ == "__main__":
    main()
