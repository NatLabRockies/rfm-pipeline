"""Fit, export, and reload a small reduced-form model."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from rfm_pipeline import (
    CanonicalWorkflowRun,
    load_postfit_bundle,
    predict_from_postfit_bundle,
    run_canonical_workflow,
    write_postfit_bundle,
)


def make_example_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create a deterministic train/holdout dataset with two outputs."""
    x_train = pd.DataFrame(
        {
            "feedstock_cost": [12, 14, 15, 18, 20, 21, 24, 26],
            "conversion_yield": [0.55, 0.62, 0.58, 0.70, 0.74, 0.79, 0.82, 0.88],
            "policy_case": [0, 0, 1, 0, 1, 0, 1, 1],
        }
    )
    y_train = pd.DataFrame(
        {
            "fuel_cost": 4.0 + 0.08 * x_train["feedstock_cost"] - x_train["conversion_yield"],
            "fuel_volume": 2.0 + 3.5 * x_train["conversion_yield"] + 0.2 * x_train["policy_case"],
        }
    )
    x_holdout = pd.DataFrame(
        {
            "feedstock_cost": [16, 22, 25],
            "conversion_yield": [0.66, 0.77, 0.85],
            "policy_case": [0, 1, 0],
        }
    )
    y_holdout = pd.DataFrame(
        {
            "fuel_cost": 4.0
            + 0.08 * x_holdout["feedstock_cost"]
            - x_holdout["conversion_yield"]
            + [0.03, -0.02, 0.01],
            "fuel_volume": 2.0
            + 3.5 * x_holdout["conversion_yield"]
            + 0.2 * x_holdout["policy_case"]
            + [-0.02, 0.01, 0.03],
        }
    )
    return x_train, y_train, x_holdout, y_holdout


def run_example(output_dir: Path) -> tuple[CanonicalWorkflowRun, dict[str, pd.DataFrame]]:
    """Run the canonical API and return the in-memory run and reloaded bundle."""
    x_train, y_train, x_holdout, y_holdout = make_example_data()
    run = run_canonical_workflow(
        x_train,
        y_train,
        x_holdout,
        y_holdout,
        dataset_tag="basic-workflow-example",
        screening_cv=3,
        screening_alphas=50,
        n_boot=25,
        artifact_format="csv",
    )
    write_postfit_bundle(run.artifacts, output_dir)
    return run, load_postfit_bundle(output_dir)


def main() -> None:
    """Run the example from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/basic-workflow"),
        help="Destination for the exported model bundle.",
    )
    args = parser.parse_args()

    run, loaded = run_example(args.output_dir)
    _, _, x_holdout, _ = make_example_data()
    predictions = predict_from_postfit_bundle(args.output_dir, x_holdout)
    print(f"Wrote {len(loaded)} tables to {args.output_dir}")
    print("\nHoldout predictions:")
    print(predictions.to_string(index=False))
    print("\nHoldout summary:")
    print(run.holdout_summary.to_string(index=False))


if __name__ == "__main__":
    main()
