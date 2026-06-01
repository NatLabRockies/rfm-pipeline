"""Run deterministic public workflow and manuscript-reproduction examples."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from rfm_pipeline import (
    build_manuscript_notebook_context,
    load_postfit_bundle,
    run_canonical_workflow,
    run_manuscript_reproduction_audit_stage,
    write_postfit_bundle,
)

DATASET_TAG = "toy-reproducibility-example"
MANUSCRIPT_REPRODUCTION_TAG = "toy-manuscript-reproduction-example"


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


def run_manuscript_reproduction_example(output_dir: Path | str) -> dict[str, object]:
    """Execute the complete demo manuscript reproduction chain and write artifacts."""
    repo_root = Path(__file__).resolve().parents[1]
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    context = build_manuscript_notebook_context(
        repo_root,
        "08_manuscript_tables_and_figures.ipynb",
    )
    runtime = replace(context.runtime, output_root=output_root)
    context = replace(context, runtime=runtime)
    audit_result = run_manuscript_reproduction_audit_stage(context)
    reproduction = audit_result.reproduction
    return {
        "manuscript_output_root": output_root,
        "manuscript_runtime_mode": context.runtime.mode,
        "manuscript_reproduction": reproduction,
        "manuscript_artifact_paths": reproduction.artifact_paths,
        "manuscript_audit": audit_result.audit,
        "manuscript_audit_paths": audit_result.artifact_paths,
        "manuscript_audit_summary": audit_result.audit.summary,
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
    parser.add_argument(
        "--run-manuscript-chain",
        action="store_true",
        help="Also run the complete demo manuscript reproduction stage chain.",
    )
    parser.add_argument(
        "--manuscript-output-dir",
        type=Path,
        default=Path("artifacts") / MANUSCRIPT_REPRODUCTION_TAG,
        help="Directory where manuscript reproduction artifacts should be written.",
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

    if args.run_manuscript_chain:
        manuscript_result = run_manuscript_reproduction_example(args.manuscript_output_dir)
        artifact_families = sorted(manuscript_result["manuscript_artifact_paths"])
        print(f"Wrote manuscript reproduction artifacts to: {args.manuscript_output_dir}")
        print(f"Manuscript artifact families: {', '.join(artifact_families)}")
        audit_summary = manuscript_result["manuscript_audit_summary"]
        print(f"Manuscript audit status: {audit_summary.loc[0, 'qa_status']}")


if __name__ == "__main__":
    main()
