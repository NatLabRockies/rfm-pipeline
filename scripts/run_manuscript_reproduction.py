r"""Run manuscript reproduction workflow from config files.

This script provides a simple config-driven interface for running manuscript reproduction.
Pass a single config file that contains all paths needed for the run.

Usage:
    pixi run manuscript-reproduce --config configs/datasets/real_data.yml
    pixi run manuscript-reproduce --config configs/datasets/real_data.yml \\
        --output-dir artifacts/my-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# Make package importable from repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from bsm_rfm import run_manuscript_reproduction_audit_stage  # noqa: E402
from bsm_rfm.manuscript_runtime import (  # noqa: E402
    ManuscriptNotebookContext,
    ManuscriptRuntimeContext,
)


def load_simple_config(config_path: Path) -> dict[str, str]:
    """Load a simple config file with all paths in one place.

    Parameters
    ----------
    config_path
        Path to YAML config file containing all required artifact paths.

    Returns
    -------
    dict
        Configuration dictionary with paths.
    """
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Validate required fields
    required_fields = [
        "case_study_input_matrix",
        "case_study_output_matrix",
        "input_metadata",
        "output_metadata",
        "manuscript_feature_catalog",
        "fixed_holdout_assignments",
    ]

    missing = [f for f in required_fields if f not in config]
    if missing:
        raise ValueError(f"Config file missing required fields: {missing}")

    return config


def build_context_from_config(
    config_path: Path,
    output_dir: Path | None = None,
) -> ManuscriptNotebookContext:
    """Build a manuscript context from a simple config file.

    Parameters
    ----------
    config_path
        Path to config file.
    output_dir
        Optional output directory override.

    Returns
    -------
    ManuscriptNotebookContext
        Context ready for manuscript reproduction.
    """
    import pandas as pd

    config = load_simple_config(config_path)

    # Convert string paths to Path objects
    artifact_paths = {key: Path(value) for key, value in config.items() if key != "output_root"}

    # Determine output root
    if output_dir is not None:
        output_root = output_dir
    elif "output_root" in config:
        output_root = Path(config["output_root"])
    else:
        output_root = REPO_ROOT / "artifacts" / "manuscript-output"

    output_root.mkdir(parents=True, exist_ok=True)

    # Validate that files exist
    for key, path in artifact_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{key} not found at: {path}")

    # Load required tables
    tables = {}

    # Load metadata tables
    for key in [
        "input_metadata",
        "output_metadata",
        "manuscript_feature_catalog",
        "fixed_holdout_assignments",
    ]:
        path = artifact_paths[key]
        tables[key] = pd.read_parquet(path)

    # Load data matrices
    tables["case_study_input_matrix"] = pd.read_parquet(artifact_paths["case_study_input_matrix"])
    tables["case_study_output_matrix"] = pd.read_parquet(artifact_paths["case_study_output_matrix"])

    # Load case study config (provides workflow parameters)
    case_study_config_path = REPO_ROOT / "configs" / "manuscript_case_study.yml"
    if case_study_config_path.exists():
        with open(case_study_config_path, encoding="utf-8") as f:
            case_study_config = yaml.safe_load(f)
    else:
        case_study_config = {}

    # Load runtime manifest (workflow needs this for notebook ordering)
    runtime_manifest_path = REPO_ROOT / "configs" / "manuscript_runtime.yml"
    if runtime_manifest_path.exists():
        with open(runtime_manifest_path, encoding="utf-8") as f:
            runtime_manifest = yaml.safe_load(f)
    else:
        runtime_manifest = {"notebook_order": ["08_manuscript_tables_and_figures.ipynb"]}

    # Create runtime context
    runtime = ManuscriptRuntimeContext(
        mode="real",
        repo_root=REPO_ROOT,
        artifact_paths=artifact_paths,
        output_root=output_root,
        unresolved_placeholders=(),
        local_override_used=True,
        runtime_dir=None,
    )

    # Create notebook context
    context = ManuscriptNotebookContext(
        notebook_name="08_manuscript_tables_and_figures.ipynb",
        runtime=runtime,
        tables=tables,
        case_study_config=case_study_config,
        runtime_manifest=runtime_manifest,
    )

    return context


def run_manuscript_from_config(
    config_path: Path,
    output_dir: Path | None = None,
) -> dict[str, object]:
    """Run manuscript reproduction using a simple config file.

    Parameters
    ----------
    config_path
        Path to config file with all required artifact paths.
    output_dir
        Optional output directory override.

    Returns
    -------
    dict
        Dictionary containing reproduction and audit results.
    """
    # Build context from config
    context = build_context_from_config(config_path, output_dir)

    # Report configuration
    print("=" * 70)
    print("MANUSCRIPT REPRODUCTION WORKFLOW")
    print("=" * 70)
    print(f"Config file:         {config_path}")
    print(f"Output directory:    {context.runtime.output_root}")
    print()

    # Run the full reproduction and audit chain
    print("Running manuscript reproduction audit stage...")
    print("(This will take several minutes: output conditioning, empirical null")
    print(" screening, interaction discovery, nonlinear discovery, sparse selection,")
    print(" final artifacts, and audit checks)\n")

    result = run_manuscript_reproduction_audit_stage(context)

    # Extract results
    summary = result.audit.summary.iloc[0]
    artifact_families = sorted(result.reproduction.artifact_paths.keys())

    # Report results
    print("\n" + "=" * 70)
    print("MANUSCRIPT REPRODUCTION COMPLETE")
    print("=" * 70)
    print(f"Output root: {context.runtime.output_root.resolve()}")
    print(f"\nArtifact families generated ({len(artifact_families)}):")
    for family in artifact_families:
        n_files = len(result.reproduction.artifact_paths[family])
        print(f"  • {family:<40} ({n_files:>2} files)")

    print("\nAudit Results:")
    print(f"  QA status:            {summary['qa_status']}")
    print(f"  Missing artifacts:    {int(summary['n_missing_artifacts'])}")
    print(f"  Empty artifacts:      {int(summary['n_empty_artifacts'])}")
    print(f"  Failed metric checks: {int(summary['n_failed_metric_checks'])}")
    print(f"  Passed metric checks: {int(summary['n_passed_metric_checks'])}")
    print("=" * 70)

    return {
        "context": context,
        "reproduction": result.reproduction,
        "audit": result.audit,
        "summary": summary,
        "artifact_families": artifact_families,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run manuscript reproduction workflow from a config file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with a config file
  pixi run manuscript-reproduce --config configs/datasets/real_data.yml

  # Run with custom output directory
  pixi run manuscript-reproduce --config configs/datasets/real_data.yml \\
      --output-dir artifacts/my-run

Config File Format:
  The config file should be YAML with these required fields:

  case_study_input_matrix: /path/to/X.parquet
  case_study_output_matrix: /path/to/Y.parquet
  input_metadata: /path/to/input_metadata.parquet
  output_metadata: /path/to/output_metadata.parquet
  manuscript_feature_catalog: /path/to/feature_catalog.parquet
  fixed_holdout_assignments: /path/to/holdout_assignments.parquet
  output_root: /path/to/default/output  # optional

Environment Variables (recommended for memory efficiency):
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
  NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \\
  pixi run manuscript-reproduce --config your-config.yml
        """,
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to config YAML file with all required artifact paths",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for artifacts (overrides config)",
    )
    return parser


def main() -> int:
    """Run the manuscript reproduction workflow from the command line."""
    args = _build_parser().parse_args()

    if not args.config.exists():
        print(f"✗ ERROR: Config file not found: {args.config}", file=sys.stderr)
        return 1

    try:
        result = run_manuscript_from_config(
            config_path=args.config,
            output_dir=args.output_dir,
        )

        # Check for failures
        if result["summary"]["qa_status"] != "pass":
            print("\n⚠ WARNING: Audit did not pass all checks", file=sys.stderr)
            return 1

        print("\n✓ Manuscript reproduction completed successfully!")
        return 0

    except Exception as e:
        print(f"\n✗ ERROR: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
