"""Preprocess raw data matrices to add sample_id column.

This script converts data matrices with MultiIndex (scenario, run_id) to the format
expected by the manuscript reproduction workflow (sample_id column).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def preprocess_multiindex_data(
    input_x: Path,
    input_y: Path,
    output_x: Path,
    output_y: Path,
    index_columns: tuple[str, ...] = ("scenario", "run_id"),
    separator: str = "_",
) -> None:
    """Convert MultiIndex data to sample_id format.

    Parameters
    ----------
    input_x
        Path to input X matrix (features) with MultiIndex
    input_y
        Path to input Y matrix (responses) with MultiIndex
    output_x
        Path to write preprocessed X matrix
    output_y
        Path to write preprocessed Y matrix
    index_columns
        Names of index columns to combine into sample_id
    separator
        String to join index values
    """
    print(f"Loading X matrix: {input_x}")
    x = pd.read_parquet(input_x)
    print(f"  Shape: {x.shape}")
    print(f"  Index: {x.index.names}")

    print(f"\nLoading Y matrix: {input_y}")
    y = pd.read_parquet(input_y)
    print(f"  Shape: {y.shape}")
    print(f"  Index: {y.index.names}")

    # Validate index structure
    if x.index.names != list(index_columns):
        raise ValueError(
            f"X matrix index {x.index.names} does not match expected {list(index_columns)}"
        )
    if y.index.names != list(index_columns):
        raise ValueError(
            f"Y matrix index {y.index.names} does not match expected {list(index_columns)}"
        )

    # Reset index to convert MultiIndex to columns
    print("\nCreating sample_id from MultiIndex...")
    x_reset = x.reset_index()
    y_reset = y.reset_index()

    # Create sample_id by joining index columns
    sample_id_x = x_reset[index_columns[0]].astype(str)
    sample_id_y = y_reset[index_columns[0]].astype(str)
    for col in index_columns[1:]:
        sample_id_x = sample_id_x + separator + x_reset[col].astype(str)
        sample_id_y = sample_id_y + separator + y_reset[col].astype(str)

    x_reset.insert(0, "sample_id", sample_id_x)
    y_reset.insert(0, "sample_id", sample_id_y)

    # Extract binary scenario factors from scenario string
    # Format: "AFSCoff_UAEOROoff" -> AFSC=0, UAEORO=0
    if "scenario" in x_reset.columns:
        scenario_str = x_reset["scenario"].astype(str)
        x_reset["AFSC"] = scenario_str.str.contains("AFSCon").astype(int)
        x_reset["UAEORO"] = scenario_str.str.contains("UAEOROon").astype(int)
        print(f"  Extracted AFSC: {x_reset['AFSC'].unique()}")
        print(f"  Extracted UAEORO: {x_reset['UAEORO'].unique()}")

    # Keep scenario and run_id as features, only drop if explicitly requested
    # These are important scenario factors for the manuscript workflow
    x_final = x_reset.drop(columns=[])  # Keep all columns including scenario/run_id
    y_final = y_reset.drop(columns=list(index_columns))

    print(f"\nPreprocessed X shape: {x_final.shape}")
    print(f"  Columns (first 10): {x_final.columns.tolist()[:10]}")
    print(f"Preprocessed Y shape: {y_final.shape}")
    print(f"Sample IDs (first 3): {list(x_final['sample_id'].head(3))}")

    # Save preprocessed data
    output_x.parent.mkdir(parents=True, exist_ok=True)
    output_y.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving preprocessed X to: {output_x}")
    x_final.to_parquet(output_x)

    print(f"Saving preprocessed Y to: {output_y}")
    y_final.to_parquet(output_y)

    print("\n✓ Preprocessing complete!")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess raw data matrices to add sample_id column",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  pixi run python scripts/preprocess_multiindex_data.py \\
    --input-x /path/to/raw_X.parquet \\
    --input-y /path/to/raw_Y.parquet \\
    --output-x artifacts/preprocessed/X.parquet \\
    --output-y artifacts/preprocessed/Y.parquet
        """,
    )
    parser.add_argument(
        "--input-x",
        type=Path,
        required=True,
        help="Path to input X matrix (features) with MultiIndex",
    )
    parser.add_argument(
        "--input-y",
        type=Path,
        required=True,
        help="Path to input Y matrix (responses) with MultiIndex",
    )
    parser.add_argument(
        "--output-x",
        type=Path,
        required=True,
        help="Path to write preprocessed X matrix",
    )
    parser.add_argument(
        "--output-y",
        type=Path,
        required=True,
        help="Path to write preprocessed Y matrix",
    )
    parser.add_argument(
        "--index-columns",
        type=str,
        nargs="+",
        default=["scenario", "run_id"],
        help="Names of index columns to combine (default: scenario run_id)",
    )
    parser.add_argument(
        "--separator",
        type=str,
        default="_",
        help="String to join index values (default: _)",
    )
    return parser


def main() -> int:
    """Run the preprocessing script from the command line."""
    args = _build_parser().parse_args()

    try:
        preprocess_multiindex_data(
            input_x=args.input_x,
            input_y=args.input_y,
            output_x=args.output_x,
            output_y=args.output_y,
            index_columns=tuple(args.index_columns),
            separator=args.separator,
        )
        return 0
    except Exception as e:
        print(f"\n✗ ERROR: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
