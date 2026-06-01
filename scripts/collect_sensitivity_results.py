#!/usr/bin/env python3
# Copyright (c) 2026 Dylan Hettinger
"""Collect per-job sensitivity-study results into one CSV file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rfm_pipeline.sensitivity_study import collect_study_results  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True, help="Study output directory.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output CSV path (default: <study-dir>/results.csv).",
    )
    return parser.parse_args()


def main() -> int:
    """Collect sensitivity study results into a tidy CSV."""
    args = _parse_args()
    results = collect_study_results(args.study_dir)
    output_path = args.output or (args.study_dir / "results.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    print(f"rows={len(results)}")
    print(f"output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
