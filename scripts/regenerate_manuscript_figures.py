#!/usr/bin/env python
"""Regenerate manuscript SVG figures from pre-collected run artifacts.

Reads the CSV data files produced by the final_manuscript_artifacts stage and
re-renders all SVG figures using the current (updated) rendering functions in
manuscript_stages.py.  Use this after collecting a run locally to apply any
styling or rendering improvements without re-running the full pipeline.

Usage:
    pixi run python scripts/regenerate_manuscript_figures.py \
        --artifacts-dir artifacts/publication_full_dataset_distributed_results/\
publication_full_dataset_distributed_20260526_short_hp1

The script overwrites the SVG files in
<artifacts-dir>/artifacts/final_manuscript_artifacts/figures/
and prints a summary of what was written.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Ensure repo src is on the path when run directly
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from rfm_pipeline.manuscript_stages import (  # noqa: E402
    _build_model_performance_figure_data,
    _render_feature_pruning_curve_svg,
    _render_horizontal_bar_svg,
    _render_module_pair_heatmap_svg,
    _render_nrmse_summary_svg,
    _render_per_output_nrmse_distribution_svg,
)


def _load(fig_dir: Path, name: str) -> pd.DataFrame:
    path = fig_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Required CSV not found: {path}")
    return pd.read_csv(path)


def _load_table(table_dir: Path, name: str) -> pd.DataFrame:
    path = table_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Required table CSV not found: {path}")
    return pd.read_csv(path)


def regenerate(artifacts_root: Path) -> dict[str, Path]:
    """Regenerate all manuscript SVG figures from collected artifact CSVs."""
    fig_dir = artifacts_root / "artifacts" / "final_manuscript_artifacts" / "figures"
    table_dir = artifacts_root / "artifacts" / "final_manuscript_artifacts" / "tables"

    if not fig_dir.exists():
        raise FileNotFoundError(f"Figures directory not found: {fig_dir}")

    # ------------------------------------------------------------------ load data
    ablation_table = _load_table(table_dir, "ablation_table.csv")
    mp_data = _build_model_performance_figure_data(ablation_table)
    sup_data = _load(fig_dir, "figure_support_composition_data.csv")
    mod_data = _load(fig_dir, "figure_selected_by_module_data.csv")
    nrmse_data = _load(fig_dir, "figure_nrmse_summary_data.csv")
    pruning_data = _load(fig_dir, "figure_feature_pruning_curve_data.csv")
    feat_types = _load(fig_dir, "feature_type_counts.csv")
    influential = _load(fig_dir, "influential_counts_by_module.csv")
    pair_df = _load(fig_dir, "interaction_counts_by_module_pair.csv")  # noqa: F841
    matrix_df = _load(fig_dir, "interaction_density_module_matrix.csv")
    total_iact = _load(fig_dir, "module_total_interactions.csv")
    per_output = _load_table(table_dir, "per_output_nrmse.csv")

    # pruning counts from feature_pruning_summary
    pruning_summary = _load_table(table_dir, "feature_pruning_summary.csv")
    auto_remove = int(pruning_summary["auto_remove_count"].iloc[0])
    effective_remove = int(pruning_summary["effective_remove_count"].iloc[0])

    # interaction density matrix needs the module column as index
    matrix_indexed = matrix_df.set_index(matrix_df.columns[0])
    matrix_indexed.index.name = None

    # ------------------------------------------------------------------ render
    svg_figures = {
        "figure_model_performance": _render_horizontal_bar_svg(
            mp_data,
            label_column="display_name",
            value_column="nrmse",
            title="Holdout macro nRMSE",
            ci_lower_column="ci_lower",
            ci_upper_column="ci_upper",
        ),
        "figure_support_composition": _render_horizontal_bar_svg(
            sup_data,
            label_column="feature_type",
            value_column="n_features",
            title="Final support composition",
        ),
        "figure_selected_by_module_count": _render_horizontal_bar_svg(
            mod_data,
            label_column="module",
            value_column="n_selected_inputs",
            title="Selected inputs by module (count)",
        ),
        "figure_selected_by_module_share": _render_horizontal_bar_svg(
            mod_data,
            label_column="module",
            value_column="share_selected_support",
            title="Selected inputs by module (share)",
        ),
        "figure_nrmse_bootstrap_summary": _render_nrmse_summary_svg(nrmse_data),
        "fig_feature_type_distribution": _render_horizontal_bar_svg(
            feat_types,
            label_column="feature_type",
            value_column="count",
            title="Distribution of Influential Feature Types",
        ),
        "fig_influential_by_module": _render_horizontal_bar_svg(
            influential.sort_values("module", ignore_index=True),
            label_column="module",
            value_column="count",
            title="Influential Features by Module",
        ),
        "fig_module_pair_heatmap": _render_module_pair_heatmap_svg(matrix_indexed),
        "fig_module_total_interactions": _render_horizontal_bar_svg(
            total_iact,
            label_column="module",
            value_column="total_interactions",
            title="Total Interactions by Module",
        ),
        "figure_feature_pruning_curve": _render_feature_pruning_curve_svg(
            pruning_data,
            auto_remove_count=auto_remove,
            effective_remove_count=effective_remove,
        ),
        "figure_per_output_nrmse_distribution": _render_per_output_nrmse_distribution_svg(
            per_output
        ),
    }

    # ------------------------------------------------------------------ write
    written: dict[str, Path] = {}
    for name, svg_text in svg_figures.items():
        path = fig_dir / f"{name}.svg"
        path.write_text(svg_text, encoding="utf-8")
        written[name] = path
        print(f"  wrote {path.name}  ({len(svg_text):,} bytes)")

    return written


def main() -> None:
    """Parse arguments and regenerate manuscript figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts-dir",
        required=True,
        help="Local root of the collected run (e.g. artifacts/publication_full_dataset_distributed_results/...)",  # noqa: E501
    )
    args = parser.parse_args()

    artifacts_root = Path(args.artifacts_dir).expanduser().resolve()
    if not artifacts_root.exists():
        print(f"error: artifacts directory not found: {artifacts_root}", file=sys.stderr)
        sys.exit(1)

    print(f"Regenerating figures from: {artifacts_root}")
    written = regenerate(artifacts_root)
    print(f"\nDone — {len(written)} figures written.")


if __name__ == "__main__":
    main()
