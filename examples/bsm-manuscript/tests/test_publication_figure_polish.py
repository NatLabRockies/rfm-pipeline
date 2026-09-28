"""Regression tests for final manuscript figure text polish."""

from __future__ import annotations

from pathlib import Path

from scripts.reproduce_artifacts import polish_publication_svgs


def test_polish_keeps_vetted_figures_but_fixes_two_text_defects(
    tmp_path: Path,
) -> None:
    output = tmp_path / "figures"
    tables = tmp_path / "tables"
    output.mkdir()
    tables.mkdir()
    heatmap = output / "fig_module_pair_heatmap.svg"
    distribution = output / "figure_per_output_nrmse_distribution.svg"
    untouched = output / "figure_support_composition.svg"
    heatmap.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg">'
        '<text x="24" y="36">Interaction Density by Module Pair</text>'
        "</svg>",
        encoding="utf-8",
    )
    distribution.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg">'
        '<text x="60" y="32">Per-output holdout nRMSE distribution</text>'
        '<text x="793.3" y="44" font-size="12" text-anchor="end">'
        "A very long and truncated output name (0.317); another output (0.303)"
        "</text></svg>",
        encoding="utf-8",
    )
    untouched.write_text("<svg>unchanged</svg>", encoding="utf-8")
    (tables / "per_output_nrmse.csv").write_text(
        "output_name,nrmse\noutput-a,0.303\noutput-b,0.317\nexcluded,\n",
        encoding="utf-8",
    )

    polish_publication_svgs(
        {
            "fig_module_pair_heatmap": heatmap,
            "figure_per_output_nrmse_distribution": distribution,
            "figure_support_composition": untouched,
        },
        tables_dir=tables,
    )

    heatmap_text = heatmap.read_text(encoding="utf-8")
    assert "Interaction endpoint counts" in heatmap_text
    assert "Interaction Density by Module Pair" not in heatmap_text
    distribution_text = distribution.read_text(encoding="utf-8")
    assert "max nRMSE=0.317" in distribution_text
    assert 'y="68"' in distribution_text
    assert "truncated output name" not in distribution_text
    assert untouched.read_text(encoding="utf-8") == "<svg>unchanged</svg>"
