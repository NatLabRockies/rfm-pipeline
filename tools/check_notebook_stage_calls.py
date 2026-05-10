#!/usr/bin/env python
"""Validate manuscript notebooks call the expected stage entrypoints."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_DIR = REPO_ROOT / "notebooks" / "manuscript"

EXPECTED_TOKENS: dict[str, list[str]] = {
    "00_case_study_data_intake.ipynb": [
        "build_manuscript_notebook_context",
        "manuscript_runtime_summary_table",
    ],
    "01_candidate_library_audit.ipynb": [
        "build_manuscript_notebook_context",
        'context.tables["manuscript_feature_catalog"]',
    ],
    "02_output_conditioning.ipynb": [
        "build_manuscript_notebook_context",
        "run_output_conditioning_stage",
    ],
    "03_empirical_null_screen.ipynb": [
        "build_manuscript_notebook_context",
        "run_empirical_null_screening_stage",
    ],
    "04_interaction_discovery.ipynb": [
        "build_manuscript_notebook_context",
        "run_interaction_discovery_stage",
    ],
    "05_nonlinear_discovery.ipynb": [
        "build_manuscript_notebook_context",
        "run_nonlinear_discovery_stage",
    ],
    "06_sparse_selection_and_stability.ipynb": [
        "build_manuscript_notebook_context",
        "run_sparse_selection_stability_stage",
    ],
    "07_final_ols_and_bundle_export.ipynb": [
        "build_manuscript_notebook_context",
        "run_final_manuscript_artifacts_stage",
    ],
    "08_manuscript_tables_and_figures.ipynb": [
        "build_manuscript_notebook_context",
        "run_final_manuscript_artifacts_stage",
    ],
}


def _source_blob(notebook_path: Path) -> str:
    payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = payload.get("cells", [])
    chunks: list[str] = []
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        if isinstance(source, list):
            chunks.append("".join(source))
        elif isinstance(source, str):
            chunks.append(source)
    return "\n".join(chunks)


def main() -> int:
    """Validate expected notebook tokens and return process exit code."""
    missing: list[str] = []
    for nb_name, tokens in EXPECTED_TOKENS.items():
        nb_path = NOTEBOOK_DIR / nb_name
        if not nb_path.exists():
            missing.append(f"{nb_name}: notebook missing")
            continue
        blob = _source_blob(nb_path)
        for token in tokens:
            if token not in blob:
                missing.append(f"{nb_name}: missing token `{token}`")
    if missing:
        print("Notebook stage-call check failed:")
        for item in missing:
            print(f" - {item}")
        return 1
    print(f"Notebook stage-call check passed for {len(EXPECTED_TOKENS)} notebooks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
