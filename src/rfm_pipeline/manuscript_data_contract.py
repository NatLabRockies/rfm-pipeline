"""Phase 1 data-contract helpers for the manuscript reproduction layer."""

from __future__ import annotations

import pandas as pd


def required_manuscript_artifacts() -> tuple[str, ...]:
    """Return the required real-data artifacts for the manuscript reproduction package."""
    return (
        "input_metadata",
        "output_metadata",
        "case_study_input_matrix",
        "case_study_output_matrix",
        "manuscript_feature_catalog",
        "fixed_holdout_assignments",
    )


def manuscript_placeholder_path_policy() -> dict[str, str]:
    """Return the frozen placeholder-path policy for local real-data filenames."""
    return {
        "template_file": "configs/manuscript_paths.template.yml",
        "local_override_file": "configs/local/manuscript_paths.local.yml",
        "placeholder_prefix": "REPLACE_WITH_REAL_FILE/",
    }


def manuscript_notebook_order() -> tuple[str, ...]:
    """Return the frozen execution order for manuscript reproduction notebooks."""
    return (
        "00_case_study_data_intake.ipynb",
        "01_candidate_library_audit.ipynb",
        "02_output_conditioning.ipynb",
        "03_empirical_null_screen.ipynb",
        "04_interaction_discovery.ipynb",
        "05_nonlinear_discovery.ipynb",
        "06_sparse_selection_and_stability.ipynb",
        "07_final_ols_and_bundle_export.ipynb",
        "08_manuscript_tables_and_figures.ipynb",
    )


def manuscript_required_artifact_table() -> pd.DataFrame:
    """Return the required real-data artifacts as a table."""
    return pd.DataFrame({"artifact_name": list(required_manuscript_artifacts())})


def manuscript_notebook_manifest_table() -> pd.DataFrame:
    """Return the ordered manuscript notebooks as a table."""
    names = manuscript_notebook_order()
    return pd.DataFrame({"order": list(range(len(names))), "notebook_name": list(names)})
