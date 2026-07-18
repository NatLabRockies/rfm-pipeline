"""Phase 1 data-contract helpers for the manuscript reproduction layer."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

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


# ---------------------------------------------------------------------------
# Data-contract claim validator (F3)
# ---------------------------------------------------------------------------


class ClaimMismatchError(AssertionError):
    """Raised when a manuscript claim does not match its source artifact."""


class MissingSourceError(FileNotFoundError):
    """Raised when a source artifact referenced by a claim is not found."""


def _resolve_source_value(
    source_path: str | Path,
    column: str,
    row_filter: dict[str, Any] | None = None,
) -> Any:
    """Read *column* from a CSV source artifact, optionally filtered to one row.

    Parameters
    ----------
    source_path:
        Path to a CSV file containing the source data.
    column:
        Column name whose value to return.
    row_filter:
        Optional dict of ``{col: value}`` pairs used to select exactly one row.
        When omitted the first row is used.

    Raises
    ------
    MissingSourceError
        When the file does not exist.
    KeyError
        When *column* is not present in the file.
    ValueError
        When *row_filter* matches zero or more than one row.
    """
    path = Path(source_path)
    if not path.exists():
        raise MissingSourceError(f"Source artifact not found: {path}")
    df = pd.read_csv(path)
    if row_filter:
        mask = pd.Series([True] * len(df))
        for col, val in row_filter.items():
            mask &= df[col] == val
        rows = df[mask]
        if len(rows) == 0:
            raise ValueError(f"row_filter {row_filter!r} matched no rows in {path}")
        if len(rows) > 1:
            raise ValueError(f"row_filter {row_filter!r} matched {len(rows)} rows in {path}")
        return rows.iloc[0][column]
    return df.iloc[0][column]


def validate_claims(
    claims: list[dict[str, Any]],
    *,
    tolerance: float = 1e-6,
) -> list[str]:
    """Validate a list of manuscript claims against their source artifacts.

    Each claim dict must have:
      - ``label``: human-readable identifier for the claim.
      - ``expected``: the value asserted in the manuscript.
      - ``source_path``: path to the CSV artifact.
      - ``column``: column name to read from the artifact.
      - ``row_filter`` (optional): ``{col: value}`` dict to select the row.

    Returns a list of mismatch messages (empty on full pass).

    Raises
    ------
    MissingSourceError
        On the first claim whose source file does not exist.
    """
    mismatches: list[str] = []
    for claim in claims:
        label = claim["label"]
        expected = claim["expected"]
        actual = _resolve_source_value(
            claim["source_path"],
            claim["column"],
            claim.get("row_filter"),
        )
        if isinstance(expected, float) or isinstance(actual, float):
            try:
                exp_f = float(expected)
                act_f = float(actual)
            except (TypeError, ValueError):
                mismatches.append(f"{label}: cannot compare {expected!r} and {actual!r} as floats")
                continue
            if not math.isclose(exp_f, act_f, abs_tol=tolerance, rel_tol=tolerance):
                mismatches.append(
                    f"{label}: expected {exp_f} but source has {act_f} "
                    f"(diff={abs(act_f - exp_f):.6g}, tol={tolerance})"
                )
        else:
            if expected != actual:
                mismatches.append(f"{label}: expected {expected!r} but source has {actual!r}")
    return mismatches


def assert_claims(
    claims: list[dict[str, Any]],
    *,
    tolerance: float = 1e-6,
) -> None:
    """Validate claims and raise :class:`ClaimMismatchError` if any fail.

    Parameters
    ----------
    claims:
        List of claim dicts; see :func:`validate_claims`.
    tolerance:
        Absolute and relative tolerance for float comparisons.

    Raises
    ------
    ClaimMismatchError
        When one or more claims do not match their source artifacts.
    MissingSourceError
        When a source artifact file does not exist.
    """
    mismatches = validate_claims(claims, tolerance=tolerance)
    if mismatches:
        detail = "\n  ".join(mismatches)
        raise ClaimMismatchError(f"{len(mismatches)} claim(s) failed:\n  {detail}")
