"""Selected-feature parsing utilities for recovered pipeline naming conventions."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

KNOWN_TRANSFORMATIONS = {
    # Legacy labels (used by pre-existing artifact parsing)
    "quadratic",
    "inverse",
    "log",
    "log1p",
    "square",
    "cubic",
    "cube",
    "reciprocal",
    "abs",
    "logarithmic",
    # Current labels from DEFAULT_TRANSFORM_LIBRARY (TransformDef.label values)
    "sq",  # quadratic  x**2
    "inv",  # inverse    1/x
    "sqrt",  # square root
}


def _strip_outer_brackets(text: str) -> str:
    """Strip one matching outer bracket pair from a feature label."""
    text = str(text).strip()
    if text.startswith("[") and text.endswith("]"):
        return text[1:-1].strip()
    return text


def _split_interaction(name: str) -> tuple[str | None, str | None]:
    """Split a second-order interaction into its component terms."""
    cleaned = _strip_outer_brackets(name)
    if "*" not in cleaned:
        return None, None
    left, right = cleaned.split("*", 1)
    return left.strip(), right.strip()


def _split_transformation(
    name: str,
    known_transformations: set[str] | None = None,
) -> tuple[str | None, str | None]:
    """Split a transformed feature into its base feature and transform label."""
    known_transformations = known_transformations or KNOWN_TRANSFORMATIONS
    cleaned = _strip_outer_brackets(name)
    for suffix in sorted(known_transformations, key=len, reverse=True):
        marker = f"_{suffix}"
        if cleaned.endswith(marker):
            return cleaned[: -len(marker)].strip(), suffix
    return None, None


def parse_selected_input_structure(
    feature_names: Iterable[str],
    known_transformations: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Classify selected inputs into first-order, interaction, and nonlinear terms.

    Parameters
    ----------
    feature_names
        Selected feature names in their original order.
    known_transformations
        Optional set of recognized transformation suffixes.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]
        Structure table, type summary table, and transformation summary table.
    """
    records: list[dict[str, object]] = []
    known_transformations = known_transformations or KNOWN_TRANSFORMATIONS

    for raw_name in feature_names:
        name = str(raw_name)
        left, right = _split_interaction(name)
        if left is not None:
            records.append(
                {
                    "feature_name": name,
                    "input_type": "second_order",
                    "term_1": left,
                    "term_2": right,
                    "transformation": None,
                }
            )
            continue

        base, transformation = _split_transformation(
            name,
            known_transformations=known_transformations,
        )
        if base is not None:
            records.append(
                {
                    "feature_name": name,
                    "input_type": "nonlinear_transformation",
                    "term_1": base,
                    "term_2": None,
                    "transformation": transformation,
                }
            )
            continue

        records.append(
            {
                "feature_name": name,
                "input_type": "first_order",
                "term_1": name,
                "term_2": None,
                "transformation": None,
            }
        )

    structure_df = pd.DataFrame.from_records(records)
    type_summary_df = (
        structure_df.groupby("input_type", dropna=False)
        .size()
        .rename("n_inputs")
        .reset_index()
        .sort_values("input_type")
        .reset_index(drop=True)
    )
    transformation_summary_df = (
        structure_df.dropna(subset=["transformation"])
        .groupby("transformation", dropna=False)
        .size()
        .rename("n_inputs")
        .reset_index()
        .sort_values(["n_inputs", "transformation"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return structure_df, type_summary_df, transformation_summary_df


def canonical_module_from_factor_name(name: str) -> str:
    """Return the coarse module prefix implied by a factor name.

    Parameters
    ----------
    name
        Full factor or feature label.

    Returns
    -------
    str
        ``Scenario`` for special scenario flags, the left-hand scope before the first
        dot for scoped names, or ``Unscoped`` otherwise.
    """
    name = str(name)
    if name in {"AFSC", "UAEORO"}:
        return "Scenario"
    if "." not in name:
        return "Unscoped"
    return name.split(".", 1)[0].strip() or "Unscoped"
