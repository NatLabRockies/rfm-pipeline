"""Explicit feature-expansion boundary recovered from the audited notebook workflow.

The recovered archive includes a notebook that expands influential first-order inputs
into a wider modeling matrix with scenario flags, nonlinear transforms, and second-
order interactions. This module captures that boundary as an explicit, tested package
contract without claiming that the notebook itself is canonical source code.

Transforms are expressed as :class:`~rfm_pipeline.transforms.TransformDef` objects
(SymPy expression strings) rather than hardcoded string labels.  Invalid values
produced by a transform (e.g. log of a negative number) become ``NaN`` in the
expanded matrix; a :class:`UserWarning` is issued listing the affected features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from rfm_pipeline.transforms import TransformDef, warn_nan_transforms

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class FeatureExpansionSpec:
    """Explicit feature-expansion specification.

    Parameters
    ----------
    base_features
        Ordered first-order feature names that seed the expansion.
    scenario_flags
        Ordered scenario-indicator feature names preserved as first-order terms.
    transforms
        Mapping from base feature name to an ordered tuple of
        :class:`~rfm_pipeline.transforms.TransformDef` objects.  Each transform is
        applied to its base feature; the resulting column name is
        ``TransformDef.column_name(base_feature)`` (i.e. ``"{base}_{label}"``).
    interactions
        Ordered pairs of feature names whose product terms should be created.
    provenance
        Provenance label describing the strength of the recovered source.
    source_artifact
        Supporting artifact for the recovered specification.
    """

    base_features: tuple[str, ...]
    scenario_flags: tuple[str, ...]
    transforms: dict[str, tuple[TransformDef, ...]]
    interactions: tuple[tuple[str, str], ...]
    provenance: str
    source_artifact: str


@dataclass(frozen=True)
class FeatureExpansionResult:
    """Materialized expanded feature matrix and its ordered column catalog."""

    expanded_frame: pd.DataFrame
    ordered_columns: tuple[str, ...]


def _ordered_unique(values: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return tuple(ordered)


def _ordered_unique_transform_defs(values: list[TransformDef]) -> tuple[TransformDef, ...]:
    """Deduplicate TransformDef objects by label, preserving insertion order."""
    seen: set[str] = set()
    ordered: list[TransformDef] = []
    for td in values:
        if td.label not in seen:
            seen.add(td.label)
            ordered.append(td)
    return tuple(ordered)


def default_feature_expansion_spec(
    *,
    base_features: list[str],
    scenario_flags: tuple[str, ...] = ("AFSC", "UAEORO"),
    add_transforms: dict[str, list[TransformDef]] | None = None,
    interaction_pairs: tuple[tuple[str, str], ...] = (),
) -> FeatureExpansionSpec:
    """Build a simple explicit feature-expansion specification.

    Parameters
    ----------
    base_features
        Ordered influential first-order feature names.
    scenario_flags
        Scenario-indicator features preserved in the expanded matrix.
    add_transforms
        Mapping from base feature name to a list of
        :class:`~rfm_pipeline.transforms.TransformDef` objects to apply.
        Features not listed receive no transforms.
        Example::

            from rfm_pipeline.transforms import QUADRATIC, INVERSE

            add_transforms = {"x1": [QUADRATIC], "x2": [INVERSE]}

    interaction_pairs
        Ordered feature pairs used to create product terms.

    Returns
    -------
    FeatureExpansionSpec
        Explicit feature-expansion contract labeled as notebook-derived.
    """
    ordered_base = _ordered_unique(list(base_features))
    transforms: dict[str, tuple[TransformDef, ...]] = {}
    if add_transforms:
        for name in ordered_base:
            if name in add_transforms:
                transforms[name] = _ordered_unique_transform_defs(add_transforms[name])
    return FeatureExpansionSpec(
        base_features=ordered_base,
        scenario_flags=scenario_flags,
        transforms=transforms,
        interactions=tuple(interaction_pairs),
        provenance="notebook-derived",
        source_artifact="make_nonlinear_features.ipynb",
    )


def ordered_expanded_feature_names(spec: FeatureExpansionSpec) -> tuple[str, ...]:
    """Return the ordered expanded feature catalog implied by a specification.

    Returns
    -------
    tuple of str
        Ordered expanded feature names.
    """
    ordered: list[str] = [*spec.base_features, *spec.scenario_flags]
    for base_feature in spec.base_features:
        for td in spec.transforms.get(base_feature, ()):
            ordered.append(td.column_name(base_feature))
    ordered.extend(f"{left}*{right}" for left, right in spec.interactions)
    return _ordered_unique(ordered)


def apply_feature_expansion(
    frame: pd.DataFrame,
    spec: FeatureExpansionSpec,
) -> FeatureExpansionResult:
    """Materialize an expanded feature matrix from an explicit specification.

    Parameters
    ----------
    frame
        Input frame containing the base features and any scenario flags referenced by
        the specification.
    spec
        Explicit feature-expansion specification.

    Returns
    -------
    FeatureExpansionResult
        Expanded feature matrix and ordered column catalog.

    Raises
    ------
    KeyError
        Raised when the frame does not contain all required source columns.

    Warns
    -----
    UserWarning
        Issued when a transform produces ``NaN`` values for any row.  The expanded
        matrix still contains the column with ``NaN`` entries; callers that cannot
        tolerate missing values should impute or drop those rows downstream.
    """
    required_columns = set(spec.base_features) | set(spec.scenario_flags)
    required_columns |= {name for pair in spec.interactions for name in pair}
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise KeyError(f"Input frame is missing required feature columns: {missing}")

    expanded = pd.DataFrame(index=frame.index)
    for name in spec.base_features:
        expanded[name] = frame[name]
    for name in spec.scenario_flags:
        expanded[name] = frame[name]

    nan_report: dict[str, list[str]] = {}
    for base_feature in spec.base_features:
        source_values = frame[base_feature].to_numpy(dtype=float)
        for td in spec.transforms.get(base_feature, ()):
            col_name = td.column_name(base_feature)
            result = td.apply(source_values)
            expanded[col_name] = result
            if np.isnan(result).any():
                nan_report.setdefault(td.label, []).append(base_feature)

    for left, right in spec.interactions:
        expanded[f"{left}*{right}"] = frame[left] * frame[right]

    if nan_report:
        warn_nan_transforms(nan_report, stacklevel=2)

    ordered_columns = ordered_expanded_feature_names(spec)
    expanded = expanded.loc[:, list(ordered_columns)]
    return FeatureExpansionResult(
        expanded_frame=expanded,
        ordered_columns=ordered_columns,
    )
