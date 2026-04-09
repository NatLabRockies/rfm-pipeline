"""Explicit feature-expansion boundary recovered from the audited notebook workflow.

The recovered archive includes a notebook that expands influential first-order inputs
into a wider modeling matrix with scenario flags, nonlinear transforms, and second-
order interactions. This module captures that boundary as an explicit, tested package
contract without claiming that the notebook itself is canonical source code.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


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
        Mapping from base feature name to an ordered tuple of transform labels.
        Supported labels are ``"quadratic"`` and ``"inverse"``.
    interactions
        Ordered pairs of feature names whose product terms should be created.
    provenance
        Provenance label describing the strength of the recovered source.
    source_artifact
        Supporting artifact for the recovered specification.
    """

    base_features: tuple[str, ...]
    scenario_flags: tuple[str, ...]
    transforms: dict[str, tuple[str, ...]]
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


def _ordered_unique_transforms(values: list[str]) -> tuple[str, ...]:
    allowed = {"quadratic", "inverse"}
    filtered = [value for value in values if value in allowed]
    return _ordered_unique(filtered)


def default_feature_expansion_spec(
    *,
    base_features: list[str],
    scenario_flags: tuple[str, ...] = ("AFSC", "UAEORO"),
    add_quadratic_for: tuple[str, ...] = (),
    add_inverse_for: tuple[str, ...] = (),
    interaction_pairs: tuple[tuple[str, str], ...] = (),
) -> FeatureExpansionSpec:
    """Build a simple explicit feature-expansion specification.

    Parameters
    ----------
    base_features
        Ordered influential first-order feature names.
    scenario_flags
        Scenario-indicator features preserved in the expanded matrix.
    add_quadratic_for
        Base features that should also receive quadratic terms.
    add_inverse_for
        Base features that should also receive inverse terms.
    interaction_pairs
        Ordered feature pairs used to create product terms.

    Returns
    -------
    FeatureExpansionSpec
        Explicit feature-expansion contract labeled as notebook-derived.
    """
    ordered_base = _ordered_unique(list(base_features))
    quadratic_targets = set(add_quadratic_for)
    inverse_targets = set(add_inverse_for)
    transform_targets = quadratic_targets | inverse_targets

    transforms = {
        name: _ordered_unique_transforms(
            [
                *(["quadratic"] if name in quadratic_targets else []),
                *(["inverse"] if name in inverse_targets else []),
            ]
        )
        for name in ordered_base
        if name in transform_targets
    }
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
        for transform in spec.transforms.get(base_feature, ()):
            if transform == "quadratic":
                ordered.append(f"{base_feature}_quadratic")
            elif transform == "inverse":
                ordered.append(f"{base_feature}_inverse")
    ordered.extend(f"{left}*{right}" for left, right in spec.interactions)
    return _ordered_unique(ordered)


def _inverse_series(series: pd.Series) -> pd.Series:
    if (series == 0).any():
        raise ValueError("Cannot create inverse-transformed features from zero-valued rows.")
    return 1.0 / series


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
    ValueError
        Raised when an inverse transform is requested for a feature containing zeros.
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

    for base_feature in spec.base_features:
        source = frame[base_feature]
        for transform in spec.transforms.get(base_feature, ()):
            if transform == "quadratic":
                expanded[f"{base_feature}_quadratic"] = source**2
            elif transform == "inverse":
                expanded[f"{base_feature}_inverse"] = _inverse_series(source)

    for left, right in spec.interactions:
        expanded[f"{left}*{right}"] = frame[left] * frame[right]

    ordered_columns = ordered_expanded_feature_names(spec)
    expanded = expanded.loc[:, list(ordered_columns)]
    return FeatureExpansionResult(
        expanded_frame=expanded,
        ordered_columns=ordered_columns,
    )
