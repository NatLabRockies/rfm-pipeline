"""Explicit feature-expansion boundary for notebook-derived modeling stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

SUPPORTED_TRANSFORMS = ("quadratic", "inverse")


@dataclass(frozen=True)
class FeatureExpansionTerm:
    """One explicit derived-feature definition.

    Attributes
    ----------
    name
        Output feature name.
    source_columns
        Input columns consumed by this term.
    term_type
        ``first_order``, ``nonlinear_transformation``, or ``second_order``.
    transform
        Transform label for nonlinear terms, otherwise ``None``.
    provenance
        Provenance label for the term definition.
    """

    name: str
    source_columns: tuple[str, ...]
    term_type: str
    transform: str | None
    provenance: str


@dataclass(frozen=True)
class FeatureExpansionSpec:
    """Explicit feature-expansion configuration.

    Attributes
    ----------
    base_features
        First-order features retained directly in the expanded matrix.
    transforms
        Mapping from base feature name to transform labels.
    interactions
        Ordered interaction pairs to materialize.
    provenance
        Provenance label for the specification.
    """

    base_features: tuple[str, ...]
    transforms: dict[str, tuple[str, ...]]
    interactions: tuple[tuple[str, str], ...]
    provenance: str = "notebook-derived"


@dataclass(frozen=True)
class FeatureExpansionResult:
    """Expanded feature matrix and explicit feature catalog."""

    expanded_frame: pd.DataFrame
    catalog: pd.DataFrame


def make_feature_expansion_spec(
    base_features: list[str] | tuple[str, ...],
    *,
    scenario_flags: tuple[str, ...] = ("AFSC", "UAEORO"),
    add_quadratic_for: list[str] | tuple[str, ...] = (),
    add_inverse_for: list[str] | tuple[str, ...] = (),
    interaction_pairs: list[tuple[str, str]] | tuple[tuple[str, str], ...] = (),
    interaction_with_scenario_flags: bool = True,
    provenance: str = "notebook-derived",
) -> FeatureExpansionSpec:
    """Build an explicit notebook-derived feature-expansion specification.

    Parameters
    ----------
    base_features
        First-order features entering the modeling stage before expansion.
    scenario_flags
        Scenario-flag columns appended if absent from ``base_features``.
    add_quadratic_for
        Base features receiving a ``_quadratic`` derived term.
    add_inverse_for
        Base features receiving a ``_inverse`` derived term.
    interaction_pairs
        Explicit interaction pairs to materialize.
    interaction_with_scenario_flags
        Whether to add interactions between each non-flag base feature and each
        scenario flag.
    provenance
        Provenance label attached to the resulting specification.

    Returns
    -------
    FeatureExpansionSpec
        Explicit feature-expansion specification.

    Raises
    ------
    ValueError
        Raised when the specification contains duplicate features, unsupported
        transforms, or interaction pairs referencing unknown columns.
    """
    ordered_base = tuple(dict.fromkeys([*base_features, *scenario_flags]))
    unknown_quadratic = sorted(set(add_quadratic_for) - set(ordered_base))
    unknown_inverse = sorted(set(add_inverse_for) - set(ordered_base))
    if unknown_quadratic or unknown_inverse:
        missing = sorted(set(unknown_quadratic + unknown_inverse))
        raise ValueError(f"Transform targets are not present in base features: {missing}.")

    interactions: list[tuple[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for pair in interaction_pairs:
        left, right = pair
        if left not in ordered_base or right not in ordered_base:
            raise ValueError(f"Interaction pair {pair!r} references unknown base features.")
        normalized = _normalize_pair(left, right)
        if normalized not in seen_pairs:
            interactions.append(normalized)
            seen_pairs.add(normalized)

    if interaction_with_scenario_flags:
        non_flags = [name for name in ordered_base if name not in scenario_flags]
        for left in non_flags:
            for right in scenario_flags:
                normalized = _normalize_pair(left, right)
                if normalized not in seen_pairs:
                    interactions.append(normalized)
                    seen_pairs.add(normalized)

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
        transforms=transforms,
        interactions=tuple(interactions),
        provenance=provenance,
    )


def feature_catalog_from_spec(spec: FeatureExpansionSpec) -> pd.DataFrame:
    """Render the explicit feature-expansion catalog for a specification."""
    terms: list[FeatureExpansionTerm] = []
    for name in spec.base_features:
        terms.append(
            FeatureExpansionTerm(
                name=name,
                source_columns=(name,),
                term_type="first_order",
                transform=None,
                provenance=spec.provenance,
            )
        )
    for base_feature in spec.base_features:
        for transform in spec.transforms.get(base_feature, ()):
            terms.append(
                FeatureExpansionTerm(
                    name=f"{base_feature}_{transform}",
                    source_columns=(base_feature,),
                    term_type="nonlinear_transformation",
                    transform=transform,
                    provenance=spec.provenance,
                )
            )
    for left, right in spec.interactions:
        terms.append(
            FeatureExpansionTerm(
                name=f"{left}*{right}",
                source_columns=(left, right),
                term_type="second_order",
                transform=None,
                provenance=spec.provenance,
            )
        )
    return pd.DataFrame(asdict(term) for term in terms)


def apply_feature_expansion(
    frame: pd.DataFrame,
    spec: FeatureExpansionSpec,
) -> FeatureExpansionResult:
    """Materialize an expanded feature matrix from an explicit specification.

    Parameters
    ----------
    frame
        Input frame containing the base columns required by ``spec``.
    spec
        Explicit feature-expansion specification.

    Returns
    -------
    FeatureExpansionResult
        Expanded matrix and its aligned feature catalog.

    Raises
    ------
    ValueError
        Raised when required input columns are missing or inverse transforms would
        divide by zero.
    """
    missing = [name for name in spec.base_features if name not in frame.columns]
    if missing:
        raise ValueError(f"Input frame is missing required base columns: {missing}.")

    expanded = pd.DataFrame(index=frame.index)
    catalog = feature_catalog_from_spec(spec)
    for row in catalog.itertuples(index=False):
        source_columns = tuple(row.source_columns)
        if row.term_type == "first_order":
            expanded[row.name] = frame[source_columns[0]]
        elif row.term_type == "nonlinear_transformation":
            expanded[row.name] = _apply_transform(frame[source_columns[0]], row.transform)
        elif row.term_type == "second_order":
            expanded[row.name] = frame[source_columns[0]] * frame[source_columns[1]]
        else:
            raise ValueError(f"Unsupported term type {row.term_type!r} in catalog.")
    return FeatureExpansionResult(expanded_frame=expanded, catalog=catalog)


def _normalize_pair(left: str, right: str) -> tuple[str, str]:
    """Normalize interaction order while preserving distinct terms."""
    if left == right:
        raise ValueError("Self-interactions are not supported in the explicit specification.")
    return tuple(sorted((left, right)))  # type: ignore[return-value]


def _ordered_unique_transforms(transforms: list[str]) -> tuple[str, ...]:
    """Validate and de-duplicate transform labels while preserving order."""
    ordered: list[str] = []
    seen: set[str] = set()
    for transform in transforms:
        if transform not in SUPPORTED_TRANSFORMS:
            raise ValueError(f"Unsupported transform {transform!r}.")
        if transform not in seen:
            ordered.append(transform)
            seen.add(transform)
    return tuple(ordered)


def _apply_transform(series: pd.Series, transform: str | None) -> pd.Series:
    """Apply a supported nonlinear transform to one feature series."""
    if transform == "quadratic":
        return series.astype(float) ** 2
    if transform == "inverse":
        numeric = series.astype(float)
        if np.isclose(numeric, 0.0).any():
            raise ValueError("Inverse transform is undefined for zero-valued inputs.")
        return 1.0 / numeric
    raise ValueError(f"Unsupported transform {transform!r}.")
