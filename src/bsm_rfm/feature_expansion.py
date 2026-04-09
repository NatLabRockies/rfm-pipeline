"""Explicit feature-expansion contracts for the reduced-form workflow.

This module captures the notebook-derived boundary between upstream null
screening and downstream modeling. It records *what* feature expansion should do
without claiming the original notebook is itself canonical source code.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class FeatureExpansionSpec:
    """Explicit feature-expansion contract.

    Parameters
    ----------
    base_features
        Ordered first-order features entering the expansion step.
    transforms
        Mapping from base feature to ordered transform labels such as
        ``quadratic`` or ``inverse``.
    interaction_pairs
        Ordered second-order interaction pairs.
    scenario_flags
        Scenario-indicator columns expected downstream.
    source_status
        Provenance label for this spec.
    primary_source
        Script, notebook, or audit artifact anchoring this spec.
    notes
        Short explanatory note.
    """

    base_features: tuple[str, ...]
    transforms: dict[str, tuple[str, ...]] = field(default_factory=dict)
    interaction_pairs: tuple[tuple[str, str], ...] = ()
    scenario_flags: tuple[str, ...] = ("AFSC", "UAEORO")
    source_status: str = "notebook-derived"
    primary_source: str = "make_nonlinear_features.ipynb"
    notes: str = (
        "Notebook-derived feature-expansion boundary extracted into an explicit package contract."
    )


@dataclass(frozen=True)
class FeatureExpansionResult:
    """Materialized feature-expansion output.

    Attributes
    ----------
    expanded_frame
        Expanded matrix containing base features, transformed features,
        interactions, and required scenario flags.
    feature_order
        Ordered expanded feature names in the returned frame.
    feature_metadata
        Row-wise metadata describing each expanded feature.
    provenance
        Minimal provenance describing how the expansion was produced.
    """

    expanded_frame: pd.DataFrame
    feature_order: tuple[str, ...]
    feature_metadata: pd.DataFrame
    provenance: dict[str, object]


SUPPORTED_TRANSFORMS = {"quadratic", "inverse"}


def _ordered_unique_strings(values: list[str]) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            ordered.append(value)
            seen.add(value)
    return tuple(ordered)


def _ordered_unique_pairs(values: list[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
    ordered: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for left, right in values:
        pair = (str(left), str(right))
        if pair not in seen:
            ordered.append(pair)
            seen.add(pair)
    return tuple(ordered)


def make_feature_expansion_spec(
    base_features: list[str],
    *,
    add_quadratic_for: list[str] | None = None,
    add_inverse_for: list[str] | None = None,
    interaction_pairs: list[tuple[str, str]] | None = None,
    scenario_flags: list[str] | None = None,
) -> FeatureExpansionSpec:
    """Build an explicit notebook-derived feature-expansion spec.

    Parameters
    ----------
    base_features
        Ordered first-order features.
    add_quadratic_for
        Features receiving a quadratic term.
    add_inverse_for
        Features receiving an inverse term.
    interaction_pairs
        Ordered second-order interaction pairs.
    scenario_flags
        Scenario-indicator columns expected downstream.

    Returns
    -------
    FeatureExpansionSpec
        Explicit feature-expansion contract.

    Raises
    ------
    ValueError
        Raised if a requested transform is attached to an unknown base feature.
    """
    ordered_base = _ordered_unique_strings([str(name) for name in base_features])
    quadratic_targets = set(add_quadratic_for or [])
    inverse_targets = set(add_inverse_for or [])
    unknown_targets = (quadratic_targets | inverse_targets) - set(ordered_base)
    if unknown_targets:
        unknown = ", ".join(sorted(unknown_targets))
        raise ValueError(f"Transform requested for unknown base feature(s): {unknown}.")

    transforms: dict[str, tuple[str, ...]] = {}
    for name in ordered_base:
        labels: list[str] = []
        if name in quadratic_targets:
            labels.append("quadratic")
        if name in inverse_targets:
            labels.append("inverse")
        if labels:
            transforms[name] = _ordered_unique_strings(labels)

    return FeatureExpansionSpec(
        base_features=ordered_base,
        transforms=transforms,
        interaction_pairs=_ordered_unique_pairs(interaction_pairs or []),
        scenario_flags=tuple(scenario_flags or ["AFSC", "UAEORO"]),
    )


def expanded_feature_names(spec: FeatureExpansionSpec) -> tuple[str, ...]:
    """Return the ordered expanded feature names implied by a spec."""
    names: list[str] = list(spec.base_features)
    for base_feature in spec.base_features:
        for transform in spec.transforms.get(base_feature, ()):
            names.append(f"{base_feature}_{transform}")
    for left, right in spec.interaction_pairs:
        names.append(f"[{left} * {right}]")
    for flag in spec.scenario_flags:
        if flag not in names:
            names.append(flag)
    return tuple(names)


def apply_feature_expansion(
    frame: pd.DataFrame,
    spec: FeatureExpansionSpec,
) -> FeatureExpansionResult:
    """Materialize an expanded feature matrix from an explicit specification.

    Parameters
    ----------
    frame
        Input data containing the base features and required scenario flags.
    spec
        Explicit feature-expansion contract.

    Returns
    -------
    FeatureExpansionResult
        Expanded data, feature ordering, metadata table, and provenance.

    Raises
    ------
    KeyError
        Raised if required base features or scenario flags are missing.
    ValueError
        Raised if an unsupported transform is requested.
    ZeroDivisionError
        Raised when an inverse term is requested for a feature containing zeros.
    """
    required_columns = set(spec.base_features) | set(spec.scenario_flags)
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        missing_text = ", ".join(missing)
        raise KeyError(f"Missing required feature-expansion columns: {missing_text}.")

    expanded = pd.DataFrame(index=frame.index)
    metadata: list[dict[str, object]] = []

    for base_feature in spec.base_features:
        expanded[base_feature] = frame[base_feature]
        metadata.append(
            {
                "feature_name": base_feature,
                "feature_type": "first_order",
                "base_feature": base_feature,
                "transform": None,
                "interaction_left": None,
                "interaction_right": None,
            }
        )

    for base_feature in spec.base_features:
        for transform in spec.transforms.get(base_feature, ()):
            if transform not in SUPPORTED_TRANSFORMS:
                raise ValueError(f"Unsupported transform requested: {transform}.")
            feature_name = f"{base_feature}_{transform}"
            series = frame[base_feature].astype(float)
            if transform == "quadratic":
                expanded[feature_name] = series.pow(2)
            elif transform == "inverse":
                if (series == 0).any():
                    raise ZeroDivisionError(
                        f"Cannot form inverse term for {base_feature} with zero entries."
                    )
                expanded[feature_name] = 1.0 / series
            metadata.append(
                {
                    "feature_name": feature_name,
                    "feature_type": "nonlinear_transformation",
                    "base_feature": base_feature,
                    "transform": transform,
                    "interaction_left": None,
                    "interaction_right": None,
                }
            )

    for left, right in spec.interaction_pairs:
        if left not in frame.columns or right not in frame.columns:
            raise KeyError(f"Missing interaction input(s): {left}, {right}.")
        feature_name = f"[{left} * {right}]"
        expanded[feature_name] = frame[left] * frame[right]
        metadata.append(
            {
                "feature_name": feature_name,
                "feature_type": "second_order",
                "base_feature": None,
                "transform": None,
                "interaction_left": left,
                "interaction_right": right,
            }
        )

    for flag in spec.scenario_flags:
        if flag not in expanded.columns:
            expanded[flag] = frame[flag]
            metadata.append(
                {
                    "feature_name": flag,
                    "feature_type": "scenario_flag",
                    "base_feature": flag,
                    "transform": None,
                    "interaction_left": None,
                    "interaction_right": None,
                }
            )

    feature_order = tuple(expanded.columns.tolist())
    provenance = {
        "source_status": spec.source_status,
        "primary_source": spec.primary_source,
        "n_base_features": int(len(spec.base_features)),
        "n_expanded_features": int(len(feature_order)),
        "supported_transforms": tuple(sorted(SUPPORTED_TRANSFORMS)),
    }
    feature_metadata = pd.DataFrame.from_records(metadata)
    return FeatureExpansionResult(
        expanded_frame=expanded,
        feature_order=feature_order,
        feature_metadata=feature_metadata,
        provenance=provenance,
    )


__all__ = [
    "FeatureExpansionResult",
    "FeatureExpansionSpec",
    "SUPPORTED_TRANSFORMS",
    "apply_feature_expansion",
    "expanded_feature_names",
    "make_feature_expansion_spec",
]
