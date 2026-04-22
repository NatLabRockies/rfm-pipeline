"""Artifact-schema helpers for the reduced-form workflow.

The functions in this module preserve original ordering information so that downstream
visualization code and manuscript tables can reconstruct feature and output provenance
without re-running modeling code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd


def canonical_manifest_position_map_keys() -> list[str]:
    """Return the stable position-map keys stored in the manifest.

    Returns
    -------
    list of str
        Canonical position-map payload keys written into ``manifest.json``.
    """
    return [
        "all_input_position_map",
        "selected_input_position_map",
        "retained_input_position_map",
        "output_position_map",
    ]


def canonical_manifest_top_level_keys() -> list[str]:
    """Return the stable top-level keys stored in the manifest.

    Returns
    -------
    list of str
        Canonical top-level manifest keys for exported bundles.
    """
    return [
        "dataset_tag",
        "n_all_input_features",
        "n_selected_features",
        "n_retained_features",
        "n_outputs",
        "all_input_features",
        "selected_features",
        "retained_features",
        "output_names",
        "files",
        "metrics",
        "evaluation",
        "upstream_provenance",
        *canonical_manifest_position_map_keys(),
    ]


@dataclass(frozen=True)
class PipelineManifest:
    """Top-level manifest for a canonical pipeline export bundle.

    Parameters
    ----------
    dataset_tag
        Human-readable label for the dataset or run family.
    n_all_input_features
        Total number of candidate inputs available to the modeling stage.
    n_selected_features
        Number of inputs surviving the screening stage.
    n_retained_features
        Number of inputs retained in the final fitted model.
    n_outputs
        Total number of modeled outputs.
    all_input_features
        Candidate input names in their original order.
    selected_features
        Screening-stage selected input names in their original order.
    retained_features
        Final retained input names in their original order.
    output_names
        Output names in their original order.
    files
        Mapping from logical artifact name to relative file path.
    metrics
        Summary metric payloads associated with the run.
    evaluation
        Evaluation metadata associated with the run.
    upstream_provenance
        Provenance notes describing upstream scripts, notebooks, and settings.
    """

    dataset_tag: str
    n_all_input_features: int
    n_selected_features: int
    n_retained_features: int
    n_outputs: int
    all_input_features: list[str]
    selected_features: list[str]
    retained_features: list[str]
    output_names: list[str]
    files: dict[str, str]
    metrics: dict[str, Any] = field(default_factory=dict)
    evaluation: dict[str, Any] = field(default_factory=dict)
    upstream_provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the manifest as a JSON-serializable dictionary.

        Returns
        -------
        dict[str, Any]
            Manifest payload enriched with position maps for all ordered feature and
            output collections.
        """
        payload = asdict(self)
        payload["all_input_position_map"] = build_position_map(self.all_input_features)
        payload["selected_input_position_map"] = build_position_map(self.selected_features)
        payload["retained_input_position_map"] = build_position_map(self.retained_features)
        payload["output_position_map"] = build_position_map(self.output_names)
        return payload


def build_position_map(names: list[str]) -> dict[str, int]:
    """Build a zero-based position map for an ordered list of names.

    Parameters
    ----------
    names
        Ordered names whose original positions must be preserved.

    Returns
    -------
    dict[str, int]
        Mapping from each name to its zero-based original position.
    """
    return {name: int(i) for i, name in enumerate(names)}


def make_metadata_frame(names: list[str], name_column: str) -> pd.DataFrame:
    """Create a two-column metadata frame preserving original order.

    Parameters
    ----------
    names
        Ordered feature or output names.
    name_column
        Column name to use for the names in the returned table.

    Returns
    -------
    pandas.DataFrame
        Metadata table with the name column and an ``original_position`` column.
    """
    return pd.DataFrame(
        {
            name_column: list(names),
            "original_position": list(range(len(names))),
        }
    )
