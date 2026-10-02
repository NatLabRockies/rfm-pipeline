"""Artifact-schema helpers for the reduced-form workflow.

The functions in this module preserve original ordering information so downstream
tools can reconstruct feature and output provenance without rerunning model fitting.

It also provides frozen-config provenance stamping (F2 closure).  A resolved
configuration must be cryptographically stamped before any sealed-test evaluation
is permitted.  The stamp records the config hash and the freeze timestamp so that
post-hoc config drift is detectable.
"""

from __future__ import annotations

import datetime
import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Frozen-config provenance API (F2)
# ---------------------------------------------------------------------------


class FrozenConfigRequiredError(RuntimeError):
    """Raised when sealed-test evaluation is attempted without a frozen-config stamp."""


class ConfigDriftError(RuntimeError):
    """Raised when the config hash does not match the frozen stamp."""


@dataclass(frozen=True)
class FrozenConfigStamp:
    """Immutable provenance stamp for a resolved configuration.

    Parameters
    ----------
    config_hash:
        SHA-256 hex digest of the canonical JSON representation of the config.
    frozen_at:
        ISO-8601 UTC timestamp recorded at freeze time.
    """

    config_hash: str
    frozen_at: str


def _config_hash(config: Any) -> str:
    """Return the SHA-256 hex digest of the canonical JSON serialisation of *config*."""
    canonical = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(canonical).hexdigest()


def freeze_config(config: Any) -> FrozenConfigStamp:
    """Freeze *config* and return an auditable provenance stamp.

    The stamp records the SHA-256 hash of the full resolved config and the UTC
    timestamp of the freeze.  Call this once, before any sealed-test evaluation.

    Parameters
    ----------
    config:
        Any JSON-serialisable resolved-config object (dict, dataclass, etc.).

    Returns
    -------
    FrozenConfigStamp
        Immutable stamp suitable for passing to :func:`require_frozen_stamp`.
    """
    return FrozenConfigStamp(
        config_hash=_config_hash(config),
        frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )


def require_frozen_stamp(config: Any, stamp: FrozenConfigStamp | None) -> None:
    """Assert that *config* was frozen before this evaluation point.

    Raises
    ------
    FrozenConfigRequiredError
        When *stamp* is ``None`` — no freeze has been recorded.
    ConfigDriftError
        When the hash of *config* does not match *stamp.config_hash*, indicating
        that the config was modified after the freeze.

    Parameters
    ----------
    config:
        The resolved config that will be used for the sealed-test evaluation.
    stamp:
        The :class:`FrozenConfigStamp` returned by :func:`freeze_config`, or
        ``None`` if no freeze has been performed.
    """
    if stamp is None:
        raise FrozenConfigRequiredError(
            "Sealed-test evaluation requires a frozen-config stamp. "
            "Call freeze_config(config) before evaluating on the sealed test set."
        )
    current_hash = _config_hash(config)
    if current_hash != stamp.config_hash:
        raise ConfigDriftError(
            f"Config drift detected: current hash {current_hash!r} does not match "
            f"frozen stamp hash {stamp.config_hash!r}. "
            "Re-freeze the config if the change is intentional."
        )


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


def canonical_manifest_position_map_keys() -> list[str]:
    """Return the canonical position-map keys stored in the manifest payload."""
    return [
        "all_input_position_map",
        "selected_input_position_map",
        "retained_input_position_map",
        "output_position_map",
    ]


def canonical_manifest_top_level_keys() -> list[str]:
    """Return the canonical top-level keys in the manifest payload."""
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
