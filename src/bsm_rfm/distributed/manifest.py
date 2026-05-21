"""Shard manifest schema and I/O for distributed BSM pipeline execution.

A manifest is a JSONL file where each line is a JSON object representing
one ShardManifest record. The SLURM array runner selects a record by
SLURM_ARRAY_TASK_ID (0-indexed).
"""

from __future__ import annotations

import fcntl
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)


@dataclass
class ShardManifest:
    """Describes one shard of distributed pipeline work.

    A shard is the unit of work assigned to a single SLURM array task.
    For the BSM manuscript pipeline, each shard corresponds to one
    interaction-scoring batch (a subset of feature pairs) or one
    output-conditioning chunk.

    Attributes
    ----------
    shard_id
        Unique identifier, e.g. "task-0042". Used to name output directories.
    stage
        Pipeline stage this shard belongs to: output_conditioning,
        empirical_null_screening, interaction_discovery, nonlinear_discovery,
        sparse_selection, or final_manuscript_artifacts.
    input_paths
        Parquet/NPY/CSV file paths this shard should read.
    output_path
        Durable output directory for this shard's results.
    expected_rows
        Expected number of rows in input (for validation).
    expected_columns
        Expected number of feature columns in input (for validation).
    feature_start_idx
        First feature column index covered by this shard (for pair sharding).
    feature_end_idx
        Last feature column index (exclusive) covered by this shard.
    scenario_id
        Optional scenario/case identifier for multi-scenario runs.
    year
        Optional year for temporally sharded runs.
    status
        Current status: pending, running, completed, failed.
    attempt
        Number of times this shard has been attempted (for retry tracking).
    created_at
        ISO 8601 UTC timestamp when this record was created.
    started_at
        ISO 8601 UTC timestamp when the latest attempt started (or None).
    completed_at
        ISO 8601 UTC timestamp when the attempt completed (or None).
    error_message
        Error message if status == 'failed' (or None).
    """

    shard_id: str
    stage: str
    input_paths: list[str]
    output_path: str
    expected_rows: int = 0
    expected_columns: int = 0
    feature_start_idx: int | None = None
    feature_end_idx: int | None = None
    scenario_id: str | None = None
    year: int | None = None
    status: Literal["pending", "running", "completed", "failed"] = "pending"
    attempt: int = 0
    created_at: str = field(default_factory=lambda: _now_utc())
    started_at: str | None = None
    completed_at: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict:
        """Serialize to a plain dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ShardManifest:
        """Deserialize from a plain dict, ignoring unknown keys."""
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def build_manifest(
    stage: str,
    input_paths: list[str],
    output_root: str,
    n_shards: int,
    expected_rows: int = 0,
    expected_columns: int = 0,
    scenario_id: str | None = None,
) -> list[ShardManifest]:
    """Build a list of ShardManifest records for a given stage.

    For interaction_discovery, shards are split by feature-pair range.
    For other stages, each shard gets all inputs (embarrassingly parallel
    by resample index or permutation index, controlled by the runner).

    Parameters
    ----------
    stage
        Pipeline stage name.
    input_paths
        All input file paths for this stage.
    output_root
        Root directory where shard outputs will be written.
    n_shards
        Number of shards to generate.
    expected_rows
        Expected row count in input data (for shard validation).
    expected_columns
        Expected feature column count (used to compute pair ranges for
        interaction_discovery sharding).
    scenario_id
        Optional scenario identifier embedded in each record.

    Returns
    -------
    list[ShardManifest]
        One manifest entry per shard, all with status='pending'.
    """
    shards: list[ShardManifest] = []
    effective_n_shards = max(1, int(n_shards))
    if expected_columns > 0:
        effective_n_shards = min(effective_n_shards, int(expected_columns))

    for i in range(effective_n_shards):
        shard_id = f"task-{i:04d}"
        if expected_columns > 0:
            feat_start = int(i * expected_columns / effective_n_shards)
            feat_end = int((i + 1) * expected_columns / effective_n_shards)
        else:
            feat_start = None
            feat_end = None

        shards.append(
            ShardManifest(
                shard_id=shard_id,
                stage=stage,
                input_paths=list(input_paths),
                output_path=str(Path(output_root) / shard_id),
                expected_rows=expected_rows,
                expected_columns=expected_columns,
                feature_start_idx=feat_start,
                feature_end_idx=feat_end,
                scenario_id=scenario_id,
            )
        )
    return shards


def save_manifest(shards: list[ShardManifest], path: str | Path) -> None:
    """Write shards to a JSONL manifest file (one JSON object per line).

    Parameters
    ----------
    shards
        List of ShardManifest records.
    path
        Output path (will be overwritten if it exists).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for shard in shards:
            f.write(json.dumps(shard.to_dict()) + "\n")


def load_manifest(path: str | Path) -> list[ShardManifest]:
    """Load shards from a JSONL manifest file.

    Parameters
    ----------
    path
        Path to manifest JSONL file.

    Returns
    -------
    list[ShardManifest]
        Deserialized shard records.
    """
    path = Path(path)
    shards = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                shards.append(ShardManifest.from_dict(json.loads(line)))
    return shards


def update_shard_status(
    manifest_path: str | Path,
    shard_id: str,
    status: Literal["pending", "running", "completed", "failed"],
    error_message: str | None = None,
) -> None:
    """Update the status of a single shard in the manifest file (in-place).

    Uses file locking to safely handle concurrent writes from multiple SLURM
    array tasks. Each task updates its own shard_id, so lock contention is
    minimal, but locking prevents read-during-write corruption that broke the
    first run.

    Parameters
    ----------
    manifest_path
        Path to the JSONL manifest file.
    shard_id
        Which shard to update.
    status
        New status value.
    error_message
        Optional error string (set when status='failed').
    """
    manifest_path = Path(manifest_path)
    lock_path = manifest_path.parent / f"{manifest_path.name}.lock"

    # Use lock file to serialize writes
    with lock_path.open("a") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            shards = load_manifest(manifest_path)
            now = _now_utc()
            for shard in shards:
                if shard.shard_id == shard_id:
                    shard.status = status
                    if status == "running":
                        shard.attempt += 1
                        shard.started_at = now
                        shard.completed_at = None
                        shard.error_message = None
                    elif status in {"completed", "failed"}:
                        shard.completed_at = now
                        shard.error_message = error_message
                    break
            save_manifest(shards, manifest_path)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def manifest_summary(shards: list[ShardManifest]) -> dict[str, int]:
    """Return counts by status."""
    counts: dict[str, int] = {"pending": 0, "running": 0, "completed": 0, "failed": 0}
    for s in shards:
        counts[s.status] = counts.get(s.status, 0) + 1
    return counts


def resolve_interaction_discovery_shard_inputs(
    artifact_dir: str | Path,
    dataset_path: str | Path | None = None,
) -> dict[str, str]:
    """Resolve required shard inputs for interaction_discovery stage.

    Locates prior-stage artifacts (pca_scores from output_conditioning,
    retained_terms from empirical_null_screen) and raw dataset files
    (X.parquet, holdout_assignments.parquet, actual_input_feature_catalog.parquet).

    Parameters
    ----------
    artifact_dir
        Artifact root directory (typically workflow output_root). Must contain
        subdirectories output_conditioning/ and empirical_null_screen/.
    dataset_path
        Optional path to the raw dataset directory containing X.parquet,
        holdout_assignments.parquet, and actual_input_feature_catalog.parquet.
        If None, these files are looked for in artifact_dir (legacy behavior).

    Returns
    -------
    dict[str, str]
        Mapping from symbolic name to absolute file path:
        - "pca_scores": path to output_conditioning/pca_scores.csv
        - "retained_terms": path to empirical_null_screen/retained_terms.csv
        - "X": path to X.parquet
        - "holdout_assignments": path to holdout_assignments.parquet
        - "feature_catalog": path to actual_input_feature_catalog.parquet

    Raises
    ------
    FileNotFoundError
        If any required file is missing.
    """
    artifact_dir = Path(artifact_dir)
    data_root = Path(dataset_path) if dataset_path else artifact_dir

    required_files = {
        "pca_scores": artifact_dir / "output_conditioning" / "pca_scores.csv",
        "retained_terms": artifact_dir / "empirical_null_screen" / "retained_terms.csv",
        "X": data_root / "X.parquet",
        "holdout_assignments": data_root / "holdout_assignments.parquet",
        "feature_catalog": data_root / "actual_input_feature_catalog.parquet",
    }

    # feature_catalog may also be named manuscript_feature_catalog.parquet
    if not required_files["feature_catalog"].exists():
        alt = data_root / "manuscript_feature_catalog.parquet"
        if alt.exists():
            required_files["feature_catalog"] = alt

    result = {}
    for name, path in required_files.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Required artifact '{name}' not found at {path}. "
                f"Ensure prior stages (output_conditioning, empirical_null_screen) have completed."
            )
        result[name] = str(path.resolve())

    return result
