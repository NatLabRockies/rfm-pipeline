"""Canonical pre-execution controls and score-only interaction artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np


def _canonical_json(value: Any) -> bytes:
    """Encode JSON-compatible values deterministically."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    """Hash an array's dtype, shape, and C-order payload."""
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(repr(array.shape).encode("utf-8"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _sha256_names(names: tuple[str, ...]) -> str:
    return _sha256_bytes(_canonical_json(list(names)))


def _seed_schedule_identity(
    *, stage_seed: int, permutation_draws: int, n_training_rows: int
) -> tuple[str, str]:
    """Return the exact score-seed and response-row schedule identities."""
    schedule_rng = np.random.default_rng(stage_seed)
    score_seeds = np.asarray(
        [int(schedule_rng.integers(0, 2**31)) for _ in range(permutation_draws + 1)],
        dtype=np.int64,
    )
    permutation_digest = hashlib.sha256()
    permutation_digest.update(repr((permutation_draws, n_training_rows)).encode("utf-8"))
    for score_seed in score_seeds[1:]:
        row_order = np.random.default_rng(int(score_seed)).permutation(n_training_rows)
        permutation_digest.update(np.asarray(row_order, dtype=np.int64).tobytes(order="C"))
    return _sha256_array(score_seeds), permutation_digest.hexdigest()


def _current_source_and_lock_hashes() -> tuple[str, str]:
    """Hash the live implementation tree and dependency lock for reuse checks."""
    repo_root = Path(__file__).resolve().parents[2]
    source_digest = hashlib.sha256()
    for source_path in sorted((repo_root / "src" / "rfm_pipeline").rglob("*.py")):
        source_digest.update(str(source_path.relative_to(repo_root)).encode("utf-8"))
        source_digest.update(b"\0")
        source_digest.update(source_path.read_bytes())
        source_digest.update(b"\0")
    lock_path = repo_root / "pixi.lock"
    if not lock_path.is_file():
        raise ValueError("Control snapshot requires the repository dependency lockfile.")
    return source_digest.hexdigest(), _sha256_bytes(lock_path.read_bytes())


def _plain_dataclass_dict(value: Any) -> dict[str, Any]:
    """Return a deterministic JSON-compatible dataclass mapping."""
    if not is_dataclass(value):
        raise TypeError(f"Expected dataclass controls, got {type(value).__name__}.")
    result = asdict(value)
    return {field.name: result[field.name] for field in fields(value)}


@dataclass(frozen=True)
class CanonicalExecutionContract:
    """Frozen controls required before interaction scoring may begin.

    The contract binds interaction-selection controls to terminal HC3, pruning,
    and OLS-refit controls.  Its checksum is persisted by every score-only
    artifact and must match at reduction time.
    """

    schema_version: int
    source_sha256: str
    interaction_controls: dict[str, Any]
    terminal_controls: dict[str, Any]

    def __post_init__(self) -> None:
        """Validate exact maxT and terminal-refit controls."""
        method = self.interaction_controls.get("selection_method")
        if method != "max_stat_adjusted_p_mc":
            raise ValueError(
                "Canonical interaction controls require "
                f"selection_method='max_stat_adjusted_p_mc'; got {method!r}."
            )
        alpha = float(self.interaction_controls.get("selection_alpha", 0.0))
        draws = int(self.interaction_controls.get("permutation_count_B", 0))
        minimum_draws = int(self.interaction_controls.get("minimum_selection_draws", 0))
        if not 0.0 < alpha < 1.0:
            raise ValueError(
                "Canonical interaction selection_alpha must be in the open interval (0, 1)."
            )
        if draws < minimum_draws:
            raise ValueError(
                "Canonical interaction controls fail pre-execution draw adequacy: "
                f"permutation_count_B={draws} < minimum_selection_draws={minimum_draws}."
            )
        if minimum_draws < 199:
            raise ValueError(
                "Canonical interaction controls require minimum_selection_draws >= 199."
            )
        if 1.0 / (draws + 1) > alpha:
            raise ValueError(
                "Canonical interaction controls cannot resolve selection_alpha with the "
                "configured finite permutation draw count."
            )
        terminal = self.terminal_controls
        if terminal.get("refit_method") != "ordinary_least_squares_after_pruning":
            raise ValueError(
                "Canonical terminal controls require the ordinary_least_squares_after_pruning "
                "refit method."
            )
        if terminal.get("inferential_filter_interval_method") != (
            "hc3_wald_95_percent_drop_if_zero_compatible_for_all_outputs"
        ):
            raise ValueError(
                "Canonical terminal controls require the frozen 95% HC3 Wald inferential filter."
            )
        inferential_alpha = float(terminal.get("inferential_filter_alpha", 0.0))
        if not 0.0 < inferential_alpha < 1.0:
            raise ValueError("Canonical terminal HC3 inferential-filter alpha must be in (0, 1).")
        valid_subset_modes = {"all", "random_fraction", "target_list", "top_variance"}
        subset_mode = terminal.get("hc3_output_subset_mode", "all")
        if subset_mode not in valid_subset_modes:
            raise ValueError(
                "Canonical terminal HC3 output subset mode must be one of "
                f"{sorted(valid_subset_modes)}."
            )
        output_fraction = terminal.get("hc3_output_fraction")
        if output_fraction is not None and not 0.0 < float(output_fraction) <= 1.0:
            raise ValueError("Canonical terminal HC3 output fraction must be in (0, 1].")
        if subset_mode == "random_fraction" and output_fraction is None:
            raise ValueError(
                "Canonical terminal HC3 random_fraction mode requires an output fraction."
            )
        if subset_mode == "target_list" and not terminal.get("hc3_output_names", ()):
            raise ValueError(
                "Canonical terminal HC3 target_list mode requires at least one output name."
            )
        max_outputs = terminal.get("hc3_output_max_outputs")
        if max_outputs is not None and int(max_outputs) <= 0:
            raise ValueError("Canonical terminal HC3 max outputs must be positive.")
        if terminal.get("hc3_output_subset_metric", "variance") != "variance":
            raise ValueError("Canonical terminal HC3 subset metric must be 'variance'.")
        error_scale_quantile = float(terminal.get("pruning_error_scale_quantile", 0.95))
        if not 0.0 < error_scale_quantile <= 1.0:
            raise ValueError("Canonical terminal pruning error-scale quantile must be in (0, 1].")
        delta_override = terminal.get("pruning_delta_threshold_override")
        remove_count_override = terminal.get("pruning_remove_count_override")
        if delta_override is not None and float(delta_override) < 0.0:
            raise ValueError(
                "Canonical terminal pruning delta-threshold override must be nonnegative."
            )
        if remove_count_override is not None and int(remove_count_override) < 0:
            raise ValueError(
                "Canonical terminal pruning remove-count override must be nonnegative."
            )
        if delta_override is not None and remove_count_override is not None:
            raise ValueError("Canonical terminal controls allow at most one pruning override.")

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical payload used for checksumming."""
        return {
            "schema_version": int(self.schema_version),
            "source_sha256": str(self.source_sha256),
            "interaction_controls": self.interaction_controls,
            "terminal_controls": self.terminal_controls,
        }

    @property
    def checksum(self) -> str:
        """Return the immutable contract checksum."""
        return _sha256_bytes(_canonical_json(self.to_dict()))


def canonical_execution_contract_from_specs(
    interaction_spec: Any,
    final_spec: Any | None = None,
    *,
    source_sha256: str | None = None,
) -> CanonicalExecutionContract:
    """Build the canonical contract from typed interaction and terminal specs."""
    interaction_controls = _plain_dataclass_dict(interaction_spec)
    terminal_controls = (
        _plain_dataclass_dict(final_spec)
        if final_spec is not None
        else {
            "inferential_filter_interval_method": (
                "hc3_wald_95_percent_drop_if_zero_compatible_for_all_outputs"
            ),
            "inferential_filter_alpha": 0.05,
            "hc3_output_subset_mode": "all",
            "hc3_output_fraction": None,
            "hc3_output_names": (),
            "hc3_output_max_outputs": None,
            "hc3_output_random_seed": 123,
            "hc3_output_subset_metric": "variance",
            "pruning_error_scale_quantile": 0.95,
            "pruning_delta_threshold_override": None,
            "pruning_remove_count_override": None,
        }
    )
    terminal_controls["refit_method"] = "ordinary_least_squares_after_pruning"
    source = source_sha256
    if source is None:
        source = _sha256_bytes(
            _canonical_json(
                {
                    "interaction_controls": interaction_controls,
                    "terminal_controls": terminal_controls,
                }
            )
        )
    return CanonicalExecutionContract(
        schema_version=1,
        source_sha256=source,
        interaction_controls=interaction_controls,
        terminal_controls=terminal_controls,
    )


def load_canonical_execution_contract(config_path: str | Path) -> CanonicalExecutionContract:
    """Load one generic workflow config and freeze its effective controls.

    No default-config fallback is permitted: the supplied path is the source of
    record for the exact selection and terminal-model settings.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Canonical execution config does not exist: {path}")

    from .config import apply_fast_mode_overrides, load_config
    from .manuscript_pipeline_helpers import config_to_legacy_case_study
    from .manuscript_stages import (
        final_manuscript_artifacts_spec_from_case_study_config,
        interaction_discovery_spec_from_case_study_config,
    )

    workflow_config = apply_fast_mode_overrides(load_config(path))
    case_study_config = config_to_legacy_case_study(workflow_config)
    interaction_spec = interaction_discovery_spec_from_case_study_config(case_study_config)
    final_spec = final_manuscript_artifacts_spec_from_case_study_config(case_study_config)
    return canonical_execution_contract_from_specs(
        interaction_spec,
        final_spec,
        source_sha256=_sha256_bytes(path.read_bytes()),
    )


@dataclass(frozen=True)
class ControlSnapshot:
    """Identity snapshot for one complete interaction candidate family."""

    schema_version: int
    contract_sha256: str
    implementation_source_sha256: str
    dependency_lock_sha256: str
    stage_seed: int
    score_seed_schedule_sha256: str
    permutation_index_schedule_sha256: str
    candidate_pair_names: tuple[str, ...]
    candidate_family_sha256: str
    training_sample_ids_sha256: str
    feature_matrix_sha256: str
    response_matrix_sha256: str
    component_names: tuple[str, ...]
    n_training_rows: int
    permutation_draws: int

    def __post_init__(self) -> None:
        """Validate the frozen family identity."""
        if self.n_training_rows < 1:
            raise ValueError("Control snapshot requires at least one training row.")
        if self.permutation_draws < 1:
            raise ValueError("Control snapshot requires at least one permutation draw.")
        if len(self.candidate_pair_names) != len(set(self.candidate_pair_names)):
            raise ValueError("Control snapshot candidate-family contains duplicate pair names.")
        if self.candidate_family_sha256 != _sha256_names(self.candidate_pair_names):
            raise ValueError(
                "Control snapshot candidate-family checksum does not match pair order."
            )
        for name, value in (
            ("contract_sha256", self.contract_sha256),
            ("implementation_source_sha256", self.implementation_source_sha256),
            ("dependency_lock_sha256", self.dependency_lock_sha256),
            ("score_seed_schedule_sha256", self.score_seed_schedule_sha256),
            ("permutation_index_schedule_sha256", self.permutation_index_schedule_sha256),
        ):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"Control snapshot {name} is not a SHA-256 digest.")

    def to_dict(self) -> dict[str, Any]:
        """Return a serialization-safe snapshot payload."""
        return {
            "schema_version": int(self.schema_version),
            "contract_sha256": self.contract_sha256,
            "implementation_source_sha256": self.implementation_source_sha256,
            "dependency_lock_sha256": self.dependency_lock_sha256,
            "stage_seed": int(self.stage_seed),
            "score_seed_schedule_sha256": self.score_seed_schedule_sha256,
            "permutation_index_schedule_sha256": self.permutation_index_schedule_sha256,
            "candidate_pair_names": list(self.candidate_pair_names),
            "candidate_family_sha256": self.candidate_family_sha256,
            "training_sample_ids_sha256": self.training_sample_ids_sha256,
            "feature_matrix_sha256": self.feature_matrix_sha256,
            "response_matrix_sha256": self.response_matrix_sha256,
            "component_names": list(self.component_names),
            "n_training_rows": int(self.n_training_rows),
            "permutation_draws": int(self.permutation_draws),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ControlSnapshot:
        """Deserialize and validate a persisted snapshot."""
        required = {
            "schema_version",
            "contract_sha256",
            "implementation_source_sha256",
            "dependency_lock_sha256",
            "stage_seed",
            "score_seed_schedule_sha256",
            "permutation_index_schedule_sha256",
            "candidate_pair_names",
            "candidate_family_sha256",
            "training_sample_ids_sha256",
            "feature_matrix_sha256",
            "response_matrix_sha256",
            "component_names",
            "n_training_rows",
            "permutation_draws",
        }
        missing = required.difference(data)
        if missing:
            raise ValueError(f"Control snapshot is missing fields: {sorted(missing)}")
        return cls(
            schema_version=int(data["schema_version"]),
            contract_sha256=str(data["contract_sha256"]),
            implementation_source_sha256=str(data["implementation_source_sha256"]),
            dependency_lock_sha256=str(data["dependency_lock_sha256"]),
            stage_seed=int(data["stage_seed"]),
            score_seed_schedule_sha256=str(data["score_seed_schedule_sha256"]),
            permutation_index_schedule_sha256=str(data["permutation_index_schedule_sha256"]),
            candidate_pair_names=tuple(str(name) for name in data["candidate_pair_names"]),
            candidate_family_sha256=str(data["candidate_family_sha256"]),
            training_sample_ids_sha256=str(data["training_sample_ids_sha256"]),
            feature_matrix_sha256=str(data["feature_matrix_sha256"]),
            response_matrix_sha256=str(data["response_matrix_sha256"]),
            component_names=tuple(str(name) for name in data["component_names"]),
            n_training_rows=int(data["n_training_rows"]),
            permutation_draws=int(data["permutation_draws"]),
        )

    @property
    def checksum(self) -> str:
        """Return a full snapshot identity checksum."""
        return _sha256_bytes(_canonical_json(self.to_dict()))


def build_control_snapshot(
    contract: CanonicalExecutionContract,
    *,
    candidate_pair_names: tuple[str, ...],
    training_sample_ids: np.ndarray,
    feature_matrix: np.ndarray,
    response_matrix: np.ndarray,
    component_names: tuple[str, ...],
) -> ControlSnapshot:
    """Capture immutable data and control identities before score execution."""
    training_ids = np.asarray(training_sample_ids)
    features = np.asarray(feature_matrix)
    response = np.asarray(response_matrix)
    if training_ids.ndim != 1 or training_ids.size < 1:
        raise ValueError("Control snapshot requires a non-empty one-dimensional training ID array.")
    if features.ndim != 2 or response.ndim != 2:
        raise ValueError("Control snapshot feature and response matrices must be two-dimensional.")
    if features.shape[0] != training_ids.size or response.shape[0] != training_ids.size:
        raise ValueError(
            "Control snapshot training IDs, feature matrix, and response matrix must have the "
            "same row count."
        )
    if response.shape[1] != len(component_names):
        raise ValueError(
            "Control snapshot component names must match the response-matrix column count."
        )
    if len(component_names) != len(set(component_names)):
        raise ValueError("Control snapshot component names must be unique.")
    draws = int(contract.interaction_controls["permutation_count_B"])
    stage_seed = int(contract.interaction_controls["random_seed"])
    score_seed_hash, permutation_schedule_hash = _seed_schedule_identity(
        stage_seed=stage_seed,
        permutation_draws=draws,
        n_training_rows=int(training_ids.size),
    )
    source_hash, lock_hash = _current_source_and_lock_hashes()
    return ControlSnapshot(
        schema_version=1,
        contract_sha256=contract.checksum,
        implementation_source_sha256=source_hash,
        dependency_lock_sha256=lock_hash,
        stage_seed=stage_seed,
        score_seed_schedule_sha256=score_seed_hash,
        permutation_index_schedule_sha256=permutation_schedule_hash,
        candidate_pair_names=tuple(str(name) for name in candidate_pair_names),
        candidate_family_sha256=_sha256_names(tuple(str(name) for name in candidate_pair_names)),
        training_sample_ids_sha256=_sha256_array(training_ids),
        feature_matrix_sha256=_sha256_array(features),
        response_matrix_sha256=_sha256_array(response),
        component_names=tuple(str(name) for name in component_names),
        n_training_rows=int(training_ids.shape[0]),
        permutation_draws=draws,
    )


def verify_control_snapshot(
    snapshot: ControlSnapshot,
    contract: CanonicalExecutionContract,
) -> None:
    """Reject a snapshot whose controls or identity drift from the contract."""
    if snapshot.schema_version != 1:
        raise ValueError(f"Unsupported control snapshot schema: {snapshot.schema_version}")
    if snapshot.contract_sha256 != contract.checksum:
        raise ValueError(
            "Control snapshot contract checksum differs from the canonical execution contract."
        )
    current_source_hash, current_lock_hash = _current_source_and_lock_hashes()
    if snapshot.implementation_source_sha256 != current_source_hash:
        raise ValueError("Control snapshot implementation source hash differs from live bytes.")
    if snapshot.dependency_lock_sha256 != current_lock_hash:
        raise ValueError("Control snapshot dependency lock hash differs from live bytes.")
    expected_draws = int(contract.interaction_controls["permutation_count_B"])
    if snapshot.permutation_draws != expected_draws:
        raise ValueError(
            "Control snapshot permutation draw count differs from the canonical execution contract."
        )
    expected_stage_seed = int(contract.interaction_controls["random_seed"])
    if snapshot.stage_seed != expected_stage_seed:
        raise ValueError(
            "Control snapshot stage seed differs from the canonical execution contract."
        )
    expected_score_hash, expected_permutation_hash = _seed_schedule_identity(
        stage_seed=expected_stage_seed,
        permutation_draws=expected_draws,
        n_training_rows=snapshot.n_training_rows,
    )
    if snapshot.score_seed_schedule_sha256 != expected_score_hash:
        raise ValueError("Control snapshot score-seed schedule differs from the frozen seed.")
    if snapshot.permutation_index_schedule_sha256 != expected_permutation_hash:
        raise ValueError(
            "Control snapshot permutation-index schedule differs from the frozen seed and rows."
        )
    expected_family_checksum = _sha256_names(snapshot.candidate_pair_names)
    if snapshot.candidate_family_sha256 != expected_family_checksum:
        raise ValueError("Control snapshot candidate-family checksum does not match pair order.")


def _readonly_array(value: np.ndarray, *, dtype: np.dtype[Any]) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True, order="C")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class ScoreOnlyInteractionArtifact:
    """Persistable score-only output for one contiguous permutation-draw block.

    Each completed artifact scores the complete ordered candidate family.  The
    block beginning at draw zero owns the single observed-score fit; all blocks
    contain only their assigned null-draw rows.  Pair-range shards are
    deliberately unsupported because they duplicate model fits and cannot
    represent a complete-family inferential unit.
    """

    status: str
    draw_range_start: int
    draw_range_end: int
    pair_names: tuple[str, ...]
    observed_scores: np.ndarray
    null_scores: np.ndarray
    draw_ids: np.ndarray
    control_snapshot: ControlSnapshot

    _STATUS_COMPLETED = "score_only_completed"
    _STATUS_EMPTY = "empty_candidate_family"
    _SCHEMA_VERSION = 2
    _NPZ_NAME = "score_only_interaction.npz"
    _METADATA_NAME = "score_only_interaction.json"

    def __post_init__(self) -> None:
        """Normalize immutable arrays and validate their complete identity."""
        pair_names = tuple(str(name) for name in self.pair_names)
        observed = _readonly_array(self.observed_scores, dtype=np.dtype(np.float64))
        null = _readonly_array(self.null_scores, dtype=np.dtype(np.float64))
        draw_ids = _readonly_array(self.draw_ids, dtype=np.dtype(np.int64))
        object.__setattr__(self, "pair_names", pair_names)
        object.__setattr__(self, "observed_scores", observed)
        object.__setattr__(self, "null_scores", null)
        object.__setattr__(self, "draw_ids", draw_ids)

        if self.status not in {self._STATUS_COMPLETED, self._STATUS_EMPTY}:
            raise ValueError(f"Unknown score-only artifact status: {self.status!r}")
        if self.draw_range_start < 0 or self.draw_range_end < self.draw_range_start:
            raise ValueError("Score-only artifact has an invalid draw range.")
        if pair_names != self.control_snapshot.candidate_pair_names:
            raise ValueError(
                "Score-only artifact must contain the complete ordered canonical candidate family."
            )
        draws = self.control_snapshot.permutation_draws
        if not np.array_equal(
            draw_ids, np.arange(self.draw_range_start, self.draw_range_end, dtype=np.int64)
        ):
            raise ValueError(
                "Score-only artifact draw IDs must be its canonical contiguous draw range."
            )

        if self.status == self._STATUS_EMPTY:
            if (
                self.control_snapshot.candidate_pair_names
                or self.draw_range_start != 0
                or self.draw_range_end != draws
                or pair_names
                or observed.shape != (0,)
                or null.shape != (draws, 0)
            ):
                raise ValueError(
                    "empty_candidate_family is valid only for a globally empty candidate family."
                )
            return

        width = len(pair_names)
        if width < 1:
            raise ValueError(
                "Completed score-only artifacts must cover at least one candidate pair."
            )
        expected_observed_shape = (width,) if self.draw_range_start == 0 else (0,)
        if observed.shape != expected_observed_shape:
            raise ValueError(
                "Only the draw block beginning at zero may contain complete observed scores."
            )
        if null.shape != (self.draw_range_end - self.draw_range_start, width):
            raise ValueError("Score-only artifact null-score shape does not match its draw range.")
        if not np.isfinite(observed).all() or not np.isfinite(null).all():
            raise ValueError("Score-only artifact contains non-finite scores.")

    @classmethod
    def empty_terminal(cls, control_snapshot: ControlSnapshot) -> ScoreOnlyInteractionArtifact:
        """Return the sole valid terminal artifact for an empty candidate family."""
        draws = control_snapshot.permutation_draws
        return cls(
            status=cls._STATUS_EMPTY,
            draw_range_start=0,
            draw_range_end=draws,
            pair_names=(),
            observed_scores=np.empty(0, dtype=np.float64),
            null_scores=np.empty((draws, 0), dtype=np.float64),
            draw_ids=np.arange(draws, dtype=np.int64),
            control_snapshot=control_snapshot,
        )

    @property
    def payload_sha256(self) -> str:
        """Hash draw-block metadata, names, snapshot, and score arrays."""
        digest = hashlib.sha256()
        digest.update(
            _canonical_json(
                {
                    "status": self.status,
                    "draw_range_start": self.draw_range_start,
                    "draw_range_end": self.draw_range_end,
                    "pair_names": list(self.pair_names),
                    "control_snapshot_sha256": self.control_snapshot.checksum,
                }
            )
        )
        for array in (self.observed_scores, self.null_scores, self.draw_ids):
            digest.update(_sha256_array(array).encode("ascii"))
        return digest.hexdigest()

    def write_to(self, directory: str | Path) -> dict[str, Path]:
        """Persist one self-verifying score-only artifact."""
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        npz_path = root / self._NPZ_NAME
        metadata_path = root / self._METADATA_NAME
        temporary_npz = root / "score_only_interaction.pending.npz"
        np.savez_compressed(
            temporary_npz,
            observed_scores=self.observed_scores,
            null_scores=self.null_scores,
            draw_ids=self.draw_ids,
        )
        temporary_npz.replace(npz_path)
        metadata = {
            "schema_version": self._SCHEMA_VERSION,
            "status": self.status,
            "draw_range_start": self.draw_range_start,
            "draw_range_end": self.draw_range_end,
            "pair_names": list(self.pair_names),
            "control_snapshot": self.control_snapshot.to_dict(),
            "payload_sha256": self.payload_sha256,
            "npz_sha256": _sha256_bytes(npz_path.read_bytes()),
        }
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return {"scores": npz_path, "metadata": metadata_path}

    @classmethod
    def read_from(cls, directory: str | Path) -> ScoreOnlyInteractionArtifact:
        """Load and verify a persisted score-only artifact without recovery fallback."""
        root = Path(directory)
        npz_path = root / cls._NPZ_NAME
        metadata_path = root / cls._METADATA_NAME
        if not npz_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(
                f"Missing score-only artifact files: {npz_path} and {metadata_path} are required."
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("schema_version") != cls._SCHEMA_VERSION:
            raise ValueError("Unsupported score-only artifact schema.")
        required = {
            "status",
            "draw_range_start",
            "draw_range_end",
            "pair_names",
            "control_snapshot",
            "payload_sha256",
            "npz_sha256",
        }
        missing = required.difference(metadata)
        if missing:
            raise ValueError(f"Score-only artifact metadata is incomplete: {sorted(missing)}")
        expected_npz_checksum = str(metadata.get("npz_sha256", ""))
        actual_npz_checksum = _sha256_bytes(npz_path.read_bytes())
        if actual_npz_checksum != expected_npz_checksum:
            raise ValueError("Score-only artifact NPZ checksum verification failed.")
        with np.load(npz_path, allow_pickle=False) as arrays:
            artifact = cls(
                status=str(metadata["status"]),
                draw_range_start=int(metadata["draw_range_start"]),
                draw_range_end=int(metadata["draw_range_end"]),
                pair_names=tuple(str(name) for name in metadata["pair_names"]),
                observed_scores=arrays["observed_scores"],
                null_scores=arrays["null_scores"],
                draw_ids=arrays["draw_ids"],
                control_snapshot=ControlSnapshot.from_dict(metadata["control_snapshot"]),
            )
        if artifact.payload_sha256 != metadata.get("payload_sha256"):
            raise ValueError("Score-only artifact payload checksum verification failed.")
        return artifact


def replace_contract_source(
    contract: CanonicalExecutionContract,
    source_sha256: str,
) -> CanonicalExecutionContract:
    """Return a contract with a source checksum while preserving controls."""
    return replace(contract, source_sha256=source_sha256)


__all__ = [
    "CanonicalExecutionContract",
    "ControlSnapshot",
    "ScoreOnlyInteractionArtifact",
    "build_control_snapshot",
    "canonical_execution_contract_from_specs",
    "load_canonical_execution_contract",
    "replace_contract_source",
    "verify_control_snapshot",
]
