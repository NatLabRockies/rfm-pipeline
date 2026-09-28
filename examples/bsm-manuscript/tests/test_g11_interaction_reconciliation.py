"""Fail-closed tests for the G11 mixed-host interaction reconciliation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from scripts.reconcile_g11_interaction_runtime_snapshots import (
    classify_snapshot_runtime,
    compare_diagnostic_bundles,
    validate_contract_runtime_drift,
    validate_snapshot_runtime_drift,
)


def _snapshot() -> dict[str, object]:
    return {
        "schema_version": 2,
        "contract_sha256": "a" * 64,
        "implementation_source_sha256": "b" * 64,
        "dependency_lock_sha256": "c" * 64,
        "stage_seed": 123,
        "score_seed_schedule_sha256": "d" * 64,
        "permutation_index_schedule_sha256": "e" * 64,
        "candidate_pair_names": ["x:y"],
        "candidate_family_sha256": "f" * 64,
        "candidate_pair_detectors": ["tree_shap"],
        "candidate_detector_sha256": "1" * 64,
        "training_sample_ids_sha256": "2" * 64,
        "feature_matrix_sha256": "3" * 64,
        "response_matrix_sha256": "4" * 64,
        "component_names": ["PC1"],
        "n_training_rows": 2,
        "permutation_draws": 999,
    }


def test_snapshot_runtime_drift_allows_only_predeclared_nonsemantic_fields() -> None:
    reference = _snapshot()
    candidate = deepcopy(reference)
    candidate["contract_sha256"] = "5" * 64
    candidate["training_sample_ids_sha256"] = "6" * 64
    candidate["response_matrix_sha256"] = "7" * 64

    differences = validate_snapshot_runtime_drift(reference, candidate)

    assert differences == {
        "contract_sha256",
        "response_matrix_sha256",
        "training_sample_ids_sha256",
    }


def test_snapshot_runtime_drift_rejects_scientific_identity_change() -> None:
    reference = _snapshot()
    candidate = deepcopy(reference)
    candidate["candidate_family_sha256"] = "9" * 64

    with pytest.raises(ValueError, match="candidate_family_sha256"):
        validate_snapshot_runtime_drift(reference, candidate)


def test_contract_runtime_drift_requires_n_jobs_as_only_difference() -> None:
    reference = {"random_seed": 123, "permutation_count_B": 999, "n_jobs": 65}
    candidate = {"random_seed": 123, "permutation_count_B": 999, "n_jobs": 20}

    assert validate_contract_runtime_drift(reference, candidate) == (65, 20)

    candidate["random_seed"] = 456
    with pytest.raises(ValueError, match="random_seed"):
        validate_contract_runtime_drift(reference, candidate)


def test_worker_snapshot_binds_runtime_with_process_specific_string_hash() -> None:
    reference = _snapshot()
    candidate = deepcopy(reference)
    candidate["contract_sha256"] = "5" * 64
    candidate["response_matrix_sha256"] = "6" * 64
    candidate["training_sample_ids_sha256"] = "7" * 64
    worker = deepcopy(candidate)
    worker["training_sample_ids_sha256"] = "8" * 64
    worker["response_matrix_sha256"] = "9" * 64

    assert (
        classify_snapshot_runtime(
            worker, reference=reference, candidate=candidate
        )
        == "candidate"
    )

    worker["feature_matrix_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="exactly one"):
        classify_snapshot_runtime(worker, reference=reference, candidate=candidate)


def _write_bundle(
    path: Path,
    *,
    train_ids: np.ndarray,
    features: np.ndarray,
    response: np.ndarray,
) -> None:
    np.savez_compressed(
        path,
        training_sample_ids=train_ids,
        feature_matrix=features,
        response_matrix=response,
    )


def test_diagnostic_bundle_comparison_accepts_only_roundoff_scale_response_drift(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.npz"
    candidate = tmp_path / "candidate.npz"
    train_ids = np.asarray(["sample-1", "sample-2"])
    features = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    response = np.asarray([[0.25, -1.0], [0.5, 2.0]])
    _write_bundle(
        reference,
        train_ids=train_ids,
        features=features,
        response=response,
    )
    _write_bundle(
        candidate,
        train_ids=train_ids.astype("U32"),
        features=features.copy(),
        response=response + np.finfo(np.float64).eps,
    )

    report = compare_diagnostic_bundles(reference, candidate)

    assert report["training_sample_ids_equal"] is True
    assert report["feature_matrix_equal"] is True
    assert report["response_matrix_allclose"] is True
    assert report["response_matrix_max_abs_diff"] <= np.finfo(np.float64).eps


def test_diagnostic_bundle_comparison_rejects_material_response_drift(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.npz"
    candidate = tmp_path / "candidate.npz"
    _write_bundle(
        reference,
        train_ids=np.asarray(["sample-1"]),
        features=np.asarray([[1.0]]),
        response=np.asarray([[1.0]]),
    )
    _write_bundle(
        candidate,
        train_ids=np.asarray(["sample-1"]),
        features=np.asarray([[1.0]]),
        response=np.asarray([[1.0 + 1e-8]]),
    )

    with pytest.raises(ValueError, match="response matrices"):
        compare_diagnostic_bundles(reference, candidate)


def test_diagnostic_bundle_loader_rejects_object_arrays(tmp_path: Path) -> None:
    reference = tmp_path / "reference.npz"
    candidate = tmp_path / "candidate.npz"
    object_ids = np.asarray(["sample-1"], dtype=object)
    _write_bundle(
        reference,
        train_ids=object_ids,
        features=np.asarray([[1.0]]),
        response=np.asarray([[1.0]]),
    )
    _write_bundle(
        candidate,
        train_ids=object_ids,
        features=np.asarray([[1.0]]),
        response=np.asarray([[1.0]]),
    )

    with pytest.raises(ValueError, match="Object arrays"):
        compare_diagnostic_bundles(reference, candidate)
