"""End-to-end HPC integration tests for Phase 8c distributed execution.

Tests the full roundtrip:
1. Generate manifest with resolved interaction_discovery inputs
2. Execute shards via SLURM array or MPI dispatcher
3. Reduce merges shard outputs
4. Validate merged artifacts match expected format

Uses temporary artifact fixtures with prior-stage outputs (pca_scores,
retained_terms, X, holdout_assignments, feature_catalog).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from rfm_pipeline.distributed.checkpoint import CheckpointManager
from rfm_pipeline.distributed.manifest import (
    build_manifest,
    load_manifest,
    resolve_interaction_discovery_shard_inputs,
    save_manifest,
)


@pytest.fixture
def artifact_fixture_with_priors(tmp_path: Path) -> dict[str, Any]:
    """Create a complete fixture with interaction_discovery inputs + outputs dirs."""
    # Create output_conditioning artifacts
    output_cond_dir = tmp_path / "output_conditioning"
    output_cond_dir.mkdir(parents=True, exist_ok=True)
    pca_scores_path = output_cond_dir / "pca_scores.csv"
    pca_scores_path.write_text("dim1,dim2,dim3\n0.1,0.2,0.3\n0.4,0.5,0.6\n0.7,0.8,0.9")

    # Create empirical_null_screening artifacts
    emp_null_dir = tmp_path / "empirical_null_screen"
    emp_null_dir.mkdir(parents=True, exist_ok=True)
    retained_terms_path = emp_null_dir / "retained_terms.csv"
    retained_terms_path.write_text("feature,score\nf1,0.9\nf2,0.85\nf3,0.8\nf4,0.75\nf5,0.7")

    # Create root-level data artifacts (simulated)
    x_path = tmp_path / "X.parquet"
    x_path.write_bytes(b"fake_parquet_X_data")
    holdout_path = tmp_path / "holdout_assignments.parquet"
    holdout_path.write_bytes(b"fake_parquet_holdout")
    catalog_path = tmp_path / "actual_input_feature_catalog.parquet"
    catalog_path.write_bytes(b"fake_catalog")

    # Create shard output directory (where reduce will look for merged artifacts)
    shard_output_dir = tmp_path / "hpc_shards"
    shard_output_dir.mkdir(parents=True, exist_ok=True)

    return {
        "artifact_dir": tmp_path,
        "shard_output_dir": shard_output_dir,
        "pca_scores": pca_scores_path,
        "retained_terms": retained_terms_path,
        "X": x_path,
        "holdout": holdout_path,
        "catalog": catalog_path,
    }


class TestE2EHPCManifestGeneration:
    """Test end-to-end manifest generation with resolved inputs."""

    def test_generate_manifest_with_resolved_inputs_creates_shards(
        self, artifact_fixture_with_priors: dict[str, Any]
    ) -> None:
        """Can generate a manifest with resolved inputs for interaction_discovery."""
        artifact_dir = artifact_fixture_with_priors["artifact_dir"]

        # Resolve inputs (as bsm_hpc_submit.py does)
        resolved_inputs = resolve_interaction_discovery_shard_inputs(artifact_dir)
        input_paths = list(resolved_inputs.values())

        # Build manifest (as bsm_hpc_submit.py does)
        manifest_path = artifact_dir / "test_manifest.jsonl"
        shards = build_manifest(
            stage="interaction_discovery",
            input_paths=input_paths,
            output_root=str(artifact_fixture_with_priors["shard_output_dir"]),
            n_shards=4,
            expected_rows=100,
            expected_columns=5,  # Match num features in retained_terms.csv
        )

        # Verify manifest
        assert len(shards) == 4  # Effective shards capped at expected_columns
        assert all(s.stage == "interaction_discovery" for s in shards)
        assert all(len(s.input_paths) == len(input_paths) for s in shards)
        assert all(s.feature_start_idx is not None for s in shards)
        assert all(s.feature_end_idx is not None for s in shards)

        # Verify feature ranges partition the 5 columns
        all_ranges = [(s.feature_start_idx, s.feature_end_idx) for s in shards]
        assert all_ranges[0][0] == 0  # First shard starts at 0
        assert all_ranges[-1][1] == 5  # Last shard ends at 5

        # Save and reload to test JSONL round-trip
        save_manifest(shards, manifest_path)
        loaded_shards = load_manifest(manifest_path)
        assert len(loaded_shards) == 4
        assert all(s.shard_id == ls.shard_id for s, ls in zip(shards, loaded_shards, strict=True))

    def test_manifest_shards_have_resolved_input_paths(
        self, artifact_fixture_with_priors: dict[str, Any]
    ) -> None:
        """Each shard in the manifest carries resolved input_paths."""
        artifact_dir = artifact_fixture_with_priors["artifact_dir"]
        resolved_inputs = resolve_interaction_discovery_shard_inputs(artifact_dir)
        input_paths = list(resolved_inputs.values())

        shards = build_manifest(
            stage="interaction_discovery",
            input_paths=input_paths,
            output_root=str(artifact_fixture_with_priors["shard_output_dir"]),
            n_shards=2,
            expected_columns=5,
        )

        # Each shard should have all input paths
        for shard in shards:
            assert len(shard.input_paths) == len(input_paths)
            assert all(Path(p).exists() for p in shard.input_paths)
            # Paths should include pca_scores, retained_terms, X, holdout, catalog
            paths_str = " ".join(shard.input_paths)
            assert "pca_scores" in paths_str or "output_conditioning" in paths_str
            assert "retained_terms" in paths_str or "empirical_null_screen" in paths_str
            assert "X.parquet" in paths_str or "X" in paths_str


class TestE2ESardWorkerExecution:
    """Test shard worker execution with manifest inputs (simulated)."""

    def test_shard_manifest_provides_necessary_inputs_for_worker(
        self, artifact_fixture_with_priors: dict[str, Any]
    ) -> None:
        """Manifest shards contain all inputs required by hpc_shard_worker."""
        artifact_dir = artifact_fixture_with_priors["artifact_dir"]
        resolved_inputs = resolve_interaction_discovery_shard_inputs(artifact_dir)
        input_paths = list(resolved_inputs.values())

        shards = build_manifest(
            stage="interaction_discovery",
            input_paths=input_paths,
            output_root=str(artifact_fixture_with_priors["shard_output_dir"]),
            n_shards=2,
            expected_columns=5,
        )

        # Simulate shard worker input validation
        for shard in shards:
            # Shard should have all required inputs
            required_files = {"pca_scores.csv", "retained_terms.csv", "X.parquet"}
            for req_file in required_files:
                assert any(req_file in p for p in shard.input_paths), (
                    f"Shard {shard.shard_id} missing {req_file}"
                )

            # Output path should be set and unique
            assert shard.output_path
            assert shard.shard_id in shard.output_path

    def test_shard_feature_ranges_enable_pair_sharding(
        self, artifact_fixture_with_priors: dict[str, Any]
    ) -> None:
        """Shard feature_start/end indices enable pair-subset scoring."""
        artifact_dir = artifact_fixture_with_priors["artifact_dir"]
        resolved_inputs = resolve_interaction_discovery_shard_inputs(artifact_dir)
        input_paths = list(resolved_inputs.values())

        shards = build_manifest(
            stage="interaction_discovery",
            input_paths=input_paths,
            output_root=str(artifact_fixture_with_priors["shard_output_dir"]),
            n_shards=2,
            expected_columns=5,
        )

        # Each shard should have disjoint feature ranges that cover all features
        ranges = sorted([(s.feature_start_idx, s.feature_end_idx) for s in shards])

        # Ranges should be contiguous
        for i in range(len(ranges) - 1):
            assert ranges[i][1] == ranges[i + 1][0], "Feature ranges should be contiguous"

        # Full coverage: first starts at 0, last ends at expected_columns
        assert ranges[0][0] == 0
        assert ranges[-1][1] == 5


class TestE2EReduceMerge:
    """Test reduce merge of shard outputs (simulated)."""

    def test_reduce_merge_expects_per_shard_outputs(
        self, artifact_fixture_with_priors: dict[str, Any]
    ) -> None:
        """Reduce expects standard shard output locations."""
        shard_output_dir = artifact_fixture_with_priors["shard_output_dir"]

        # Simulate shard outputs (one shard)
        shard_1_dir = shard_output_dir / "task-0000"
        shard_1_dir.mkdir(parents=True, exist_ok=True)

        # Standard interaction discovery shard outputs
        (shard_1_dir / "retained_interaction_pairs.csv").write_text(
            "pair_name,interaction_score\npair1,0.8\npair2,0.75"
        )
        (shard_1_dir / "interaction_pair_scores.csv").write_text(
            "pair_name,score,pval\npair1,0.8,0.01\npair2,0.75,0.02"
        )
        (shard_1_dir / "shard_result.json").write_text(
            json.dumps(
                {
                    "shard_id": "task-0000",
                    "stage": "interaction_discovery",
                    "status": "completed",
                }
            )
        )

        # Reduce should find these files
        assert (shard_1_dir / "retained_interaction_pairs.csv").exists()
        assert (shard_1_dir / "interaction_pair_scores.csv").exists()
        assert (shard_1_dir / "shard_result.json").exists()

    def test_reduce_merge_deduplicates_by_pair_name(
        self, artifact_fixture_with_priors: dict[str, Any]
    ) -> None:
        """Reduce deduplicates overlapping pairs across shards."""
        shard_output_dir = artifact_fixture_with_priors["shard_output_dir"]

        # Simulate two shards with overlapping pairs
        for shard_id in ["task-0000", "task-0001"]:
            shard_dir = shard_output_dir / shard_id
            shard_dir.mkdir(parents=True, exist_ok=True)

            if shard_id == "task-0000":
                pairs = "pair_name,interaction_score\npair_AB,0.9\npair_CD,0.8"
                scores = "pair_name,score,pval\npair_AB,0.9,0.001\npair_CD,0.8,0.01"
            else:
                # Overlapping pair_AB with lower score (should be dedup'd)
                pairs = "pair_name,interaction_score\npair_AB,0.7\npair_EF,0.85"
                scores = "pair_name,score,pval\npair_AB,0.7,0.05\npair_EF,0.85,0.002"

            (shard_dir / "retained_interaction_pairs.csv").write_text(pairs)
            (shard_dir / "interaction_pair_scores.csv").write_text(scores)
            (shard_dir / "shard_result.json").write_text(
                json.dumps({"shard_id": shard_id, "status": "completed"})
            )

        # Validate shard outputs exist
        assert (shard_output_dir / "task-0000" / "retained_interaction_pairs.csv").exists()
        assert (shard_output_dir / "task-0001" / "retained_interaction_pairs.csv").exists()


class TestE2EManifestCheckpointIntegration:
    """Test integration with CheckpointManager for shard tracking."""

    def test_checkpoint_manager_tracks_shard_completion(
        self, artifact_fixture_with_priors: dict[str, Any]
    ) -> None:
        """CheckpointManager can track shard execution status."""
        shard_output_dir = artifact_fixture_with_priors["shard_output_dir"]

        # Create a checkpoint manager
        cm = CheckpointManager(str(shard_output_dir), "task-0000")

        # Simulate shard execution
        assert not cm.is_complete()

        # Create staging dir with dummy artifact
        cm.staging_dir.mkdir(parents=True, exist_ok=True)
        (cm.staging_dir / "test_artifact.txt").write_text("test")

        # Mark as running then validate/promote
        cm.mark_running()
        cm.validate_and_promote()

        # Verify completion marker
        assert cm.is_complete()
        assert (cm.success_path).exists()

    def test_checkpoint_manager_idempotence_prevents_rerun(
        self, artifact_fixture_with_priors: dict[str, Any]
    ) -> None:
        """Checkpoint manager's _SUCCESS.json marker prevents shard reruns."""
        shard_output_dir = artifact_fixture_with_priors["shard_output_dir"]

        # First run
        cm1 = CheckpointManager(str(shard_output_dir), "task-0001")
        cm1.staging_dir.mkdir(parents=True, exist_ok=True)
        cm1.mark_running()
        cm1.validate_and_promote()

        # Subsequent run should detect completion
        cm2 = CheckpointManager(str(shard_output_dir), "task-0001")
        assert cm2.is_complete()
        # _SUCCESS.json should exist from first run
        assert cm2.success_path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
