"""Tests for HPC artifact input path resolution for shard manifest building.

This module tests the resolver that finds prior-stage outputs (X.parquet,
holdout assignments, feature catalog, PCA scores, retained terms) and populates
the shard manifest's input_paths array for interaction_discovery sharding.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from rfm_pipeline.distributed.manifest import (
    resolve_interaction_discovery_shard_inputs,
)


@pytest.fixture
def temp_artifact_tree(tmp_path: Path) -> dict[str, Any]:
    """Create a minimal artifact tree with prior-stage outputs."""
    # output_conditioning outputs
    output_cond_dir = tmp_path / "output_conditioning"
    output_cond_dir.mkdir(parents=True, exist_ok=True)
    (output_cond_dir / "pca_scores.csv").write_text("dim1,dim2\n0.1,0.2\n0.3,0.4")

    # empirical_null_screening outputs
    emp_null_dir = tmp_path / "empirical_null_screen"
    emp_null_dir.mkdir(parents=True, exist_ok=True)
    (emp_null_dir / "retained_terms.csv").write_text("feature\nf1\nf2\nf3")
    (emp_null_dir / "feature_screening_statistics.csv").write_text("feature,score\nf1,0.5\nf2,0.6")

    # Assume X.parquet and holdout assignments exist in the root
    (tmp_path / "X.parquet").write_bytes(b"fake_parquet_X")
    (tmp_path / "holdout_assignments.parquet").write_bytes(b"fake_parquet_holdout")
    (tmp_path / "actual_input_feature_catalog.parquet").write_bytes(b"fake_catalog")

    return {
        "artifact_dir": tmp_path,
        "output_cond_dir": output_cond_dir,
        "emp_null_dir": emp_null_dir,
    }


class TestResolveInteractionDiscoveryShardinputs:
    """Test suite for interaction_discovery shard input resolution."""

    def test_resolve_interaction_inputs_finds_all_required_files(
        self, temp_artifact_tree: dict[str, Any]
    ) -> None:
        """resolve_interaction_discovery_shard_inputs finds pca_scores, retained_terms, X, etc."""
        artifact_dir = temp_artifact_tree["artifact_dir"]

        result = resolve_interaction_discovery_shard_inputs(artifact_dir)

        # Should resolve all required paths
        assert result is not None
        assert isinstance(result, dict)
        assert "pca_scores" in result
        assert "retained_terms" in result
        assert "X" in result
        assert "holdout_assignments" in result
        assert "feature_catalog" in result

        # Paths should exist
        assert Path(result["pca_scores"]).exists()
        assert Path(result["retained_terms"]).exists()
        assert Path(result["X"]).exists()
        assert Path(result["holdout_assignments"]).exists()
        assert Path(result["feature_catalog"]).exists()

    def test_resolve_interaction_inputs_raises_on_missing_pca_scores(
        self, temp_artifact_tree: dict[str, Any]
    ) -> None:
        """Should raise if pca_scores.csv is missing."""
        artifact_dir = temp_artifact_tree["artifact_dir"]
        (artifact_dir / "output_conditioning" / "pca_scores.csv").unlink()

        with pytest.raises(FileNotFoundError, match="pca_scores"):
            resolve_interaction_discovery_shard_inputs(artifact_dir)

    def test_resolve_interaction_inputs_raises_on_missing_retained_terms(
        self, temp_artifact_tree: dict[str, Any]
    ) -> None:
        """Should raise if retained_terms.csv is missing."""
        artifact_dir = temp_artifact_tree["artifact_dir"]
        (artifact_dir / "empirical_null_screen" / "retained_terms.csv").unlink()

        with pytest.raises(FileNotFoundError, match="retained_terms"):
            resolve_interaction_discovery_shard_inputs(artifact_dir)

    def test_resolve_interaction_inputs_raises_on_missing_x_parquet(
        self, temp_artifact_tree: dict[str, Any]
    ) -> None:
        """Should raise if X.parquet is missing."""
        artifact_dir = temp_artifact_tree["artifact_dir"]
        (artifact_dir / "X.parquet").unlink()

        with pytest.raises(FileNotFoundError, match="X.parquet"):
            resolve_interaction_discovery_shard_inputs(artifact_dir)

    def test_resolve_interaction_inputs_accepts_pathlib_paths(
        self, temp_artifact_tree: dict[str, Any]
    ) -> None:
        """Should accept both str and Path for artifact_dir."""
        artifact_dir_str = str(temp_artifact_tree["artifact_dir"])
        artifact_dir_path = temp_artifact_tree["artifact_dir"]

        # Both should work
        result_str = resolve_interaction_discovery_shard_inputs(artifact_dir_str)
        result_path = resolve_interaction_discovery_shard_inputs(artifact_dir_path)

        assert result_str is not None
        assert result_path is not None
        assert result_str.keys() == result_path.keys()

    def test_resolve_returns_absolute_paths(self, temp_artifact_tree: dict[str, Any]) -> None:
        """Returned paths should be absolute."""
        artifact_dir = temp_artifact_tree["artifact_dir"]
        result = resolve_interaction_discovery_shard_inputs(artifact_dir)

        for key, path_str in result.items():
            p = Path(path_str)
            assert p.is_absolute(), f"{key} returned relative path: {path_str}"

    def test_resolve_with_separate_dataset_path(self, tmp_path: Path) -> None:
        """dataset_path causes X/holdout/catalog to load from dataset_path, not artifact_dir."""
        artifact_dir = tmp_path / "study_artifacts"
        dataset_dir = tmp_path / "dataset"

        # stage artifacts in artifact_dir
        (artifact_dir / "output_conditioning").mkdir(parents=True)
        (artifact_dir / "empirical_null_screen").mkdir(parents=True)
        (artifact_dir / "output_conditioning" / "pca_scores.csv").write_text("dim1\n0.1")
        (artifact_dir / "empirical_null_screen" / "retained_terms.csv").write_text("feature\nf1")

        # raw data files in dataset_dir (NOT in artifact_dir)
        dataset_dir.mkdir(parents=True)
        (dataset_dir / "X.parquet").write_bytes(b"fake_X")
        (dataset_dir / "holdout_assignments.parquet").write_bytes(b"fake_holdout")
        (dataset_dir / "actual_input_feature_catalog.parquet").write_bytes(b"fake_catalog")

        result = resolve_interaction_discovery_shard_inputs(artifact_dir, dataset_path=dataset_dir)

        assert Path(result["X"]).parent == dataset_dir.resolve()
        assert Path(result["holdout_assignments"]).parent == dataset_dir.resolve()
        assert Path(result["feature_catalog"]).parent == dataset_dir.resolve()
        assert Path(result["pca_scores"]).parent == (artifact_dir / "output_conditioning").resolve()

    def test_resolve_with_dataset_path_raises_if_raw_files_missing(self, tmp_path: Path) -> None:
        """When dataset_path is provided, missing raw files raise FileNotFoundError."""
        artifact_dir = tmp_path / "study_artifacts"
        dataset_dir = tmp_path / "dataset"

        (artifact_dir / "output_conditioning").mkdir(parents=True)
        (artifact_dir / "empirical_null_screen").mkdir(parents=True)
        (artifact_dir / "output_conditioning" / "pca_scores.csv").write_text("dim1\n0.1")
        (artifact_dir / "empirical_null_screen" / "retained_terms.csv").write_text("feature\nf1")
        dataset_dir.mkdir(parents=True)
        # X.parquet intentionally absent

        with pytest.raises(FileNotFoundError, match="X"):
            resolve_interaction_discovery_shard_inputs(artifact_dir, dataset_path=dataset_dir)


class TestManifestInputPathsPopulation:
    """Test that manifest builder uses resolved inputs."""

    def test_build_manifest_with_interaction_inputs_includes_input_paths(
        self, temp_artifact_tree: dict[str, Any]
    ) -> None:
        """build_manifest should populate interaction_discovery input_paths."""
        from rfm_pipeline.distributed.manifest import build_manifest

        artifact_dir = temp_artifact_tree["artifact_dir"]
        inputs = resolve_interaction_discovery_shard_inputs(artifact_dir)

        # Convert dict to list of (key, value) tuples for input_paths
        input_paths = [str(p) for p in inputs.values()]

        shards = build_manifest(
            stage="interaction_discovery",
            input_paths=input_paths,
            output_root=str(artifact_dir / "hpc_shards"),
            n_shards=2,
        )

        assert len(shards) > 0
        # Each shard should have the resolved input paths
        for shard in shards:
            assert hasattr(shard, "input_paths") or "input_paths" in shard
            if hasattr(shard, "input_paths"):
                shard_inputs = shard.input_paths
            else:
                shard_inputs = shard.get("input_paths", [])
            assert len(shard_inputs) > 0


class TestHpcSubmitIntegration:
    """Test bsm_hpc_submit.py integration with artifact resolution."""

    def test_bsm_hpc_submit_can_resolve_inputs_from_artifact_dir(
        self, tmp_path: Path, temp_artifact_tree: dict[str, Any]
    ) -> None:
        """Auto-populate input_paths from artifact_dir in submit script."""
        artifact_dir = temp_artifact_tree["artifact_dir"]
        inputs = resolve_interaction_discovery_shard_inputs(artifact_dir)

        # Verify we got a dict with all required keys
        required_keys = {
            "pca_scores",
            "retained_terms",
            "X",
            "holdout_assignments",
            "feature_catalog",
        }
        assert set(inputs.keys()) == required_keys

    def test_manifest_with_resolved_inputs_passes_to_shards(
        self, temp_artifact_tree: dict[str, Any]
    ) -> None:
        """When input_paths are resolved and passed to build_manifest, shards inherit them."""
        from rfm_pipeline.distributed.manifest import build_manifest

        artifact_dir = temp_artifact_tree["artifact_dir"]
        inputs = resolve_interaction_discovery_shard_inputs(artifact_dir)
        input_paths = list(inputs.values())

        shards = build_manifest(
            stage="interaction_discovery",
            input_paths=input_paths,
            output_root=str(artifact_dir / "hpc_shards"),
            n_shards=3,
            expected_columns=100,
        )

        assert len(shards) == 3
        for shard in shards:
            # Each shard should have copies of all input_paths
            assert len(shard.input_paths) == len(input_paths)
            assert all(Path(p).exists() for p in shard.input_paths)
            # Shards should also have feature ranges defined
            assert shard.feature_start_idx is not None
            assert shard.feature_end_idx is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
