"""Tests for config loading and validation."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import yaml

from rfm_pipeline.config import (
    DatasetConfig,
    WorkflowConfig,
    apply_fast_mode_overrides,
    load_config,
)
from rfm_pipeline.distributed.stage_loaders import config_to_legacy_case_study

if TYPE_CHECKING:
    from rfm_pipeline.config import WorkflowConfig


class TestLoadConfig:
    """Test configuration loading and parsing."""

    def test_load_minimal_config(self):
        """Test loading config with minimal required fields."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            yaml.dump(
                {
                    "dataset": {"type": "synthetic_300_sample"},
                    "output": {"artifact_dir": "./artifacts/test/"},
                },
                f,
            )
            f.flush()

            try:
                config = load_config(f.name)
                assert config.dataset.type == "synthetic_300_sample"
                assert config.output.artifact_dir == "./artifacts/test/"
                assert config.runtime.n_jobs == 1  # default
                assert config.validation.fast_mode is False
            finally:
                Path(f.name).unlink()

    def test_public_workflow_template_loads(self):
        """The user-facing staged-workflow template must match the live schema."""
        config = load_config("configs/workflow.template.yml")

        assert config.dataset.type == "custom"
        assert config.dataset.path == "/absolute/path/to/dataset"
        assert config.output.artifact_dir == "./artifacts/my-workflow-run/"

    def test_load_full_config(self):
        """Test loading config with all parameters specified."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            yaml.dump(
                {
                    "dataset": {"type": "synthetic_300_sample", "path": "./data/test/"},
                    "algorithm": {"variance_threshold": 0.95, "retained_components": 20},
                    "runtime": {
                        "n_jobs": -1,
                        "output_batch_size": 500,
                        "max_loaded_table_mb": 4096.0,
                        "oom_output_cap": 5000,
                        "out_of_core": {
                            "enabled": True,
                            "chunk_size_mb": 64,
                            "max_memory_budget_mb": 1024,
                            "enable_spill_to_disk": True,
                        },
                    },
                    "stages": {
                        "empirical_null_screening": {
                            "n_permutations": 500,
                            "bh_q_threshold": 0.05,
                            "max_retained_terms": 63,
                        },
                        "interaction_discovery": {
                            "p_threshold": 0.01,
                            "n_permutations": 120,
                            "n_tree_estimators": 200,
                            "parallel_batch_timeout_seconds": 3600,
                        },
                        "sparse_selection": {
                            "n_stability_subsamples": 50,
                            "max_candidate_terms": 1000,
                        },
                        "final_artifacts": {
                            "bootstrap_count": 100,
                            "hc3_output_subset_mode": "random_fraction",
                            "hc3_output_fraction": 0.2,
                            "hc3_output_names": ["Y1", "Y2"],
                            "hc3_output_max_outputs": 300,
                            "hc3_output_random_seed": 777,
                            "hc3_output_subset_metric": "variance",
                        },
                    },
                    "output": {"artifact_dir": "./artifacts/full/", "seed": 42},
                },
                f,
            )
            f.flush()

            try:
                config = load_config(f.name)
                assert config.dataset.type == "synthetic_300_sample"
                assert config.algorithm.retained_components == 20
                assert config.runtime.n_jobs == -1
                assert config.runtime.max_loaded_table_mb == 4096.0
                assert config.runtime.oom_output_cap == 5000
                assert config.runtime.out_of_core.enabled is True
                assert config.runtime.out_of_core.chunk_size_mb == 64
                assert config.stages.empirical_null_screening.n_permutations == 500
                assert config.stages.empirical_null_screening.max_retained_terms == 63
                assert config.stages.interaction_discovery.n_permutations == 120
                assert config.stages.interaction_discovery.parallel_batch_timeout_seconds == 3600
                assert config.stages.sparse_selection.max_candidate_terms == 1000
                assert config.stages.final_artifacts.hc3_output_subset_mode == "random_fraction"
                assert config.stages.final_artifacts.hc3_output_fraction == pytest.approx(0.2)
                assert config.stages.final_artifacts.hc3_output_names == ["Y1", "Y2"]
                assert config.stages.final_artifacts.hc3_output_max_outputs == 300
                assert config.stages.final_artifacts.hc3_output_random_seed == 777
                assert config.stages.final_artifacts.hc3_output_subset_metric == "variance"
                assert config.output.seed == 42
            finally:
                Path(f.name).unlink()

    def test_load_legacy_chunked_runtime_keys(self):
        """Test backward compatibility for runtime.use_chunked_io/chunked_io_config."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            yaml.dump(
                {
                    "dataset": {"type": "synthetic_300_sample"},
                    "runtime": {
                        "use_chunked_io": True,
                        "chunked_io_config": {
                            "chunk_size_mb": 32,
                            "max_memory_budget_mb": 512,
                            "enable_spill_to_disk": False,
                        },
                    },
                },
                f,
            )
            f.flush()

            try:
                config = load_config(f.name)
                assert config.runtime.use_chunked_io is True
                assert config.runtime.out_of_core.enabled is True
                assert config.runtime.out_of_core.chunk_size_mb == 32
                assert config.runtime.out_of_core.max_memory_budget_mb == 512
                assert config.runtime.out_of_core.enable_spill_to_disk is False
            finally:
                Path(f.name).unlink()

    def test_load_config_file_not_found(self):
        """Test error handling for missing config file."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.yml")

    def test_load_config_empty_file_raises_error(self):
        """Test loading from empty YAML file raises ValueError (missing dataset type)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write("")
            f.flush()

            try:
                with pytest.raises(ValueError, match="dataset.type"):
                    load_config(f.name)
            finally:
                Path(f.name).unlink()


class TestFastModeOverrides:
    """Test fast-mode override application."""

    def test_apply_fast_mode_overrides(self):
        """Test that fast-mode overrides are applied correctly."""
        config = WorkflowConfig(
            dataset=DatasetConfig(type="synthetic_300_sample"),
        )
        config.validation.fast_mode = True
        config.validation.fast_mode_overrides.n_permutations = 5
        config.validation.fast_mode_overrides.bootstrap_count = 20

        config = apply_fast_mode_overrides(config)
        assert config.stages.empirical_null_screening.n_permutations == 5
        assert config.stages.final_artifacts.bootstrap_count == 20

    def test_fast_mode_disabled_no_changes(self):
        """Test that no overrides are applied when fast_mode is False."""
        config = WorkflowConfig(
            dataset=DatasetConfig(type="synthetic_300_sample"),
        )
        original_perms = config.stages.empirical_null_screening.n_permutations

        config = apply_fast_mode_overrides(config)
        assert config.stages.empirical_null_screening.n_permutations == original_perms

    def test_partial_overrides(self):
        """Test applying only some overrides."""
        config = WorkflowConfig(
            dataset=DatasetConfig(type="synthetic_300_sample"),
        )
        config.validation.fast_mode = True
        config.validation.fast_mode_overrides.n_permutations = 10
        # bootstrap_count is None, so it should not be overridden

        original_bootstrap = config.stages.final_artifacts.bootstrap_count
        config = apply_fast_mode_overrides(config)

        assert config.stages.empirical_null_screening.n_permutations == 10
        assert config.stages.final_artifacts.bootstrap_count == original_bootstrap


class TestLegacyConfigMapping:
    """Test WorkflowConfig -> legacy case-study mapping."""

    def test_maps_variance_threshold_to_temporary_reduction(self):
        """variance_threshold should map to output_conditioning.temporary_reduction."""
        config = WorkflowConfig(dataset=DatasetConfig(type="synthetic_300_sample"))
        config.algorithm.variance_threshold = 0.85

        legacy = config_to_legacy_case_study(config)
        reduction = legacy["case_study"]["output_conditioning"]["temporary_reduction"]
        assert reduction["retained_variance_fraction"] == pytest.approx(0.85)

    def test_maps_retained_component_override_to_temporary_reduction(self):
        """retained_components override should map to temporary_reduction key used by stages."""
        config = WorkflowConfig(dataset=DatasetConfig(type="synthetic_300_sample"))
        config.algorithm.retained_components = 5

        legacy = config_to_legacy_case_study(config)
        reduction = legacy["case_study"]["output_conditioning"]["temporary_reduction"]
        assert reduction["retained_components"] == 5

    def test_maps_empirical_null_max_retained_terms_override(self):
        """max_retained_terms should map to case-study empirical-null section."""
        config = WorkflowConfig(dataset=DatasetConfig(type="synthetic_300_sample"))
        config.stages.empirical_null_screening.max_retained_terms = 63

        legacy = config_to_legacy_case_study(config)
        section = legacy["case_study"]["empirical_null_screen"]
        assert section["max_retained_terms"] == 63

    def test_maps_interaction_permutation_override(self):
        """Interaction n_permutations should map to interaction_discovery.permutation_count_B."""
        config = WorkflowConfig(dataset=DatasetConfig(type="synthetic_300_sample"))
        config.stages.interaction_discovery.n_permutations = 21

        legacy = config_to_legacy_case_study(config)
        section = legacy["case_study"]["interaction_discovery"]
        assert section["permutation_count_B"] == 20

    def test_maps_interaction_parallel_timeout_override(self):
        """Interaction timeout override should map to interaction_discovery timeout key."""
        config = WorkflowConfig(dataset=DatasetConfig(type="synthetic_300_sample"))
        config.stages.interaction_discovery.parallel_batch_timeout_seconds = 0

        legacy = config_to_legacy_case_study(config)
        section = legacy["case_study"]["interaction_discovery"]
        assert section["parallel_batch_timeout_seconds"] == 0

    def test_maps_final_hc3_subset_overrides(self):
        """HC3 output subset options should map to final_inferential_filter."""
        config = WorkflowConfig(dataset=DatasetConfig(type="synthetic_300_sample"))
        final = config.stages.final_artifacts
        final.hc3_output_subset_mode = "target_list"
        final.hc3_output_fraction = 0.25
        final.hc3_output_names = ["Y1", "Y2"]
        final.hc3_output_max_outputs = 75
        final.hc3_output_random_seed = 999
        final.hc3_output_subset_metric = "variance"

        legacy = config_to_legacy_case_study(config)
        section = legacy["case_study"]["final_inferential_filter"]
        assert section["output_subset_mode"] == "target_list"
        assert section["output_fraction"] == pytest.approx(0.25)
        assert section["output_names"] == ["Y1", "Y2"]
        assert section["max_outputs"] == 75
        assert section["random_seed"] == 999
        assert section["subset_metric"] == "variance"

    def test_maps_runtime_parallelism_to_interaction_discovery(self):
        """runtime.parallelism should map to interaction_discovery distributed keys."""
        config = WorkflowConfig(dataset=DatasetConfig(type="synthetic_300_sample"))
        config.runtime.parallelism.enabled = True
        config.runtime.parallelism.backend = "dask"
        config.runtime.parallelism.dask_workers = 7
        config.runtime.parallelism.dask_cores_per_worker = 3
        config.runtime.parallelism.dask_memory_per_worker = "6 GB"

        legacy = config_to_legacy_case_study(config)
        section = legacy["case_study"]["interaction_discovery"]
        assert section["parallel_backend"] == "dask"
        assert section["dask_workers"] == 7
        assert section["dask_cores_per_worker"] == 3
        assert section["dask_memory_per_worker"] == "6 GB"

    def test_validation_full_dataset_final_cost_04_bakes_in_pruning_override(self):
        """The full-cost rung config should pin the manuscript pruning threshold explicitly."""
        config_path = Path("configs/validation_full_dataset_final_cost_04.yml")
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        assert data["stages"]["final_artifacts"]["delta_threshold_override"] == pytest.approx(0.002)
