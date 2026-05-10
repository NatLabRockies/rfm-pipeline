"""Typed configuration schema and parsing for unified manuscript pipeline runner.

Replaces dataset-specific script hardcoding with config-driven parameterization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DatasetConfig:
    """Dataset selection and loading parameters."""

    type: str
    """Dataset identifier: 'synthetic_300_sample', 'synthetic_full', etc."""
    path: str | None = None
    """Optional override path; if None, derived from type."""


@dataclass
class AlgorithmConfig:
    """Algorithm-level hyperparameters."""

    variance_threshold: float = 0.90
    """PCA variance threshold for component selection."""
    retained_components: int | None = None
    """If set, override variance_threshold-based selection."""


@dataclass
class RuntimeConfig:
    """Execution environment and performance tuning."""

    n_jobs: int = 1
    """Parallel workers: 1=serial, -1=all CPUs."""
    output_batch_size: int | None = None
    """Batch size for final OLS; None=no batching."""


@dataclass
class ScreeningStageConfig:
    """Empirical null screening parameters."""

    n_permutations: int = 1000
    """Number of null permutations (B+1)."""
    bh_q_threshold: float = 0.10
    """Benjamini-Hochberg FDR threshold."""


@dataclass
class InteractionStageConfig:
    """Interaction discovery parameters."""

    p_threshold: float = 0.05
    """Interaction significance threshold."""
    n_tree_estimators: int = 100
    """SHAP tree ensemble size."""
    max_tree_depth: int = 10
    """Max tree depth for ensemble."""


@dataclass
class NonlinearStageConfig:
    """Nonlinear transformation discovery parameters."""

    edf_threshold: float = 2.5
    """Empirical density function threshold."""
    transform_families: list[str] = field(default_factory=lambda: ["spline", "poly", "log"])
    """Transformation families to discover."""


@dataclass
class SparseStageConfig:
    """Sparse selection/stability parameters."""

    n_stability_subsamples: int = 100
    """Number of stability resamples."""
    subsample_fraction: float = 1.0
    """Fraction of data per resample."""
    lasso_alpha_percentile: int = 50
    """LASSO alpha selection percentile."""


@dataclass
class FinalArtifactsStageConfig:
    """Final OLS and artifact parameters."""

    bootstrap_count: int = 200
    """Number of bootstrap replicates."""
    bootstrap_alpha: float = 0.05
    """Two-sided bootstrap error level."""


@dataclass
class StagesConfig:
    """All pipeline stage configurations."""

    empirical_null_screening: ScreeningStageConfig = field(default_factory=ScreeningStageConfig)
    interaction_discovery: InteractionStageConfig = field(default_factory=InteractionStageConfig)
    nonlinear_discovery: NonlinearStageConfig = field(default_factory=NonlinearStageConfig)
    sparse_selection: SparseStageConfig = field(default_factory=SparseStageConfig)
    final_artifacts: FinalArtifactsStageConfig = field(default_factory=FinalArtifactsStageConfig)


@dataclass
class FastModeOverridesConfig:
    """Override values when fast mode is enabled."""

    n_permutations: int | None = None
    n_tree_estimators: int | None = None
    n_stability_subsamples: int | None = None
    bootstrap_count: int | None = None
    output_cap: int | None = None


@dataclass
class ValidationConfig:
    """Validation and testing modes."""

    fast_mode: bool = False
    """Enable fast-mode overrides (for CI/quick testing)."""
    fast_mode_overrides: FastModeOverridesConfig = field(default_factory=FastModeOverridesConfig)


@dataclass
class OutputConfig:
    """Output and artifact configuration."""

    artifact_dir: str = "./artifacts/"
    """Root directory for pipeline artifacts."""
    seed: int = 0
    """Random seed for reproducibility."""
    verbose: bool = True
    """Enable verbose logging."""


@dataclass
class WorkflowConfig:
    """Complete manuscript pipeline configuration.

    Can be loaded from YAML and converted to legacy case_study_config for
    integration with existing manuscript_stages.py functions.
    """

    dataset: DatasetConfig
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    stages: StagesConfig = field(default_factory=StagesConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def load_config(config_path: str | Path) -> WorkflowConfig:
    """Load and validate workflow configuration from YAML file.

    Parameters
    ----------
    config_path
        Path to configuration YAML file.

    Returns
    -------
    WorkflowConfig
        Typed configuration object.

    Raises
    ------
    FileNotFoundError
        If config file does not exist.
    ValueError
        If config is invalid or missing required fields.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        data = yaml.safe_load(f) or {}

    # Parse nested structures into typed dataclasses
    dataset_data = data.get("dataset", {})
    if not dataset_data or "type" not in dataset_data:
        raise ValueError(
            f"Config must include 'dataset.type' (e.g., 'synthetic_300_sample'). "
            f"Got: {dataset_data}"
        )
    dataset = DatasetConfig(**dataset_data)
    algorithm = AlgorithmConfig(**data.get("algorithm", {}))
    runtime = RuntimeConfig(**data.get("runtime", {}))

    # Parse stage configs
    stages_data = data.get("stages", {})
    stages = StagesConfig(
        empirical_null_screening=ScreeningStageConfig(
            **stages_data.get("empirical_null_screening", {})
        ),
        interaction_discovery=InteractionStageConfig(
            **stages_data.get("interaction_discovery", {})
        ),
        nonlinear_discovery=NonlinearStageConfig(**stages_data.get("nonlinear_discovery", {})),
        sparse_selection=SparseStageConfig(**stages_data.get("sparse_selection", {})),
        final_artifacts=FinalArtifactsStageConfig(**stages_data.get("final_artifacts", {})),
    )

    # Parse validation config
    validation_data = data.get("validation", {})
    validation = ValidationConfig(
        fast_mode=validation_data.get("fast_mode", False),
        fast_mode_overrides=FastModeOverridesConfig(
            **validation_data.get("fast_mode_overrides", {})
        ),
    )

    # Parse output config
    output = OutputConfig(**data.get("output", {}))

    return WorkflowConfig(
        dataset=dataset,
        algorithm=algorithm,
        runtime=runtime,
        stages=stages,
        validation=validation,
        output=output,
    )


def apply_fast_mode_overrides(config: WorkflowConfig) -> WorkflowConfig:
    """Apply fast-mode parameter overrides if enabled.

    Parameters
    ----------
    config
        Configuration to modify in-place.

    Returns
    -------
    WorkflowConfig
        Modified configuration (same object).
    """
    if not config.validation.fast_mode:
        return config

    ov = config.validation.fast_mode_overrides
    if ov.n_permutations is not None:
        config.stages.empirical_null_screening.n_permutations = ov.n_permutations
    if ov.n_tree_estimators is not None:
        config.stages.interaction_discovery.n_tree_estimators = ov.n_tree_estimators
    if ov.n_stability_subsamples is not None:
        config.stages.sparse_selection.n_stability_subsamples = ov.n_stability_subsamples
    if ov.bootstrap_count is not None:
        config.stages.final_artifacts.bootstrap_count = ov.bootstrap_count
    return config
