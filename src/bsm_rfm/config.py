"""Typed configuration schema and parsing for unified manuscript pipeline runner.

Replaces dataset-specific script hardcoding with config-driven parameterization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from bsm_rfm.transforms import DEFAULT_TRANSFORM_LIBRARY, TransformDef


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
    max_loaded_table_mb: float | None = None
    """Optional max memory budget (MB) for loaded X+Y+catalog+holdout tables."""
    oom_output_cap: int | None = None
    """If budget exceeded, reload outputs with this cap instead of failing."""
    use_chunked_io: bool = False
    """Deprecated compatibility toggle; prefer runtime.out_of_core.enabled."""
    chunked_io_config: dict | None = None
    """Deprecated compatibility map; prefer runtime.out_of_core.*."""
    out_of_core: OutOfCoreConfig = field(default_factory=lambda: OutOfCoreConfig())
    """Out-of-core settings used by chunked loading and spill-to-disk logic."""
    parallelism: ParallelismConfig = field(default_factory=lambda: ParallelismConfig())
    """Distributed execution settings (Dask, Ray, MPI)."""


@dataclass
class ParallelismConfig:
    """Distributed execution settings (Dask, Ray, MPI)."""

    backend: str = "joblib"
    """Execution backend: 'joblib' (default), 'dask', 'ray', 'mpi'."""
    enabled: bool = False
    """Enable distributed execution."""
    dask_workers: int | None = None
    """Number of Dask workers (if backend='dask')."""
    dask_cores_per_worker: int = 4
    """CPU cores per Dask worker."""
    dask_memory_per_worker: str = "4 GB"
    """RAM per Dask worker."""
    dask_scheduler: str = "threads"
    """Dask scheduler: 'threads', 'processes', or 'distributed'."""
    gpu_enabled: bool = False
    """Enable GPU acceleration if available."""
    gpu_type: str = "a100"
    """GPU type: 'a100', 'v100', 'h100', etc."""
    gpu_count: int = 1
    """Number of GPUs per node."""


@dataclass
class OutOfCoreConfig:
    """Out-of-core processing settings."""

    enabled: bool = False
    """Enable chunked loading and spill-aware paths."""
    chunk_size_mb: int = 512
    """Target chunk size for Parquet chunked readers."""
    max_memory_budget_mb: int = 8000
    """Spill buffer memory threshold before flushing chunks to disk."""
    temp_dir: str | None = None
    """Optional temp directory for spill files."""
    enable_spill_to_disk: bool = True
    """When true, buffer chunks to disk under memory pressure."""
    use_chunked_io: bool = False
    """Enable out-of-core processing with chunked I/O and spill-to-disk."""
    chunked_io_config: dict | None = None
    """Out-of-core config: {chunk_size_mb, max_memory_budget_mb, temp_dir, enable_spill_to_disk}."""


@dataclass
class ScreeningStageConfig:
    """Empirical null screening parameters."""

    n_permutations: int = 1000
    """Number of null permutations (B+1)."""
    bh_q_threshold: float = 0.10
    """Benjamini-Hochberg FDR threshold."""
    max_retained_terms: int | None = None
    """Optional deterministic top-K cap on retained first-order terms."""


@dataclass
class InteractionStageConfig:
    """Interaction discovery parameters."""

    p_threshold: float = 0.05
    """Interaction significance threshold."""
    n_permutations: int | None = None
    """Optional interaction null permutations (B+1); default inherits empirical stage."""
    n_tree_estimators: int = 100
    """SHAP tree ensemble size."""
    max_tree_depth: int = 10
    """Max tree depth for ensemble."""
    parallel_batch_timeout_seconds: int | None = None
    """Per-batch parallel timeout; <=0 disables timeout (useful for long HPC shards)."""
    max_shap_samples: int = 500
    """Max samples for SHAP interaction matrix computation (further capped adaptively)."""
    min_component_variance_fraction: float = 0.01
    """Skip PCA components below this fraction of maximum component variance."""
    max_active_components: int | None = None
    """Hard cap on active PCA components for SHAP; keeps highest-variance components."""


@dataclass
class NonlinearStageConfig:
    """Nonlinear transformation discovery parameters."""

    edf_threshold: float = 2.5
    """Empirical density function threshold."""
    transform_library: list[TransformDef] = field(
        default_factory=lambda: list(DEFAULT_TRANSFORM_LIBRARY)
    )
    """Algebraic transform library; each entry is a :class:`~bsm_rfm.transforms.TransformDef`
    with ``expr`` (SymPy expression in ``x``), ``label`` (column suffix), and optional ``name``.
    When not specified in config, defaults to the five standard families: quadratic (sq),
    logarithmic (log1p), inverse (inv), square-root (sqrt), and exponential (exp).
    """


@dataclass
class SparseStageConfig:
    """Sparse selection/stability parameters."""

    n_stability_subsamples: int = 100
    """Number of stability resamples."""
    subsample_fraction: float = 1.0
    """Fraction of data per resample."""
    lasso_alpha_percentile: int = 50
    """LASSO alpha selection percentile."""
    max_candidate_terms: int | None = None
    """Optional deterministic top-K candidate cap before sparse stability."""


@dataclass
class FinalArtifactsStageConfig:
    """Final OLS and artifact parameters."""

    bootstrap_count: int = 200
    """Number of bootstrap replicates."""
    bootstrap_alpha: float = 0.05
    """Two-sided bootstrap error level."""
    hc3_output_subset_mode: str = "all"
    """HC3 output selection mode: all, random_fraction, target_list, top_variance."""
    hc3_output_fraction: float | None = None
    """Optional output fraction for random_fraction mode."""
    hc3_output_names: list[str] | None = None
    """Optional explicit outputs for target_list mode."""
    hc3_output_max_outputs: int | None = None
    """Optional hard cap on outputs used in HC3 filtering."""
    hc3_output_random_seed: int = 123
    """Deterministic seed for random output subsetting."""
    hc3_output_subset_metric: str = "variance"
    """Ranking metric for principled downselection modes."""
    delta_threshold_override: float | None = None
    """Optional delta threshold override for feature pruning in final artifacts."""


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


def _nonlinear_stage_config_from_data(data: dict) -> NonlinearStageConfig:
    """Deserialize a :class:`NonlinearStageConfig` from a YAML-parsed dict.

    Handles the ``transform_library`` key as a list of dicts, each with
    ``expr``, ``label``, and optional ``name`` fields.  Ignores the legacy
    ``transform_families`` key if present.
    """
    raw = dict(data)
    library_data = raw.pop("transform_library", None)
    raw.pop("transform_families", None)  # drop legacy key
    cfg = NonlinearStageConfig(**raw)
    if library_data is not None:
        cfg.transform_library = [TransformDef.from_config(d) for d in library_data]
    return cfg


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
    runtime_data = dict(data.get("runtime", {}))
    out_of_core_data = dict(runtime_data.pop("out_of_core", {}) or {})
    legacy_use_chunked = bool(runtime_data.get("use_chunked_io", False))
    legacy_chunked_cfg = runtime_data.get("chunked_io_config", {}) or {}
    if not out_of_core_data and (legacy_use_chunked or legacy_chunked_cfg):
        out_of_core_data = {
            "enabled": legacy_use_chunked,
            "chunk_size_mb": legacy_chunked_cfg.get("chunk_size_mb", 512),
            "max_memory_budget_mb": legacy_chunked_cfg.get("max_memory_budget_mb", 8000),
            "temp_dir": legacy_chunked_cfg.get("temp_dir"),
            "enable_spill_to_disk": legacy_chunked_cfg.get("enable_spill_to_disk", True),
        }
    out_of_core = OutOfCoreConfig(**out_of_core_data)
    runtime = RuntimeConfig(**runtime_data, out_of_core=out_of_core)

    # Parse stage configs
    stages_data = data.get("stages", {})
    stages = StagesConfig(
        empirical_null_screening=ScreeningStageConfig(
            **stages_data.get("empirical_null_screening", {})
        ),
        interaction_discovery=InteractionStageConfig(
            **stages_data.get("interaction_discovery", {})
        ),
        nonlinear_discovery=_nonlinear_stage_config_from_data(
            stages_data.get("nonlinear_discovery", {})
        ),
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
