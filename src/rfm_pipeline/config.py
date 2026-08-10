"""Typed configuration schema and parsing for unified manuscript pipeline runner.

Replaces dataset-specific script hardcoding with config-driven parameterization.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from rfm_pipeline.transforms import DEFAULT_TRANSFORM_LIBRARY, TransformDef

_VALID_INPUT_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_categorical_inputs(decls: list[CategoricalInputDecl]) -> None:
    """Validate categorical input declarations; raise ValueError on bad entries."""
    for decl in decls:
        if not _VALID_INPUT_NAME_RE.match(decl.name):
            raise ValueError(
                f"categorical_inputs: invalid predictor name {decl.name!r}. "
                "Names must be non-empty and match [A-Za-z_][A-Za-z0-9_]*."
            )
        if decl.levels is not None:
            if not decl.levels:
                raise ValueError(
                    f"categorical_inputs: levels for {decl.name!r} must be a "
                    "non-empty list when provided."
                )
            for lvl in decl.levels:
                if not isinstance(lvl, str) or not lvl.strip():
                    raise ValueError(
                        f"categorical_inputs: each level for {decl.name!r} must be "
                        f"a non-empty string; got {lvl!r}."
                    )


@dataclass
class CategoricalInputDecl:
    """Declaration of a single categorical/block predictor input.

    Attributes
    ----------
    name:
        Column name of the predictor in the design matrix. Must match
        ``[A-Za-z_][A-Za-z0-9_]*``.
    levels:
        Optional explicit level list. When provided, must be a non-empty list
        of non-empty strings. Downstream code may use levels for dummy-coding
        or contrast encoding; the config layer only validates the structure.
    """

    name: str
    levels: list[str] | None = None


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
    """Empirical null screening parameters.

    Defaults match the validated publication run
    (``n_permutations=201`` corresponds to ``B=200`` null draws plus one
    observed statistic; ``bh_q_threshold=0.05`` is the manuscript value).
    """

    n_permutations: int = 201
    """Number of null permutations (B+1); manuscript baseline is B=200."""
    bh_q_threshold: float = 0.05
    """Benjamini-Hochberg FDR threshold; manuscript baseline is q=0.05."""
    max_retained_terms: int | None = None
    """Optional deterministic top-K cap on retained first-order terms."""


@dataclass
class InteractionStageConfig:
    """Interaction discovery parameters.

    Defaults provide the canonical pre-execution maxT resolution:
    ``n_permutations=201`` (B=200 null draws plus the observed statistic),
    ``n_tree_estimators=250``, and ``max_tree_depth=5``.
    """

    p_threshold: float = 0.05
    """Interaction significance threshold."""
    selection_method: str = "max_t"
    """Canonical global interaction selector; only exact finite-permutation maxT is supported."""
    selection_alpha: float = 0.05
    """Family-wise error target for the canonical maxT selector."""
    minimum_selection_draws: int = 199
    """Minimum null draws required by the pre-execution interaction contract."""
    n_permutations: int | None = 201
    """Interaction null permutations (B+1); canonical exact-maxT baseline is B=200."""
    n_tree_estimators: int = 250
    """SHAP tree ensemble size; manuscript baseline is 250."""
    max_tree_depth: int = 5
    """Max tree depth for ensemble; manuscript baseline is 5."""
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
    """Effective degrees-of-freedom threshold."""
    transform_library: list[TransformDef] = field(
        default_factory=lambda: list(DEFAULT_TRANSFORM_LIBRARY)
    )
    """Algebraic transform library; each entry is a :class:`~rfm_pipeline.transforms.TransformDef`
    with ``expr`` (SymPy expression in ``x``), ``label`` (column suffix), and optional ``name``.
    When not specified in config, defaults to the four manuscript families: quadratic (sq),
    logarithmic (log1p), inverse (inv), and square-root (sqrt).
    """


@dataclass
class SparseStageConfig:
    """Sparse selection/stability parameters.

    Defaults match the validated publication run: ``n_stability_subsamples=50``
    resamples at ``subsample_fraction=0.8`` (80% of training rows per
    subsample) are the manuscript baseline values.
    """

    n_stability_subsamples: int = 50
    """Number of stability resamples; manuscript baseline is 50."""
    subsample_fraction: float = 0.8
    """Fraction of data per resample; manuscript baseline is 0.80."""
    jaccard_threshold: float = 0.75
    """Minimum Jaccard overlap between subsample supports; manuscript baseline 0.75."""
    spearman_threshold: float = 0.90
    """Minimum Spearman rank correlation of coefficient magnitudes; manuscript baseline 0.90."""
    lasso_alpha_grid_size: int = 40
    """EBIC alpha grid size (count of geomspaced candidate penalties); manuscript baseline 40."""
    max_candidate_terms: int | None = None
    """Optional deterministic top-K candidate cap before sparse stability."""


@dataclass
class FinalArtifactsStageConfig:
    """Final OLS and artifact parameters.

    Defaults match the validated publication run: ``bootstrap_count=100`` replicates
    at ``bootstrap_alpha=0.05`` and ``delta_threshold_override=0.002`` for
    feature pruning are the manuscript baseline values.
    """

    bootstrap_count: int = 100
    """Number of bootstrap replicates; manuscript baseline is 100."""
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
    delta_threshold_override: float | None = 0.002
    """Delta threshold for feature pruning; manuscript baseline is 0.002."""
    pruning_error_scale_quantile: float = 0.95
    """Quantile used to scale the no-refit feature-pruning error penalty."""
    pruning_remove_count_override: int | None = None
    """Optional explicit pruning count; mutually exclusive with delta threshold override."""


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
    categorical_inputs: list[CategoricalInputDecl] = field(default_factory=list)
    """Categorical/block predictor declarations.

    Each entry names a predictor column that must be treated as a categorical
    block variable (e.g. a scenario switch, a site indicator). An optional
    ``levels`` list provides explicit level ordering for downstream encoding.

    Default is an empty list, which preserves all existing behaviour.
    Names are validated on load; unknown or malformed entries raise
    :class:`ValueError`.
    """


def _nonlinear_stage_config_from_data(data: dict) -> NonlinearStageConfig:
    """Deserialize a :class:`NonlinearStageConfig` from a YAML-parsed dict.

    Handles the ``transform_library`` key as a list of dicts, each with
    ``expr``, ``label``, and optional ``name`` fields.  Raises ``ValueError``
    on the legacy ``transform_families`` key (which was silently dropped
    in prior releases and caused user-set families to have no effect).
    """
    raw = dict(data)
    library_data = raw.pop("transform_library", None)
    if "transform_families" in raw:
        raise ValueError(
            "`stages.nonlinear_discovery.transform_families` is not a "
            "recognised configuration key. The current API uses "
            "`transform_library`, a list of "
            "`{expr, label, name}` mappings (see docs/configuration_reference.md). "
            "Remove the `transform_families` entry or convert it to "
            "`transform_library`."
        )
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
    final_artifacts_data = dict(stages_data.get("final_artifacts", {}))
    if (
        final_artifacts_data.get("delta_threshold_override") is not None
        and final_artifacts_data.get("pruning_remove_count_override") is not None
    ):
        raise ValueError(
            "stages.final_artifacts allows at most one pruning override: "
            "delta_threshold_override or pruning_remove_count_override."
        )
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
        final_artifacts=FinalArtifactsStageConfig(**final_artifacts_data),
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

    # Parse categorical input declarations
    raw_cat = data.get("categorical_inputs", []) or []
    categorical_inputs: list[CategoricalInputDecl] = []
    for entry in raw_cat:
        if isinstance(entry, str):
            categorical_inputs.append(CategoricalInputDecl(name=entry))
        elif isinstance(entry, dict):
            categorical_inputs.append(
                CategoricalInputDecl(
                    name=entry["name"],
                    levels=entry.get("levels"),
                )
            )
        else:
            raise ValueError(
                f"categorical_inputs: each entry must be a string or mapping; got {entry!r}."
            )
    _validate_categorical_inputs(categorical_inputs)

    return WorkflowConfig(
        dataset=dataset,
        algorithm=algorithm,
        runtime=runtime,
        stages=stages,
        validation=validation,
        output=output,
        categorical_inputs=categorical_inputs,
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
