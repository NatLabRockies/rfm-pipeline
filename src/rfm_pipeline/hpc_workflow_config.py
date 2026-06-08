"""Config and command builders for user-facing HPC workflow orchestration."""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path

import yaml

_VALID_PULLBACK_MODES = {"manifest_only", "reporting_bundle", "full", "study_package"}
_VALID_STAGES = {
    "output_conditioning",
    "empirical_null_screening",
    "interaction_discovery",
    "nonlinear_discovery",
    "sparse_selection",
    "final_manuscript_artifacts",
}


@dataclass
class HpcClusterConfig:
    """Cluster identity details used for SSH execution and accounting context."""

    host: str = ""  # CONFIGURE: set to your HPC login node hostname
    user: str | None = None
    account: str = ""


@dataclass
class CpuTierConfig:
    """One CPU scaling tier to submit on HPC."""

    nodes: int
    config_path: str


def _default_cpu_tiers() -> list[CpuTierConfig]:
    return [
        CpuTierConfig(nodes=2, config_path="configs/hpc/kestrel_cpu_scale_2.yml"),
        CpuTierConfig(nodes=10, config_path="configs/hpc/kestrel_cpu_scale_10.yml"),
        CpuTierConfig(nodes=1000, config_path="configs/hpc/kestrel_cpu_scale_1000.yml"),
    ]


@dataclass
class GpuWorkflowConfig:
    """Optional GPU submission controls."""

    enabled: bool = False
    config_path: str = "configs/hpc/kestrel_gpu_h100.yml"
    n_shards: int = 10


@dataclass
class HpcPathConfig:
    """Local/remote filesystem locations used by orchestration actions."""

    remote_repo_root: str = ""
    remote_artifacts_root: str = "/scratch/${USER}/rfm-pipeline/artifacts"
    remote_logs_root: str = "/scratch/${USER}/rfm-pipeline"
    remote_snapshot_root: str = "/scratch/${USER}/rfm-pipeline/snapshots"
    local_bundle_dir: str = "./artifacts/kestrel_collected_bundles"
    cpu_suite_output_root: str | None = None
    # Orchestration helper scripts. The rfm-pipeline package itself is
    # generic and does not ship cluster-specific bash drivers; the
    # consumer (e.g. bsm-public-rf) provides them at the paths below.
    # Override these in the orchestration YAML if your scripts live
    # elsewhere. ``remote_status_script`` is invoked on the HPC head
    # node via ssh; ``local_collect_script`` runs locally and pulls
    # artifacts via scp/rsync.
    remote_status_script: str = "scripts/kestrel/status_all_tests.sh"
    local_collect_script: str = "scripts/kestrel/pull_hpc_artifacts_bundle.sh"


@dataclass
class HpcExecutionConfig:
    """Execution scale and stage controls.

    Either ``stage`` (single stage, legacy) or ``stages`` (ordered list of
    stages for a full cascade) may be set. When ``stages`` is non-empty it
    takes precedence and ``build_remote_submit_commands`` emits a
    ``rfm-hpc-submit`` invocation per stage so the entire pipeline runs end
    to end on the cluster. The single ``stage`` form is retained for
    diagnostic / single-stage benchmark runs.
    """

    stage: str = "interaction_discovery"
    stages: list[str] = field(default_factory=list)
    local_cores: int = 1
    run_diagnostic: bool = True
    prepare_interaction_inputs: bool = False
    prepare_full_pipeline_artifacts: bool = False
    cpu_tiers: list[CpuTierConfig] = field(default_factory=_default_cpu_tiers)

    def effective_stages(self) -> list[str]:
        """Return the ordered stage list this execution will submit.

        Returns ``self.stages`` when set (preserving order), else a
        single-element list containing ``self.stage``.
        """
        if self.stages:
            return list(self.stages)
        return [self.stage]


@dataclass
class PullbackPolicyConfig:
    """Controls what remains locally after pullback."""

    mode: str = "reporting_bundle"
    keep_remote_snapshot: bool = False


@dataclass
class HpcWorkflowConfig:
    """Top-level HPC orchestration config."""

    cluster: HpcClusterConfig = field(default_factory=HpcClusterConfig)
    paths: HpcPathConfig = field(default_factory=HpcPathConfig)
    execution: HpcExecutionConfig = field(default_factory=HpcExecutionConfig)
    gpu: GpuWorkflowConfig = field(default_factory=GpuWorkflowConfig)
    pullback: PullbackPolicyConfig = field(default_factory=PullbackPolicyConfig)

    def validate(self) -> list[str]:
        """Return validation errors for orchestration settings."""
        errors: list[str] = []
        remote_user = _resolve_remote_user(self)
        if not self.cluster.host.strip():
            errors.append("cluster.host must be non-empty")
        if self.execution.stages:
            # Cascade mode: validate each stage in the list; the singular
            # `execution.stage` is ignored (effective_stages prefers stages).
            for st in self.execution.stages:
                if st not in _VALID_STAGES:
                    errors.append(
                        f"execution.stages contains invalid stage {st!r}; "
                        "must be one of: " + ", ".join(sorted(_VALID_STAGES))
                    )
        else:
            # Single-stage mode: validate the singular field.
            if self.execution.stage not in _VALID_STAGES:
                errors.append("execution.stage must be one of: " + ", ".join(sorted(_VALID_STAGES)))
        if self.execution.local_cores < 1:
            errors.append("execution.local_cores must be >= 1")
        if not self.execution.cpu_tiers:
            errors.append("execution.cpu_tiers must contain at least one tier")
        for tier in self.execution.cpu_tiers:
            if tier.nodes < 1:
                errors.append("execution.cpu_tiers[].nodes must be >= 1")
            if not tier.config_path.strip():
                errors.append("execution.cpu_tiers[].config_path must be non-empty")
        if self.gpu.enabled and self.gpu.n_shards < 1:
            errors.append("gpu.n_shards must be >= 1 when gpu.enabled=true")
        if self.pullback.mode not in _VALID_PULLBACK_MODES:
            errors.append(
                "pullback.mode must be one of: " + ", ".join(sorted(_VALID_PULLBACK_MODES))
            )
        for label, value in (
            ("paths.remote_artifacts_root", resolved_remote_artifacts_root(self)),
            ("paths.remote_logs_root", resolved_remote_logs_root(self)),
            ("paths.remote_snapshot_root", resolved_remote_snapshot_root(self)),
            ("paths.cpu_suite_output_root", resolved_cpu_suite_output_root(self)),
        ):
            if _is_home_scoped(value, remote_user):
                errors.append(f"{label} cannot point under /home/{remote_user}: {value}")
        return errors


def _shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def _resolve_remote_user(config: HpcWorkflowConfig) -> str:
    if config.cluster.user and config.cluster.user.strip():
        return os.path.expandvars(config.cluster.user.strip())
    return os.environ.get("USER", "").strip()


def _expand_user_token(path_value: str, remote_user: str) -> str:
    if "${USER}" not in path_value:
        return path_value
    return path_value.replace("${USER}", remote_user)


def _is_home_scoped(path_value: str, remote_user: str) -> bool:
    if not remote_user:
        return path_value.startswith("/home/")
    return path_value.startswith(f"/home/{remote_user}/")


def resolved_remote_artifacts_root(config: HpcWorkflowConfig) -> str:
    """Resolve remote artifacts root with `${USER}` token expansion."""
    return _expand_user_token(config.paths.remote_artifacts_root, _resolve_remote_user(config))


def resolved_remote_logs_root(config: HpcWorkflowConfig) -> str:
    """Resolve remote logs root with `${USER}` token expansion."""
    return _expand_user_token(config.paths.remote_logs_root, _resolve_remote_user(config))


def resolved_remote_snapshot_root(config: HpcWorkflowConfig) -> str:
    """Resolve remote snapshot root with `${USER}` token expansion."""
    return _expand_user_token(config.paths.remote_snapshot_root, _resolve_remote_user(config))


def resolved_cpu_suite_output_root(config: HpcWorkflowConfig) -> str:
    """Resolve remote CPU-suite output root from explicit or derived path."""
    if config.paths.cpu_suite_output_root:
        return _expand_user_token(config.paths.cpu_suite_output_root, _resolve_remote_user(config))
    return f"{resolved_remote_artifacts_root(config)}/kestrel_cpu_scaling_suite"


def ssh_target(config: HpcWorkflowConfig) -> str:
    """Return SSH destination string (`user@host` or `host`)."""
    user = _resolve_remote_user(config)
    if user:
        return f"{user}@{config.cluster.host}"
    return config.cluster.host


def build_remote_submit_commands(
    config: HpcWorkflowConfig,
    *,
    submit: bool,
    dry_run: bool,
) -> list[str]:
    """Build remote submit/generate commands for CPU tiers and optional GPU."""
    suite_root = resolved_cpu_suite_output_root(config)
    common_flags: list[str] = []
    if submit:
        common_flags.append("--submit")
    if dry_run:
        common_flags.append("--dry-run")

    commands: list[str] = []
    stages_to_submit = config.execution.effective_stages()
    if "interaction_discovery" in stages_to_submit:
        prep_configs: list[str] = []
        seen_configs: set[str] = set()
        for tier in config.execution.cpu_tiers:
            if tier.config_path in seen_configs:
                continue
            seen_configs.add(tier.config_path)
            prep_configs.append(tier.config_path)
        if config.execution.prepare_full_pipeline_artifacts:
            for prep_cfg in prep_configs:
                cmd = [
                    "pixi",
                    "run",
                    "python",
                    "tools/run_manuscript_pipeline.py",
                    prep_cfg,
                    "--start-stage",
                    "output_conditioning",
                    "--stop-stage",
                    "final_manuscript_artifacts",
                ]
                commands.append(_shell_join(cmd))
        elif config.execution.prepare_interaction_inputs:
            for prep_cfg in prep_configs:
                cmd = [
                    "pixi",
                    "run",
                    "python",
                    "tools/run_manuscript_pipeline.py",
                    prep_cfg,
                    "--start-stage",
                    "output_conditioning",
                    "--stop-stage",
                    "empirical_null_screen",
                ]
                commands.append(_shell_join(cmd))

    first_tier_cfg = config.execution.cpu_tiers[0].config_path
    if config.execution.run_diagnostic:
        cmd = [
            "pixi",
            "run",
            "rfm-hpc-submit",
            "--config",
            first_tier_cfg,
            "--diagnostic-only",
            "--output-dir",
            f"{suite_root}/diagnostic/hpc_scripts",
            *common_flags,
        ]
        commands.append(_shell_join(cmd))

    for stage_name in stages_to_submit:
        # Per-stage output dir suffix keeps SLURM scripts and shard outputs
        # from different stages from clobbering each other.
        stage_suffix = f"_{stage_name}" if len(stages_to_submit) > 1 else ""
        for tier in config.execution.cpu_tiers:
            cmd = [
                "pixi",
                "run",
                "rfm-hpc-submit",
                "--config",
                tier.config_path,
                "--stage",
                stage_name,
                "--n-shards",
                str(tier.nodes),
                "--output-dir",
                f"{suite_root}/cpu_nodes_{tier.nodes}{stage_suffix}/hpc_scripts",
                *common_flags,
            ]
            commands.append(_shell_join(cmd))

        if config.gpu.enabled:
            cmd = [
                "pixi",
                "run",
                "rfm-hpc-submit",
                "--config",
                config.gpu.config_path,
                "--stage",
                stage_name,
                "--n-shards",
                str(config.gpu.n_shards),
                "--output-dir",
                f"{resolved_remote_artifacts_root(config)}/kestrel_gpu_h100_run{stage_suffix}/hpc_scripts",
                *common_flags,
            ]
            commands.append(_shell_join(cmd))

    return commands


def build_remote_submit_command_groups(
    config: HpcWorkflowConfig,
    *,
    submit: bool,
    dry_run: bool,
) -> list[tuple[str | None, list[str]]]:
    """Return remote submit commands grouped by cascade stage.

    Each element is ``(stage_name | None, [cmds])``:

    - ``stage_name=None``: prep + diagnostic groups (no SLURM job ID to
      capture for dependency chaining)
    - ``stage_name=<stage>``: all per-tier (and optional GPU)
      ``rfm-hpc-submit`` commands for that stage, in submission order

    Orchestrators chain cascade stages by capturing
    ``RFM_HPC_SUBMIT_REDUCE_JOB_ID=<id>`` from the last command of each
    stage group and passing ``--depends-on-job-id <id>`` to the next
    stage's commands. The single flat ``build_remote_submit_commands``
    keeps backward compatibility for callers that don't chain.
    """
    suite_root = resolved_cpu_suite_output_root(config)
    common_flags: list[str] = []
    if submit:
        common_flags.append("--submit")
    if dry_run:
        common_flags.append("--dry-run")

    groups: list[tuple[str | None, list[str]]] = []
    stages_to_submit = config.execution.effective_stages()

    # Prep group (None marker): tied to interaction_discovery presence
    prep_cmds: list[str] = []
    if "interaction_discovery" in stages_to_submit:
        prep_configs: list[str] = []
        seen_configs: set[str] = set()
        for tier in config.execution.cpu_tiers:
            if tier.config_path in seen_configs:
                continue
            seen_configs.add(tier.config_path)
            prep_configs.append(tier.config_path)
        if config.execution.prepare_full_pipeline_artifacts:
            for prep_cfg in prep_configs:
                prep_cmds.append(
                    _shell_join(
                        [
                            "pixi",
                            "run",
                            "python",
                            "tools/run_manuscript_pipeline.py",
                            prep_cfg,
                            "--start-stage",
                            "output_conditioning",
                            "--stop-stage",
                            "final_manuscript_artifacts",
                        ]
                    )
                )
        elif config.execution.prepare_interaction_inputs:
            for prep_cfg in prep_configs:
                prep_cmds.append(
                    _shell_join(
                        [
                            "pixi",
                            "run",
                            "python",
                            "tools/run_manuscript_pipeline.py",
                            prep_cfg,
                            "--start-stage",
                            "output_conditioning",
                            "--stop-stage",
                            "empirical_null_screen",
                        ]
                    )
                )
    if prep_cmds:
        groups.append((None, prep_cmds))

    # Diagnostic group (None marker)
    if config.execution.run_diagnostic:
        first_tier_cfg = config.execution.cpu_tiers[0].config_path
        groups.append(
            (
                None,
                [
                    _shell_join(
                        [
                            "pixi",
                            "run",
                            "rfm-hpc-submit",
                            "--config",
                            first_tier_cfg,
                            "--diagnostic-only",
                            "--output-dir",
                            f"{suite_root}/diagnostic/hpc_scripts",
                            *common_flags,
                        ]
                    )
                ],
            )
        )

    # Per-stage groups
    for stage_name in stages_to_submit:
        stage_suffix = f"_{stage_name}" if len(stages_to_submit) > 1 else ""
        stage_cmds: list[str] = []
        for tier in config.execution.cpu_tiers:
            stage_cmds.append(
                _shell_join(
                    [
                        "pixi",
                        "run",
                        "rfm-hpc-submit",
                        "--config",
                        tier.config_path,
                        "--stage",
                        stage_name,
                        "--n-shards",
                        str(tier.nodes),
                        "--output-dir",
                        f"{suite_root}/cpu_nodes_{tier.nodes}{stage_suffix}/hpc_scripts",
                        *common_flags,
                    ]
                )
            )
        if config.gpu.enabled:
            stage_cmds.append(
                _shell_join(
                    [
                        "pixi",
                        "run",
                        "rfm-hpc-submit",
                        "--config",
                        config.gpu.config_path,
                        "--stage",
                        stage_name,
                        "--n-shards",
                        str(config.gpu.n_shards),
                        "--output-dir",
                        f"{resolved_remote_artifacts_root(config)}/kestrel_gpu_h100_run{stage_suffix}/hpc_scripts",
                        *common_flags,
                    ]
                )
            )
        groups.append((stage_name, stage_cmds))

    return groups


def build_remote_status_command(config: HpcWorkflowConfig) -> str:
    """Build one-shot remote status command using configured artifact/log roots."""
    cpu_tiers: list[str] = []
    seen_tiers: set[int] = set()
    tier_shard_dirs: list[str] = []

    for tier in config.execution.cpu_tiers:
        if tier.nodes in seen_tiers:
            continue
        seen_tiers.add(tier.nodes)
        cpu_tiers.append(str(tier.nodes))

        # Resolve actual shard output dir(s) from the tier's workflow config so the
        # status script checks the right path (not just the ARTIFACTS_ROOT pattern).
        try:
            from rfm_pipeline.config import load_config as _load_wf

            wf = _load_wf(tier.config_path)
            artifact_dir: str = wf.output.artifact_dir
            # Relative paths are resolved against the remote repo root
            if not os.path.isabs(artifact_dir):
                artifact_dir = (
                    config.paths.remote_repo_root.rstrip("/") + "/" + artifact_dir.lstrip("./")
                )
            # Emit one shard-dir entry per cascade stage so the status
            # script tallies per-stage progress correctly. Stage-suffixed
            # hpc_shards_<stage> matches hpc_submit.py's per-stage layout.
            for st in config.execution.effective_stages():
                shard_dir = artifact_dir.rstrip("/") + f"/hpc_shards_{st}"
                tier_shard_dirs.append(f"{tier.nodes}:{shard_dir}:{tier.nodes}")
        except Exception:
            pass  # status script falls back to derived path

    cpu_tiers_csv = ",".join(cpu_tiers) if cpu_tiers else "2,10,1000"
    include_gpu = "1" if config.gpu.enabled else "0"
    gpu_shards = str(config.gpu.n_shards)
    tier_shard_dirs_str = ",".join(tier_shard_dirs)

    return (
        f"ARTIFACTS_ROOT={shlex.quote(resolved_remote_artifacts_root(config))} "
        f"LOGS_ROOT={shlex.quote(resolved_remote_logs_root(config))} "
        f"STATUS_CPU_TIERS={shlex.quote(cpu_tiers_csv)} "
        f"STATUS_INCLUDE_GPU={shlex.quote(include_gpu)} "
        f"STATUS_GPU_SHARDS={shlex.quote(gpu_shards)} "
        f"STATUS_TIER_SHARD_DIRS={shlex.quote(tier_shard_dirs_str)} "
        f"bash {shlex.quote(config.paths.remote_status_script)}"
    )


def build_collect_command(config: HpcWorkflowConfig, *, repo_root: Path) -> list[str]:
    """Build local artifact pull command from config.

    Raises ``FileNotFoundError`` when ``config.paths.local_collect_script``
    (resolved against ``repo_root`` when relative) does not exist on
    the local filesystem — the orchestration package is generic and
    expects the consumer repo to ship its own kestrel collect script.
    """
    cpu_tier_specs = ",".join(
        f"{tier.nodes}={tier.config_path}" for tier in config.execution.cpu_tiers
    )
    include_gpu = "1" if config.gpu.enabled else "0"
    raw_script = config.paths.local_collect_script
    script_path = Path(raw_script)
    if not script_path.is_absolute():
        script_path = repo_root / script_path
    if not script_path.exists():
        raise FileNotFoundError(
            f"HPC collect script not found: {script_path}. The rfm-pipeline "
            "package does not ship cluster-specific bash drivers; set "
            "paths.local_collect_script in the orchestration YAML to a "
            "script that exists in this checkout (see the bsm-public-rf "
            "repository for a reference implementation)."
        )
    cmd = [
        "bash",
        str(script_path),
        "--hpc-host",
        ssh_target(config),
        "--hpc-repo-root",
        config.paths.remote_repo_root,
        "--hpc-artifacts-root",
        resolved_remote_artifacts_root(config),
        "--remote-snapshot-root",
        resolved_remote_snapshot_root(config),
        "--local-out-dir",
        config.paths.local_bundle_dir,
        "--pullback-mode",
        config.pullback.mode,
        "--cpu-tier-specs",
        cpu_tier_specs,
        "--include-gpu",
        include_gpu,
        "--gpu-config",
        config.gpu.config_path,
    ]
    if config.pullback.keep_remote_snapshot:
        cmd.append("--keep-remote")
    return cmd


def load_hpc_workflow_config(path: str | Path) -> HpcWorkflowConfig:
    """Load orchestration config from YAML.

    YAML can either be the config object directly or nested under `hpc_workflow`.
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data = raw.get("hpc_workflow", raw)

    cluster_data = dict(data.get("cluster", {}) or {})
    paths_data = dict(data.get("paths", {}) or {})
    execution_data = dict(data.get("execution", {}) or {})
    gpu_data = dict(data.get("gpu", {}) or {})
    pullback_data = dict(data.get("pullback", {}) or {})

    cpu_tiers_raw = execution_data.pop("cpu_tiers", None)
    if cpu_tiers_raw is None:
        cpu_tiers = _default_cpu_tiers()
    else:
        cpu_tiers = [CpuTierConfig(**tier) for tier in cpu_tiers_raw]

    config = HpcWorkflowConfig(
        cluster=HpcClusterConfig(**cluster_data),
        paths=HpcPathConfig(**paths_data),
        execution=HpcExecutionConfig(**execution_data, cpu_tiers=cpu_tiers),
        gpu=GpuWorkflowConfig(**gpu_data),
        pullback=PullbackPolicyConfig(**pullback_data),
    )
    errors = config.validate()
    if errors:
        joined = "; ".join(errors)
        raise ValueError(f"Invalid HPC workflow config: {joined}")
    return config
