#!/usr/bin/env python
"""Unified local entrypoint for Kestrel submit/status/collect operations."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rfm_pipeline.hpc_cascade import (  # noqa: E402
    CascadeChainError,
    inject_dependency_flag,
    parse_reduce_job_id,
)
from rfm_pipeline.hpc_workflow_config import (  # noqa: E402
    build_collect_command,
    build_remote_status_command,
    build_remote_submit_command_groups,
    load_hpc_workflow_config,
    resolved_remote_artifacts_root,
    resolved_remote_logs_root,
    ssh_target,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="Path to orchestration YAML (see configs/hpc/kestrel_publication_orchestration.yml)",
    )
    parser.add_argument(
        "--action",
        default="submit",
        choices=["submit", "status", "collect", "full"],
        help=(
            "submit: remote submissions, status: remote snapshot, "
            "collect: pull bundle, full: submit+status+collect"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Forward --dry-run to remote rfm-hpc-submit so SLURM scripts "
            "are generated on the remote host but no sbatch calls are "
            "made. SSH still executes by default (the remote command "
            "must run to produce the scripts). Pair with --generate-only "
            "for a fully local validation (no SSH at all; commands are "
            "printed only)."
        ),
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="For submit/full actions, generate scripts only (do not pass --submit).",
    )
    return parser.parse_args()


def _run_command(
    command: list[str],
    *,
    dry_run: bool,
    env: dict[str, str] | None = None,
) -> None:
    print(f">>> {' '.join(shlex.quote(part) for part in command)}")
    if dry_run:
        return
    subprocess.run(command, check=True, env=env)


def _run_remote_shell(
    ssh_dest: str,
    remote_repo_root: str,
    remote_command: str,
    *,
    dry_run: bool,
    capture_output: bool = False,
) -> str:
    """Run a shell command on the remote host via SSH.

    When ``capture_output=True``, the command's stdout is tee'd locally
    (so the user still sees it) and returned. Used by cascade chaining
    to capture each rfm-hpc-submit's RFM_HPC_SUBMIT_REDUCE_JOB_ID
    marker line.
    """
    remote_shell = f"cd {shlex.quote(remote_repo_root)} && {remote_command}"
    command = [
        "ssh",
        "-T",
        ssh_dest,
        f"bash -lc {shlex.quote(remote_shell)}",
    ]
    print(f">>> {' '.join(shlex.quote(part) for part in command)}")
    if dry_run:
        return ""
    if capture_output:
        proc = subprocess.run(command, check=True, capture_output=True, text=True)
        # Tee stdout/stderr so the operator still sees the remote output.
        if proc.stdout:
            sys.stdout.write(proc.stdout)
        if proc.stderr:
            sys.stderr.write(proc.stderr)
        return proc.stdout
    subprocess.run(command, check=True)
    return ""


def _load_yaml(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Expected YAML mapping at {path}")
    return raw


def _resolve_dataset_sync_specs(
    remote_repo_root: str,
    cpu_tier_configs: list[str],
) -> list[tuple[Path, str]]:
    dataset_type_paths = {
        "synthetic_300_sample": Path("artifacts/test_dataset_300"),
        "synthetic_full": Path("artifacts/test_dataset_3k"),
        "real_full_dataset": Path("artifacts/preprocessed_real_data_30k"),
    }
    specs: list[tuple[Path, str]] = []
    seen_remote: set[str] = set()

    def _add_spec(local_path: Path, remote_path: str) -> None:
        remote_norm = remote_path.replace("\\", "/")
        if remote_norm in seen_remote:
            return
        seen_remote.add(remote_norm)
        specs.append((local_path, remote_norm))

    def _remote_from_config_path(path_value: str) -> str:
        # Expand ${ENV} placeholders so e.g. ${SCRATCH_DIR}/... is treated
        # as an absolute path on the remote, not a relative subdir of the
        # repo root.
        expanded = os.path.expandvars(path_value)
        cfg_path = Path(expanded)
        if cfg_path.is_absolute():
            return cfg_path.as_posix()
        return f"{remote_repo_root.rstrip('/')}/{cfg_path.as_posix()}"

    for config_path in cpu_tier_configs:
        tier_cfg_path = Path(config_path)
        if not tier_cfg_path.is_absolute():
            tier_cfg_path = REPO_ROOT / tier_cfg_path
        data = _load_yaml(tier_cfg_path)

        dataset = data.get("dataset", {}) or {}
        local_dataset_root: Path | None = None
        rel_dataset_path: Path | None = None
        dataset_path = dataset.get("path")
        if dataset_path:
            dataset_candidate = Path(str(dataset_path))
            if dataset_candidate.is_absolute():
                local_dataset_root = dataset_candidate
            else:
                rel_dataset_path = dataset_candidate
                local_dataset_root = REPO_ROOT / rel_dataset_path
        else:
            rel_dataset_path = dataset_type_paths.get(str(dataset.get("type", "")))
            if rel_dataset_path is not None:
                local_dataset_root = REPO_ROOT / rel_dataset_path

        if local_dataset_root is None:
            continue

        if rel_dataset_path is not None:
            remote_dataset_root = (
                f"{remote_repo_root.rstrip('/')}/{rel_dataset_path.as_posix()}"
            )
        else:
            remote_dataset_root = (
                f"{remote_repo_root.rstrip('/')}/artifacts/{local_dataset_root.name}"
            )
        _add_spec(local_dataset_root, remote_dataset_root)

        output = data.get("output", {}) or {}
        artifact_dir_value = str(output.get("artifact_dir", "./artifacts"))
        remote_artifact_dir = _remote_from_config_path(artifact_dir_value).rstrip("/")

        for dataset_file in (
            "X.parquet",
            "Y.parquet",
            "fixed_holdout_assignments.parquet",
        ):
            _add_spec(
                local_dataset_root / dataset_file,
                f"{remote_artifact_dir}/{dataset_file}",
            )

        catalog_candidates = [
            local_dataset_root / "manuscript_feature_catalog.parquet",
            REPO_ROOT / "artifacts" / "manuscript_feature_catalog.parquet",
        ]
        catalog_source = next(
            (p for p in catalog_candidates if p.exists()), catalog_candidates[-1]
        )
        _add_spec(
            catalog_source,
            f"{remote_artifact_dir}/manuscript_feature_catalog.parquet",
        )

    local_catalog = REPO_ROOT / "artifacts" / "manuscript_feature_catalog.parquet"
    remote_catalog = (
        f"{remote_repo_root.rstrip('/')}/artifacts/manuscript_feature_catalog.parquet"
    )
    if local_catalog.exists():
        _add_spec(local_catalog, remote_catalog)
    return specs


def _remote_path_exists(ssh_dest: str, remote_path: str) -> bool:
    check_cmd = [
        "ssh",
        "-T",
        ssh_dest,
        f"bash -lc {shlex.quote(f'test -e {shlex.quote(remote_path)}')}",
    ]
    result = subprocess.run(
        check_cmd,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def _sync_required_datasets(
    *,
    ssh_dest: str,
    remote_repo_root: str,
    cpu_tier_configs: list[str],
    dry_run: bool,
) -> None:
    specs = _resolve_dataset_sync_specs(remote_repo_root, cpu_tier_configs)
    for local_path, remote_path in specs:
        if not local_path.exists():
            raise FileNotFoundError(
                f"Required local dataset artifact missing for HPC sync: {local_path}"
            )
        if not dry_run and _remote_path_exists(ssh_dest, remote_path):
            continue
        remote_parent = str(Path(remote_path).parent).replace("\\", "/")
        mkdir_cmd = [
            "ssh",
            "-T",
            ssh_dest,
            f"bash -lc {shlex.quote(f'mkdir -p {shlex.quote(remote_parent)}')}",
        ]
        _run_command(mkdir_cmd, dry_run=dry_run)
        scp_cmd = ["scp"]
        if local_path.is_dir():
            scp_cmd.append("-r")
        scp_cmd.extend([str(local_path), f"{ssh_dest}:{remote_path}"])
        _run_command(scp_cmd, dry_run=dry_run)


def _sync_remote_repo_via_git(
    *,
    ssh_dest: str,
    remote_repo_root: str,
    dry_run: bool,
) -> None:
    remote_cmd = (
        'if [ -n "$(git --no-pager status --porcelain)" ]; then '
        "echo 'error: remote repo has local changes; "
        "resolve them, then run git pull --ff-only.' >&2; "
        "git --no-pager status --short >&2; "
        "exit 2; "
        "fi && git --no-pager pull --ff-only"
    )
    _run_remote_shell(
        ssh_dest,
        remote_repo_root,
        remote_cmd,
        dry_run=dry_run,
    )


def main() -> int:
    args = parse_args()
    if Path(args.config).name == "kestrel_publication_orchestration.yml":
        raise SystemExit(
            "RETIRED: use the content-addressed G11 campaign package; "
            "the legacy publication orchestration is non-executable."
        )
    config = load_hpc_workflow_config(args.config)

    local_env = os.environ.copy()
    local_cores = str(config.execution.local_cores)
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        local_env[var] = local_cores

    ssh_dest = ssh_target(config)
    should_submit = not args.generate_only

    if args.action in {"submit", "status", "collect", "full"}:
        _sync_remote_repo_via_git(
            ssh_dest=ssh_dest,
            remote_repo_root=config.paths.remote_repo_root,
            dry_run=args.dry_run,
        )

    if args.action in {"submit", "full"}:
        if config.execution.prepare_interaction_inputs:
            cpu_tier_configs = [tier.config_path for tier in config.execution.cpu_tiers]
            _sync_required_datasets(
                ssh_dest=ssh_dest,
                remote_repo_root=config.paths.remote_repo_root,
                cpu_tier_configs=cpu_tier_configs,
                dry_run=args.dry_run,
            )
        # Cascade-aware submission: groups are (stage|None, [cmds]).
        # Prep/diagnostic carry stage=None and don't produce a SLURM job
        # id to chain. Per-stage groups: capture stdout of the last
        # command in each group to harvest RFM_HPC_SUBMIT_REDUCE_JOB_ID
        # and inject it as --depends-on-job-id on the next stage's
        # commands so SLURM enforces cascade ordering.
        groups = build_remote_submit_command_groups(
            config,
            submit=should_submit,
            dry_run=args.dry_run,
        )
        prev_reduce_job_ids: list[int] = []
        for stage_name, group_cmds in groups:
            chained_cmds = [
                inject_dependency_flag(c, prev_reduce_job_ids)
                if (stage_name is not None and prev_reduce_job_ids)
                else c
                for c in group_cmds
            ]

            captured_for_stage: list[str] = []
            for c in chained_cmds:
                # Capture stdout when this group represents a cascade
                # stage and we actually submitted (not dry-run, not
                # generate-only) — otherwise no real job id exists to
                # chain.
                capture = stage_name is not None and should_submit and not args.dry_run
                # When the user asked for --dry-run + --generate-only
                # together they want a fully local validation: no SSH,
                # no remote rfm-hpc-submit. Skip the remote shell
                # entirely (the per-stage commands are still printed by
                # build_remote_submit_command_groups in submit-mode docs).
                if args.dry_run and args.generate_only:
                    print(f">>> [skip ssh: --dry-run --generate-only] {c}")
                    continue
                stdout = _run_remote_shell(
                    ssh_dest,
                    config.paths.remote_repo_root,
                    c,
                    # Remote command carries --dry-run already when
                    # args.dry_run, so SSH itself must execute.
                    dry_run=False,
                    capture_output=capture,
                )
                if capture:
                    captured_for_stage.append(stdout)

            if stage_name is not None and captured_for_stage:
                # Each per-tier rfm-hpc-submit invocation in this stage
                # group emits its own RFM_HPC_SUBMIT_REDUCE_JOB_ID
                # marker; take the LAST marker from each invocation
                # (parse_reduce_job_id) and chain ALL ids into the next
                # stage's --depends-on-job-id (colon-list →
                # SLURM afterok). Chaining only the last tier's reduce
                # would let earlier tiers' reduces race ahead of the
                # next stage.
                tier_ids: list[int] = []
                for stdout in captured_for_stage:
                    tid = parse_reduce_job_id(stdout)
                    if tid is None:
                        raise CascadeChainError(
                            f"Cascade stage {stage_name!r} submitted but "
                            "one of its per-tier rfm-hpc-submit invocations "
                            "produced no RFM_HPC_SUBMIT_REDUCE_JOB_ID=<id> "
                            "marker. Refusing to chain the next stage with "
                            "a partial upstream id set (would let the "
                            "unmarked tier race ahead of this stage)."
                        )
                    tier_ids.append(tid)
                prev_reduce_job_ids = tier_ids
                joined = ":".join(str(i) for i in tier_ids)
                print(
                    f"[cascade] captured reduce jobs {joined} for stage "
                    f"{stage_name}; chaining next stage with "
                    f"--depends-on-job-id {joined}"
                )

    if args.action in {"status", "full"}:
        status_cmd = build_remote_status_command(config)
        _run_remote_shell(
            ssh_dest,
            config.paths.remote_repo_root,
            status_cmd,
            dry_run=args.dry_run,
        )

    if args.action in {"collect", "full"}:
        collect_cmd = build_collect_command(config, repo_root=REPO_ROOT)
        _run_command(collect_cmd, dry_run=args.dry_run, env=local_env)

    print(
        "HPC workflow action complete. "
        f"artifacts_root={resolved_remote_artifacts_root(config)} "
        f"logs_root={resolved_remote_logs_root(config)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
