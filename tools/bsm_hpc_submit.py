"""bsm-hpc-submit — Generate and optionally submit SLURM array jobs for the BSM pipeline.

Reads a workflow config (with an embedded 'distributed' section) and produces:
  - A JSONL shard manifest
  - sbatch scripts for the array stage and reduce job
  - A combined submit_all.sh convenience script

Usage::

    # Generate scripts only (safe to run locally)
    pixi run bsm-hpc-submit --config configs/hpc/kestrel_30k.yml --stage interaction_discovery

    # Generate and submit (requires sbatch on PATH)
    pixi run bsm-hpc-submit --config configs/hpc/kestrel_30k.yml \\
        --stage interaction_discovery --submit

    # Dry run (show sbatch commands, don't execute)
    pixi run bsm-hpc-submit --config configs/hpc/kestrel_30k.yml \\
        --stage interaction_discovery --submit --dry-run

    # Smoke test on debug partition first
    pixi run bsm-hpc-submit --config configs/hpc/kestrel_30k.yml --diagnostic-only --submit

Output directory defaults to <artifact_dir>/hpc_scripts/.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("bsm.hpc_submit")


def _find_incomplete_task_ids(output_root: Path, n_shards: int) -> list[int]:
    """Return 0-based task IDs whose ``_SUCCESS.json`` marker is absent."""
    return [
        i for i in range(n_shards) if not (output_root / f"task-{i:04d}" / "_SUCCESS.json").exists()
    ]


def _reconcile_manifest_from_fs(shards: list, output_root: Path) -> tuple[list, int]:
    """Update manifest statuses from ``_SUCCESS.json`` markers on disk.

    Returns the updated shard list and the number of records that were
    changed from a non-completed status to *completed*.
    """
    reconciled = 0
    for shard in shards:
        if (output_root / shard.shard_id / "_SUCCESS.json").exists():
            if shard.status != "completed":
                shard.status = "completed"
                reconciled += 1
    return shards, reconciled


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate and optionally submit BSM HPC SLURM scripts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--config",
        required=True,
        help="Path to workflow config YAML (must have a 'distributed' section)",
    )
    p.add_argument(
        "--stage",
        default="interaction_discovery",
        choices=[
            "output_conditioning",
            "empirical_null_screening",
            "interaction_discovery",
            "nonlinear_discovery",
            "sparse_selection",
            "final_manuscript_artifacts",
        ],
        help="Pipeline stage to shard (default: interaction_discovery)",
    )
    p.add_argument(
        "--n-shards",
        type=int,
        default=None,
        help="Number of shards to create. Defaults to slurm.max_concurrent_array_tasks.",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Directory for generated scripts and manifest (default: <artifact_dir>/hpc_scripts)",
    )
    p.add_argument(
        "--submit",
        action="store_true",
        help="Submit generated scripts via sbatch after writing them",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="With --submit: print sbatch commands but do not execute",
    )
    p.add_argument(
        "--diagnostic-only",
        action="store_true",
        help="Generate and optionally submit only the diagnostic smoke-test script",
    )
    p.add_argument(
        "--reduce-walltime",
        default="02:00:00",
        help="Walltime for reduce job (default: 02:00:00)",
    )
    p.add_argument(
        "--reduce-memory-gb",
        type=int,
        default=32,
        help="Memory (GB) for reduce job (default: 32)",
    )
    p.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild manifest from scratch even if one already exists "
        "(ignores any previously-completed shard state)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Load configs
    from bsm_rfm.config import load_config
    from bsm_rfm.distributed.config_distributed import load_distributed_config
    from bsm_rfm.distributed.manifest import (
        load_manifest,
        save_manifest,
    )
    from bsm_rfm.distributed.slurm_array_runner import SlurmArrayRunner

    # Load workflow config for artifact_dir
    workflow = load_config(args.config)

    # Load distributed config (from the same file's 'distributed' section)
    dist_cfg = load_distributed_config(args.config)

    if not dist_cfg.enabled:
        logger.warning(
            "distributed.enabled is false in config — generating scripts anyway, but they "
            "are not needed for local single-node execution"
        )

    # Determine output directory
    artifact_dir = Path(workflow.output.artifact_dir)
    script_dir = Path(args.output_dir) if args.output_dir else artifact_dir / "hpc_scripts"
    script_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = script_dir / "manifest.jsonl"
    output_root = artifact_dir / "hpc_shards"

    # Build shard manifest
    n_shards = args.n_shards
    if not n_shards:
        # For stages with data-driven cardinality, estimate from prior stage outputs
        dynamic_stages = (
            "interaction_discovery",
            "nonlinear_discovery",
            "sparse_selection",
            "final_manuscript_artifacts",
        )
        if args.stage in dynamic_stages:
            n_shards = _estimate_stage_work_items(args.stage, workflow, artifact_dir)
            if n_shards:
                logger.info(
                    "[hpc-submit] auto-estimated n_shards for %s: %d (from prior outputs)",
                    args.stage,
                    n_shards,
                )
        if not n_shards:
            n_shards = dist_cfg.slurm.max_concurrent_array_tasks
    logger.info(
        "[hpc-submit] stage=%s n_shards=%d run_id=%s",
        args.stage,
        n_shards,
        dist_cfg.run_id,
    )

    incomplete_ids: list[int] | None = None  # None → diagnostic-only path

    if not args.diagnostic_only:
        if manifest_path.exists() and not args.force_rebuild:
            # Reuse existing manifest: reconcile statuses from filesystem
            existing = load_manifest(manifest_path)
            if len(existing) == n_shards:
                shards, n_reconciled = _reconcile_manifest_from_fs(existing, output_root)
                if n_reconciled:
                    save_manifest(shards, manifest_path)
                    logger.info(
                        "[hpc-submit] reconciled %d stale shard(s) from _SUCCESS.json markers",
                        n_reconciled,
                    )
                logger.info("[hpc-submit] loaded existing manifest (%d shards)", n_shards)
            else:
                logger.warning(
                    "[hpc-submit] manifest has %d shards but config requests %d — rebuilding",
                    len(existing),
                    n_shards,
                )
                shards = _build_fresh_manifest(args, artifact_dir, n_shards, workflow)
                save_manifest(shards, manifest_path)
                logger.info("[hpc-submit] wrote fresh manifest (%d shards)", n_shards)
        else:
            shards = _build_fresh_manifest(args, artifact_dir, n_shards, workflow)
            save_manifest(shards, manifest_path)
            logger.info("[hpc-submit] wrote manifest with %d shards to %s", n_shards, manifest_path)

        # Determine which task IDs still need to run
        incomplete_ids = _find_incomplete_task_ids(output_root, n_shards)
        n_complete = n_shards - len(incomplete_ids)
        logger.info(
            "[hpc-submit] %d / %d shards already complete, %d still needed",
            n_complete,
            n_shards,
            len(incomplete_ids),
        )
        if not incomplete_ids:
            logger.info("[hpc-submit] all shards complete — skipping array job, reduce only")

    # Build runner and generate scripts
    runner = SlurmArrayRunner(
        config=dist_cfg,
        manifest_path=manifest_path,
        output_root=str(output_root),
        repo_root=Path.cwd(),
    )

    if args.diagnostic_only:
        scripts = {"diagnostic": script_dir / "submit_diagnostic.sh"}
        scripts["diagnostic"].write_text(runner.generate_diagnostic_script())
        _make_executable(scripts["diagnostic"])
        logger.info("[hpc-submit] wrote diagnostic script to %s", scripts["diagnostic"])
        if args.submit:
            _submit(runner, scripts["diagnostic"], args.dry_run)
        _print_summary(scripts, args.stage, dist_cfg.run_id, args.submit, args.dry_run)
        return

    # incomplete_ids=[] → reduce-only (no stage script); otherwise sparse or full array
    scripts = runner.write_scripts(
        output_dir=script_dir,
        stage=args.stage,
        reduce_walltime=args.reduce_walltime,
        reduce_memory_gb=args.reduce_memory_gb,
        task_ids=incomplete_ids,
    )

    if args.submit:
        if incomplete_ids:
            # Submit sparse array for incomplete shards only
            array_job_id = _submit(runner, _select_array_script(scripts), args.dry_run)
            if array_job_id and not args.dry_run:
                # Regenerate reduce with afterany dependency on the new array job
                reduce_content = runner.generate_reduce_script(
                    args.stage,
                    after_job_id=array_job_id,
                    reduce_walltime=args.reduce_walltime,
                    reduce_memory_gb=args.reduce_memory_gb,
                )
                scripts["reduce"].write_text(reduce_content)
            _submit(runner, scripts["reduce"], args.dry_run)
        else:
            # All shards done — submit reduce directly with no dependency
            logger.info(
                "[hpc-submit] all %d shards already complete; submitting reduce directly",
                n_shards,
            )
            _submit(runner, scripts["reduce"], args.dry_run)

    _print_summary(scripts, args.stage, dist_cfg.run_id, args.submit, args.dry_run)


def _build_fresh_manifest(args, artifact_dir: Path, n_shards: int, workflow) -> list:
    """Build a fresh shard manifest from pipeline inputs."""
    from bsm_rfm.distributed.manifest import (
        build_manifest,
        resolve_interaction_discovery_shard_inputs,
    )

    input_paths: list[str] = [str(Path(args.config).resolve())]
    expected_columns = _estimate_stage_work_items(args.stage, workflow, artifact_dir)
    if args.stage == "interaction_discovery":
        try:
            dataset_path = getattr(workflow.dataset, "path", None)
            resolved_inputs = resolve_interaction_discovery_shard_inputs(
                artifact_dir, dataset_path=dataset_path
            )
            input_paths.extend(list(resolved_inputs.values()))
            logger.info(
                "[hpc-submit] resolved %d interaction_discovery inputs: %s",
                len(resolved_inputs),
                ", ".join(k for k in resolved_inputs.keys()),
            )
        except FileNotFoundError as e:
            logger.error("[hpc-submit] failed to resolve interaction_discovery inputs: %s", e)
            raise

    return build_manifest(
        stage=args.stage,
        input_paths=input_paths,
        output_root=str(artifact_dir / "hpc_shards"),
        n_shards=n_shards,
        expected_columns=expected_columns,
    )


def _estimate_stage_work_items(stage: str, workflow, artifact_dir: Path) -> int:
    """Estimate stage work-item cardinality for shard-range partitioning."""
    if stage == "output_conditioning":
        return 1
    if stage == "empirical_null_screening":
        return max(1, int(workflow.stages.empirical_null_screening.n_permutations) - 1)
    if stage == "interaction_discovery":
        retained_terms_path = artifact_dir / "empirical_null_screen" / "retained_terms.csv"
        if not retained_terms_path.exists():
            return 0
        try:
            import pandas as pd

            retained_terms = pd.read_csv(retained_terms_path)
            if "feature_name" not in retained_terms.columns:
                return 0
            return int(retained_terms["feature_name"].astype(str).nunique())
        except Exception:
            return 0
    if stage == "nonlinear_discovery":
        retained_terms_path = artifact_dir / "empirical_null_screen" / "retained_terms.csv"
        if not retained_terms_path.exists():
            return 0
        try:
            import pandas as pd

            retained_terms = pd.read_csv(retained_terms_path)
            if "feature_name" not in retained_terms.columns:
                return 0
            return int(retained_terms["feature_name"].astype(str).nunique())
        except Exception:
            return 0
    if stage == "sparse_selection":
        return max(1, int(workflow.stages.sparse_selection.n_stability_subsamples))
    if stage == "final_manuscript_artifacts":
        return max(1, int(workflow.stages.final_artifacts.bootstrap_count))
    return 0


def _submit(runner, script_path: Path, dry_run: bool) -> int | None:
    try:
        job_id = runner.submit(script_path, dry_run=dry_run)
        if job_id:
            logger.info("[hpc-submit] submitted %s → job %d", script_path.name, job_id)
        return job_id
    except Exception as e:
        logger.error("[hpc-submit] submission failed: %s", e)
        return None


def _select_array_script(scripts: dict[str, Path]) -> Path:
    """Choose the preferred array script path from generated scripts.

    Prefer GPU array submissions when GPU scripts are present, otherwise
    use the CPU stage array script.
    """
    if "gpu_stage" in scripts:
        return scripts["gpu_stage"]
    if "stage" in scripts:
        return scripts["stage"]
    raise KeyError("No stage or gpu_stage script found for submission")


def _make_executable(path: Path) -> None:
    import stat

    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _print_summary(scripts: dict, stage: str, run_id: str, submitted: bool, dry_run: bool) -> None:
    print()
    print("=" * 60)
    print(f"BSM HPC Scripts — stage={stage}  run_id={run_id}")
    print("=" * 60)
    for name, path in scripts.items():
        print(f"  {name:15s}: {path}")
    print()
    if not submitted:
        print("To submit jobs:")
        if "submit_all" in scripts:
            print(f"  bash {scripts['submit_all']}")
        else:
            for name, path in scripts.items():
                if name != "submit_all":
                    print(f"  sbatch {path}")
    elif dry_run:
        print("[dry-run] sbatch commands shown above; not actually submitted")
    else:
        print("Jobs submitted. Monitor with:")
        print("  squeue -u $USER")
        log_parent = scripts.get(
            "gpu_stage",
            scripts.get("stage", scripts.get("diagnostic", "")),
        ).parent
        print(f"  tail -f {log_parent}/bsm_{stage}_*.out")
    print()


if __name__ == "__main__":
    main()
