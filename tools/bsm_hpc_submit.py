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
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Load configs
    from bsm_rfm.config import load_config
    from bsm_rfm.distributed.config_distributed import load_distributed_config
    from bsm_rfm.distributed.manifest import (
        build_manifest,
        resolve_interaction_discovery_shard_inputs,
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

    # Build shard manifest
    n_shards = args.n_shards or dist_cfg.slurm.max_concurrent_array_tasks
    logger.info(
        "[hpc-submit] stage=%s n_shards=%d run_id=%s",
        args.stage,
        n_shards,
        dist_cfg.run_id,
    )

    if not args.diagnostic_only:
        # Auto-resolve input paths for interaction_discovery stage
        input_paths = []
        if args.stage == "interaction_discovery":
            try:
                resolved_inputs = resolve_interaction_discovery_shard_inputs(artifact_dir)
                input_paths = list(resolved_inputs.values())
                logger.info(
                    "[hpc-submit] resolved %d interaction_discovery inputs: %s",
                    len(input_paths),
                    ", ".join(k for k in resolved_inputs.keys()),
                )
            except FileNotFoundError as e:
                logger.error("[hpc-submit] failed to resolve interaction_discovery inputs: %s", e)
                raise

        shards = build_manifest(
            stage=args.stage,
            input_paths=input_paths,
            output_root=str(artifact_dir / "hpc_shards"),
            n_shards=n_shards,
        )
        save_manifest(shards, manifest_path)
        logger.info("[hpc-submit] wrote manifest with %d shards to %s", len(shards), manifest_path)

    # Build runner and generate scripts
    runner = SlurmArrayRunner(
        config=dist_cfg,
        manifest_path=manifest_path if not args.diagnostic_only else manifest_path,
        output_root=str(artifact_dir / "hpc_shards"),
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

    scripts = runner.write_scripts(
        output_dir=script_dir,
        stage=args.stage,
        reduce_walltime=args.reduce_walltime,
        reduce_memory_gb=args.reduce_memory_gb,
    )

    if args.submit:
        # Submit array job, then reduce with dependency
        array_job_id = _submit(runner, _select_array_script(scripts), args.dry_run)
        if array_job_id and not args.dry_run:
            # Regenerate reduce script with actual dependency
            reduce_content = runner.generate_reduce_script(
                args.stage,
                after_job_id=array_job_id,
                reduce_walltime=args.reduce_walltime,
                reduce_memory_gb=args.reduce_memory_gb,
            )
            scripts["reduce"].write_text(reduce_content)
        _submit(runner, scripts["reduce"], args.dry_run)

    _print_summary(scripts, args.stage, dist_cfg.run_id, args.submit, args.dry_run)


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
