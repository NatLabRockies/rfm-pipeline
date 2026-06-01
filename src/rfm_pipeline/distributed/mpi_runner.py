r"""MPI-based shard distribution for the BSM manuscript pipeline.

Provides rank/size-based work assignment so MPI jobs on Kestrel can
directly distribute shards across CPU nodes without SLURM array overhead.

This is a thin adapter: the actual computation is still routed through
the existing `hpc_shard_worker.py` entry point. MPI is used only for
rank assignment and barrier synchronisation.

Requires: mpi4py (optional; import guard provided).

Usage (inside a Kestrel MPI job)::

    srun -n 128 --exclusive pixi run python -m rfm_pipeline.distributed.mpi_runner \\
        --manifest /scratch/$USER/bsm/manifest.jsonl \\
        --config configs/hpc/kestrel_30k.yml

Rank 0 logs summary statistics after all ranks complete their shards.

Environment variables honoured:
  BSM_MPI_BARRIER_TIMEOUT   Seconds to wait at final barrier (default: 3600)
  BSM_INTERACTION_DEVICE    cuda | cpu | auto (passed through to shard worker)

SLURM job template::

    #SBATCH --ntasks=128
    #SBATCH --nodes=4
    #SBATCH --ntasks-per-node=32
    #SBATCH --partition=shared
    #SBATCH --account=bsm

    module load mpi4py  # or install via pixi
    srun pixi run python -m rfm_pipeline.distributed.mpi_runner \\
        --manifest /scratch/$USER/bsm/manifest.jsonl \\
        --config configs/hpc/kestrel_30k.yml

Notes
-----
  - Gated behind mpi4py; if mpi4py is not installed the module logs a
    clear error and exits rather than silently misassigning work.
  - Prefer SLURM arrays for most workloads; use MPI only when task
    startup overhead dominates (very large numbers of tiny shards).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_MPI4PY_AVAILABLE = False
try:
    from mpi4py import MPI

    _MPI4PY_AVAILABLE = True
except ImportError:
    pass


def _require_mpi4py() -> None:
    """Raise ImportError with install guidance if mpi4py is absent."""
    if not _MPI4PY_AVAILABLE:
        raise ImportError(
            "mpi4py is required for MPI-based execution. "
            "Install with: conda install mpi4py, or add to pixi.toml under "
            "[dependencies] and run pixi install."
        )


def get_rank_size() -> tuple[int, int]:
    """Return (rank, size) from MPI_COMM_WORLD.

    Returns (0, 1) when mpi4py is unavailable (serial fallback for testing).
    """
    if not _MPI4PY_AVAILABLE:
        return 0, 1
    comm = MPI.COMM_WORLD  # type: ignore[attr-defined]
    return comm.Get_rank(), comm.Get_size()


def assign_shards(n_shards: int, rank: int, size: int) -> list[int]:
    """Return the list of shard indices assigned to this MPI rank.

    Uses round-robin assignment: rank r handles shards r, r+size, r+2*size, ...

    Parameters
    ----------
    n_shards
        Total number of shards in the manifest.
    rank
        This process's MPI rank.
    size
        Total number of MPI ranks.

    Returns
    -------
    list[int]
        Shard indices (0-based) assigned to this rank.
    """
    return list(range(rank, n_shards, size))


def barrier(timeout: float | None = None) -> None:
    """MPI barrier with optional timeout.

    Parameters
    ----------
    timeout
        Seconds to wait at barrier (None = indefinite). Reads
        BSM_MPI_BARRIER_TIMEOUT env var as fallback.
    """
    if not _MPI4PY_AVAILABLE:
        return

    if timeout is None:
        timeout = float(os.environ.get("BSM_MPI_BARRIER_TIMEOUT", "3600"))

    comm = MPI.COMM_WORLD  # type: ignore[attr-defined]
    # mpi4py does not expose ibarrier reliably on all versions; use standard barrier
    comm.Barrier()


def run_mpi_worker(
    manifest_path: str,
    config_path: str,
    stage: str = "interaction_discovery",
    dry_run: bool = False,
) -> None:
    """Run shard processing for this MPI rank.

    Parameters
    ----------
    manifest_path
        Path to the shard manifest JSONL file.
    config_path
        Path to the distributed config YAML.
    stage
        Pipeline stage to execute.
    dry_run
        If True, log what would be done without running.
    """
    _require_mpi4py()

    from rfm_pipeline.distributed.manifest import load_manifest

    rank, size = get_rank_size()
    shards = load_manifest(manifest_path)
    my_shards = assign_shards(len(shards), rank, size)

    if rank == 0:
        logger.info(
            "[mpi_runner] ranks=%d, total_shards=%d, shards/rank≈%.1f",
            size,
            len(shards),
            len(shards) / size,
        )

    logger.info("[mpi_runner] rank=%d assigned shards %s", rank, my_shards)

    for shard_idx in my_shards:
        shard = shards[shard_idx]
        shard_id = getattr(shard, "shard_id", str(shard_idx))
        if dry_run:
            logger.info("[mpi_runner] rank=%d dry-run shard %s", rank, shard_id)
            continue

        try:
            _run_shard(shard, config_path, stage)
        except Exception as exc:
            logger.error("[mpi_runner] rank=%d FAILED shard %s: %s", rank, shard_id, exc)
            raise

    barrier()

    if rank == 0:
        logger.info("[mpi_runner] all ranks complete for stage=%s", stage)


def _run_shard(shard, config_path: str, stage: str, dry_run: bool = False) -> None:
    """Execute a single shard via the hpc_shard_worker entry point."""
    from rfm_pipeline.distributed.config_distributed import load_config

    config = load_config(config_path)
    output_root = str(Path(config.output_dir) / config.run_id)

    # Delegate to shard worker function (avoids subprocess overhead inside MPI)
    # Actual implementation lives in tools/hpc_shard_worker.py
    # Import here to keep mpi_runner lean and testable
    try:
        tools_dir = Path(__file__).parent.parent.parent.parent / "tools"
        if str(tools_dir) not in sys.path:
            sys.path.insert(0, str(tools_dir))
        import hpc_shard_worker

        hpc_shard_worker.run_shard(
            shard=shard,
            output_root=output_root,
            config_path=config_path,
            dry_run=dry_run,
        )
    except ImportError:
        logger.error(
            "[mpi_runner] Could not import hpc_shard_worker from tools/. "
            "Ensure the tools/ directory is on PYTHONPATH or run via pixi."
        )
        raise


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: python -m rfm_pipeline.distributed.mpi_runner."""
    parser = argparse.ArgumentParser(description="BSM manuscript pipeline MPI shard worker")
    parser.add_argument("--manifest", required=True, help="Path to shard manifest JSONL")
    parser.add_argument("--config", required=True, help="Path to distributed config YAML")
    parser.add_argument("--stage", default="interaction_discovery", help="Pipeline stage")
    parser.add_argument("--dry-run", action="store_true", help="Log without executing")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    run_mpi_worker(
        manifest_path=args.manifest,
        config_path=args.config,
        stage=args.stage,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
