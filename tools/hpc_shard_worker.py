"""Shim: delegates to rfm_pipeline.hpc_shard_worker (now an installed package module).

Kept here for backward compatibility with existing SLURM scripts that call
  pixi run python tools/hpc_shard_worker.py ...
"""

from rfm_pipeline.hpc_shard_worker import main

if __name__ == "__main__":
    main()
