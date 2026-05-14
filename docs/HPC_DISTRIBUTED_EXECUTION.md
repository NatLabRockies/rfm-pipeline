# HPC and Distributed Execution Guide

This guide explains how to run the BSM manuscript pipeline on HPC clusters (NREL Kestrel and generic SLURM) using the Phase 8a distributed execution infrastructure.

## Quick Start

### 0. Unified local orchestration entrypoint (recommended)

Use one local command to submit, check status, and collect artifacts without manually juggling HPC scripts:

```bash
# Submit CPU/GPU tiers defined in orchestration config
pixi run hpc-workflow -- \
    --config configs/hpc/kestrel_workflow_orchestration.yml \
    --action submit

# Check status snapshot
pixi run hpc-workflow -- \
    --config configs/hpc/kestrel_workflow_orchestration.yml \
    --action status

# Pull artifact bundle back locally
pixi run hpc-workflow -- \
    --config configs/hpc/kestrel_workflow_orchestration.yml \
    --action collect
```

The orchestration config centralizes user/host/account, remote repo path, scratch/projects roots, artifact roots, local core count, CPU tier node counts, and optional GPU submission.

`pullback.mode` controls local bundle size:

- `manifest_only`: run manifest + status summary only
- `reporting_bundle` (default): manifest + merged outputs + latest logs
- `full`: full shard outputs and logs

### 0a. One-command small distributed smoke test

```bash
bash scripts/kestrel/run_small_distributed_test_local.sh
```

This uses `configs/hpc/kestrel_workflow_small_distributed.yml` (2-node tier only, GPU disabled, manifest-only pullback).
That orchestration config now targets `configs/hpc/kestrel_cpu_scale_2_smoke.yml`, which is tuned for lightweight end-to-end validation with a 30-minute SLURM walltime budget.
It also enables `execution.prepare_interaction_inputs: true` to materialize `output_conditioning` + `empirical_null_screen` artifacts before submitting distributed `interaction_discovery`.

### 1. Validate your environment (debug partition smoke test)

```bash
pixi run bsm-hpc-submit \
    --config configs/hpc/kestrel_debug_smoke.yml \
    --diagnostic-only \
    --submit
```

This submits a lightweight job to the `debug` partition to verify that Pixi, BSM dependencies, and cluster connectivity all work before committing to a full run.

### 2. Generate production scripts (without submitting)

```bash
pixi run bsm-hpc-submit \
    --config configs/hpc/kestrel_30k.yml \
    --stage interaction_discovery \
    --n-shards 50
```

This writes scripts to `artifacts/kestrel_30k_run/hpc_scripts/` without submitting anything.

### 3. Generate and submit in one step

```bash
pixi run bsm-hpc-submit \
    --config configs/hpc/kestrel_30k.yml \
    --stage interaction_discovery \
    --n-shards 50 \
    --submit
```

### 4. Monitor running jobs

```bash
squeue -u $USER
tail -f /scratch/$USER/bsm/bsm_kestrel_30k/logs/bsm_interaction_discovery_*.out
```

### 5. Check shard completion status

```bash
# After jobs complete, count _SUCCESS.json markers
ls -la artifacts/kestrel_30k_run/hpc_shards/task-*/  | grep _SUCCESS | wc -l
```

______________________________________________________________________

## Architecture

The distributed execution follows a **shard → reduce** pattern:

```
manifest.jsonl           (one record per shard)
      │
      ├── SLURM array job (one task per shard)
      │         │
      │         └── hpc_shard_worker.py
      │               ├── select shard by SLURM_ARRAY_TASK_ID
      │               ├── run stage computation for shard's feature range
      │               ├── validate outputs
      │               └── write _SUCCESS.json
      │
      └── SLURM reduce job (--dependency=afterok:$ARRAY_JOB_ID)
                │
                └── hpc_reduce.py
                      ├── verify all _SUCCESS.json markers
                      ├── load shard_result.json from each shard
                      └── merge into combined stage artifact
```

**Idempotent re-runs**: Each shard checks for `_SUCCESS.json` before computing. Resubmitting after partial failure skips completed shards automatically.

______________________________________________________________________

## Configuration

### Adding a `distributed` section to your config

Add a `distributed:` block to any workflow config YAML:

```yaml
# Standard workflow settings
dataset:
  type: synthetic_full
runtime:
  n_jobs: 104
# ... (rest of workflow config)

# Distributed execution settings
distributed:
  enabled: true
  backend: slurm_array
  run_id: bsm_kestrel_30k
  pixi_env_path: /projects/bsm/.pixi

  slurm:
    account: bsm
    partition: shared
    walltime: "08:00:00"
    memory_gb: 240
    cpus_per_task: 104
    max_concurrent_array_tasks: 50

  kestrel:
    projects_root: /projects/bsm
    pixi_cache_dir: /projects/bsm/.cache/pixi

  spill:
    scratch_root: "/scratch/${USER}/bsm"
    min_free_gb: 20.0
```

### Pre-built configs

| Config                                | Purpose                            | Partition |
| ------------------------------------- | ---------------------------------- | --------- |
| `configs/hpc/kestrel_30k.yml`         | Full 30k production run on Kestrel | `shared`  |
| `configs/hpc/kestrel_debug_smoke.yml` | Environment validation             | `debug`   |

______________________________________________________________________

## Directory Layout on Kestrel

```
/projects/bsm/
├── .pixi/                          ← shared Pixi installation
├── .cache/pixi/                    ← shared package cache
└── bsm-public-rf/
    ├── configs/hpc/                ← HPC configs
    └── artifacts/
        └── kestrel_30k_run/
            ├── hpc_scripts/        ← generated sbatch scripts + manifest
            │   ├── manifest.jsonl
            │   ├── submit_interaction_discovery_array.sh
            │   ├── submit_interaction_discovery_reduce.sh
            │   ├── submit_diagnostic.sh
            │   └── submit_all.sh
            └── hpc_shards/
                ├── task-0000/
                │   ├── _SUCCESS.json
                │   └── shard_result.json
                ├── task-0001/
                └── ...

/scratch/$USER/bsm/
└── bsm_kestrel_30k/
    ├── logs/                       ← SLURM stdout/stderr
    │   ├── bsm_interaction_discovery_12345_0.out
    │   └── ...
    └── task-0000/                  ← scratch working dir per task
        └── ...
```

______________________________________________________________________

## Kestrel-Specific Notes

- **Account**: `bsm` (set in `slurm.account`)
- **Production partition**: `shared` (up to 48 hours)
- **Debug partition**: `debug` (up to 1 hour, use for smoke tests)
- **GPU partition**: `gpu-h100s` (for GPU-accelerated stages)
- **Max array size**: 11,000 tasks
- **$TMPDIR warning**: On Kestrel CPU nodes, `$TMPDIR` can be RAM-backed. Use `/scratch/$USER/` for spill instead (the default in all provided configs).
- **Network interface**: Use `hsn0` for Dask distributed communication (Phase 8c).

______________________________________________________________________

## First-Time Kestrel Setup

Before running any jobs:

```bash
# 1. Clone the repo to /projects/bsm/
cd /projects/bsm
git clone https://github.com/NatLabRockies/bsm-public-rf.git
cd bsm-public-rf

# 2. Set up shared Pixi environment
export PIXI_HOME=/projects/bsm/.pixi
export PIXI_CACHE_DIR=/projects/bsm/.cache/pixi
pixi install --locked

# 3. Verify the environment
pixi run python -c "import bsm_rfm; print('BSM OK')"

# 4. Submit the diagnostic smoke test
pixi run bsm-hpc-submit \
    --config configs/hpc/kestrel_debug_smoke.yml \
    --diagnostic-only \
    --submit
```

______________________________________________________________________

## CLI Reference

### `pixi run bsm-hpc-submit`

```
usage: bsm_hpc_submit.py [-h] --config CONFIG
                         [--stage {output_conditioning,...}]
                         [--n-shards N_SHARDS]
                         [--output-dir OUTPUT_DIR]
                         [--submit] [--dry-run]
                         [--diagnostic-only]
                         [--reduce-walltime REDUCE_WALLTIME]
                         [--reduce-memory-gb REDUCE_MEMORY_GB]

options:
  --config         Path to workflow config YAML (must have 'distributed' section)
  --stage          Pipeline stage to shard (default: interaction_discovery)
  --n-shards       Number of shards (default: slurm.max_concurrent_array_tasks)
  --output-dir     Script output directory (default: <artifact_dir>/hpc_scripts)
  --submit         Submit generated scripts via sbatch
  --dry-run        Print sbatch commands but do not execute (use with --submit)
  --diagnostic-only  Generate/submit diagnostic smoke test only
  --reduce-walltime  Walltime for reduce job (default: 02:00:00)
  --reduce-memory-gb Memory (GB) for reduce job (default: 32)
```

### `pixi run bsm-hpc-shard-worker`

Run by SLURM inside each array task. Not typically called directly.

### `pixi run bsm-hpc-reduce`

Run by the SLURM reduce job after all array tasks complete. Can be run manually to check/retry reduction.

```bash
pixi run bsm-hpc-reduce \
    --manifest artifacts/kestrel_30k_run/hpc_scripts/manifest.jsonl \
    --output-root artifacts/kestrel_30k_run/hpc_shards \
    --stage interaction_discovery
```

### Migration notes: legacy scripts → unified runner

| Legacy operation                               | Unified command                                                                                     |
| ---------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `scripts/kestrel/submit_cpu_scaling_suite.sh`  | `pixi run hpc-workflow -- --config configs/hpc/kestrel_workflow_orchestration.yml --action submit`  |
| `scripts/kestrel/status_all_tests.sh`          | `pixi run hpc-workflow -- --config configs/hpc/kestrel_workflow_orchestration.yml --action status`  |
| `scripts/kestrel/pull_hpc_artifacts_bundle.sh` | `pixi run hpc-workflow -- --config configs/hpc/kestrel_workflow_orchestration.yml --action collect` |

Legacy scripts remain available for low-level operations and debugging, but the orchestration runner is the default user-facing path.

______________________________________________________________________

## Troubleshooting

### Array task fails with "already complete"

If you re-submit after a partial failure, the worker will skip shards that already have `_SUCCESS.json`. Only failed/pending shards will recompute.

### Reduce job fails with "pending shards"

Some array tasks did not complete. Check their log files:

```bash
grep -l "FAILED\|Error\|Traceback" /scratch/$USER/bsm/bsm_kestrel_30k/logs/*.err
```

### Out of memory

Increase `slurm.memory_gb` in your config. For standard Kestrel CPU nodes, 240 GB leaves headroom below the 256 GB limit.

### Jobs not starting

Check partition availability:

```bash
sinfo -p shared --noheader -O partition,avail,nodes,time
```

### Pixi not found on compute nodes

Ensure `pixi_env_path` in your config points to the shared filesystem. Compute nodes must be able to read `/projects/bsm/.pixi`.
