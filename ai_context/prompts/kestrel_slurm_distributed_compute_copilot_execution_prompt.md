# Copilot Execution Prompt: Kestrel SLURM Distributed Compute Support

You are working inside the local repository. Implement Kestrel SLURM distributed compute support using the repo-local research artifacts listed below. You do **not** have web search. Do not invent external API behavior. Treat the cited artifacts as the source of truth unless the live repo contradicts them.

## Required context files to read first

Read these files completely before editing code:

```text
ai_context/literature/kestrel_slurm_distributed_compute_literature_bundle.md
ai_context/methods/kestrel_slurm_distributed_compute_method_manifest.md
ai_context/api_docs/kestrel_slurm_scheduler_interfaces_api_docs.md
ai_context/api_docs/kestrel_distributed_runtime_decision_matrix.md
ai_context/api_docs/runtime_api_bundle_mpi4py_kestrel.md
ai_context/api_docs/runtime_api_bundle_dask_jobqueue_kestrel.md
ai_context/api_docs/runtime_api_bundle_ray_kestrel.md
ai_context/manifests/kestrel_slurm_distributed_compute_engineering_manifest.md
```

## Hard rules

1. Start with a repo audit. Do not patch from assumed state.
1. Use test-first development.
1. Make the smallest robust implementation slice that advances the manifest.
1. Do not add hacks, compatibility shims, or silent fallbacks.
1. Do not make Ray a default backend.
1. Do not assume `/home` can hold Pixi environments or large caches.
1. Do not assume `$TMPDIR` is disk-backed.
1. Do not rely on automatic SLURM requeue for correctness.
1. Do not use deprecated dask-jobqueue parameters.
1. Do not use OpenMPI as the default MPI assumption for Kestrel.

## Target behavior

Implement multi-runtime support in this order:

```text
1. SLURM array baseline.
2. Dask + dask-jobqueue backend.
3. MPI/mpi4py backend.
4. Ray experimental backend, disabled by default.
```

Default Kestrel values:

```yaml
account: bsm
project_root: /projects/bsm
cpu_probe_partition: debug
gpu_probe_partition: gpu-h100s
production_array_partition: shared
dask_network_interface: hsn0
max_array_size: 11000
job_requeue_default: false
preemption_enabled: false
kill_wait_seconds: 30
job_accounting_frequency_seconds: 30
```

## Initial audit tasks

Before editing, inspect:

```bash
find . -maxdepth 3 -type f | sort
grep -R "slurm\|sbatch\|dask\|ray\|mpi\|distributed\|pixi\|parquet" -n . \
  --exclude-dir=.git \
  --exclude-dir=.pixi \
  --exclude-dir=__pycache__ || true
```

Identify package structure, tests, scripts, Pixi config, CI hooks, and any prior distributed-compute code.

## Implementation slice 1: Kestrel config and SLURM script rendering

Write tests first for:

```text
- default Kestrel config uses account=bsm
- CPU probe uses partition=debug
- GPU probe uses partition=gpu-h100s
- every rendered sbatch script includes --account and --time
- Pixi/cache exports use /projects/bsm, not /home
- thread-control exports set OMP/MKL/OPENBLAS/NUMEXPR from SLURM_CPUS_PER_TASK
```

Then implement, adapting paths to the existing repo package layout:

```text
src/<package>/distributed/config.py
src/<package>/distributed/slurm.py
src/<package>/distributed/resources.py
```

## Implementation slice 2: Filesystem and spill path selection

Write tests first for:

```text
- no TMPDIR -> scratch fallback
- tmpfs TMPDIR -> scratch fallback
- disk-backed TMPDIR -> local spill path
- scratch path includes user and run_id
```

Then implement:

```text
src/<package>/distributed/paths.py
```

Rules:

```text
/projects/bsm        durable project root and Pixi/cache root
/scratch/$USER/...   temporary high-throughput working sets
$TMPDIR              use only after filesystem check proves it is not tmpfs
```

## Implementation slice 3: SLURM array manifest

Write tests first for:

```text
- manifest schema validation
- SLURM_ARRAY_TASK_ID selects the expected shard
- output attempt directory is unique
- final output is promoted only after validation
- _SUCCESS marker is written only after validation
- completed shard is skipped on rerun
```

Expected command shape:

```bash
pixi run python -m <package>.distributed.run_shard \
  --manifest manifests/shards.parquet \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --run-id "${SLURM_ARRAY_JOB_ID}"
```

## Implementation slice 4: Dask + dask-jobqueue

Write tests first for:

```text
- Dask config uses account=bsm
- debug smoke queue is debug
- production queue can be shared or nvme
- interface defaults to hsn0
- local_directory comes from spill selector
- rendered code does not use deprecated project/env_extra/job_extra args
```

Allowed dask-jobqueue parameters include:

```text
account
queue
cores
memory
processes
walltime
interface
local_directory
log_directory
job_script_prologue
job_extra_directives
```

Avoid deprecated:

```text
project
env_extra
job_extra
```

## Implementation slice 5: MPI/mpi4py

Write tests first for:

```text
- deterministic rank-to-shard mapping
- unique rank output paths
- generated script uses srun
- generated script does not assume OpenMPI
```

Do not make MPI the default dataframe backend.

## Implementation slice 6: Ray experimental

Write tests first for:

```text
- Ray backend is disabled by default
- Ray requires BSM_ENABLE_RAY_EXPERIMENTAL=1
- Ray script uses ray symmetric-run
- Ray GPU script uses gpu-h100s
- cleanup trap is present
```

Implement script generation only. Do not wire Ray into production defaults.

## Validation commands

Use the repo’s established gate. If none exists, run at minimum:

```bash
pixi run ruff check .
pixi run ruff format --check .
pixi run pytest -q
```

If the repo has `test_repo.sh`, inspect it first and use the appropriate check/fix mode according to the repo’s documented workflow.

## Deliverables after implementation

Report:

```text
- files changed
- tests added
- commands run
- exact validation results
- unresolved Kestrel-only manual checks
- suggested git commit message
```

Do not claim Kestrel runtime support is production-ready until manual Kestrel smoke scripts have been run and their logs inspected.
