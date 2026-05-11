# Kestrel Distributed Runtime Decision Matrix

**Purpose:** Compare SLURM arrays, Dask+dask-jobqueue, MPI/mpi4py, and Ray on SLURM for Kestrel distributed compute support.
**Retrieval date:** 2026-05-10.
**Default account:** `bsm`.

## 1. Summary recommendation

| Rank | Runtime              | Recommendation                                                                 |
| ---: | -------------------- | ------------------------------------------------------------------------------ |
|    1 | SLURM arrays         | Implement first as deterministic production baseline.                          |
|    2 | Dask + dask-jobqueue | Recommended default distributed dataframe/runtime layer after baseline arrays. |
|    3 | MPI / mpi4py         | Add as low-level fallback for rank-sharded tasks and reductions.               |
|    4 | Ray on SLURM         | Keep experimental until Kestrel smoke tests pass.                              |

## 2. Evidence matrix

| Criterion                 | SLURM arrays                                                                                                                                                         | Dask + dask-jobqueue                                                                                                                                                                                                         | MPI / mpi4py                                                                                                                                                                               | Ray on SLURM                                                                                                                                                             |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Kestrel-specific docs     | Strong: NLR documents arrays and recommends shared partition for arrays. Source: `https://nrel.github.io/HPC/Documentation/Slurm/job_arrays/`; retrieved 2026-05-10. | Strong: NLR provides Kestrel Dask examples using `SLURMCluster`, `SLURMRunner`, `account`, `queue`, and `hsn0`. Source: `https://nrel.github.io/HPC/Documentation/Development/Languages/Python/dask/`; retrieved 2026-05-10. | Strong for MPI stack: NLR recommends Cray MPICH and warns against OpenMPI. Source: `https://nrel.github.io/HPC/Documentation/Development/Programming-Environments/`; retrieved 2026-05-10. | Weak for Kestrel-specific docs; Ray has official SLURM docs. Source: `https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm.html`; retrieved 2026-05-10. |
| Best workload shape       | Embarrassingly parallel shards                                                                                                                                       | Parquet/dataframe/array workloads with task graphs                                                                                                                                                                           | Explicit rank-sharded compute and collectives                                                                                                                                              | Task/actor workloads; potential heterogeneous CPU/GPU orchestration                                                                                                      |
| Restart model             | Strong if repo implements `_SUCCESS` markers                                                                                                                         | Medium; needs checkpointing and scheduler cleanup                                                                                                                                                                            | Medium; repo must implement checkpointing                                                                                                                                                  | Medium; repo must manage Ray lifecycle                                                                                                                                   |
| Dynamic scheduling        | Low                                                                                                                                                                  | Strong                                                                                                                                                                                                                       | Low                                                                                                                                                                                        | Strong                                                                                                                                                                   |
| Operational complexity    | Low                                                                                                                                                                  | Medium                                                                                                                                                                                                                       | Medium/high                                                                                                                                                                                | High until validated                                                                                                                                                     |
| Dataframe ergonomics      | Low                                                                                                                                                                  | Strong                                                                                                                                                                                                                       | Low                                                                                                                                                                                        | Medium                                                                                                                                                                   |
| First implementation role | Baseline executor                                                                                                                                                    | Default distributed runtime                                                                                                                                                                                                  | Specialized fallback                                                                                                                                                                       | Experimental backend                                                                                                                                                     |

## 3. Selection criteria

### Use SLURM arrays when

The repo can precompute a shard manifest and every shard can run independently or with a final reduce step. Examples: one model per output block, one feature block per task, independent bootstrap replicates, or validation over Parquet shards.

### Use Dask + dask-jobqueue when

The workload benefits from lazy task graphs, partitioned dataframe operations, distributed memory, or interactive scaling. Avoid large shuffles unless explicitly benchmarked.

### Use MPI / mpi4py when

Rank-to-rank communication, deterministic collectives, or explicit reductions are clearer than a task scheduler. Use Cray MPICH on Kestrel unless HPC support directs otherwise.

### Use Ray only after smoke tests when

The repo needs actors, task orchestration not well served by Dask, or mixed CPU/GPU experiments. Use Ray 2.49+ `symmetric-run` if Ray is tested. Do not make Ray a default backend before Kestrel smoke tests pass.

## 4. Risk matrix

| Risk                                              | Runtime(s) affected | Mitigation                                                                                                                                                                                                        |
| ------------------------------------------------- | ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `$TMPDIR` may be RAM-backed on non-NVMe CPU nodes | All                 | Verify `df -T "$TMPDIR"`; use `/scratch` fallback or request `nvme --tmp`. Source: `https://nrel.github.io/HPC/Documentation/Systems/Kestrel/Filesystems/`; retrieved 2026-05-10.                                 |
| Lustre metadata storms                            | All                 | Avoid huge flat directories, Python `os.walk`, `os.scandir`, wildcard copies, and many small files. Source: `https://nrel.github.io/HPC/Documentation/Systems/Kestrel/Filesystems/lustre/`; retrieved 2026-05-10. |
| Requeue not enabled by default                    | All                 | Implement idempotent outputs; do not rely on SLURM requeue. User live config reports `JobRequeue=0`.                                                                                                              |
| Dask communication errors                         | Dask                | Prefer dask-jobqueue over dask-mpi if Kestrel `CommClosedError` occurs, matching NLR note. Source: `https://nrel.github.io/HPC/Documentation/Development/Languages/Python/dask/`; retrieved 2026-05-10.           |
| OpenMPI instability/performance concerns          | MPI                 | Use Cray MPICH, not OpenMPI, unless directed otherwise. Source: `https://nrel.github.io/HPC/Documentation/Development/Programming-Environments/`; retrieved 2026-05-10.                                           |
| Ray launch mismatch with SLURM                    | Ray                 | Use Ray 2.49+ `symmetric-run`; validate locally before enabling. Source: `https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm.html`; retrieved 2026-05-10.                                      |

## 5. Minimal implementation decision

Implement these repo backends or equivalent adapters:

```text
bsm_rfm.distributed.backends.slurm_array
bsm_rfm.distributed.backends.dask_jobqueue
bsm_rfm.distributed.backends.mpi4py
bsm_rfm.distributed.backends.ray_experimental
```

The Ray backend must remain opt-in and excluded from default tests unless `BSM_ENABLE_RAY_EXPERIMENTAL=1` is set.
