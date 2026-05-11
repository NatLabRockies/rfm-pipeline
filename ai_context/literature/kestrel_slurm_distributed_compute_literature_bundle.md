# Kestrel SLURM Distributed Compute Literature Bundle

**Purpose:** Repo-local research packet for planning distributed Python support on NREL/NLR Kestrel using SLURM.
**Repo context:** Python/Pixi pipeline processing large Parquet-derived feature matrices with out-of-core, spill-to-disk, and distributed execution requirements.
**Default project/allocation handle for examples:** `bsm`.
**Retrieval date for external web sources:** 2026-05-10.
**User-provided live cluster probe date:** 2026-05-10.

## 1. Source-status legend

| Label                  | Meaning                                                                  |
| ---------------------- | ------------------------------------------------------------------------ |
| Verified external fact | Documented in cited official documentation retrieved 2026-05-10.         |
| Verified live fact     | User-provided live Kestrel command output from 2026-05-10.               |
| Assumption             | Engineering default inferred from docs and repo needs; validate locally. |
| Risk / unknown         | Not confirmed by public docs or likely site-policy dependent.            |

## 2. Kestrel/SLURM facts

### Scheduler requirements

**Verified external fact.** Kestrel uses SLURM. NLR examples show project accounting with `--account=<project-handle>`, and Kestrel job examples include walltime through `--time/-t`. Treat `--account` and `--time` as required for all generated jobs. Source URL: `https://nrel.github.io/HPC/Documentation/Slurm/batch_jobs/`; retrieved 2026-05-10.

**Verified live fact.** User probe on Kestrel reported:

```text
slurm 25.05.5
SchedulerType=sched/backfill
SelectType=select/cons_tres
SelectTypeParameters=CR_CORE_MEMORY
TaskPlugin=task/cgroup,task/affinity
AccountingStorageType=accounting_storage/slurmdbd
JobAcctGatherType=jobacct_gather/linux
JobAcctGatherFrequency=30
MaxArraySize=11000
MaxJobCount=65000
JobRequeue=0
MaxBatchRequeue=5
PreemptMode=OFF
PreemptType=(null)
KillWait=30 sec
MinJobAge=300 sec
PriorityType=priority/multifactor
PriorityDecayHalfLife=7-00:00:00
```

**Implementation consequence.** Use `--account=bsm` and explicit `--time` everywhere. Use `debug` for CPU probes and `gpu-h100s` for short GPU probes when GPU probes are present.

### Partitions and topology

**Verified external fact.** Kestrel documentation lists CPU partitions such as `debug`, `short`, `standard`, `long`, `shared`, `nvme`, `medmem`, `bigmem`, and `hbw`, plus H100 GPU partitions including `gpu-h100`, `gpu-h100s`, and `gpu-h100l`. Source URL: `https://nrel.github.io/HPC/Documentation/Systems/Kestrel/Running/`; retrieved 2026-05-10.

**Verified external fact.** Kestrel standard CPU nodes are documented as 104-core nodes with approximately 240 GB usable RAM. Kestrel also has CPU nodes with local NVMe, high-memory nodes, and big-memory nodes. Source URL: `https://nrel.github.io/HPC/Documentation/Systems/Kestrel/Running/`; retrieved 2026-05-10.

**Verified external fact.** Kestrel GPU nodes are documented as H100 GPU nodes with four NVIDIA H100 80 GB GPUs and dual AMD Genoa 64-core CPUs. Source URL: `https://nrel.github.io/HPC/Documentation/Systems/Kestrel/Running/`; retrieved 2026-05-10.

### Filesystems and I/O

**Verified external fact.** Kestrel provides `/home`, ProjectFS under `/projects`, ScratchFS under `/scratch`, and node-local temporary storage through `$TMPDIR=/tmp/scratch/<JOBID>` when local disk exists. Most CPU nodes do not have local disk; on nodes without local disk, `$TMPDIR` consumes RAM. Local storage is cleaned after job completion and is not visible from other nodes. Source URL: `https://nrel.github.io/HPC/Documentation/Systems/Kestrel/Filesystems/`; retrieved 2026-05-10.

**Verified external fact.** NLR documents `/projects` as durable project storage for data, configuration, and applications. `/scratch` is temporary high-throughput storage subject to purge/inactivity policy. Source URL: `https://nrel.github.io/HPC/Documentation/Systems/Kestrel/Filesystems/`; retrieved 2026-05-10.

**Verified external fact.** NLR Lustre guidance warns against metadata-heavy patterns including many small files in one directory, `ls -l` on large directories, Python `os.walk`/`os.scandir` over large trees, wildcard operations over many files, and many processes hammering one file. Source URL: `https://nrel.github.io/HPC/Documentation/Systems/Kestrel/Filesystems/lustre/`; retrieved 2026-05-10.

**Implementation consequence.** Use:

```text
/projects/bsm/...                  durable Pixi installs, package caches, manifests, final artifacts
/scratch/$USER/bsm_<run_id>/...     high-throughput temporary distributed working set
$TMPDIR/...                         local spill only after confirming it is not RAM-backed/tmpfs
```

## 3. Candidate distributed runtimes

### MPI / mpi4py

**Verified external fact.** NLR recommends Cray MPICH on Kestrel and indicates OpenMPI should not be the default due to performance/stability concerns on Kestrel. Source URL: `https://nrel.github.io/HPC/Documentation/Development/Programming-Environments/`; retrieved 2026-05-10.

**Verified external fact.** mpi4py provides Python bindings to MPI with lower-case pickle-oriented calls such as `send`, `recv`, `bcast`, `scatter`, `gather`, and upper-case buffer-oriented calls such as `Send`, `Recv`, `Bcast`, `Scatter`, and `Gather`. Source URL: `https://mpi4py.readthedocs.io/en/stable/tutorial.html`; retrieved 2026-05-10.

**Engineering interpretation.** MPI/mpi4py is best for deterministic rank-sharded computation and explicit reductions. It does not provide a dataframe scheduler, automatic Parquet planning, spill management, or restart model.

### Dask + dask-jobqueue

**Verified external fact.** NLR provides Kestrel-specific Dask documentation using `dask-jobqueue`, `SLURMCluster`, `SLURMRunner`, `account`, `walltime`, `queue`, `processes`, `memory`, and `interface='hsn0'`. Source URL: `https://nrel.github.io/HPC/Documentation/Development/Languages/Python/dask/`; retrieved 2026-05-10.

**Verified external fact.** NLR’s Dask page notes observed `distributed.comm.core.CommClosedError` with `dask-mpi` and recommends using `dask-jobqueue` if issues occur. Source URL: `https://nrel.github.io/HPC/Documentation/Development/Languages/Python/dask/`; retrieved 2026-05-10.

**Verified API fact.** `dask_jobqueue.SLURMCluster` documents current parameters including `queue`, `account`, `cores`, `memory`, `processes`, `interface`, `local_directory`, `walltime`, `job_script_prologue`, and `job_extra_directives`; older parameters such as `project`, `env_extra`, and `job_extra` are deprecated. Source URL: `https://jobqueue.dask.org/en/stable/generated/dask_jobqueue.SLURMCluster.html`; retrieved 2026-05-10.

**Verified API fact.** Dask DataFrame is a parallel dataframe composed of many pandas DataFrames and supports larger-than-memory tabular workloads through lazy task graphs. Source URL: `https://docs.dask.org/en/stable/dataframe.html`; retrieved 2026-05-10.

**Engineering interpretation.** Dask+dask-jobqueue is the most Kestrel-documented distributed Python runtime and the best default candidate for Parquet/dataframe/task-graph workloads.

### Ray on SLURM

**Verified API fact.** Ray provides official SLURM deployment documentation. Ray 2.49+ includes `ray symmetric-run`, intended to start Ray on SLURM nodes and run the user entry point only on the head node. Source URL: `https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm.html`; retrieved 2026-05-10.

**Verified API fact.** Ray tasks are Python functions decorated with `@ray.remote`; `.remote()` returns object references and `ray.get()` retrieves results. Source URL: `https://docs.ray.io/en/latest/ray-core/tasks.html`; retrieved 2026-05-10.

**Risk / unknown.** No Kestrel-specific Ray guidance was found in NLR docs. Treat Ray as experimental until CPU debug, multi-node, and optional GPU short smoke tests pass.

## 4. Parquet/data layout evidence

**Verified external fact.** Apache Parquet recommends large row groups, commonly 512 MB to 1 GB, for efficient sequential I/O. Source URL: `https://parquet.apache.org/docs/file-format/configurations/`; retrieved 2026-05-10.

**Verified API fact.** PyArrow `write_table` defines `row_group_size` as a number of rows, not bytes. Source URL: `https://arrow.apache.org/docs/python/generated/pyarrow.parquet.write_table.html`; retrieved 2026-05-10.

**Engineering interpretation.** For very wide matrices, byte-targeted row groups must be translated into rows per group. Single ultra-wide Parquet tables with millions of columns are risky; benchmark feature-blocked layouts.

## 5. Recommended research conclusion

**Recommended baseline:** SLURM arrays with deterministic shard manifests and idempotent `_SUCCESS` markers.

**Recommended distributed dataframe/runtime default:** Dask + dask-jobqueue because NLR has Kestrel-specific guidance and the API maps to Parquet/dataframe workflows.

**Recommended low-level fallback:** MPI/mpi4py for rank-sharded computation and explicit reductions, using Kestrel’s Cray MPICH-oriented environment.

**Experimental path:** Ray on SLURM only after isolated smoke tests validate Kestrel startup, networking, GPU resource mapping if needed, and cleanup.
