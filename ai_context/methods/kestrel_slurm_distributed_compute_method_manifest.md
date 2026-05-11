# Kestrel SLURM Distributed Compute Method Manifest

**Purpose:** Convert Kestrel/SLURM/runtime research into implementation guidance for a local repo agent.\
**Default allocation/account:** `bsm`.\
**Retrieval date for external sources:** 2026-05-10.

## 1. Method decision principle

Do not lock the repo to one distributed runtime before validating runtime-specific failure modes on Kestrel. Implement a small execution abstraction that can support:

1. SLURM array shard execution.
1. Dask + dask-jobqueue distributed execution.
1. MPI/mpi4py rank-sharded execution.
1. Ray experimental execution, disabled by default.

## 2. Recommended runtime order

### 2.1 Baseline: SLURM arrays

**Status:** Implement first as the deterministic production baseline.

**Rationale.** NLR documents SLURM job arrays, array ranges/lists/steps, and concurrency throttling with `%N`; the user’s live Kestrel probe reports `MaxArraySize=11000`. Slurm official documentation defines the maximum array index as `MaxArraySize - 1`. Sources: `https://nrel.github.io/HPC/Documentation/Slurm/job_arrays/` and `https://slurm.schedmd.com/job_array.html`; retrieved 2026-05-10.

**Method.** Create a manifest-driven shard runner:

```text
manifest.parquet or manifest.jsonl:
  shard_id
  input_paths
  output_path
  expected_rows
  expected_columns
  feature_block_id
  scenario_id
  year
  status
```

Each array task selects shard rows using `SLURM_ARRAY_TASK_ID`, writes to an attempt directory, validates outputs, atomically promotes results, and writes `_SUCCESS.json` only after validation.

### 2.2 Default distributed dataframe/runtime path: Dask + dask-jobqueue

**Status:** Recommended default for dataframe/Parquet-oriented distributed Python after the SLURM array baseline.

**Rationale.** NLR provides Kestrel-specific Dask documentation with `SLURMCluster`, `SLURMRunner`, `account`, `walltime`, `queue`, `memory`, and `interface='hsn0'`. Dask DataFrame is documented for larger-than-memory tabular workflows composed of pandas partitions. Sources: `https://nrel.github.io/HPC/Documentation/Development/Languages/Python/dask/` and `https://docs.dask.org/en/stable/dataframe.html`; retrieved 2026-05-10.

**Method.** Use Dask for partition-wise Parquet/dataframe operations and bounded reductions. Avoid large shuffles or materializing full distributed dataframes into local pandas objects.

### 2.3 Low-level fallback: MPI / mpi4py

**Status:** Add after baseline and Dask config tests.

**Rationale.** NLR recommends Cray MPICH on Kestrel and warns against OpenMPI as the default. mpi4py documents rank/size, point-to-point, collective, and buffer communication APIs. Sources: `https://nrel.github.io/HPC/Documentation/Development/Programming-Environments/` and `https://mpi4py.readthedocs.io/en/stable/tutorial.html`; retrieved 2026-05-10.

**Method.** Use mpi4py when rank identity, collectives, or deterministic reductions are simpler than a task scheduler.

### 2.4 Experimental: Ray on SLURM

**Status:** Document and smoke-test only; disabled by default.

**Rationale.** Ray has official SLURM documentation and `ray symmetric-run` for Ray 2.49+, but no Kestrel-specific Ray documentation was found. Source: `https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm.html`; retrieved 2026-05-10.

**Method.** Keep Ray isolated behind an opt-in adapter controlled by `BSM_ENABLE_RAY_EXPERIMENTAL=1`.

## 3. Kestrel resource method

Use these defaults for generated Kestrel smoke/probe scripts:

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

**Verified external fact.** Kestrel partition names and GPU H100 partition names are documented by NLR. Source URL: `https://nrel.github.io/HPC/Documentation/Systems/Kestrel/Running/`; retrieved 2026-05-10.

## 4. Filesystem and spill method

**Verified external fact.** NLR documents `/projects`, `/scratch`, and `$TMPDIR`, and warns that `$TMPDIR` can consume RAM on CPU nodes without local disk. Source URL: `https://nrel.github.io/HPC/Documentation/Systems/Kestrel/Filesystems/`; retrieved 2026-05-10.

**Method requirement.** Use:

```text
/projects/bsm/.pixi                    Pixi installation state
/projects/bsm/.cache/pixi              Pixi package cache
/projects/bsm/<repo_or_run>/artifacts  durable manifests/final artifacts
/scratch/$USER/bsm_<run_id>/work       temporary distributed working set
/scratch/$USER/bsm_<run_id>/logs       SLURM logs and scheduler files
$TMPDIR/bsm_<run_id>/spill             local spill only if disk-backed
```

Implement `choose_spill_root()` to reject missing `$TMPDIR` and `tmpfs`/RAM-backed `$TMPDIR`, falling back to `/scratch/$USER/...`.

## 5. Failure and restart method

**Verified live fact.** User-provided Kestrel config reports `JobRequeue=0`, `PreemptMode=OFF`, `KillWait=30 sec`, and `JobAcctGatherFrequency=30`.

**Verified external fact.** Slurm documents `--signal=[{R|B}:]<signal>[@time]` and notes signals may arrive earlier than requested; Slurm also documents `--requeue`, but behavior depends on cluster configuration. Source URL: `https://slurm.schedmd.com/sbatch.html`; retrieved 2026-05-10.

**Engineering rule.** Do not rely on automatic requeue. Use idempotent task outputs:

```text
attempt output -> validation -> atomic promotion -> _SUCCESS marker
```

A rerun must skip completed shards and clean or overwrite only incomplete attempt directories.

## 6. Parquet/data layout method

**Verified external fact.** Apache Parquet recommends large row groups, commonly 512 MB to 1 GB, while PyArrow specifies `row_group_size` in rows. Sources: `https://parquet.apache.org/docs/file-format/configurations/` and `https://arrow.apache.org/docs/python/generated/pyarrow.parquet.write_table.html`; retrieved 2026-05-10.

**Method requirement.** Compute `row_group_size` from estimated bytes per row:

```python
target_row_group_bytes = 512 * 1024**2
estimated_bytes_per_row = n_columns * bytes_per_value
row_group_size = max(1, target_row_group_bytes // estimated_bytes_per_row)
```

Benchmark single ultra-wide Parquet files against feature-blocked layouts before standardizing.

## 7. Recommended implementation sequence

1. Add scheduler/resource config objects with `account=bsm` and Kestrel queue defaults.
1. Add script rendering tests for account, walltime, logs, partition, CPU, memory, and thread controls.
1. Implement filesystem/spill path selection.
1. Implement manifest-driven SLURM array execution with `_SUCCESS` semantics.
1. Add Dask local and Kestrel `SLURMCluster` smoke scripts.
1. Add MPI rank mapping and smoke scripts.
1. Add Ray experimental smoke scripts, disabled by default.
1. Add Parquet/spill/accounting benchmark harness.
1. Add documentation warning that production values must be checked with live `scontrol`, `sacctmgr`, and `lfs` probes.
