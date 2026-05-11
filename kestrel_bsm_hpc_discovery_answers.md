# Kestrel/NLR HPC discovery answers and implementation constraints for BSM Slurm/Pixi workflow

Generated for the `bsm` allocation/project handle. The accompanying scripts in this bundle are configured to use:

- Slurm account/allocation: `bsm`
- CPU discovery partition: `debug`
- Optional GPU discovery partition: `gpu-h100s`
- Project software/env root: `/projects/bsm`
- Project Pixi root: `/projects/bsm/.pixi`
- Project Pixi cache: `/projects/bsm/.cache/pixi`
- Scratch run root: `/scratch/$USER/bsm_hpc_discovery`

Primary public sources used:

- NLR Kestrel running/jobs and partitions: <https://natlabrockies.github.io/HPC/Documentation/Systems/Kestrel/Running/>
- NLR Kestrel filesystems: <https://natlabrockies.github.io/HPC/Documentation/Systems/Kestrel/Filesystems/>
- NLR Lustre best practices: <https://natlabrockies.github.io/HPC/Documentation/Systems/Kestrel/Filesystems/lustre/>
- NLR user basics: <https://www.nlr.gov/hpc/user-basics>
- NLR Slurm job arrays: <https://natlabrockies.github.io/HPC/Documentation/Slurm/job_arrays/>
- NLR Dask documentation: <https://natlabrockies.github.io/HPC/Documentation/Development/Languages/Python/dask/>
- NLR Apptainer documentation: <https://natlabrockies.github.io/HPC/Documentation/Development/Containers/apptainer/>
- NLR data security policy: <https://www.nlr.gov/hpc/data-security-policy>
- NLR appropriate use policy: <https://www.nlr.gov/hpc/appropriate-use-policy>
- Apache Parquet file-format configuration guidance: <https://parquet.apache.org/docs/file-format/configurations/>

## Run order

From the directory containing the updated scripts:

```bash
chmod +x ./*.sh

# 1. Capture login/control-plane state.
./discovery_collect_env.sh artifacts/hpc_discovery

# 2. Install Pixi and, if pixi.toml exists in the current repo, install the env.
./install_pixi_project_env.sh artifacts/hpc_discovery
source artifacts/hpc_discovery/pixi_kestrel_env.sh

# 3. Run the CPU debug Slurm topology probe.
./discovery_collect_slurm_job.sh artifacts/hpc_discovery

# 4. Run Python/Pixi spill + Parquet probes on CPU debug.
./discovery_submit_python_probes.sh artifacts/hpc_discovery

# 5. Optional GPU probe. This skips cleanly if gpu-h100s is not visible.
./discovery_gpu_probe.sh artifacts/hpc_discovery

# Or run the suite wrapper:
./run_kestrel_discovery_suite.sh artifacts/hpc_discovery
```

## User-provided live Slurm configuration

You reported:

```text
scontrol --version
slurm 25.05.5

AccountingStorageType   = accounting_storage/slurmdbd
CompleteWait            = 0 sec
JobAcctGatherFrequency  = 30
JobAcctGatherType       = jobacct_gather/linux
JobRequeue              = 0
KillWait                = 30 sec
MaxArraySize            = 11000
MaxBatchRequeue         = 5
MaxJobCount             = 65000
MinJobAge               = 300 sec
PreemptMode             = OFF
PreemptType             = (null)
PriorityDecayHalfLife   = 7-00:00:00
PriorityType            = priority/multifactor
SchedulerType           = sched/backfill
SelectType              = select/cons_tres
SelectTypeParameters    = CR_CORE_MEMORY
SlurmctldHost[0]        = kmgmt1(10.150.0.23)
SlurmctldHost[1]        = kmgmt2(10.150.0.2)
SlurmctldPort           = 6817
TaskPlugin              = task/cgroup,task/affinity
```

Operational interpretation:

- Slurm is modern (`25.05.5`) and uses `select/cons_tres` with `CR_CORE_MEMORY`, so CPU and memory requests are consumable resources and should be requested explicitly.
- `task/cgroup,task/affinity` are enabled, so CPU and memory containment/binding are enforced by Slurm/cgroups.
- Job accounting is enabled through `slurmdbd`; job-accounting sampling frequency is 30 seconds.
- Preemption is off at the Slurm controller level (`PreemptMode=OFF`, `PreemptType=(null)`).
- Automatic job requeue is not globally enabled (`JobRequeue=0`). Do not design the workflow assuming `--requeue` works unless a project/QOS/partition test proves otherwise.
- `KillWait=30 sec` means after Slurm sends its termination signal, the configured grace period before final kill is 30 seconds. Checkpointing should happen well before walltime, not only during final termination.
- Max array size is 11,000. Use `%N` throttling for concurrency, e.g. `--array=0-10999%100`.
- `MaxJobCount=65000` is a controller-level maximum, not necessarily a per-user/project submission limit. Verify user/project limits with `sacctmgr`.

## 1. Scheduler + policy

### Known values

| Item                             | Value / policy                                                                     |
| -------------------------------- | ---------------------------------------------------------------------------------- |
| Scheduler                        | Slurm                                                                              |
| Slurm version                    | `25.05.5` from live probe                                                          |
| Accounting                       | `accounting_storage/slurmdbd`; `jobacct_gather/linux`; sample frequency 30 seconds |
| Required account                 | Use `--account=bsm` / `-A bsm`                                                     |
| CPU discovery partition          | `debug`, per user request                                                          |
| Optional GPU discovery partition | `gpu-h100s`, per user request for short GPU queue                                  |
| Max array size                   | `MaxArraySize=11000`                                                               |
| Controller max job count         | `MaxJobCount=65000`; not a per-user limit                                          |
| Preemption                       | `PreemptMode=OFF`; `PreemptType=(null)`                                            |
| Requeue                          | `JobRequeue=0`; do not assume `--requeue`                                          |
| Kill grace                       | `KillWait=30 sec`                                                                  |
| Scheduler                        | `sched/backfill`                                                                   |
| Resource selection               | `select/cons_tres`, `CR_CORE_MEMORY`                                               |
| Task plugins                     | `task/cgroup,task/affinity`                                                        |

### Partition limits from public Kestrel docs

Important entries for this workflow:

| Partition            | Use                             | Key limits / placement                                                                 |
| -------------------- | ------------------------------- | -------------------------------------------------------------------------------------- |
| `debug`              | troubleshooting and development | max walltime 4 hours; 1 job and max 2 nodes per user; 4 GPUs per user where applicable |
| `short`              | short CPU jobs                  | walltime `<= 4:00:00`                                                                  |
| `standard`           | normal CPU jobs                 | walltime `<= 2-00`; max 1050 nodes per user                                            |
| `long`               | long CPU jobs                   | walltime `> 2-00` and `<= 10-00`; max 215 nodes per user                               |
| `shared`             | shared CPU nodes                | max 2 days; default about 1 GB/core unless `--mem` or `--mem-per-cpu` is requested     |
| `nvme`               | CPU nodes with local NVMe       | walltime `<= 2-00`; 256 nodes with 1.7 TB local NVMe; max 128 nodes per user           |
| `medmem`             | 1 TB nodes                      | walltime up to 10 days; 32 nodes per user                                              |
| `bigmem` / `bigmeml` | 2 TB memory + local NVMe        | `bigmem` up to 2 days; `bigmeml` longer, up to 10 days                                 |
| `gpu-h100s`          | short H100 GPU jobs             | walltime `<= 4:00:00`; `1 <= --gpus <= 4`                                              |
| `gpu-h100`           | regular H100 GPU jobs           | walltime `<= 2-00`; `1 <= --gpus <= 4`                                                 |
| `gpu-h100l`          | long H100 GPU jobs              | walltime `> 2-00`; `1 <= --gpus <= 4`                                                  |

Kestrel docs recommend job arrays on the `shared` partition, but this discovery bundle uses `debug` for CPU probes because that was explicitly requested. For production arrays, reconsider `shared`, `short`, `standard`, or `nvme` depending on runtime and I/O profile.

### Commands to verify live policy

```bash
scontrol --version
scontrol show config | egrep -i \
  'SlurmctldHost|SlurmctldPort|SchedulerType|SelectType|TaskPlugin|AccountingStorageType|JobAcctGather|JobAcctGatherFrequency|MaxArraySize|MaxJobCount|KillWait|CompleteWait|MinJobAge|Preempt|Requeue|Priority|PriorityDecayHalfLife'

sinfo -s
scontrol show partition -o
scontrol show partition debug
scontrol show partition gpu-h100s
sacctmgr show assoc user="$USER" account=bsm -P
sacctmgr show qos -P
sprio -u "$USER" -l
```

### Dependency patterns

Use standard Slurm dependencies:

```bash
stage1="$(sbatch --parsable -A bsm -p shared -t 01:00:00 --array=0-999%50 stage1_array.sbatch)"

sbatch --parsable \
  -A bsm \
  -p shared \
  -t 00:30:00 \
  --dependency=afterok:"$stage1" \
  stage2_reduce.sbatch

sbatch --parsable \
  -A bsm \
  -p shared \
  -t 00:30:00 \
  --dependency=afterany:"$stage1" \
  stage1_cleanup_or_diagnostics.sbatch
```

Recommended workflow dependencies:

- `afterok`: downstream reduce/aggregation stages that require all upstream shards to succeed.
- `afterany`: cleanup, telemetry collection, or failure-summary jobs.
- `afternotok`: failure handler jobs.
- `singleton`: named singleton jobs where only one instance should run.

## 2. Node topology

### Public hardware information

Kestrel CPU nodes:

- Standard CPU nodes: 104 cores and about 240 GB usable RAM.
- 256 standard CPU nodes have 1.7 TB local NVMe and are exposed through `nvme`.
- 32 single-NIC `medmem` nodes have 1 TB RAM.
- 10 `bigmem` nodes have 2 TB RAM and 5.6 TB local NVMe.
- Some high-bandwidth CPU nodes have dual NICs and require multi-node jobs.

Kestrel GPU nodes:

- 156 GPU nodes.
- 4 NVIDIA H100 GPUs per node, 80 GB per GPU.
- Dual-socket AMD Genoa 64-core CPUs, 128 CPU cores total.
- GPU nodes have local NVMe; common classes include about 3.4 TB and 14 TB local disk.

### Commands to verify topology inside a job

```bash
hostname
cat /etc/redhat-release || true
lscpu
numactl --hardware || true
free -h
nproc

echo "SLURM_CPUS_ON_NODE=${SLURM_CPUS_ON_NODE:-}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-}"
echo "SLURM_MEM_PER_NODE=${SLURM_MEM_PER_NODE:-}"
echo "SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION:-}"
echo "TMPDIR=${TMPDIR:-}"

df -hT "${TMPDIR:-/tmp}" /scratch /projects /projects/bsm 2>/dev/null || true
mount | egrep 'lustre|scratch|projects|tmp|nvme|kfs' || true
```

For GPU jobs:

```bash
nvidia-smi
nvidia-smi --query-gpu=index,name,uuid,driver_version,cuda_version,memory.total,memory.free --format=csv
```

### Binding recommendations

For single-process threaded Python jobs:

```bash
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

srun --cpu-bind=cores pixi run python script.py
```

For task-parallel array jobs, prefer one process per Slurm task and set BLAS/OpenMP thread counts to 1 unless each task is intentionally multi-threaded.

## 3. Storage + I/O

### Filesystem roles

| Filesystem                         | Role                                                               | Notes                                                                                  |
| ---------------------------------- | ------------------------------------------------------------------ | -------------------------------------------------------------------------------------- |
| `/home/$USER`                      | small config, shell files, source snippets                         | 50 GB quota; not for job data                                                          |
| `/projects/bsm`                    | durable project data, Pixi env/cache, checkpoints, final artifacts | ProjectFS; quota is allocation-specific                                                |
| `/scratch/$USER`                   | high-throughput temporary job data                                 | ScratchFS; data subject to 28-day inactivity purge                                     |
| `$TMPDIR` / `/tmp/scratch/<JOBID>` | node-local temporary data when local disk exists                   | cleaned after job; not shared across nodes; on non-local-disk nodes it can consume RAM |

Public NLR filesystem values:

- ProjectFS: `/projects`, 68 PB capacity, approximately 200 GB/s IOR bandwidth.
- ScratchFS: `/scratch`, 27 PB capacity, approximately 354 GB/s IOR bandwidth.
- Standard CPU local NVMe nodes: 1.7 TB local NVMe on 256 nodes.
- Bigmem local NVMe: 5.6 TB.
- GPU nodes: local NVMe available, typically 3.4 TB or larger.

### I/O anti-patterns to avoid

Do not:

- keep important files only in `/scratch`;
- create many small files in one flat directory;
- run `ls -l` on huge directories;
- use Python `os.walk` or `os.scandir` at large scale on Lustre;
- have many processes hammer the same file;
- use broad wildcard copies such as `cp * target/` for very large file counts.

Do:

- store durable state in `/projects/bsm`;
- run high-I/O jobs from `/scratch/$USER`;
- use `$TMPDIR` for many temporary files only when the job has real node-local disk;
- use `lfs find`, `lfs quota`, `lfs getstripe`, and `lfs setstripe` where appropriate;
- compress/tar large small-file collections before moving them to durable storage.

### Verification commands

```bash
lfs project -d /projects/bsm
PROJECT_ID="$(lfs project -d /projects/bsm | awk 'NR==1 {print $1}')"
lfs quota -hp "$PROJECT_ID" /projects/bsm
lfs quota -h -u "$USER" /scratch
lfs quota -uh "$USER" "/home/$USER"
lfs df -h /scratch /projects/bsm
lfs getstripe "/scratch/$USER"
lfs getstripe /projects/bsm
```

## 4. Python execution model

### Recommended approach for this workflow

Use Pixi, but keep Pixi itself and the package/cache state under `/projects/bsm`, not `/home`:

```bash
export NLR_ACCOUNT=bsm
export KESTREL_PROJECT_ROOT=/projects/bsm
export PIXI_HOME=/projects/bsm/.pixi
export PIXI_CACHE_DIR=/projects/bsm/.cache/pixi
export XDG_CACHE_HOME=/projects/bsm/.cache
export CONDA_PKGS_DIRS=/projects/bsm/.conda/pkgs
export PATH="$PIXI_HOME/bin:$PATH"

./install_pixi_project_env.sh artifacts/hpc_discovery
source artifacts/hpc_discovery/pixi_kestrel_env.sh
```

Do package resolution and environment installation before production jobs. Production jobs should run locked environments, not dynamic package installs.

### Dask / distributed Python

NLR documents Dask on Kestrel, including Dask Jobqueue and Dask-MPI. The NLR Dask page notes observed `distributed.comm.core.CommClosedError` issues with `dask-mpi` and encourages users experiencing those issues to use `dask-jobqueue` instead.

For this reduced-form workflow, the safest initial distributed execution model is:

1. Slurm job arrays for embarrassingly parallel shard/block work.
1. Dependency-driven reduce/merge jobs.
1. Dask Jobqueue only if dynamic scheduling, worker reuse, or distributed dataframe operations become necessary.
1. Avoid Ray until a live Kestrel support/probe confirms it is available and acceptable for your project.

### Containers

NLR documents Apptainer as the supported/deprecated-Singularity replacement. Docker itself is not the direct runtime model on HPC. If containers become necessary, use Apptainer images stored under `/projects/bsm` or `/scratch/$USER`, and use `--nv` for GPU containers.

GPU host values from NLR documentation:

- Kestrel H100 partition GPU driver: `550.54.15`
- CUDA version from `nvidia-smi`: `12.4`

Verify live:

```bash
module avail apptainer
module load apptainer
apptainer --version
nvidia-smi
```

## 5. Monitoring + observability

Live config says Slurm accounting is enabled with `JobAcctGatherFrequency=30`, so `sacct` should be useful for completed jobs, with about 30-second sampling granularity for gathered job metrics.

Recommended commands:

```bash
squeue -u "$USER" \
  -o "%.18i %.9P %.8q %.12a %.8T %.10M %.10l %.6D %R"

sprio -u "$USER" -l

sacct -j "$JOBID" \
  --format=JobID,JobName%24,Partition,Account,State,ExitCode,Elapsed,Timelimit,AllocTRES,ReqTRES,MaxRSS,AveRSS,MaxVMSize,MaxDiskRead,MaxDiskWrite,NodeList%40 \
  -P

sstat -j "$JOBID.batch" \
  --fields=JobID,AveCPU,AveRSS,MaxRSS,MaxVMSize,AveDiskRead,AveDiskWrite,MaxDiskRead,MaxDiskWrite \
  -P
```

Caveat: `MaxDiskRead` and `MaxDiskWrite` can be unavailable or incomplete depending on Slurm accounting and kernel support. The discovery scripts record them but should not treat missing disk metrics as a fatal failure.

Log strategy:

```text
/projects/bsm/<workflow>/runs/<run_id>/logs/      durable Slurm logs and summaries
/scratch/$USER/bsm_hpc_discovery/<run_id>/       large temporary probe/job output
```

Keep Slurm stdout/stderr paths unique with `%j` for job ID and `%A_%a` for arrays.

## 6. Reliability + restart

Because `PreemptMode=OFF`, preemption is not the main restart driver under the current Slurm configuration. The major failure modes are more likely to be walltime, node failure, memory limit, bad filesystem assumptions, or application exceptions.

Because `JobRequeue=0`, build restartability into the workflow rather than relying on automatic Slurm requeue.

Recommended idempotent layout:

```text
inputs/
  shard_manifest.parquet or shard_manifest.jsonl

work/
  task_id=<id>/
    attempt=<jobid>/
      partial outputs
      telemetry.json

outputs/
  task_id=<id>/
    data.parquet
    schema.json
    checksum.json
    _SUCCESS.json

logs/
  slurm-%A_%a.out
  slurm-%A_%a.err
```

Rules:

- write partial outputs to attempt-specific directories;
- validate row counts, schema, and checksums;
- atomically promote completed outputs into `outputs/task_id=<id>/`;
- write `_SUCCESS.json` last;
- skip tasks that already have a valid success marker;
- use `afterok` for reducers and `afterany` for diagnostics;
- checkpoint before walltime based on internal timers rather than relying on Slurm's final `KillWait=30 sec` grace interval.

Optional signal pattern, if desired:

```bash
#SBATCH --signal=B:USR1@300

trap 'echo "Caught USR1; writing checkpoint"; python checkpoint.py; exit 99' USR1
```

Do not add `#SBATCH --requeue` until a live test proves the account/QOS/partition honors it.

## 7. Security/compliance constraints

NLR policy states that HPC systems are research systems with a low FIPS-199/NIST 800-53 security baseline. NLR also states that network and storage systems provide no explicit encryption and that users are responsible for protecting their data. NLR data-security policy says export-controlled data may not be stored or processed on these systems.

Implications:

- Do not put export-controlled, classified, PII, HIPAA, or other controlled/sensitive data into this workflow unless NLR policy and project authorization explicitly permit it.
- Do not assume encryption at rest or in transit for `/projects`, `/scratch`, or node-local scratch.
- Treat `$TMPDIR` as ephemeral and cleaned after the job.
- Store durable audit manifests and final results in `/projects/bsm`.
- Keep `/scratch` for temporary high-throughput job data only.

## 8. Scaling guidance for huge feature matrices and Parquet

I found no Kestrel-specific Parquet row-group or file-count rule. Use Kestrel's Lustre guidance plus Apache Parquet layout guidance.

Starting recommendations:

- Avoid one monolithic ultra-wide Parquet table if the matrix may reach millions of columns.
- Prefer feature-blocked and/or output-blocked shards.
- Keep file counts controlled; avoid millions of small files and huge flat directories.
- Target row groups around 512 MiB to 1 GiB uncompressed where feasible.
- Remember PyArrow `row_group_size` is rows, not bytes. The updated benchmark estimates rows per group from target MiB and matrix width.
- Benchmark full reads and column-projected reads separately.
- Record schema, row count, row groups, file size, write time, full-read time, projection-read time, Slurm memory, and Slurm disk metrics.

For a float32 table, a rough row-group estimate is:

```python
row_group_rows = max(1, target_row_group_bytes // (n_columns * 4))
```

For millions of columns, this becomes very small. That is a warning that the physical layout should likely be redesigned around feature blocks rather than a single table with all features as columns.

## Updated script inventory

| Script                              | Purpose                                                                                                                                                       |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `hpc_kestrel_config.sh`             | Shared `bsm` account, partition, Pixi, project, scratch, and threading defaults                                                                               |
| `install_pixi_project_env.sh`       | Installs Pixi under `/projects/bsm/.pixi`, configures caches under `/projects/bsm`, and runs `pixi install` / `pixi install --locked` if a `pixi.toml` exists |
| `discovery_collect_env.sh`          | Collects login/control-plane Slurm, storage, module, Pixi, and environment details                                                                            |
| `discovery_collect_slurm_job.sh`    | Submits a CPU debug Slurm topology/accounting probe using `--account=bsm --partition=debug`                                                                   |
| `discovery_submit_python_probes.sh` | Submits Python/Pixi spill and Parquet probes to CPU `debug` using `--account=bsm`                                                                             |
| `discovery_chunk_spill_probe.sh`    | Runs a NumPy chunk spill/read benchmark; uses real `$TMPDIR` only when it is not RAM-backed, otherwise falls back to `/scratch/$USER/bsm_hpc_discovery`       |
| `discovery_parquet_bench.sh`        | Runs a PyArrow Parquet write/full-read/projection-read benchmark across `/scratch`, `/projects/bsm`, and real `$TMPDIR` when available                        |
| `discovery_gpu_probe.sh`            | Optionally submits a short H100 GPU probe to `gpu-h100s` if the partition is visible                                                                          |
| `run_kestrel_discovery_suite.sh`    | Convenience wrapper for the full discovery sequence                                                                                                           |

## Final implementation constraints summary for engineering

1. Use `--account=bsm` on every Slurm job.
1. Use `debug` only for small discovery/probe jobs; production arrays should likely use `shared`, `short`, `standard`, or `nvme` depending on runtime and I/O.
1. Use `gpu-h100s` for short GPU probes; request `--gpus=1` unless more are needed.
1. Do not rely on preemption/requeue handling: `PreemptMode=OFF` and `JobRequeue=0` in the observed config.
1. Use Slurm arrays up to the observed `MaxArraySize=11000`, but throttle concurrency with `%N` and verify account/QOS submit limits.
1. Use `/projects/bsm` for Pixi installs, caches, durable checkpoints, final outputs, and logs.
1. Use `/scratch/$USER` for high-throughput temporary distributed job data; treat it as purgeable.
1. Use `$TMPDIR` for spill only when the node has real local disk; otherwise it may consume RAM.
1. Avoid Lustre metadata storms: no huge flat small-file directories, no `os.walk`/`os.scandir` at scale, no `ls -l` on huge directories.
1. Make every shard job idempotent: attempt directories, validation metadata, atomic promotion, and `_SUCCESS.json` markers.
1. Set Python/BLAS thread counts from `SLURM_CPUS_PER_TASK`.
1. Benchmark `/scratch`, `/projects/bsm`, and real `$TMPDIR` separately before choosing a production spill/data layout.
1. Treat ultra-wide Parquet as risky; use feature-blocked/output-blocked shards if columns approach millions.
1. Do not assume encryption or controlled-data suitability on NLR HPC storage.
