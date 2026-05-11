# Kestrel SLURM Scheduler Interfaces API Docs

**Purpose:** Current scheduler syntax and Kestrel-specific constraints for local implementation agents.\
**Default account:** `bsm`.\
**External retrieval date:** 2026-05-10.\
**Live Kestrel config source:** user-provided terminal output, 2026-05-10.

## 1. Required submission fields

**Verified external fact.** NLR Kestrel examples use SLURM batch submission with `--account=<project-handle>` and walltime through `--time/-t`. Treat both as required in generated scripts. Source URL: `https://nrel.github.io/HPC/Documentation/Slurm/batch_jobs/`; retrieved 2026-05-10.

Minimum batch header:

```bash
#!/usr/bin/env bash
#SBATCH --account=bsm
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
```

## 2. CPU interactive debug allocation

**Verified external fact.** NLR documents `salloc` for interactive jobs and debug-queue usage. Source URL: `https://nrel.github.io/HPC/Documentation/Slurm/interactive_jobs/`; retrieved 2026-05-10.

```bash
salloc -A bsm -p debug -t 00:30:00 \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=4 \
  --mem=8G
```

Then start a shell on the allocation if needed:

```bash
srun --pty bash -l
```

## 3. CPU batch debug probe

```bash
mkdir -p logs

sbatch --parsable \
  -A bsm \
  -p debug \
  -t 00:30:00 \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=4 \
  --mem=8G \
  --output=logs/cpu-debug-%j.out \
  --error=logs/cpu-debug-%j.err \
  scripts/kestrel_cpu_probe.sbatch
```

## 4. Short GPU probe

**Verified external fact.** Kestrel documents H100 GPU partitions including `gpu-h100`, `gpu-h100s`, and `gpu-h100l`; GPU jobs request GPUs with `--gpus=<quantity>`. Source URL: `https://nrel.github.io/HPC/Documentation/Systems/Kestrel/Running/`; retrieved 2026-05-10.

```bash
mkdir -p logs

sbatch --parsable \
  -A bsm \
  -p gpu-h100s \
  -t 00:30:00 \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=16 \
  --mem=64G \
  --gpus=1 \
  --output=logs/gpu-short-%j.out \
  --error=logs/gpu-short-%j.err \
  scripts/kestrel_gpu_probe.sbatch
```

Inside GPU job:

```bash
nvidia-smi
python - <<'PY'
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda_available:", torch.cuda.is_available())
except Exception as exc:
    print("torch_probe_error:", repr(exc))
PY
```

## 5. SLURM arrays

**Verified external fact.** NLR documents `--array=<ARRAY_VALS>`, ranges, lists, steps, and throttling with `%N`; NLR recommends job arrays on the `shared` partition. Source URL: `https://nrel.github.io/HPC/Documentation/Slurm/job_arrays/`; retrieved 2026-05-10.

**Verified live fact.** User probe reports `MaxArraySize=11000`.

**Version-sensitive fact.** Slurm official docs define the maximum array index as `MaxArraySize - 1`. Source URL: `https://slurm.schedmd.com/job_array.html`; retrieved 2026-05-10.

```bash
sbatch --parsable \
  -A bsm \
  -p shared \
  -t 02:00:00 \
  --array=0-999%50 \
  --cpus-per-task=4 \
  --mem=16G \
  --output=logs/array-%A_%a.out \
  --error=logs/array-%A_%a.err \
  scripts/run_shard_array.sbatch
```

Inside the array script:

```bash
echo "array_job_id=${SLURM_ARRAY_JOB_ID}"
echo "array_task_id=${SLURM_ARRAY_TASK_ID}"

pixi run python -m bsm_rfm.distributed.run_shard \
  --manifest manifests/shards.parquet \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --run-id "${SLURM_ARRAY_JOB_ID}"
```

## 6. Dependencies

**Verified external fact.** Slurm supports dependency types including `after`, `afterany`, `afterok`, `afternotok`, and `singleton`. Source URL: `https://slurm.schedmd.com/sbatch.html`; retrieved 2026-05-10.

```bash
array_job_id="$(
  sbatch --parsable \
    -A bsm -p shared -t 02:00:00 \
    --array=0-999%50 \
    scripts/run_shard_array.sbatch
)"

reduce_job_id="$(
  sbatch --parsable \
    -A bsm -p debug -t 00:30:00 \
    --dependency=afterok:"${array_job_id}" \
    scripts/reduce_shards.sbatch
)"

echo "array_job_id=${array_job_id}"
echo "reduce_job_id=${reduce_job_id}"
```

Use `afterok` for stages that require successful upstream completion. Use `afterany` for cleanup or diagnostics.

## 7. Signals, requeue, and failure behavior

**Verified live fact.** User probe reports `JobRequeue=0`, `PreemptMode=OFF`, `KillWait=30 sec`, and `MaxBatchRequeue=5`.

**Verified external fact.** Slurm documents `--signal=[{R|B}:]<sig_num>[@sig_time]` and notes the signal may be sent up to 60 seconds earlier than requested. Slurm documents `--requeue`, but behavior depends on cluster configuration. Source URL: `https://slurm.schedmd.com/sbatch.html`; retrieved 2026-05-10.

Conservative signal option:

```bash
#SBATCH --signal=B:USR1@300
```

Do not rely on automatic requeue. Implement idempotent rerun semantics.

## 8. Monitoring and accounting

**Verified external fact.** NLR documents `squeue`, `scontrol`, `scancel`, `sinfo`, `sacct`, and `sprio`. Source URL: `https://nrel.github.io/HPC/Documentation/Slurm/monitor_and_control/`; retrieved 2026-05-10.

```bash
squeue -u "$USER" \
  -o "%.18i %.9P %.8q %.12a %.8T %.10M %.10l %.6D %R"

sacct -j "$JOBID" \
  --format=JobID,JobName%24,State,ExitCode,Elapsed,Timelimit,AllocTRES,ReqTRES,MaxRSS,AveRSS,MaxVMSize,MaxDiskRead,MaxDiskWrite,NodeList%40 \
  -P

scontrol show job "$JOBID"
sprio -u "$USER" -l
```

**Verified live fact.** Kestrel accounting collection frequency is `JobAcctGatherFrequency=30`; expect accounting values at approximately 30-second granularity.

## 9. Environment bootstrap

**Verified external fact.** NLR documents `/projects` for durable project data/applications, `/scratch` for temporary high-throughput data, and `$TMPDIR` for node-local storage when present. Source URL: `https://nrel.github.io/HPC/Documentation/Systems/Kestrel/Filesystems/`; retrieved 2026-05-10.

```bash
export BSM_PROJECT_ROOT="/projects/bsm"
export PIXI_HOME="/projects/bsm/.pixi"
export PIXI_CACHE_DIR="/projects/bsm/.cache/pixi"
export CONDA_PKGS_DIRS="/projects/bsm/.conda/pkgs"
export XDG_CACHE_HOME="/projects/bsm/.cache"
export BSM_SCRATCH_ROOT="/scratch/$USER/bsm_runs"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
```

## 10. Live verification commands

```bash
scontrol --version
scontrol show config | egrep -i \
  'SchedulerType|SelectType|SelectTypeParameters|TaskPlugin|AccountingStorageType|JobAcctGather|JobAcctGatherFrequency|MaxArraySize|MaxJobCount|KillWait|CompleteWait|MinJobAge|Preempt|Requeue|Priority'
scontrol show partition -o
sacctmgr show assoc user="$USER" -P || true
sacctmgr show qos -P || true
lfs quota -h -u "$USER" /scratch || true
lfs quota -h -p bsm /projects || true
lfs df -h /scratch /projects || true
```
