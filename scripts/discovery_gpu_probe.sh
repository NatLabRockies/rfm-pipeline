#!/usr/bin/env bash
# Kestrel/NLR GPU debug topology probe for the BSM allocation.
#
# Defaults:
#   account:      bsm
#   partition:    debug  (Kestrel GPU debug jobs use --partition=debug plus --gpus)
#   walltime:     00:30:00
#   gpus:         1
#   cpus/task:    16
#   memory:       64G
#
# Usage:
#   ./discovery_gpu_debug_probe.sh [out_dir]
#
# Optional overrides:
#   NLR_ACCOUNT=bsm
#   KESTREL_GPU_DEBUG_PARTITION=debug
#   GPU_DEBUG_TIME=00:30:00
#   GPU_DEBUG_GPUS=1
#   GPU_DEBUG_CPUS_PER_TASK=16
#   GPU_DEBUG_MEM=64G
#   MONITOR_INTERVAL_SECONDS=10
#   MONITOR_TAIL_LINES=12

set -euo pipefail

NLR_ACCOUNT="${NLR_ACCOUNT:-bsm}"
KESTREL_GPU_DEBUG_PARTITION="${KESTREL_GPU_DEBUG_PARTITION:-debug}"
GPU_DEBUG_TIME="${GPU_DEBUG_TIME:-00:30:00}"
GPU_DEBUG_GPUS="${GPU_DEBUG_GPUS:-1}"
GPU_DEBUG_CPUS_PER_TASK="${GPU_DEBUG_CPUS_PER_TASK:-16}"
GPU_DEBUG_MEM="${GPU_DEBUG_MEM:-64G}"
MONITOR_INTERVAL_SECONDS="${MONITOR_INTERVAL_SECONDS:-10}"
MONITOR_TAIL_LINES="${MONITOR_TAIL_LINES:-12}"

OUT_DIR="${1:-artifacts/hpc_discovery}"
OUT_DIR="$(python3 - <<PY
from pathlib import Path
print(Path("${OUT_DIR}").expanduser().resolve())
PY
)"
mkdir -p "$OUT_DIR"

JOB_SCRIPT="$OUT_DIR/gpu_debug_probe_job.sbatch"

cat > "$JOB_SCRIPT" <<'SBATCH'
#!/usr/bin/env bash
#SBATCH --job-name=bsm-gpu-debug-probe
#SBATCH --account=bsm
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpus=1

set -euo pipefail

echo "step=job_started timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "job_id=${SLURM_JOB_ID:-}"
echo "job_name=${SLURM_JOB_NAME:-}"
echo "partition=${SLURM_JOB_PARTITION:-}"
echo "account=${SLURM_JOB_ACCOUNT:-}"
echo "node=${SLURMD_NODENAME:-}"
echo "nodelist=${SLURM_JOB_NODELIST:-}"
echo "gpus_requested=${SLURM_GPUS:-${SLURM_GPUS_ON_NODE:-}}"
echo "gpus_on_node=${SLURM_GPUS_ON_NODE:-}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
echo "cpus_per_task=${SLURM_CPUS_PER_TASK:-}"
echo "cpus_on_node=${SLURM_CPUS_ON_NODE:-}"
echo "mem_per_node=${SLURM_MEM_PER_NODE:-}"
echo "tmpdir=${TMPDIR:-}"
echo "submit_dir=${SLURM_SUBMIT_DIR:-}"
echo

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

export PIXI_HOME="${PIXI_HOME:-/projects/bsm/.pixi}"
export PIXI_CACHE_DIR="${PIXI_CACHE_DIR:-/projects/bsm/.cache/pixi}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-/projects/bsm/.conda/pkgs}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/projects/bsm/.cache}"
export PATH="${PIXI_HOME}/bin:${PATH}"

echo "step=scontrol_show_job"
scontrol show job "${SLURM_JOB_ID}" || true
echo

echo "step=host_topology"
hostname
uname -a
cat /etc/redhat-release || true
lscpu || true
numactl --hardware || true
free -h || true
echo

echo "step=gpu_topology"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
  echo
  nvidia-smi topo -m || true
  echo
  nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,memory.total,memory.used,driver_version,compute_cap --format=csv || true
else
  echo "warning=nvidia-smi_not_found"
fi
echo

echo "step=filesystem_probe"
df -hT "${TMPDIR:-/tmp}" /scratch /projects /projects/bsm "/scratch/${USER}" 2>/dev/null || true
df -ih "${TMPDIR:-/tmp}" /scratch /projects /projects/bsm "/scratch/${USER}" 2>/dev/null || true
mount | egrep 'lustre|scratch|projects|tmp|nvme|kfs' || true
echo

echo "step=python_gpu_import_probe"
python3 - <<'PY' || true
from __future__ import annotations

import importlib.util
import json
import os
import platform
import subprocess
import tempfile
from pathlib import Path

payload = {
    "platform": platform.platform(),
    "python": platform.python_version(),
    "cpu_count": os.cpu_count(),
    "tmpdir_env": os.environ.get("TMPDIR"),
    "tempfile_gettempdir": tempfile.gettempdir(),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "slurm_job_partition": os.environ.get("SLURM_JOB_PARTITION"),
    "slurm_gpus": os.environ.get("SLURM_GPUS"),
    "slurm_gpus_on_node": os.environ.get("SLURM_GPUS_ON_NODE"),
    "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
    "torch_available": importlib.util.find_spec("torch") is not None,
    "cupy_available": importlib.util.find_spec("cupy") is not None,
    "numba_available": importlib.util.find_spec("numba") is not None,
}
print(json.dumps(payload, indent=2, sort_keys=True))

if importlib.util.find_spec("torch") is not None:
    import torch
    print(json.dumps({
        "torch_version": torch.__version__,
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_device_count": torch.cuda.device_count(),
        "torch_cuda_devices": [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ],
    }, indent=2, sort_keys=True))

if importlib.util.find_spec("numba") is not None:
    try:
        from numba import cuda
        print(json.dumps({
            "numba_cuda_available": cuda.is_available(),
            "numba_gpus": [str(gpu) for gpu in cuda.gpus],
        }, indent=2, sort_keys=True))
    except Exception as exc:
        print(json.dumps({"numba_cuda_probe_error": repr(exc)}, indent=2))
PY
echo

echo "step=srun_gpu_visibility_probe"
srun --ntasks=1 --gpus="${SLURM_GPUS:-1}" bash -lc '
  echo "srun_host=$(hostname)"
  echo "srun_cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.used --format=csv
  fi
' || true
echo

echo "step=job_completed timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
SBATCH

echo "step=submit_gpu_debug_probe account=$NLR_ACCOUNT partition=$KESTREL_GPU_DEBUG_PARTITION gpus=$GPU_DEBUG_GPUS cpus_per_task=$GPU_DEBUG_CPUS_PER_TASK mem=$GPU_DEBUG_MEM time=$GPU_DEBUG_TIME out_dir=$OUT_DIR"

JOBID="$(
  sbatch --parsable \
    --account="$NLR_ACCOUNT" \
    --partition="$KESTREL_GPU_DEBUG_PARTITION" \
    --time="$GPU_DEBUG_TIME" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="$GPU_DEBUG_CPUS_PER_TASK" \
    --mem="$GPU_DEBUG_MEM" \
    --gpus="$GPU_DEBUG_GPUS" \
    --output="$OUT_DIR/gpu_debug_probe_%j.out" \
    --error="$OUT_DIR/gpu_debug_probe_%j.err" \
    "$JOB_SCRIPT"
)"

echo "$JOBID" > "$OUT_DIR/gpu_debug_probe_jobid.txt"
echo "status=SUBMITTED job_id=$JOBID partition=$KESTREL_GPU_DEBUG_PARTITION account=$NLR_ACCOUNT"

OUT_FILE="$OUT_DIR/gpu_debug_probe_${JOBID}.out"
ERR_FILE="$OUT_DIR/gpu_debug_probe_${JOBID}.err"

last_state=""
last_reason=""
reported_running=0

while true; do
  queue_line="$(
    squeue -h -j "$JOBID" \
      -o "%T|%R|%N|%M|%l" 2>/dev/null || true
  )"

  if [[ -z "$queue_line" ]]; then
    echo "status=LEFT_QUEUE job_id=$JOBID checking_final_accounting=1"
    break
  fi

  IFS='|' read -r state reason nodelist elapsed timelimit <<< "$queue_line"

  if [[ "$state" != "$last_state" || "$reason" != "$last_reason" ]]; then
    case "$state" in
      PENDING)
        echo "status=PENDING job_id=$JOBID step=waiting_for_gpu_debug_node reason=$reason elapsed=$elapsed timelimit=$timelimit"
        ;;
      RUNNING)
        echo "status=RUNNING job_id=$JOBID step=gpu_debug_node_allocated nodes=$nodelist elapsed=$elapsed timelimit=$timelimit"
        reported_running=1
        ;;
      CONFIGURING|COMPLETING)
        echo "status=$state job_id=$JOBID step=slurm_transition nodes=$nodelist elapsed=$elapsed timelimit=$timelimit"
        ;;
      *)
        echo "status=$state job_id=$JOBID reason=$reason nodes=$nodelist elapsed=$elapsed timelimit=$timelimit"
        ;;
    esac
    last_state="$state"
    last_reason="$reason"
  elif [[ "$state" == "PENDING" ]]; then
    echo "status=PENDING job_id=$JOBID step=still_waiting_for_gpu_debug_node reason=$reason elapsed=$elapsed timelimit=$timelimit"
  elif [[ "$state" == "RUNNING" ]]; then
    echo "status=RUNNING job_id=$JOBID step=still_running_gpu_probe nodes=$nodelist elapsed=$elapsed timelimit=$timelimit"
  fi

  if [[ "$reported_running" -eq 1 && -f "$OUT_FILE" ]]; then
    echo "status=JOB_OUTPUT_TAIL job_id=$JOBID file=$OUT_FILE"
    tail -n "$MONITOR_TAIL_LINES" "$OUT_FILE" || true
  fi

  if [[ "$reported_running" -eq 1 && -f "$ERR_FILE" && -s "$ERR_FILE" ]]; then
    echo "status=JOB_ERROR_TAIL job_id=$JOBID file=$ERR_FILE"
    tail -n "$MONITOR_TAIL_LINES" "$ERR_FILE" || true
  fi

  sleep "$MONITOR_INTERVAL_SECONDS"
done

sacct -j "$JOBID" \
  --format=JobID,JobName%24,Partition,Account,State,ExitCode,Elapsed,Timelimit,AllocTRES,ReqTRES,MaxRSS,AveRSS,MaxVMSize,MaxDiskRead,MaxDiskWrite,NodeList%40 \
  -P > "$OUT_DIR/gpu_debug_probe_sacct.txt" || true

final_state="$(
  sacct -n -j "$JOBID" \
    --format=State,ExitCode,Elapsed,NodeList \
    -P 2>/dev/null | head -n 1 || true
)"
echo "status=FINAL job_id=$JOBID sacct=$final_state"
echo "done_gpu_debug_probe=$JOBID"
echo "outputs=$OUT_DIR"

