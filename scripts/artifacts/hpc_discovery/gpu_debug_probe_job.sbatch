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
