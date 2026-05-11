#!/usr/bin/env bash
set -euo pipefail

echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "job_id=${SLURM_JOB_ID:-}"
echo "partition=${SLURM_JOB_PARTITION:-}"
echo "account=${SLURM_JOB_ACCOUNT:-}"
echo "node=${SLURMD_NODENAME:-}"
echo "gpus=${SLURM_GPUS:-}"
echo "gpu_ids=${SLURM_JOB_GPUS:-}"
echo "cpus_per_task=${SLURM_CPUS_PER_TASK:-}"
echo "mem_per_node=${SLURM_MEM_PER_NODE:-}"
echo "tmpdir=${TMPDIR:-}"
echo

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

scontrol show job "${SLURM_JOB_ID}" || true
hostname
lscpu || true
free -h || true
df -hT "${TMPDIR:-/tmp}" /scratch /projects /projects/bsm 2>/dev/null || true
mount | egrep 'lustre|scratch|projects|tmp|nvme|kfs' || true

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "### nvidia-smi"
  nvidia-smi
  echo "### nvidia-smi query"
  nvidia-smi --query-gpu=index,name,uuid,driver_version,cuda_version,memory.total,memory.free --format=csv || true
else
  echo "nvidia-smi not found"
fi

python - <<'PY' || true
from __future__ import annotations
import json
import os
payload = {
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "slurm_gpus": os.environ.get("SLURM_GPUS"),
    "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS"),
    "tmpdir": os.environ.get("TMPDIR"),
}
print(json.dumps(payload, indent=2, sort_keys=True))
PY
