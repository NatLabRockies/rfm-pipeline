#!/usr/bin/env bash
set -euo pipefail

echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "job_id=${SLURM_JOB_ID:-}"
echo "job_name=${SLURM_JOB_NAME:-}"
echo "partition=${SLURM_JOB_PARTITION:-}"
echo "account=${SLURM_JOB_ACCOUNT:-}"
echo "node=${SLURMD_NODENAME:-}"
echo "nodelist=${SLURM_JOB_NODELIST:-}"
echo "cpus_per_task=${SLURM_CPUS_PER_TASK:-}"
echo "cpus_on_node=${SLURM_CPUS_ON_NODE:-}"
echo "mem_per_node=${SLURM_MEM_PER_NODE:-}"
echo "tmpdir=${TMPDIR:-}"
echo "submit_dir=${SLURM_SUBMIT_DIR:-}"
echo "DISCOVERY_OUT_DIR=${DISCOVERY_OUT_DIR:-}"
echo

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "### scontrol show job"
scontrol show job "${SLURM_JOB_ID}" || true

echo "### srun affinity smoke test"
srun --cpu-bind=cores bash -lc 'echo task=$SLURM_PROCID host=$(hostname) cpus_allowed=$(grep Cpus_allowed_list /proc/self/status | awk "{print \$2}")'

echo "### host"
hostname
uname -a
cat /etc/redhat-release || true

echo "### cpu"
lscpu || true
numactl --hardware || true

echo "### memory"
free -h || true
cat /proc/meminfo || true

echo "### filesystems"
df -hT "${TMPDIR:-/tmp}" /scratch /projects /projects/bsm "/scratch/${USER}" 2>/dev/null || true
df -ih "${TMPDIR:-/tmp}" /scratch /projects /projects/bsm "/scratch/${USER}" 2>/dev/null || true
mount | egrep 'lustre|scratch|projects|tmp|nvme|kfs' || true

echo "### lustre"
lfs quota -h -u "${USER}" /scratch 2>/dev/null || true
lfs project -d /projects/bsm 2>/dev/null || true
lfs getstripe "/scratch/${USER}" 2>/dev/null || true
lfs getstripe /projects/bsm 2>/dev/null || true

echo "### slurm accounting config inside job"
scontrol show config | egrep -i 'JobAcctGather|JobAcctGatherFrequency|AccountingStorage|KillWait|Preempt|Requeue|MaxArraySize|MaxJobCount' || true

echo "### python minimal topology"
python - <<'PY' || true
from __future__ import annotations
import json
import os
import platform
import tempfile
from pathlib import Path

payload = {
    "platform": platform.platform(),
    "python": platform.python_version(),
    "cpu_count": os.cpu_count(),
    "tmpdir_env": os.environ.get("TMPDIR"),
    "tempfile_gettempdir": tempfile.gettempdir(),
    "cwd": str(Path.cwd()),
    "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
    "slurm_mem_per_node": os.environ.get("SLURM_MEM_PER_NODE"),
}
print(json.dumps(payload, indent=2, sort_keys=True))
PY
