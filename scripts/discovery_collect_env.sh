#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=hpc_kestrel_config.sh
source "${SCRIPT_DIR}/hpc_kestrel_config.sh"

OUT_DIR="${1:-${DISCOVERY_OUT_DIR}}"
mkdir -p "${OUT_DIR}"

run_block() {
  local label="$1"
  shift
  {
    echo "### ${label}"
    echo ">>> $*"
    "$@" || true
    echo
  }
}

{
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname=$(hostname)"
  echo "user=$(whoami)"
  echo "pwd=$(pwd)"
  echo "kernel=$(uname -a)"
  echo "NLR_ACCOUNT=${NLR_ACCOUNT}"
  echo "KESTREL_PROJECT_ROOT=${KESTREL_PROJECT_ROOT}"
  echo "KESTREL_SCRATCH_ROOT=${KESTREL_SCRATCH_ROOT}"
  echo "KESTREL_CPU_PARTITION=${KESTREL_CPU_PARTITION}"
  echo "KESTREL_GPU_PARTITION=${KESTREL_GPU_PARTITION}"
  echo "PIXI_HOME=${PIXI_HOME}"
  echo "PIXI_CACHE_DIR=${PIXI_CACHE_DIR}"
  echo "CONDA_PKGS_DIRS=${CONDA_PKGS_DIRS}"
  echo "PATH=${PATH}"
} > "${OUT_DIR}/system.txt"

{
  run_block "slurm version" scontrol --version
  run_block "cluster config filtered" bash -lc "scontrol show config | egrep -i 'SlurmctldHost|SlurmctldPort|SchedulerType|SelectType|TaskPlugin|AccountingStorageType|JobAcctGather|JobAcctGatherFrequency|MaxArraySize|MaxJobCount|KillWait|CompleteWait|MinJobAge|Preempt|Requeue|Priority|PriorityDecayHalfLife'"
  run_block "partition summary" sinfo -o "%P %a %D %c %m %l %f"
  run_block "debug partition detail" scontrol show partition "${KESTREL_CPU_PARTITION}"
  if kestrel_have_partition "${KESTREL_GPU_PARTITION}"; then
    run_block "short GPU partition detail" scontrol show partition "${KESTREL_GPU_PARTITION}"
  else
    echo "### short GPU partition detail"
    echo "Partition ${KESTREL_GPU_PARTITION} not visible from this login/session."
    echo
  fi
  run_block "all partitions one-line" scontrol show partition -o
  run_block "association for user" sacctmgr show assoc user="${USER}" -P
  run_block "qos" sacctmgr show qos -P
  run_block "current jobs" squeue -u "${USER}" -o "%.18i %.9P %.8q %.12a %.8T %.10M %.10l %.6D %R"
  run_block "priority" sprio -u "${USER}" -l
} > "${OUT_DIR}/slurm_control.txt" 2>&1

{
  run_block "cpu topology" lscpu
  run_block "memory" free -h
  run_block "numa" numactl --hardware
  run_block "ulimits" ulimit -a
  run_block "filesystems" df -hT
  run_block "inodes" df -ih
  run_block "mounts filtered" bash -lc "mount | egrep 'lustre|scratch|projects|tmp|nvme|kfs'"
} > "${OUT_DIR}/node_local.txt" 2>&1

{
  run_block "project id" lfs project -d "${KESTREL_PROJECT_ROOT}"
  PROJECT_ID="$(lfs project -d "${KESTREL_PROJECT_ROOT}" 2>/dev/null | awk 'NR==1 {print $1}')"
  if [[ -n "${PROJECT_ID}" ]]; then
    run_block "project quota" lfs quota -hp "${PROJECT_ID}" "${KESTREL_PROJECT_ROOT}"
  fi
  run_block "scratch quota" lfs quota -h -u "${USER}" /scratch
  run_block "home quota" lfs quota -uh "${USER}" "/home/${USER}"
  run_block "lustre df" lfs df -h /scratch "${KESTREL_PROJECT_ROOT}"
  run_block "scratch stripe" lfs getstripe "/scratch/${USER}"
  run_block "project stripe" lfs getstripe "${KESTREL_PROJECT_ROOT}"
} > "${OUT_DIR}/storage.txt" 2>&1

{
  run_block "module list" module list
  run_block "module avail python/mamba/conda/apptainer" bash -lc "module avail python mamba conda apptainer 2>&1"
  run_block "pixi path" bash -lc "command -v pixi || true"
  run_block "pixi version" bash -lc "pixi --version || true"
  run_block "python path" bash -lc "command -v python || true"
  run_block "python version" bash -lc "python --version || true"
} > "${OUT_DIR}/software.txt" 2>&1

env | sort > "${OUT_DIR}/environment_vars.txt"

echo "Wrote artifacts to ${OUT_DIR}"
