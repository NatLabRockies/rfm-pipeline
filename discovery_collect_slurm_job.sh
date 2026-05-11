#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=hpc_kestrel_config.sh
source "${SCRIPT_DIR}/hpc_kestrel_config.sh"

OUT_DIR="${1:-${DISCOVERY_OUT_DIR}}"
OUT_DIR="$(kestrel_realpath "${OUT_DIR}")"
mkdir -p "${OUT_DIR}"

kestrel_status "script=discovery_collect_slurm_job.sh step=prepare_cpu_debug_probe out_dir=${OUT_DIR} account=${NLR_ACCOUNT} partition=${KESTREL_CPU_PARTITION}"

JOB_SCRIPT="${OUT_DIR}/slurm_cpu_debug_probe_job.sbatch"
cat > "${JOB_SCRIPT}" <<'SBATCH'
#!/usr/bin/env bash
set -euo pipefail

job_step() {
  echo "job_status_timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ) job_id=${SLURM_JOB_ID:-unknown} step=$1"
}

job_step "start_cpu_topology_probe"

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

job_step "scontrol_show_job"
scontrol show job "${SLURM_JOB_ID}" || true

job_step "srun_affinity_smoke_test"
srun --cpu-bind=cores bash -lc 'echo task=$SLURM_PROCID host=$(hostname) cpus_allowed=$(grep Cpus_allowed_list /proc/self/status | awk "{print \$2}")'

job_step "collect_host_details"
hostname
uname -a
cat /etc/redhat-release || true

job_step "collect_cpu_topology"
lscpu || true
numactl --hardware || true

job_step "collect_memory"
free -h || true
cat /proc/meminfo || true

job_step "collect_filesystems"
df -hT "${TMPDIR:-/tmp}" /scratch /projects /projects/bsm "/scratch/${USER}" 2>/dev/null || true
df -ih "${TMPDIR:-/tmp}" /scratch /projects /projects/bsm "/scratch/${USER}" 2>/dev/null || true
mount | egrep 'lustre|scratch|projects|tmp|nvme|kfs' || true

job_step "collect_lustre_details"
lfs quota -h -u "${USER}" /scratch 2>/dev/null || true
lfs project -d /projects/bsm 2>/dev/null || true
lfs getstripe "/scratch/${USER}" 2>/dev/null || true
lfs getstripe /projects/bsm 2>/dev/null || true

job_step "collect_slurm_accounting_config"
scontrol show config | egrep -i 'JobAcctGather|JobAcctGatherFrequency|AccountingStorage|KillWait|Preempt|Requeue|MaxArraySize|MaxJobCount' || true

job_step "python_minimal_topology"
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

job_step "finish_cpu_topology_probe"
SBATCH

SUBMIT_CMD=(
  sbatch --parsable
  --account="${NLR_ACCOUNT}"
  --partition="${KESTREL_CPU_PARTITION}"
  --time="${DISCOVERY_CPU_TIME:-00:10:00}"
  --nodes=1
  --ntasks=1
  --cpus-per-task="${DISCOVERY_CPU_CPUS:-4}"
  --mem="${DISCOVERY_CPU_MEM:-8G}"
  --job-name="bsm-cpu-probe"
  --output="${OUT_DIR}/slurm_cpu_debug_probe_%j.out"
  --error="${OUT_DIR}/slurm_cpu_debug_probe_%j.err"
  --export="ALL,DISCOVERY_OUT_DIR=${OUT_DIR},NLR_ACCOUNT=${NLR_ACCOUNT},KESTREL_PROJECT_ROOT=${KESTREL_PROJECT_ROOT},PIXI_HOME=${PIXI_HOME},PIXI_CACHE_DIR=${PIXI_CACHE_DIR},XDG_CACHE_HOME=${XDG_CACHE_HOME},CONDA_PKGS_DIRS=${CONDA_PKGS_DIRS}"
  "${JOB_SCRIPT}"
)

SUBMIT_LOG="${OUT_DIR}/slurm_cpu_debug_probe_submit_command.txt"
kestrel_write_submit_command "${SUBMIT_LOG}" "${SUBMIT_CMD[@]}"
kestrel_status "script=discovery_collect_slurm_job.sh step=submit_cpu_debug_probe submit_command_file=${SUBMIT_LOG}"
JOBID="$("${SUBMIT_CMD[@]}")"
echo "${JOBID}" > "${OUT_DIR}/slurm_cpu_debug_probe_jobid.txt"
STDOUT_FILE="${OUT_DIR}/slurm_cpu_debug_probe_${JOBID}.out"
SACCT_FILE="${OUT_DIR}/slurm_cpu_debug_probe_sacct.txt"
kestrel_monitor_slurm_job "${JOBID}" "cpu_debug_topology_probe" "${STDOUT_FILE}" "${SACCT_FILE}"
kestrel_status "script=discovery_collect_slurm_job.sh step=done_cpu_debug_probe job_id=${JOBID} out_dir=${OUT_DIR}"
