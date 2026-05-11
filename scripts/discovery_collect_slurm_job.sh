#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=hpc_kestrel_config.sh
source "${SCRIPT_DIR}/hpc_kestrel_config.sh"

OUT_DIR="${1:-${DISCOVERY_OUT_DIR}}"
OUT_DIR="$(kestrel_realpath "${OUT_DIR}")"
mkdir -p "${OUT_DIR}"

JOB_SCRIPT="${OUT_DIR}/slurm_cpu_debug_probe_job.sbatch"
cat > "${JOB_SCRIPT}" <<'SBATCH'
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

printf '%q ' "${SUBMIT_CMD[@]}" > "${OUT_DIR}/slurm_cpu_debug_probe_submit_command.txt"
printf '\n' >> "${OUT_DIR}/slurm_cpu_debug_probe_submit_command.txt"
JOBID="$("${SUBMIT_CMD[@]}")"
echo "${JOBID}" > "${OUT_DIR}/slurm_cpu_debug_probe_jobid.txt"
echo "submitted_cpu_debug_job_id=${JOBID}"

while squeue -j "${JOBID}" -h 2>/dev/null | grep -q .; do
  sleep 5
done

sacct -j "${JOBID}" \
  --format=JobID,JobName%24,Partition,Account,State,ExitCode,Elapsed,Timelimit,AllocTRES,ReqTRES,MaxRSS,AveRSS,MaxVMSize,MaxDiskRead,MaxDiskWrite,NodeList%40 \
  -P > "${OUT_DIR}/slurm_cpu_debug_probe_sacct.txt" || true

echo "done_cpu_debug_probe=${JOBID}"
