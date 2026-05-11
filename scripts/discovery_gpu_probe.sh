#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=hpc_kestrel_config.sh
source "${SCRIPT_DIR}/hpc_kestrel_config.sh"

OUT_DIR="${1:-${DISCOVERY_OUT_DIR}}"
OUT_DIR="$(kestrel_realpath "${OUT_DIR}")"
mkdir -p "${OUT_DIR}"

if ! kestrel_have_partition "${KESTREL_GPU_PARTITION}"; then
  echo "GPU partition ${KESTREL_GPU_PARTITION} is not visible; skipping GPU probe." | tee "${OUT_DIR}/gpu_probe_skipped.txt"
  exit 0
fi

JOB_SCRIPT="${OUT_DIR}/slurm_gpu_short_probe_job.sbatch"
cat > "${JOB_SCRIPT}" <<'SBATCH'
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
SBATCH

SUBMIT_CMD=(
  sbatch --parsable
  --account="${NLR_ACCOUNT}"
  --partition="${KESTREL_GPU_PARTITION}"
  --time="${DISCOVERY_GPU_TIME:-00:10:00}"
  --nodes=1
  --ntasks=1
  --cpus-per-task="${DISCOVERY_GPU_CPUS:-4}"
  --mem="${DISCOVERY_GPU_MEM:-16G}"
  --gpus="${DISCOVERY_GPUS:-1}"
  --job-name="bsm-gpu-probe"
  --output="${OUT_DIR}/slurm_gpu_short_probe_%j.out"
  --error="${OUT_DIR}/slurm_gpu_short_probe_%j.err"
  --export="ALL,DISCOVERY_OUT_DIR=${OUT_DIR},NLR_ACCOUNT=${NLR_ACCOUNT},KESTREL_PROJECT_ROOT=${KESTREL_PROJECT_ROOT},PIXI_HOME=${PIXI_HOME},PIXI_CACHE_DIR=${PIXI_CACHE_DIR},XDG_CACHE_HOME=${XDG_CACHE_HOME},CONDA_PKGS_DIRS=${CONDA_PKGS_DIRS}"
  "${JOB_SCRIPT}"
)

printf '%q ' "${SUBMIT_CMD[@]}" > "${OUT_DIR}/slurm_gpu_short_probe_submit_command.txt"
printf '\n' >> "${OUT_DIR}/slurm_gpu_short_probe_submit_command.txt"
JOBID="$("${SUBMIT_CMD[@]}")"
echo "${JOBID}" > "${OUT_DIR}/slurm_gpu_short_probe_jobid.txt"
echo "submitted_gpu_short_job_id=${JOBID}"

while squeue -j "${JOBID}" -h 2>/dev/null | grep -q .; do
  sleep 5
done

sacct -j "${JOBID}" \
  --format=JobID,JobName%24,Partition,Account,State,ExitCode,Elapsed,Timelimit,AllocTRES,ReqTRES,MaxRSS,AveRSS,MaxVMSize,MaxDiskRead,MaxDiskWrite,NodeList%40 \
  -P > "${OUT_DIR}/slurm_gpu_short_probe_sacct.txt" || true

echo "done_gpu_short_probe=${JOBID}"
