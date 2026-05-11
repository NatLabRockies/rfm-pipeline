#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=hpc_kestrel_config.sh
source "${SCRIPT_DIR}/hpc_kestrel_config.sh"

OUT_DIR="${1:-${DISCOVERY_OUT_DIR}}"
OUT_DIR="$(kestrel_realpath "${OUT_DIR}")"
mkdir -p "${OUT_DIR}"

JOB_SCRIPT="${OUT_DIR}/slurm_cpu_debug_python_probes.sbatch"
cat > "${JOB_SCRIPT}" <<SBATCH
#!/usr/bin/env bash
set -euo pipefail

cd "${PWD}"
source "${SCRIPT_DIR}/hpc_kestrel_config.sh"
export DISCOVERY_OUT_DIR="${OUT_DIR}"
export PATH="${PIXI_HOME}/bin:\${PATH}"

if ! command -v pixi >/dev/null 2>&1; then
  echo "ERROR: pixi not found in job. Run ./install_pixi_project_env.sh before submitting Python probes." >&2
  exit 2
fi

bash "${SCRIPT_DIR}/discovery_chunk_spill_probe.sh" "${OUT_DIR}"
bash "${SCRIPT_DIR}/discovery_parquet_bench.sh" "${OUT_DIR}"
SBATCH

SUBMIT_CMD=(
  sbatch --parsable
  --account="${NLR_ACCOUNT}"
  --partition="${KESTREL_CPU_PARTITION}"
  --time="${DISCOVERY_PYTHON_TIME:-00:30:00}"
  --nodes=1
  --ntasks=1
  --cpus-per-task="${DISCOVERY_PYTHON_CPUS:-8}"
  --mem="${DISCOVERY_PYTHON_MEM:-32G}"
  --tmp="${DISCOVERY_CPU_TMP:-100G}"
  --job-name="bsm-python-probes"
  --output="${OUT_DIR}/slurm_cpu_debug_python_probes_%j.out"
  --error="${OUT_DIR}/slurm_cpu_debug_python_probes_%j.err"
  --export="ALL,DISCOVERY_OUT_DIR=${OUT_DIR},NLR_ACCOUNT=${NLR_ACCOUNT},KESTREL_PROJECT_ROOT=${KESTREL_PROJECT_ROOT},PIXI_HOME=${PIXI_HOME},PIXI_CACHE_DIR=${PIXI_CACHE_DIR},XDG_CACHE_HOME=${XDG_CACHE_HOME},CONDA_PKGS_DIRS=${CONDA_PKGS_DIRS}"
  "${JOB_SCRIPT}"
)

printf '%q ' "${SUBMIT_CMD[@]}" > "${OUT_DIR}/slurm_cpu_debug_python_probes_submit_command.txt"
printf '\n' >> "${OUT_DIR}/slurm_cpu_debug_python_probes_submit_command.txt"
JOBID="$("${SUBMIT_CMD[@]}")"
echo "${JOBID}" > "${OUT_DIR}/slurm_cpu_debug_python_probes_jobid.txt"
echo "submitted_cpu_debug_python_probes_job_id=${JOBID}"

while squeue -j "${JOBID}" -h 2>/dev/null | grep -q .; do
  sleep 5
done

sacct -j "${JOBID}" \
  --format=JobID,JobName%24,Partition,Account,State,ExitCode,Elapsed,Timelimit,AllocTRES,ReqTRES,MaxRSS,AveRSS,MaxVMSize,MaxDiskRead,MaxDiskWrite,NodeList%40 \
  -P > "${OUT_DIR}/slurm_cpu_debug_python_probes_sacct.txt" || true

echo "done_cpu_debug_python_probes=${JOBID}"
