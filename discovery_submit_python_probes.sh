#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=hpc_kestrel_config.sh
source "${SCRIPT_DIR}/hpc_kestrel_config.sh"

OUT_DIR="${1:-${DISCOVERY_OUT_DIR}}"
OUT_DIR="$(kestrel_realpath "${OUT_DIR}")"
mkdir -p "${OUT_DIR}"

kestrel_status "script=discovery_submit_python_probes.sh step=prepare_python_probe_job out_dir=${OUT_DIR} account=${NLR_ACCOUNT} partition=${KESTREL_CPU_PARTITION}"

JOB_SCRIPT="${OUT_DIR}/slurm_cpu_debug_python_probes.sbatch"
cat > "${JOB_SCRIPT}" <<SBATCH
#!/usr/bin/env bash
set -euo pipefail

job_step() {
  echo "job_status_timestamp=\$(date -u +%Y-%m-%dT%H:%M:%SZ) job_id=\${SLURM_JOB_ID:-unknown} step=\$1"
}

job_step "start_python_pixi_probes"
cd "${PWD}"
source "${SCRIPT_DIR}/hpc_kestrel_config.sh"
export DISCOVERY_OUT_DIR="${OUT_DIR}"
export PATH="${PIXI_HOME}/bin:\${PATH}"

job_step "check_pixi_available"
if ! command -v pixi >/dev/null 2>&1; then
  echo "ERROR: pixi not found in job. Run ./install_pixi_project_env.sh before submitting Python probes." >&2
  exit 2
fi
pixi --version || true

job_step "run_chunk_spill_probe"
bash "${SCRIPT_DIR}/discovery_chunk_spill_probe.sh" "${OUT_DIR}"

job_step "run_parquet_bench"
bash "${SCRIPT_DIR}/discovery_parquet_bench.sh" "${OUT_DIR}"

job_step "finish_python_pixi_probes"
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

SUBMIT_LOG="${OUT_DIR}/slurm_cpu_debug_python_probes_submit_command.txt"
kestrel_write_submit_command "${SUBMIT_LOG}" "${SUBMIT_CMD[@]}"
kestrel_status "script=discovery_submit_python_probes.sh step=submit_python_probe_job submit_command_file=${SUBMIT_LOG}"
JOBID="$("${SUBMIT_CMD[@]}")"
echo "${JOBID}" > "${OUT_DIR}/slurm_cpu_debug_python_probes_jobid.txt"
STDOUT_FILE="${OUT_DIR}/slurm_cpu_debug_python_probes_${JOBID}.out"
SACCT_FILE="${OUT_DIR}/slurm_cpu_debug_python_probes_sacct.txt"
kestrel_monitor_slurm_job "${JOBID}" "cpu_debug_python_probes" "${STDOUT_FILE}" "${SACCT_FILE}"
kestrel_status "script=discovery_submit_python_probes.sh step=done_cpu_debug_python_probes job_id=${JOBID} out_dir=${OUT_DIR}"
