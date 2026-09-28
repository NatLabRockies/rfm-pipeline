#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

HPC_SCRIPTS_DIR="${1:-${REPO_ROOT}/artifacts/kestrel_gpu_h100_run/hpc_scripts}"
PARTITION="${PARTITION:-gpu-h100s}"
WALLTIME="${WALLTIME:-04:00:00}"

if [[ ! -d "${HPC_SCRIPTS_DIR}" ]]; then
  echo "error: missing hpc script dir: ${HPC_SCRIPTS_DIR}" >&2
  echo "generate scripts first with:" >&2
  echo "  pixi run rfm-hpc-submit --config configs/hpc/dev/kestrel_gpu_h100.yml --stage interaction_discovery --dry-run" >&2
  exit 2
fi

cd "${HPC_SCRIPTS_DIR}"

ARRAY_SCRIPT="submit_interaction_discovery_gpu_array.sh"
if [[ ! -f "${ARRAY_SCRIPT}" ]]; then
  ARRAY_SCRIPT="submit_interaction_discovery_array.sh"
fi

if [[ ! -f "${ARRAY_SCRIPT}" ]]; then
  echo "error: no GPU or CPU array submit script found in ${HPC_SCRIPTS_DIR}" >&2
  exit 2
fi

if [[ ! -f "submit_interaction_discovery_reduce.sh" ]]; then
  echo "error: missing reduce script in ${HPC_SCRIPTS_DIR}" >&2
  exit 2
fi

ARRAY_JOB_ID="$(sbatch --parsable --partition="${PARTITION}" --time="${WALLTIME}" "${ARRAY_SCRIPT}")"
REDUCE_JOB_ID="$(sbatch --parsable --partition="${PARTITION}" --time="${WALLTIME}" --dependency=afterok:${ARRAY_JOB_ID} submit_interaction_discovery_reduce.sh)"

echo "GPU queued: array=${ARRAY_JOB_ID} reduce=${REDUCE_JOB_ID} partition=${PARTITION} time=${WALLTIME} script=${ARRAY_SCRIPT}"
echo "monitor: squeue -u \$USER"
