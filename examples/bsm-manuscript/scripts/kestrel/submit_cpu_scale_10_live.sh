#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

AFTER_JOB_ID="${1:-}"
HPC_SCRIPTS_DIR="${2:-${REPO_ROOT}/artifacts/kestrel_cpu_scaling_suite/cpu_nodes_10/hpc_scripts}"
PARTITION="${PARTITION:-shared}"
WALLTIME="${WALLTIME:-04:00:00}"

if [[ ! -d "${HPC_SCRIPTS_DIR}" ]]; then
  echo "error: missing hpc script dir: ${HPC_SCRIPTS_DIR}" >&2
  echo "generate scripts first with:" >&2
  echo "  bash scripts/kestrel/submit_cpu_scaling_suite.sh --dry-run --stage interaction_discovery" >&2
  exit 2
fi

cd "${HPC_SCRIPTS_DIR}"

ARRAY_CMD=(sbatch --parsable --partition="${PARTITION}" --time="${WALLTIME}")
if [[ -n "${AFTER_JOB_ID}" ]]; then
  ARRAY_CMD+=(--dependency=afterok:${AFTER_JOB_ID})
fi
ARRAY_CMD+=(submit_interaction_discovery_array.sh)

ARRAY_JOB_ID="$("${ARRAY_CMD[@]}")"
REDUCE_JOB_ID="$(sbatch --parsable --partition="${PARTITION}" --time="${WALLTIME}" --dependency=afterok:${ARRAY_JOB_ID} submit_interaction_discovery_reduce.sh)"

echo "10-node queued: array=${ARRAY_JOB_ID} reduce=${REDUCE_JOB_ID} partition=${PARTITION} time=${WALLTIME} after=${AFTER_JOB_ID:-none}"
echo "monitor: squeue -u \$USER"
