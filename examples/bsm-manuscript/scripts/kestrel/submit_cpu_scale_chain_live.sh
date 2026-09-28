#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

S2_DIR="${1:-${REPO_ROOT}/artifacts/kestrel_cpu_scaling_suite/cpu_nodes_2/hpc_scripts}"
S10_DIR="${2:-${REPO_ROOT}/artifacts/kestrel_cpu_scaling_suite/cpu_nodes_10/hpc_scripts}"
S1000_DIR="${3:-${REPO_ROOT}/artifacts/kestrel_cpu_scaling_suite/cpu_nodes_1000/hpc_scripts}"

for d in "${S2_DIR}" "${S10_DIR}" "${S1000_DIR}"; do
  if [[ ! -d "${d}" ]]; then
    echo "error: missing hpc script dir: ${d}" >&2
    echo "generate scripts first with:" >&2
    echo "  bash scripts/kestrel/submit_cpu_scaling_suite.sh --dry-run --stage interaction_discovery" >&2
    exit 2
  fi
done

cd "${S2_DIR}"
A2="$(sbatch --parsable --partition=debug --time=01:00:00 submit_interaction_discovery_array.sh)"
R2="$(sbatch --parsable --partition=debug --time=01:00:00 --dependency=afterok:${A2} submit_interaction_discovery_reduce.sh)"

cd "${S10_DIR}"
A10="$(sbatch --parsable --partition=shared --time=04:00:00 --dependency=afterok:${R2} submit_interaction_discovery_array.sh)"
R10="$(sbatch --parsable --partition=shared --time=04:00:00 --dependency=afterok:${A10} submit_interaction_discovery_reduce.sh)"

cd "${S1000_DIR}"
A1000="$(sbatch --parsable --partition=shared --time=08:00:00 --dependency=afterok:${R10} submit_interaction_discovery_array.sh)"
R1000="$(sbatch --parsable --partition=shared --time=08:00:00 --dependency=afterok:${A1000} submit_interaction_discovery_reduce.sh)"

echo "2-node:    array=${A2} reduce=${R2}"
echo "10-node:   array=${A10} reduce=${R10}"
echo "1000-node: array=${A1000} reduce=${R1000}"
echo "monitor: squeue -u \$USER"
