#!/usr/bin/env bash
#SBATCH --job-name=bsm-final-cost-ladder
#SBATCH --account=bsm
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=104
#SBATCH --mem=220G
#SBATCH --time=01:00:00
#SBATCH --output=artifacts/hpc_final_cost_ladder/slurm-%j.out
#SBATCH --error=artifacts/hpc_final_cost_ladder/slurm-%j.err
#SBATCH --signal=TERM@120

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR}"
cd "${REPO_ROOT}"

mkdir -p artifacts/hpc_final_cost_ladder

echo ">>> $(date -u +%Y-%m-%dT%H:%M:%SZ) job=${SLURM_JOB_ID} start"
echo ">>> account=${SLURM_JOB_ACCOUNT:-bsm} partition=${SLURM_JOB_PARTITION:-shared} cpus=${SLURM_CPUS_PER_TASK:-unset} mem=${SLURM_MEM_PER_NODE:-unset}"

bash scripts/kestrel/run_final_cost_ladder_on_node.sh artifacts/hpc_final_cost_ladder

echo ">>> $(date -u +%Y-%m-%dT%H:%M:%SZ) job=${SLURM_JOB_ID} done"
