#!/usr/bin/env bash
#SBATCH --job-name=probe-env
#SBATCH --output=artifacts/hpc_discovery/slurm_probe_%j.out
#SBATCH --error=artifacts/hpc_discovery/slurm_probe_%j.err
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

set -euo pipefail
echo "job_id=$SLURM_JOB_ID"
echo "node=$SLURMD_NODENAME"
echo "cpus_per_task=$SLURM_CPUS_PER_TASK"
echo "mem_per_node=${SLURM_MEM_PER_NODE:-}"
echo "tmpdir=${TMPDIR:-}"
hostname
command -v lscpu >/dev/null 2>&1 && lscpu || true
command -v free >/dev/null 2>&1 && free -h || true
df -h || true
df -ih || true
command -v numactl >/dev/null 2>&1 && numactl --hardware || true
cat /proc/meminfo || true
