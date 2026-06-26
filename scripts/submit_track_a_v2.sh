#!/bin/bash
#SBATCH --job-name=rfm_track_a_v2
#SBATCH --account=bsm
#SBATCH --partition=shared
#SBATCH --time=24:00:00
#SBATCH --mem=220G
#SBATCH --cpus-per-task=104
#SBATCH --output=/home/dhetting/src/bsm-public-rf/logs/track_a_v2_%j.out
#SBATCH --error=/home/dhetting/src/bsm-public-rf/logs/track_a_v2_%j.err

# Track A v2: improved measurement-based meta-model fit
# Three improvements:
#   1. proxy_n_outputs_cap=9000 (BSM has 9954 outputs; accurate spectrum stats)
#   2. Hybrid oracle Ridge as primary model (predict eta not nrmse)
#   3. GP-ARD with optimizer enabled (learns feature relevance)
# Submit from login node: sbatch scripts/submit_track_a_v2.sh
# Do NOT run fit_track_a_v2.py directly on a login node.

set -euo pipefail

export PIXI_HOME="/projects/bsm/.pixi"
export PIXI_CACHE_DIR="/projects/bsm/.cache/pixi"

REPO="/home/dhetting/src/bsm-public-rf"
LOG_DIR="${REPO}/logs"
mkdir -p "${LOG_DIR}"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] rfm track-a v2 starting on $(hostname)"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] CPUs=${SLURM_CPUS_PER_TASK} MEM=${SLURM_MEM_PER_NODE}"

# Limit numpy/sklearn threading so sklearn's own joblib parallelism works cleanly
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd "${REPO}"

# Ensure pixi env is up to date
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Checking pixi environment..."
pixi install --locked 2>&1 | tail -3

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Running fit_track_a_v2.py ..."
pixi run python -u scripts/fit_track_a_v2.py \
    --input  artifacts/sensitivity/wave5_results.csv \
    --output-dir artifacts/sensitivity/wave5_measurement_models \
    --proxy-n-runs-cap 10000 \
    --proxy-n-outputs-cap 9000 \
    --max-outputs-for-pca 9000 \
    --max-inputs-for-corr 500 \
    --max-rows-for-corr 10000

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Done."
