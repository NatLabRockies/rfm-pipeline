#!/bin/bash
#SBATCH --job-name=rfm_track_a_v3
#SBATCH --account=bsm
#SBATCH --partition=shared
#SBATCH --time=24:00:00
#SBATCH --mem=220G
#SBATCH --cpus-per-task=104
#SBATCH --output=/home/dhetting/src/bsm-public-rf/logs/track_a_v3_%j.out
#SBATCH --error=/home/dhetting/src/bsm-public-rf/logs/track_a_v3_%j.err

set -euo pipefail

export PIXI_HOME="/projects/bsm/.pixi"
export PIXI_CACHE_DIR="/projects/bsm/.cache/pixi"

REPO="/home/dhetting/src/bsm-public-rf"
LOG_DIR="${REPO}/logs"
mkdir -p "${LOG_DIR}"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] rfm track-a v3 starting on $(hostname)"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] CPUs=${SLURM_CPUS_PER_TASK} MEM=${SLURM_MEM_PER_NODE}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd "${REPO}"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Checking pixi environment..."
pixi install --locked 2>&1 | tail -3

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Running fit_track_a_v3.py ..."
pixi run python -u scripts/fit_track_a_v3.py \
    --input artifacts/sensitivity/wave5_results.csv \
    --output-dir artifacts/sensitivity/wave5_measurement_models_v3 \
    --proxy-n-runs-cap 12000 \
    --proxy-n-outputs-cap 9954 \
    --max-outputs-for-pca 9954 \
    --max-inputs-for-corr 600 \
    --max-rows-for-corr 12000 \
    --knn-measurement-neighbors 8 \
    --near-bsm-weight-max 2.5 \
    --near-bsm-distance-bandwidth 2.0

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Done."
