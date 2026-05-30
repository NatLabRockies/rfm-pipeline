#!/usr/bin/env bash
# Copyright (c) 2026 Dylan Hettinger
#SBATCH --job-name=bsm_sensitivity
#SBATCH --account=bsm
#SBATCH --partition=shared
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-10999%200
#SBATCH --output=logs/sensitivity_%A_%a.out
#SBATCH --error=logs/sensitivity_%A_%a.err

# Usage:
#   # Wave 1 (short partition, single batch, 2,750 configs):
#   SENSITIVITY_SPEC=configs/sensitivity_study/study_spec_wave1.yml \
#     sbatch --partition=short --time=04:00:00 --mem=16G --cpus-per-task=4 \
#            --array=0-2749%300 scripts/submit_sensitivity_study.sh
#
#   # Original full study (shared partition, multi-batch):
#   bash scripts/submit_sensitivity_study.sh --submit-all
#
#   # Single batch with explicit offset:
#   sbatch --array=0-10999%200 --export=ALL,BATCH_OFFSET=0 scripts/submit_sensitivity_study.sh

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT_DIR"
mkdir -p logs

# Spec path: override via SENSITIVITY_SPEC env var or default to original.
SPEC_PATH="${SENSITIVITY_SPEC:-${ROOT_DIR}/configs/sensitivity_study/study_spec.yml}"

# Handle --submit-all mode: auto-detect total jobs and submit in batches
if [[ "${1:-}" == "--submit-all" ]]; then
  STUDY_DIR="$(pixi run python - "$SPEC_PATH" <<'PY'
from pathlib import Path
import sys, yaml
payload = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8")) or {}
print(payload["output"]["study_dir"])
PY
)"
  ARRAY_FILE="$STUDY_DIR/slurm_array.txt"
  TOTAL_JOBS=$(wc -l < "$ARRAY_FILE")
  BATCH_SIZE=10000
  OFFSET=0
  while [ "$OFFSET" -lt "$TOTAL_JOBS" ]; do
    END=$((OFFSET + BATCH_SIZE - 1))
    if [ "$END" -ge "$TOTAL_JOBS" ]; then END=$((TOTAL_JOBS - 1)); fi
    N_TASKS=$((END - OFFSET + 1))
    SLURM_PARAMS=()
    SLURM_PARAMS+=(--partition shared --time=02:00:00 --mem=16G --cpus-per-task=4)
    SLURM_PARAMS+=(--array="0-$((N_TASKS - 1))%200")
    SLURM_PARAMS+=(--export="ALL,BATCH_OFFSET=$OFFSET,SENSITIVITY_SPEC=$SPEC_PATH")
    sbatch "${SLURM_PARAMS[@]}" "${BASH_SOURCE[0]}"
    echo "Submitted batch offset=${OFFSET} tasks=0-$((N_TASKS - 1))"
    OFFSET=$((OFFSET + BATCH_SIZE))
  done
  exit 0
fi

STUDY_DIR="$(pixi run python - "$SPEC_PATH" <<'PY'
from pathlib import Path
import sys, yaml
payload = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8")) or {}
print(payload["output"]["study_dir"])
PY
)"
ARRAY_FILE="$STUDY_DIR/slurm_array.txt"

# BATCH_OFFSET shifts task IDs for multi-batch submissions (default 0)
ACTUAL_INDEX=$(( SLURM_ARRAY_TASK_ID + ${BATCH_OFFSET:-0} ))
LINE_NUMBER=$((ACTUAL_INDEX + 1))
LINE="$(sed -n "${LINE_NUMBER}p" "$ARRAY_FILE")"
if [[ -z "$LINE" ]]; then
  echo "No array entry for index ${SLURM_ARRAY_TASK_ID} in ${ARRAY_FILE}" >&2
  exit 1
fi
IFS=, read -r JOB_ID CONFIG_PATH ARTIFACT_DIR <<<"$LINE"
mkdir -p "$ARTIFACT_DIR"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] running ${JOB_ID}"

# OMP/BLAS threading: keep 1 thread per process so n_jobs workers
# don't oversubscribe the allocated CPUs.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

pixi run python scripts/run_sensitivity_job.py \
    --config "$CONFIG_PATH" \
    --artifact-dir "$ARTIFACT_DIR" \
    --job-id "$JOB_ID"
