#!/usr/bin/env bash
# Copyright (c) 2026 Dylan Hettinger
#SBATCH --job-name=bsm_sensitivity
#SBATCH --account=bsm
#SBATCH --partition=shared
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-10999%200
#SBATCH --output=logs/sensitivity_%A_%a.out
#SBATCH --error=logs/sensitivity_%A_%a.err

# To submit all 57,500 jobs in batches (Kestrel MaxArraySize=11000):
#   bash scripts/submit_sensitivity_study.sh --submit-all
# Or submit a single batch manually:
#   sbatch --array=0-10999%200 --export=ALL,BATCH_OFFSET=0 scripts/submit_sensitivity_study.sh
#   sbatch --array=0-10999%200 --export=ALL,BATCH_OFFSET=11000 scripts/submit_sensitivity_study.sh
#   ... etc.

set -euo pipefail

# SLURM sets SLURM_SUBMIT_DIR to the directory sbatch was invoked from.
# Fall back to relative path resolution for local testing.
ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT_DIR"
mkdir -p logs

# Handle --submit-all mode: submit 6 batches and exit
if [[ "${1:-}" == "--submit-all" ]]; then
  TOTAL_JOBS=57500
  BATCH_SIZE=10000
  OFFSET=0
  while [ "$OFFSET" -lt "$TOTAL_JOBS" ]; do
    END=$((OFFSET + BATCH_SIZE - 1))
    if [ "$END" -ge "$TOTAL_JOBS" ]; then END=$((TOTAL_JOBS - 1)); fi
    N_TASKS=$((END - OFFSET + 1))
    sbatch --array="0-$((N_TASKS - 1))%200" \
           --export="ALL,BATCH_OFFSET=$OFFSET" \
           "${BASH_SOURCE[0]}"
    echo "Submitted batch offset=${OFFSET} tasks=0-$((N_TASKS - 1))"
    OFFSET=$((OFFSET + BATCH_SIZE))
  done
  exit 0
fi

SPEC_PATH="${ROOT_DIR}/configs/sensitivity_study/study_spec.yml"
STUDY_DIR="$(pixi run python - "$SPEC_PATH" <<'PY'
from pathlib import Path
import sys
import yaml

spec_path = Path(sys.argv[1])
payload = yaml.safe_load(spec_path.read_text(encoding='utf-8')) or {}
print(payload['output']['study_dir'])
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
pixi run python scripts/run_sensitivity_job.py \
    --config "$CONFIG_PATH" \
    --artifact-dir "$ARTIFACT_DIR" \
    --job-id "$JOB_ID"
