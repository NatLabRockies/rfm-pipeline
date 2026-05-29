#!/usr/bin/env bash
# Copyright (c) 2026 Dylan Hettinger
#SBATCH --job-name=bsm_sensitivity
#SBATCH --account=bsm
#SBATCH --partition=shared
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-57499%200
#SBATCH --output=logs/sensitivity_%A_%a.out
#SBATCH --error=logs/sensitivity_%A_%a.err

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
mkdir -p logs

SPEC_PATH="${1:-$ROOT_DIR/configs/sensitivity_study/study_spec.yml}"
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
LINE_NUMBER=$((SLURM_ARRAY_TASK_ID + 1))
LINE="$(sed -n "${LINE_NUMBER}p" "$ARRAY_FILE")"
if [[ -z "$LINE" ]]; then
  echo "No array entry for index ${SLURM_ARRAY_TASK_ID} in ${ARRAY_FILE}" >&2
  exit 1
fi
IFS=, read -r JOB_ID CONFIG_PATH ARTIFACT_DIR <<<"$LINE"
mkdir -p "$ARTIFACT_DIR"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] running ${JOB_ID}"
pixi run python tools/run_manuscript_pipeline.py "$CONFIG_PATH" --output-dir "$ARTIFACT_DIR"
