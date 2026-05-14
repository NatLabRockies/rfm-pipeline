#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# shellcheck source=common_paths.sh
source "${REPO_ROOT}/scripts/kestrel/common_paths.sh"
DEFAULT_ARTIFACTS_ROOT="$(kestrel_default_artifacts_root "${REPO_ROOT}")"
ARTIFACTS_ROOT="${ARTIFACTS_ROOT:-${DEFAULT_ARTIFACTS_ROOT}}"
LOGS_ROOT="${LOGS_ROOT:-$(kestrel_default_logs_root)}"
RUN_DIR="${ARTIFACTS_ROOT}/kestrel_gpu_h100_run"
OUT_FILE="${RUN_DIR}/gpu_interaction_results_summary.csv"
RUN_DIR_SET=0
OUT_FILE_SET=0

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/kestrel/collect_gpu_interaction_results.sh [--artifacts-root DIR] [--run-dir DIR] [--out FILE]

Collects GPU interaction-discovery status from:
  - manifest + generated scripts under <run-dir>/hpc_scripts
  - shard outputs under <run-dir>/hpc_shards
  - merged outputs under <run-dir>/hpc_shards/_merged
  - SLURM logs from the configured log directory
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --artifacts-root)
      ARTIFACTS_ROOT="${2:-}"
      shift 2
      ;;
    --run-dir)
      RUN_DIR="${2:-}"
      RUN_DIR_SET=1
      shift 2
      ;;
    --out)
      OUT_FILE="${2:-}"
      OUT_FILE_SET=1
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "${RUN_DIR_SET}" -eq 0 ]]; then
  RUN_DIR="${ARTIFACTS_ROOT}/kestrel_gpu_h100_run"
fi
if [[ "${OUT_FILE_SET}" -eq 0 ]]; then
  OUT_FILE="${RUN_DIR}/gpu_interaction_results_summary.csv"
fi

SCRIPTS_DIR="${RUN_DIR}/hpc_scripts"
SHARDS_DIR="${RUN_DIR}/hpc_shards"
MANIFEST_PATH="${SCRIPTS_DIR}/manifest.jsonl"
STAGE_SCRIPT_GPU="${SCRIPTS_DIR}/submit_interaction_discovery_gpu_array.sh"
STAGE_SCRIPT_CPU="${SCRIPTS_DIR}/submit_interaction_discovery_array.sh"
MERGED_DIR="${SHARDS_DIR}/_merged"
MERGED_JSON="${MERGED_DIR}/interaction_discovery_merged.json"
MERGED_RETAINED="${MERGED_DIR}/retained_interaction_pairs_merged.csv"
MERGED_SCORES="${MERGED_DIR}/interaction_pair_scores_merged.csv"

mkdir -p "$(dirname "${OUT_FILE}")"

resolve_log_dir() {
  local stage_script="$1"
  local output_path=""
  if [[ -f "${stage_script}" ]]; then
    output_path="$(awk -F= '/^#SBATCH --output=/{print $2; exit}' "${stage_script}")"
  fi
  if [[ -z "${output_path}" ]]; then
    return 1
  fi
  local raw_dir
  raw_dir="$(dirname "${output_path}")"
  eval "echo \"${raw_dir}\""
}

latest_log() {
  local glob_pattern="$1"
  local newest
  newest="$(ls -1t ${glob_pattern} 2>/dev/null | head -n 1 || true)"
  printf '%s' "${newest}"
}

manifest_shards=0
[[ -f "${MANIFEST_PATH}" ]] && manifest_shards="$(wc -l < "${MANIFEST_PATH}" | tr -d ' ')"

shard_results=0
completed_shards=0
failed_shards=0
retained_shards=0
if [[ -d "${SHARDS_DIR}" ]]; then
  shard_results="$(find "${SHARDS_DIR}" -mindepth 2 -maxdepth 2 -type f -name shard_result.json | wc -l | tr -d ' ')"
  completed_shards="$(grep -R --include='shard_result.json' -h '"status": "completed"' "${SHARDS_DIR}" 2>/dev/null | wc -l | tr -d ' ')"
  failed_shards="$(grep -R --include='shard_result.json' -h '"status": "failed"' "${SHARDS_DIR}" 2>/dev/null | wc -l | tr -d ' ')"
  retained_shards="$(find "${SHARDS_DIR}" -mindepth 2 -maxdepth 2 -type f -name retained_interaction_pairs.csv | wc -l | tr -d ' ')"
fi
if [[ "${retained_shards}" -gt "${completed_shards}" ]]; then
  completed_shards="${retained_shards}"
fi
if [[ "${retained_shards}" -gt "${shard_results}" ]]; then
  shard_results="${retained_shards}"
fi

merged_retained_pairs=0
merged_pair_scores=0
[[ -f "${MERGED_RETAINED}" ]] && merged_retained_pairs="$(( $(wc -l < "${MERGED_RETAINED}") - 1 ))"
[[ -f "${MERGED_SCORES}" ]] && merged_pair_scores="$(( $(wc -l < "${MERGED_SCORES}") - 1 ))"
[[ "${merged_retained_pairs}" -lt 0 ]] && merged_retained_pairs=0
[[ "${merged_pair_scores}" -lt 0 ]] && merged_pair_scores=0

log_dir=""
if ! log_dir="$(resolve_log_dir "${STAGE_SCRIPT_GPU}")"; then
  if ! log_dir="$(resolve_log_dir "${STAGE_SCRIPT_CPU}")"; then
    log_dir="${LOGS_ROOT}/bsm_kestrel_gpu_h100/logs"
  fi
fi

latest_array_log="$(latest_log "${log_dir}/bsm_gpu_interaction_discovery_*.out")"
if [[ -z "${latest_array_log}" ]]; then
  latest_array_log="$(latest_log "${log_dir}/bsm_interaction_discovery_*.out")"
fi
latest_reduce_log="$(latest_log "${log_dir}/bsm_reduce_interaction_discovery_*.out")"

status="not_started"
if [[ ! -f "${MANIFEST_PATH}" ]]; then
  status="scripts_missing"
elif [[ "${failed_shards}" -gt 0 ]]; then
  status="failed"
elif [[ -f "${MERGED_JSON}" ]]; then
  status="complete"
elif [[ "${shard_results}" -gt 0 ]]; then
  status="running_or_waiting_reduce"
else
  status="queued_or_pending"
fi

printf '%s\n' "run_dir,manifest_shards,shard_results,completed_shards,failed_shards,merged_retained_pairs,merged_pair_scores,status,log_dir,latest_array_log,latest_reduce_log" > "${OUT_FILE}"
printf '"%s",%s,%s,%s,%s,%s,%s,%s,"%s","%s","%s"\n' \
  "${RUN_DIR}" "${manifest_shards}" "${shard_results}" "${completed_shards}" \
  "${failed_shards}" "${merged_retained_pairs}" "${merged_pair_scores}" "${status}" \
  "${log_dir}" "${latest_array_log}" "${latest_reduce_log}" >> "${OUT_FILE}"

echo "Wrote ${OUT_FILE}"
cat "${OUT_FILE}"
