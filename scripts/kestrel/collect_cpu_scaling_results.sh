#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SUITE_ROOT="${REPO_ROOT}/artifacts/kestrel_cpu_scaling_suite"
OUT_FILE="${SUITE_ROOT}/cpu_scaling_results_summary.csv"

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/kestrel/collect_cpu_scaling_results.sh [--suite-root DIR] [--out FILE]

Collects CPU scaling status for tiers 2/10/1000 from:
  - generated manifest/scripts under artifacts/kestrel_cpu_scaling_suite/
  - run artifacts under artifacts/kestrel_cpu_scale_{2,10,1000}_run/
  - merged reduce outputs under .../_merged/
  - SLURM logs under each tier's configured log directory
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --suite-root)
      SUITE_ROOT="${2:-}"
      shift 2
      ;;
    --out)
      OUT_FILE="${2:-}"
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

printf '%s\n' "tier,run_dir,manifest_shards,shard_results,completed_shards,failed_shards,merged_retained_pairs,merged_pair_scores,status,log_dir,latest_array_log,latest_reduce_log" > "${OUT_FILE}"

for tier in 2 10 1000; do
  scripts_dir="${SUITE_ROOT}/cpu_nodes_${tier}/hpc_scripts"
  run_dir="${REPO_ROOT}/artifacts/kestrel_cpu_scale_${tier}_run"
  manifest_path="${scripts_dir}/manifest.jsonl"
  merged_dir="${run_dir}/_merged"
  merged_json="${merged_dir}/interaction_discovery_merged.json"
  merged_retained="${merged_dir}/retained_interaction_pairs_merged.csv"
  merged_scores="${merged_dir}/interaction_pair_scores_merged.csv"
  stage_script="${scripts_dir}/submit_interaction_discovery_array.sh"

  manifest_shards=0
  [[ -f "${manifest_path}" ]] && manifest_shards="$(wc -l < "${manifest_path}" | tr -d ' ')"

  shard_results=0
  completed_shards=0
  failed_shards=0
  if [[ -d "${run_dir}" ]]; then
    shard_results="$(find "${run_dir}" -mindepth 2 -maxdepth 2 -type f -name shard_result.json | wc -l | tr -d ' ')"
    completed_shards="$(grep -R --include='shard_result.json' -h '"status": "completed"' "${run_dir}" 2>/dev/null | wc -l | tr -d ' ')"
    failed_shards="$(grep -R --include='shard_result.json' -h '"status": "failed"' "${run_dir}" 2>/dev/null | wc -l | tr -d ' ')"
  fi

  merged_retained_pairs=0
  merged_pair_scores=0
  [[ -f "${merged_retained}" ]] && merged_retained_pairs="$(( $(wc -l < "${merged_retained}") - 1 ))"
  [[ -f "${merged_scores}" ]] && merged_pair_scores="$(( $(wc -l < "${merged_scores}") - 1 ))"
  [[ "${merged_retained_pairs}" -lt 0 ]] && merged_retained_pairs=0
  [[ "${merged_pair_scores}" -lt 0 ]] && merged_pair_scores=0

  log_dir=""
  if ! log_dir="$(resolve_log_dir "${stage_script}")"; then
    log_dir="/scratch/${USER}/bsm/bsm_kestrel_cpu_scale_${tier}/logs"
  fi
  latest_array_log="$(latest_log "${log_dir}/bsm_interaction_discovery_*.out")"
  latest_reduce_log="$(latest_log "${log_dir}/bsm_reduce_interaction_discovery_*.out")"

  status="not_started"
  if [[ ! -f "${manifest_path}" ]]; then
    status="scripts_missing"
  elif [[ "${failed_shards}" -gt 0 ]]; then
    status="failed"
  elif [[ -f "${merged_json}" ]]; then
    status="complete"
  elif [[ "${shard_results}" -gt 0 ]]; then
    status="running_or_waiting_reduce"
  else
    status="queued_or_pending"
  fi

  printf '%s,"%s",%s,%s,%s,%s,%s,%s,%s,"%s","%s","%s"\n' \
    "${tier}" "${run_dir}" "${manifest_shards}" "${shard_results}" "${completed_shards}" \
    "${failed_shards}" "${merged_retained_pairs}" "${merged_pair_scores}" "${status}" \
    "${log_dir}" "${latest_array_log}" "${latest_reduce_log}" >> "${OUT_FILE}"
done

echo "Wrote ${OUT_FILE}"
cat "${OUT_FILE}"
