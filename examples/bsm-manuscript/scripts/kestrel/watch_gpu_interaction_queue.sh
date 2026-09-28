#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# shellcheck source=common_paths.sh
source "${REPO_ROOT}/scripts/kestrel/common_paths.sh"
DEFAULT_ARTIFACTS_ROOT="$(kestrel_default_artifacts_root "${REPO_ROOT}")"

INTERVAL_SECONDS=30
TAIL_LINES=10
RUN_ONCE=0
ARTIFACTS_ROOT="${ARTIFACTS_ROOT:-${DEFAULT_ARTIFACTS_ROOT}}"
LOGS_ROOT="${LOGS_ROOT:-$(kestrel_default_logs_root)}"

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/kestrel/watch_gpu_interaction_queue.sh [--interval SECONDS] [--tail-lines N] [--artifacts-root DIR] [--logs-root DIR] [--once]

Monitors GPU interaction-discovery jobs by:
  1) printing squeue/sacct entries for kestrel_gpu_h100 jobs
  2) tailing latest GPU array/reduce logs
  3) refreshing GPU collector summary
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --interval)
      INTERVAL_SECONDS="${2:-}"
      shift 2
      ;;
    --tail-lines)
      TAIL_LINES="${2:-}"
      shift 2
      ;;
    --artifacts-root)
      ARTIFACTS_ROOT="${2:-}"
      shift 2
      ;;
    --logs-root)
      LOGS_ROOT="${2:-}"
      shift 2
      ;;
    --once)
      RUN_ONCE=1
      shift
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

if ! [[ "${INTERVAL_SECONDS}" =~ ^[0-9]+$ ]] || [[ "${INTERVAL_SECONDS}" -lt 1 ]]; then
  echo "error: --interval must be a positive integer" >&2
  exit 2
fi
if ! [[ "${TAIL_LINES}" =~ ^[0-9]+$ ]] || [[ "${TAIL_LINES}" -lt 1 ]]; then
  echo "error: --tail-lines must be a positive integer" >&2
  exit 2
fi

print_queue_snapshot() {
  echo "== squeue (${USER}) =="
  if command -v squeue >/dev/null 2>&1; then
    squeue -u "${USER}" -o "%.18i %.10T %.12M %.9l %.35j %R" | awk 'NR==1 || /kestrel_gpu_h100/'
  else
    echo "squeue not found"
  fi
  echo

  echo "== sacct (${USER}) =="
  if command -v sacct >/dev/null 2>&1; then
    sacct -u "${USER}" --format=JobID,JobName%45,State,Elapsed,ExitCode -n \
      | grep -E 'kestrel_gpu_h100|^$' || true
  else
    echo "sacct not found"
  fi
  echo
}

tail_gpu_logs() {
  local log_dir="${LOGS_ROOT}/bsm_kestrel_gpu_h100/logs"
  local gpu_array_log
  local cpu_array_log
  local reduce_log

  gpu_array_log="$(ls -1t "${log_dir}"/bsm_gpu_interaction_discovery_*.out 2>/dev/null | head -n 1 || true)"
  cpu_array_log="$(ls -1t "${log_dir}"/bsm_interaction_discovery_*.out 2>/dev/null | head -n 1 || true)"
  reduce_log="$(ls -1t "${log_dir}"/bsm_reduce_interaction_discovery_*.out 2>/dev/null | head -n 1 || true)"

  echo "== GPU logs (${log_dir}) =="
  if [[ -n "${gpu_array_log}" ]]; then
    echo "-- gpu array: ${gpu_array_log}"
    tail -n "${TAIL_LINES}" "${gpu_array_log}"
  elif [[ -n "${cpu_array_log}" ]]; then
    echo "-- array (cpu template): ${cpu_array_log}"
    tail -n "${TAIL_LINES}" "${cpu_array_log}"
  else
    echo "-- array: none"
  fi

  if [[ -n "${reduce_log}" ]]; then
    echo "-- reduce: ${reduce_log}"
    tail -n "${TAIL_LINES}" "${reduce_log}"
  else
    echo "-- reduce: none"
  fi
  echo
}

while true; do
  date -u +"%Y-%m-%dT%H:%M:%SZ"
  print_queue_snapshot
  tail_gpu_logs
  bash "${REPO_ROOT}/scripts/kestrel/collect_gpu_interaction_results.sh" --artifacts-root "${ARTIFACTS_ROOT}" >/dev/null
  cat "${ARTIFACTS_ROOT}/kestrel_gpu_h100_run/gpu_interaction_results_summary.csv"
  echo

  if [[ "${RUN_ONCE}" -eq 1 ]]; then
    break
  fi
  sleep "${INTERVAL_SECONDS}"
done
