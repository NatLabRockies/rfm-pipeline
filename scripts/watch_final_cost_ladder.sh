#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/watch_final_cost_ladder.sh [run_root] [interval_seconds] [--once]

Examples:
  scripts/watch_final_cost_ladder.sh
  scripts/watch_final_cost_ladder.sh artifacts/final_cost_ladder 600
  scripts/watch_final_cost_ladder.sh artifacts/final_cost_ladder 600 --once
USAGE
}

RUN_ROOT="${1:-artifacts/final_cost_ladder}"
INTERVAL_SECONDS="${2:-600}"
ONCE="false"
if [[ "${3:-}" == "--once" ]]; then
  ONCE="true"
elif [[ -n "${3:-}" ]]; then
  usage
  exit 2
fi

if [[ "${RUN_ROOT}" == "--help" || "${RUN_ROOT}" == "-h" ]]; then
  usage
  exit 0
fi

if ! [[ "${INTERVAL_SECONDS}" =~ ^[0-9]+$ ]] || [[ "${INTERVAL_SECONDS}" -lt 1 ]]; then
  echo "error: interval_seconds must be a positive integer" >&2
  exit 2
fi

json_field() {
  local file="$1"
  local key="$2"
  [[ -f "${file}" ]] || return 0
  sed -nE "s/.*\"${key}\"[[:space:]]*:[[:space:]]*\"?([^\",}]*)\"?.*/\\1/p" "${file}" | head -n1
}

print_run_status() {
  local run_dir="$1"
  local run_name
  run_name="$(basename "${run_dir}")"

  local status="unknown"
  local elapsed=""
  local marker=""
  if [[ -f "${run_dir}/run_complete.json" ]]; then
    status="complete"
    elapsed="$(json_field "${run_dir}/run_complete.json" "elapsed_seconds")"
    marker="run_complete.json"
  elif [[ -f "${run_dir}/run_failed.json" ]]; then
    status="failed"
    marker="run_failed.json"
  elif [[ -f "${run_dir}/run_started.json" ]]; then
    local started_pid
    started_pid="$(json_field "${run_dir}/run_started.json" "pid")"
    if [[ -n "${started_pid}" ]] && ps -p "${started_pid}" >/dev/null 2>&1; then
      status="running"
      marker="run_started.json(pid=${started_pid})"
    elif [[ -f "${run_dir}/run_abandoned.json" ]]; then
      status="abandoned"
      marker="run_abandoned.json"
    else
      status="started_no_live_pid"
      marker="run_started.json(pid=${started_pid:-unknown})"
    fi
  elif [[ -f "${run_dir}/run_abandoned.json" ]]; then
    status="abandoned"
    marker="run_abandoned.json"
  fi

  local stage=""
  local prog=""
  local detail=""
  local progress_file="${run_dir}/runtime_diagnostics/stage_progress.json"
  if [[ -f "${progress_file}" ]]; then
    stage="$(json_field "${progress_file}" "stage")"
    local done total
    done="$(json_field "${progress_file}" "completed")"
    total="$(json_field "${progress_file}" "total")"
    detail="$(json_field "${progress_file}" "detail")"
    if [[ -n "${done}" && -n "${total}" ]]; then
      prog="${done}/${total}"
    fi
  fi

  local sparse_seconds=""
  local final_seconds=""
  local runtime_csv="${run_dir}/runtime_diagnostics/stage_runtime_summary.csv"
  if [[ -f "${runtime_csv}" ]]; then
    sparse_seconds="$(
      awk -F, '$1=="sparse_selection_and_stability"{print $2}' "${runtime_csv}" | head -n1
    )"
    final_seconds="$(
      awk -F, '$1=="final_manuscript_tables_and_figures"{print $2}' "${runtime_csv}" | head -n1
    )"
  fi

  local n_final=""
  local final_summary="${run_dir}/final_manuscript_artifacts/final_artifact_summary.csv"
  if [[ -f "${final_summary}" ]]; then
    n_final="$(awk -F, 'NR==2{print $3}' "${final_summary}")"
  fi

  printf 'run=%s status=%s marker=%s\n' "${run_name}" "${status}" "${marker}"
  [[ -n "${elapsed}" ]] && printf '  elapsed_seconds=%s\n' "${elapsed}"
  [[ -n "${stage}" || -n "${prog}" ]] && printf '  progress=%s %s\n' "${stage}" "${prog}"
  [[ -n "${detail}" ]] && printf '  detail=%s\n' "${detail}"
  [[ -n "${sparse_seconds}" ]] && printf '  sparse_seconds=%s\n' "${sparse_seconds}"
  [[ -n "${final_seconds}" ]] && printf '  final_seconds=%s\n' "${final_seconds}"
  [[ -n "${n_final}" ]] && printf '  n_final_support=%s\n' "${n_final}"
  echo
}

print_snapshot() {
  clear
  printf 'Final-cost ladder monitor\n'
  printf 'time_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'run_root=%s\n\n' "${RUN_ROOT}"

  if [[ ! -d "${RUN_ROOT}" ]]; then
    echo "No run root found yet."
    return 0
  fi

  local found="false"
  while IFS= read -r run_dir; do
    found="true"
    print_run_status "${run_dir}"
  done < <(find "${RUN_ROOT}" -mindepth 1 -maxdepth 1 -type d | sort -V)

  if [[ "${found}" == "false" ]]; then
    echo "No run directories found."
  fi
}

while true; do
  print_snapshot
  if [[ "${ONCE}" == "true" ]]; then
    break
  fi
  printf 'Refreshing in %ss (Ctrl+C to stop)\n' "${INTERVAL_SECONDS}"
  sleep "${INTERVAL_SECONDS}"
done
