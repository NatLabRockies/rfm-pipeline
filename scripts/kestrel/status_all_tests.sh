#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# shellcheck source=common_paths.sh
source "${REPO_ROOT}/scripts/kestrel/common_paths.sh"

DEFAULT_ARTIFACTS_ROOT="$(kestrel_default_artifacts_root "${REPO_ROOT}")"
ARTIFACTS_ROOT="${ARTIFACTS_ROOT:-${DEFAULT_ARTIFACTS_ROOT}}"
LOGS_ROOT="${LOGS_ROOT:-$(kestrel_default_logs_root)}"

# Scope controls (set by orchestration command; overridable for manual use):
#   STATUS_CPU_TIERS=2,10,1000
#   STATUS_INCLUDE_GPU=0|1
#   STATUS_GPU_SHARDS=10
#   STATUS_LOOKBACK_START=YYYY-MM-DD
STATUS_CPU_TIERS="${STATUS_CPU_TIERS:-2,10,1000}"
STATUS_INCLUDE_GPU="${STATUS_INCLUDE_GPU:-1}"
STATUS_GPU_SHARDS="${STATUS_GPU_SHARDS:-10}"
STATUS_LOOKBACK_START="${STATUS_LOOKBACK_START:-$(date -u +%Y-%m-%d)}"

CURRENT_USER="${USER:-$(id -un)}"

trim() {
  local value="$1"
  # shellcheck disable=SC2001
  value="$(echo "${value}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
  printf "%s" "${value}"
}

print_cpu_status() {
  local tier="$1"
  local expected_shards="$2"
  local run_dir="${ARTIFACTS_ROOT}/kestrel_cpu_scale_${tier}_run"
  local log_dir="${LOGS_ROOT}/bsm_kestrel_cpu_scale_${tier}/logs"
  local complete=0
  local percent=0
  local job_id=""
  local state=""

  echo "📊 CPU ${tier}-node:"

  job_id="$(squeue -u "${CURRENT_USER}" --name="*bsm*cpu_scale_${tier}*" --format="%i" --noheader 2>/dev/null | head -1 || true)"
  if [[ -n "${job_id}" ]]; then
    state="$(squeue -j "${job_id}" --format="%t" --noheader 2>/dev/null | head -1 || true)"
    [[ -n "${state}" ]] && echo "  State: ${state} (Job ID: ${job_id})"
  else
    sacct -u "${CURRENT_USER}" -S "${STATUS_LOOKBACK_START}" --format=JobID,State --noheader 2>/dev/null \
      | grep -i "cpu_scale_${tier}" \
      | tail -1 \
      | sed 's/^/  Last accounting: /' || true
  fi

  if [[ -d "${run_dir}/hpc_shards" ]]; then
    complete="$(find "${run_dir}/hpc_shards" -name "retained_interaction_pairs.csv" 2>/dev/null | wc -l | awk '{print $1}')"
  fi
  if [[ "${expected_shards}" -gt 0 ]]; then
    percent=$(( complete * 100 / expected_shards ))
  fi
  echo "  Progress: ${complete} / ${expected_shards} shards complete (${percent}%)"

  if [[ -d "${log_dir}" ]] && grep -q "error\|Error\|ERROR\|FAILED" "${log_dir}"/*.out 2>/dev/null; then
    echo "  ⚠️  Errors detected in logs"
    grep "error\|Error\|ERROR\|FAILED" "${log_dir}"/*.out 2>/dev/null | head -3 | sed 's/^/  /' || true
  fi
  echo ""
}

print_gpu_status() {
  local expected_shards="$1"
  local run_dir="${ARTIFACTS_ROOT}/kestrel_gpu_h100_run"
  local log_dir="${LOGS_ROOT}/bsm_kestrel_gpu_h100/logs"
  local complete=0
  local percent=0
  local job_id=""
  local state=""
  local reason=""

  echo "📊 GPU H100:"

  job_id="$(squeue -u "${CURRENT_USER}" --name="*bsm*gpu*h100*" --format="%i" --noheader 2>/dev/null | head -1 || true)"
  if [[ -n "${job_id}" ]]; then
    state="$(squeue -j "${job_id}" --format="%t" --noheader 2>/dev/null | head -1 || true)"
    reason="$(squeue -j "${job_id}" --format="%.20R" --noheader 2>/dev/null | head -1 || true)"
    [[ -n "${state}" ]] && echo "  State: ${state} (Job ID: ${job_id})"
    if [[ "${state}" == "PD" && -n "${reason}" ]]; then
      echo "  Reason: ${reason}"
    fi
  else
    sacct -u "${CURRENT_USER}" -S "${STATUS_LOOKBACK_START}" --format=JobID,State --noheader 2>/dev/null \
      | grep -i "gpu_h100" \
      | tail -1 \
      | sed 's/^/  Last accounting: /' || true
  fi

  if [[ -d "${run_dir}/hpc_shards" ]]; then
    complete="$(find "${run_dir}/hpc_shards" -name "retained_interaction_pairs.csv" 2>/dev/null | wc -l | awk '{print $1}')"
  fi
  if [[ "${expected_shards}" -gt 0 ]]; then
    percent=$(( complete * 100 / expected_shards ))
  fi
  echo "  Progress: ${complete} / ${expected_shards} shards complete (${percent}%)"

  if [[ -d "${log_dir}" ]] && grep -q "error\|Error\|ERROR\|FAILED" "${log_dir}"/*.out 2>/dev/null; then
    echo "  ⚠️  Errors detected in logs"
    grep "error\|Error\|ERROR\|FAILED" "${log_dir}"/*.out 2>/dev/null | head -3 | sed 's/^/  /' || true
  fi
  echo ""
}

echo "=== HPC Test Status $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo ""
echo "Scope: cpu_tiers=${STATUS_CPU_TIERS} include_gpu=${STATUS_INCLUDE_GPU}"
echo ""

IFS=',' read -r -a RAW_TIERS <<< "${STATUS_CPU_TIERS}"
for raw_tier in "${RAW_TIERS[@]}"; do
  tier="$(trim "${raw_tier}")"
  [[ -z "${tier}" ]] && continue
  if [[ ! "${tier}" =~ ^[0-9]+$ ]]; then
    echo "⚠️  Skipping invalid CPU tier value: ${tier}"
    echo ""
    continue
  fi
  print_cpu_status "${tier}" "${tier}"
done

if [[ "${STATUS_INCLUDE_GPU}" == "1" ]]; then
  if [[ "${STATUS_GPU_SHARDS}" =~ ^[0-9]+$ ]]; then
    print_gpu_status "${STATUS_GPU_SHARDS}"
  else
    echo "⚠️  Skipping GPU status: invalid STATUS_GPU_SHARDS=${STATUS_GPU_SHARDS}"
    echo ""
  fi
fi

echo "Legend:"
echo "  R  = Running"
echo "  PD = Pending (queued)"
echo "  CA = Cancelled"
echo "  ST = Stopped"
echo ""
echo "To see details: squeue -u \$USER"
echo "Artifacts root: ${ARTIFACTS_ROOT}"
echo "Logs root: ${LOGS_ROOT}"
echo "To tail logs: tail -f ${LOGS_ROOT}/bsm_kestrel_cpu_scale_*/logs/*.out"
