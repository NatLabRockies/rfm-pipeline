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
#   STATUS_TIER_SHARD_DIRS=nodes1:shard_dir1:count1,nodes2:shard_dir2:count2
STATUS_CPU_TIERS="${STATUS_CPU_TIERS:-2,10,1000}"
STATUS_INCLUDE_GPU="${STATUS_INCLUDE_GPU:-1}"
STATUS_GPU_SHARDS="${STATUS_GPU_SHARDS:-10}"
STATUS_LOOKBACK_START="${STATUS_LOOKBACK_START:-$(date -u +%Y-%m-%d)}"
STATUS_TIER_SHARD_DIRS="${STATUS_TIER_SHARD_DIRS:-}"

CURRENT_USER="${USER:-$(id -un)}"

trim() {
  local value="$1"
  # shellcheck disable=SC2001
  value="$(echo "${value}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
  printf "%s" "${value}"
}

# ---------------------------------------------------------------------------
# Filesystem-based shard counting helpers (always correct, even after restarts)
# ---------------------------------------------------------------------------

# Count shards with _SUCCESS.json in their final output dir (authoritative)
_count_succeeded() {
  local shards_dir="$1"
  find "${shards_dir}" -mindepth 2 -maxdepth 2 -name '_SUCCESS.json' 2>/dev/null \
    | wc -l | tr -d ' \t'
}

# Count unique shard IDs that have _FAILURE.json in an attempt dir AND no _SUCCESS.json.
# These are truly failed, not shards that failed once then succeeded on a retry.
_count_failed() {
  local shards_dir="$1"
  local failed=0
  local raw

  raw=$(find "${shards_dir}" -mindepth 3 -maxdepth 3 -name '_FAILURE.json' 2>/dev/null \
    | while IFS= read -r f; do
        attempt_dir=$(dirname "$f")
        basename "${attempt_dir}" | sed 's/_attempt_[0-9]*$//'
      done | sort -u)

  [[ -z "${raw}" ]] && echo 0 && return

  while IFS= read -r shard_id; do
    [[ -z "${shard_id}" ]] && continue
    if [[ ! -f "${shards_dir}/${shard_id}/_SUCCESS.json" ]]; then
      failed=$((failed + 1))
    fi
  done <<< "${raw}"

  echo "${failed}"
}

# Print a three-line status block for any shards directory
print_shard_status() {
  local label="$1"
  local shards_dir="$2"
  local expected="$3"

  local succeeded=0 failed=0 pending=0 pct=0

  if [[ -d "${shards_dir}" ]]; then
    succeeded=$(_count_succeeded "${shards_dir}")
    failed=$(_count_failed "${shards_dir}")
  else
    echo "  ⚠️  Shards dir not found: ${shards_dir}"
  fi

  pending=$((expected - succeeded - failed))
  [[ ${pending} -lt 0 ]] && pending=0
  [[ "${expected}" -gt 0 ]] && pct=$(( succeeded * 100 / expected ))

  echo "  Succeeded : ${succeeded} / ${expected} (${pct}%)"
  if [[ "${failed}" -gt 0 ]]; then
    echo "  Failed    : ${failed}  ⚠️"
  else
    echo "  Failed    : 0"
  fi
  echo "  Waiting   : ${pending}"

  if [[ "${succeeded}" -ge "${expected}" ]]; then
    echo "  ✅ ALL SHARDS COMPLETE"
  elif [[ "${failed}" -gt 0 && $((succeeded + failed)) -ge "${expected}" ]]; then
    echo "  ❌ ALL DONE BUT ${failed} SHARD(S) FAILED — resubmit with: pixi run rfm-hpc-submit --config <cfg> --stage <stage> --submit"
  elif [[ "${failed}" -gt 0 ]]; then
    echo "  ⚠️  IN PROGRESS WITH FAILURES"
  fi
}

# ---------------------------------------------------------------------------
# Per-tier status printers
# ---------------------------------------------------------------------------

# Resolve the actual shard dir for a given tier from STATUS_TIER_SHARD_DIRS
# Format: "nodes:shard_dir:count,...". Falls back to ARTIFACTS_ROOT pattern.
_resolve_tier_shard_dir() {
  local tier="$1"
  local default_dir="${ARTIFACTS_ROOT}/kestrel_cpu_scale_${tier}_run/hpc_shards"

  if [[ -z "${STATUS_TIER_SHARD_DIRS}" ]]; then
    echo "${default_dir}"
    return
  fi

  local entry
  while IFS= read -r entry; do
    local t sd _rest
    IFS=: read -r t sd _rest <<< "${entry}"
    if [[ "${t}" == "${tier}" ]]; then
      echo "${sd}"
      return
    fi
  done <<< "$(echo "${STATUS_TIER_SHARD_DIRS}" | tr ',' '\n')"

  echo "${default_dir}"
}

print_cpu_status() {
  local tier="$1"
  local expected_shards="$2"
  local shards_dir
  shards_dir="$(_resolve_tier_shard_dir "${tier}")"

  echo "📊 CPU ${tier}-shard run:"

  # Show current squeue state (informational only — does not affect counts)
  local job_id state reason
  job_id="$(squeue -u "${CURRENT_USER}" --name="*bsm*cpu*${tier}*" --format="%i" --noheader 2>/dev/null | head -1 || true)"
  if [[ -n "${job_id}" ]]; then
    state="$(squeue -j "${job_id}" --format="%T" --noheader 2>/dev/null | head -1 || true)"
    reason="$(squeue -j "${job_id}" --format="%.25R" --noheader 2>/dev/null | head -1 || true)"
    echo "  Active job: ${job_id}  state=$(trim "${state}")  reason=$(trim "${reason}")"
  else
    sacct -u "${CURRENT_USER}" -S "${STATUS_LOOKBACK_START}" \
      --format=JobID,State,JobName --noheader 2>/dev/null \
      | grep -i "cpu.*${tier}\|${tier}.*cpu" \
      | tail -1 \
      | sed 's/^/  Last accounting: /' || true
  fi

  print_shard_status "CPU ${tier}-shard" "${shards_dir}" "${expected_shards}"
}

print_gpu_status() {
  local expected_shards="$1"
  local shards_dir="${ARTIFACTS_ROOT}/kestrel_gpu_h100_run/hpc_shards"

  echo "📊 GPU H100:"

  local job_id state reason
  job_id="$(squeue -u "${CURRENT_USER}" --name="*bsm*gpu*h100*" --format="%i" --noheader 2>/dev/null | head -1 || true)"
  if [[ -n "${job_id}" ]]; then
    state="$(squeue -j "${job_id}" --format="%T" --noheader 2>/dev/null | head -1 || true)"
    reason="$(squeue -j "${job_id}" --format="%.25R" --noheader 2>/dev/null | head -1 || true)"
    echo "  Active job: ${job_id}  state=$(trim "${state}")  reason=$(trim "${reason}")"
    if [[ "$(trim "${reason}")" == "DependencyNeverSatisfied" ]]; then
      echo "  ⚠️  DependencyNeverSatisfied — cancel and resubmit with: scancel ${job_id} && pixi run rfm-hpc-submit --config <cfg> --submit"
    fi
  else
    sacct -u "${CURRENT_USER}" -S "${STATUS_LOOKBACK_START}" \
      --format=JobID,State --noheader 2>/dev/null \
      | grep -i "gpu_h100" \
      | tail -1 \
      | sed 's/^/  Last accounting: /' || true
  fi

  print_shard_status "GPU H100" "${shards_dir}" "${expected_shards}"
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
  echo ""
done

if [[ "${STATUS_INCLUDE_GPU}" == "1" ]]; then
  if [[ "${STATUS_GPU_SHARDS}" =~ ^[0-9]+$ ]]; then
    print_gpu_status "${STATUS_GPU_SHARDS}"
    echo ""
  else
    echo "⚠️  Skipping GPU status: invalid STATUS_GPU_SHARDS=${STATUS_GPU_SHARDS}"
    echo ""
  fi
fi

echo "Legend:"
echo "  Succeeded = shards with _SUCCESS.json (filesystem truth)"
echo "  Failed    = shards with _FAILURE.json in attempt dir and no _SUCCESS.json"
echo "  Waiting   = expected - succeeded - failed"
echo ""
echo "To see details: squeue -u \$USER"
echo "Artifacts root: ${ARTIFACTS_ROOT}"
echo "Logs root: ${LOGS_ROOT}"
