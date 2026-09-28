#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# shellcheck source=../hpc_kestrel_config.sh
source "${REPO_ROOT}/scripts/hpc_kestrel_config.sh"

cd "${REPO_ROOT}"

MODE="generate"
DRY_RUN=0
STAGE="interaction_discovery"
OUTPUT_ROOT="${REPO_ROOT}/artifacts/kestrel_cpu_scaling_suite"

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/kestrel/submit_cpu_scaling_suite.sh [--submit] [--dry-run] [--stage STAGE] [--output-root DIR]

Behavior:
  - Always generates a diagnostic script/job first (2-node config).
  - Then processes CPU scaling tiers: 2 nodes, 10 nodes, 1000 nodes.
  - Default mode generates scripts only. Use --submit to submit jobs.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --submit)
      MODE="submit"
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --stage)
      STAGE="${2:-}"
      if [[ -z "${STAGE}" ]]; then
        echo "error: --stage requires a value" >&2
        exit 2
      fi
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="${2:-}"
      if [[ -z "${OUTPUT_ROOT}" ]]; then
        echo "error: --output-root requires a value" >&2
        exit 2
      fi
      shift 2
      ;;
    --help|-h)
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

OUTPUT_ROOT="$(kestrel_realpath "${OUTPUT_ROOT}")"
mkdir -p "${OUTPUT_ROOT}"

run_submit_tool() {
  local -a cmd=("$@")
  echo ">>> ${cmd[*]}"
  "${cmd[@]}"
}

common_flags=()
if [[ "${MODE}" == "submit" ]]; then
  common_flags+=(--submit)
fi
if [[ "${DRY_RUN}" -eq 1 ]]; then
  common_flags+=(--dry-run)
fi

echo ">>> CPU scaling suite mode=${MODE} stage=${STAGE} output_root=${OUTPUT_ROOT}"

# Step 0: diagnostic smoke test (debug partition config).
run_submit_tool \
  pixi run rfm-hpc-submit \
  --config configs/hpc/dev/kestrel_cpu_scale_2.yml \
  --diagnostic-only \
  --output-dir "${OUTPUT_ROOT}/diagnostic/hpc_scripts" \
  "${common_flags[@]}"

declare -a TIER_CONFIGS=(
  "2:configs/hpc/dev/kestrel_cpu_scale_2.yml"
  "10:configs/hpc/dev/kestrel_cpu_scale_10.yml"
  "1000:configs/hpc/dev/kestrel_cpu_scale_1000.yml"
)

for tier in "${TIER_CONFIGS[@]}"; do
  nodes="${tier%%:*}"
  cfg="${tier#*:}"
  run_submit_tool \
    pixi run rfm-hpc-submit \
    --config "${cfg}" \
    --stage "${STAGE}" \
    --n-shards "${nodes}" \
    --output-dir "${OUTPUT_ROOT}/cpu_nodes_${nodes}/hpc_scripts" \
    "${common_flags[@]}"
done

echo ">>> CPU scaling suite complete"
echo ">>> Generated artifacts under: ${OUTPUT_ROOT}"
