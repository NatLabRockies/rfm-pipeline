#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

CONFIG_PATH="configs/hpc/dev/kestrel_workflow_small_distributed.yml"
POLL_COUNT=3
POLL_SECONDS=60
DRY_RUN=0
SKIP_COLLECT=0

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/kestrel/run_small_distributed_test_local.sh [options]

Runs a small distributed smoke workflow locally via the unified orchestration runner:
  1) submit (lightweight full pipeline materialization + 2-node distributed interaction smoke)
  2) status snapshots (poll loop)
  3) collect (study-package pullback by default)

Options:
  --config FILE         Orchestration config (default: configs/hpc/dev/kestrel_workflow_small_distributed.yml)
  --poll-count N        Number of status polls after submit (default: 3)
  --poll-seconds N      Seconds between status polls (default: 60)
  --skip-collect        Skip final collect action
  --dry-run             Print commands only
  -h, --help            Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="${2:-}"
      shift 2
      ;;
    --poll-count)
      POLL_COUNT="${2:-}"
      shift 2
      ;;
    --poll-seconds)
      POLL_SECONDS="${2:-}"
      shift 2
      ;;
    --skip-collect)
      SKIP_COLLECT=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
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

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "error: config file not found: ${CONFIG_PATH}" >&2
  exit 2
fi
if ! [[ "${POLL_COUNT}" =~ ^[0-9]+$ ]]; then
  echo "error: --poll-count must be a non-negative integer" >&2
  exit 2
fi
if ! [[ "${POLL_SECONDS}" =~ ^[0-9]+$ ]] || [[ "${POLL_SECONDS}" -lt 1 ]]; then
  echo "error: --poll-seconds must be a positive integer" >&2
  exit 2
fi

common_flags=()
if [[ "${DRY_RUN}" -eq 1 ]]; then
  common_flags+=(--dry-run)
fi

run_step() {
  local action="$1"
  local -a cmd=(
    pixi
    run
    hpc-workflow
    --
    --config
    "${CONFIG_PATH}"
    --action
    "${action}"
  )
  if (( ${#common_flags[@]} > 0 )); then
    cmd+=("${common_flags[@]}")
  fi
  echo ">>> ${cmd[*]}"
  "${cmd[@]}"
}

echo "==> Small distributed smoke workflow (config=${CONFIG_PATH})"
run_step submit

for ((i = 1; i <= POLL_COUNT; i++)); do
  echo "==> Status poll ${i}/${POLL_COUNT}"
  run_step status
  if [[ "${i}" -lt "${POLL_COUNT}" ]]; then
    sleep "${POLL_SECONDS}"
  fi
done

if [[ "${SKIP_COLLECT}" -eq 0 ]]; then
  echo "==> Collecting smoke-run artifacts"
  run_step collect
fi

echo "==> Smoke workflow command chain complete"
echo "If submission/status fails, review and edit:"
echo "  - ${CONFIG_PATH}: cluster.host, cluster.user, paths.remote_repo_root"
echo "  - ${CONFIG_PATH}: paths.remote_artifacts_root, paths.remote_logs_root, paths.remote_snapshot_root"
echo "  - configs/hpc/dev/kestrel_cpu_scale_2_smoke.yml: distributed.slurm.account/partition if your allocation differs"
