#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$REPO_ROOT/$(basename "${BASH_SOURCE[0]}")"
MODE="${1:-}"
PIXI_BIN="${PIXI_BIN:-pixi}"

PREP_TASKS=(
  clean-transients
  format-python
  format-markdown
  fix-notebooks
)

VALIDATION_TASKS=(
  build-import-smoke
  clean-transients
  repo-hygiene
  lint
  format-check
  markdown-check
  notebook-check
  compile-check
  unit-tests
  workflow-tests
  manuscript-reproduction-smoke
  notebook-tests
  docs
  package-build
  clean-transients
  repo-hygiene
  git-diff-check
)

usage() {
  cat <<'USAGE'
Usage:
  ./test_repo.sh
  ./test_repo.sh --fix
  ./test_repo.sh --check
  ./test_repo.sh --ci
  ./test_repo.sh --clean
USAGE
}

ensure_pixi() {
  if ! command -v "$PIXI_BIN" >/dev/null 2>&1; then
    echo "error: pixi executable not found: $PIXI_BIN" >&2
    exit 1
  fi
}

run_task_list() {
  local task
  for task in "$@"; do
    echo ">>> pixi run $task"
    "$PIXI_BIN" run "$task"
  done
}

if [[ "${MODE}" == "--help" || "${MODE}" == "-h" ]]; then
  usage
  exit 0
fi

if [[ -f "$REPO_ROOT/pixi.toml" ]]; then
  ensure_pixi
  if [[ -z "${CAW_RUNNING_UNDER_PIXI:-}" ]]; then
    exec "$PIXI_BIN" run env CAW_RUNNING_UNDER_PIXI=1 bash "$SCRIPT_PATH" "${MODE}"
  fi
fi

if [[ "${MODE}" == "--check-only" ]]; then
  MODE="--check"
fi

case "${MODE}" in
--fix|"")
  run_task_list "${PREP_TASKS[@]}"
  run_task_list "${VALIDATION_TASKS[@]}"
  ;;
--check|--ci)
  run_task_list "${VALIDATION_TASKS[@]}"
  ;;
--clean)
  if [[ -d "$REPO_ROOT/.pixi" ]]; then
    rm -rf "$REPO_ROOT/.pixi"
  fi
  "$PIXI_BIN" install --locked
  bash "$SCRIPT_PATH" --fix
  ;;
*)
  echo "Unknown mode: ${MODE}" >&2
  usage >&2
  exit 2
  ;;
esac
