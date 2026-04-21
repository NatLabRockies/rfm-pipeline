#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$REPO_ROOT/$(basename "${BASH_SOURCE[0]}")"
cd "$REPO_ROOT"

discover_pixi() {
  if [[ -n "${PIXI_BIN:-}" ]]; then
    printf '%s\n' "$PIXI_BIN"
    return 0
  fi
  command -v pixi 2>/dev/null || true
}

PIXI_BIN="$(discover_pixi)"

if [[ -z "${PIXI_BIN:-}" ]] || [[ ! -x "$PIXI_BIN" ]]; then
  echo "error: pixi executable not found" >&2
  exit 1
fi

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
  notebook-tests
  docs
  package-build
  clean-transients
  repo-hygiene
  git-diff-check
)

run_task() {
  echo ">>> $PIXI_BIN run $1"
  "$PIXI_BIN" run "$1"
}

run_python_smoke() {
  echo '>>> verifying build import in pixi environment'
  "$PIXI_BIN" run python -c "import sys, importlib.util; print(sys.executable); print(importlib.util.find_spec('build'))"
}

run_task_list() {
  local task
  for task in "$@"; do
    run_task "$task"
  done
}

sync_env_fix() {
  echo ">>> $PIXI_BIN install"
  "$PIXI_BIN" install
}

sync_env_locked() {
  echo ">>> $PIXI_BIN install --locked"
  "$PIXI_BIN" install --locked
}

main() {
  local mode="${1:---fix}"

  case "$mode" in
    --fix|"")
      sync_env_fix
      run_python_smoke
      run_task_list "${PREP_TASKS[@]}"
      run_task_list "${VALIDATION_TASKS[@]}"
      ;;
    --check|--ci)
      sync_env_locked
      run_python_smoke
      run_task_list "${VALIDATION_TASKS[@]}"
      ;;
    --clean)
      echo ">>> $PIXI_BIN clean"
      "$PIXI_BIN" clean
      rm -rf .pixi
      bash "$SCRIPT_PATH" --fix
      ;;
    *)
      echo "usage: ./test_repo.sh [--fix|--check|--ci|--clean]" >&2
      exit 2
      ;;
  esac
}

main "$@"
