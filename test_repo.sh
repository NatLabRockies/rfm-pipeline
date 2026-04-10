#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

PIXI_BIN="${PIXI_BIN:-$(command -v pixi)}"

if [[ -z "${PIXI_BIN:-}" ]] || [[ ! -x "$PIXI_BIN" ]]; then
  echo "error: pixi executable not found" >&2
  exit 1
fi

run_task() {
  echo ">>> $PIXI_BIN run $1"
  "$PIXI_BIN" run "$1"
}

run_python_smoke() {
  echo '>>> verifying build import in pixi environment'
  "$PIXI_BIN" run python -c "import sys, importlib.util; print(sys.executable); print(importlib.util.find_spec('build'))"
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
      run_task build-import-smoke
      run_task clean-transients
      run_task format-python
      run_task format-markdown
      run_task fix-notebooks
      run_task clean-transients
      run_task repo-hygiene
      run_task lint
      run_task format-check
      run_task markdown-check
      run_task notebook-check
      run_task compile-check
      run_task unit-tests
      run_task workflow-tests
      run_task notebook-tests
      run_task docs
      run_task package-build
      run_task clean-transients
      run_task repo-hygiene
      run_task git-diff-check
      ;;
    --check|--ci)
      sync_env_locked
      run_python_smoke
      run_task build-import-smoke
      run_task clean-transients
      run_task repo-hygiene
      run_task lint
      run_task format-check
      run_task markdown-check
      run_task notebook-check
      run_task compile-check
      run_task unit-tests
      run_task workflow-tests
      run_task notebook-tests
      run_task docs
      run_task package-build
      run_task clean-transients
      run_task repo-hygiene
      run_task git-diff-check
      ;;
    --clean)
      echo ">>> $PIXI_BIN clean"
      "$PIXI_BIN" clean
      rm -rf .pixi
      "$0" --fix
      ;;
    *)
      echo "usage: ./test_repo.sh [--fix|--check|--ci|--clean]" >&2
      exit 2
      ;;
  esac
}

main "$@"
