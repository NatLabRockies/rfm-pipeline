#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

PIXI_BIN="${PIXI_BIN:-$(command -v pixi)}"

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

fail_on_untracked_markdown() {
  echo ">>> checking for untracked markdown files that CI would validate after commit"
  python - <<'PY'
from pathlib import Path
import subprocess
import sys

EXCLUDED_PARTS = {
    ".git",
    ".pixi",
    ".pytest_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
    ".venv",
    "build",
    "dist",
    "_build",
    "__MACOSX",
}
MARKDOWN_SUFFIXES = {".md", ".mdx", ".markdown"}

result = subprocess.run(
    ["git", "status", "--short", "--untracked-files=all"],
    check=True,
    capture_output=True,
    text=True,
)

bad: list[str] = []
for line in result.stdout.splitlines():
    if not line.startswith("?? "):
        continue
    rel = line[3:]
    path = Path(rel)
    if path.suffix.lower() not in MARKDOWN_SUFFIXES:
        continue
    if any(part in EXCLUDED_PARTS for part in path.parts):
        continue
    bad.append(rel)

if bad:
    print("error: untracked markdown files detected; these can bypass local formatting and fail CI after commit:")
    for item in bad:
        print(f" - {item}")
    print("run `pixi run format-markdown` after adding them, or add them before running the gate.")
    sys.exit(1)
PY
}

main() {
  local mode="${1:---fix}"

  if [[ -z "${PIXI_BIN:-}" ]] || [[ ! -x "$PIXI_BIN" ]]; then
    echo "error: pixi executable not found" >&2
    exit 1
  fi

  echo ">>> repo root: $REPO_ROOT"
  echo ">>> pixi bin: $PIXI_BIN"
  "$PIXI_BIN" --version

  export PYTHONDONTWRITEBYTECODE=1

  case "$mode" in
    --fix|"")
      sync_env_fix
      run_python_smoke
      run_task build-import-smoke
      fail_on_untracked_markdown
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
      fail_on_untracked_markdown
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
      rm -rf "$REPO_ROOT/.pixi"
      "$0" --fix
      ;;
    *)
      echo "usage: ./test_repo.sh [--fix|--check|--ci|--clean]" >&2
      exit 2
      ;;
  esac
}

main "$@"
