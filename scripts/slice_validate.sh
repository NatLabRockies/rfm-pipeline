#!/usr/bin/env bash
# slice-runner validation wrapper.
# DAG slice IDs are hyphenated (P0-S01) for dependency parsing; pytest -k needs
# underscores (P0_S01). Translate, then run only the alignment test suite.
set -euo pipefail
raw_id="${1:?usage: slice_validate.sh <SLICE-ID>}"
kexpr="${raw_id//-/_}"
exec pixi run python -m pytest tests/alignment -k "${kexpr}" --tb=short -q
