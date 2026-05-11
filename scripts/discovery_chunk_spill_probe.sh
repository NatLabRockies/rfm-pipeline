#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=hpc_kestrel_config.sh
source "${SCRIPT_DIR}/hpc_kestrel_config.sh"

OUT_DIR="${1:-${DISCOVERY_OUT_DIR}}"
OUT_DIR="$(kestrel_realpath "${OUT_DIR}")"
mkdir -p "${OUT_DIR}"
export DISCOVERY_OUT_DIR="${OUT_DIR}"

SPILL_DIR="$(kestrel_choose_spill_dir)"
mkdir -p "${SPILL_DIR}/tmp"
export DISCOVERY_SPILL_DIR="${SPILL_DIR}"
export TMPDIR="${SPILL_DIR}/tmp"

if ! command -v pixi >/dev/null 2>&1; then
  echo "ERROR: pixi not found. Run ./install_pixi_project_env.sh first or source ${OUT_DIR}/pixi_kestrel_env.sh." >&2
  exit 2
fi

pixi run python - <<'PY'
from __future__ import annotations
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

out = Path(os.environ["DISCOVERY_OUT_DIR"])
out.mkdir(parents=True, exist_ok=True)

chunks = int(os.environ.get("DISC_CHUNKS", "20"))
chunk_rows = int(os.environ.get("DISC_CHUNK_ROWS", "200000"))
chunk_cols = int(os.environ.get("DISC_CHUNK_COLS", "64"))
spill_dir = Path(os.environ["DISCOVERY_SPILL_DIR"])
spill_dir.mkdir(parents=True, exist_ok=True)

tmp_root = spill_dir / f"chunks_{os.environ.get('SLURM_JOB_ID', 'manual')}_{os.environ.get('SLURM_ARRAY_TASK_ID', '0')}"
tmp_root.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)
paths: list[Path] = []
t0 = time.perf_counter()
for i in range(chunks):
    arr = rng.standard_normal((chunk_rows, chunk_cols), dtype=np.float32)
    p = tmp_root / f"chunk_{i:04d}.npy"
    np.save(p, arr)
    paths.append(p)
t1 = time.perf_counter()

total = np.zeros((chunk_cols,), dtype=np.float64)
t2 = time.perf_counter()
for p in paths:
    arr = np.load(p, mmap_mode="r")
    total += arr.mean(axis=0)
t3 = time.perf_counter()

usage = shutil.disk_usage(spill_dir)
result = {
    "job_id": os.environ.get("SLURM_JOB_ID"),
    "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    "chunks": chunks,
    "chunk_shape": [chunk_rows, chunk_cols],
    "spill_dir": str(tmp_root),
    "tmpdir": tempfile.gettempdir(),
    "spill_files": len(paths),
    "spill_total_bytes": int(sum(p.stat().st_size for p in paths)),
    "spill_write_seconds": t1 - t0,
    "stream_read_seconds": t3 - t2,
    "disk_total_bytes": usage.total,
    "disk_used_bytes": usage.used,
    "disk_free_bytes": usage.free,
    "mean_checksum": float(total.sum()),
}
name = f"chunk_spill_probe_{os.environ.get('SLURM_JOB_ID', 'manual')}_{os.environ.get('SLURM_ARRAY_TASK_ID', '0')}.json"
(out / name).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
(out / "chunk_spill_probe_latest.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(result, indent=2, sort_keys=True))
PY
