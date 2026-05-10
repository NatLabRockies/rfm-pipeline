#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-artifacts/hpc_discovery}"
mkdir -p "$OUT_DIR"

pixi run python - <<'PY'
from __future__ import annotations
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np

out = Path(os.environ.get("DISCOVERY_OUT_DIR", "artifacts/hpc_discovery"))
out.mkdir(parents=True, exist_ok=True)

chunks = int(os.environ.get("DISC_CHUNKS", "20"))
chunk_rows = int(os.environ.get("DISC_CHUNK_ROWS", "200000"))
chunk_cols = int(os.environ.get("DISC_CHUNK_COLS", "64"))

tmp_root = Path(tempfile.gettempdir()) / "spill_probe"
tmp_root.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)
paths = []
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

result = {
    "chunks": chunks,
    "chunk_shape": [chunk_rows, chunk_cols],
    "spill_dir": str(tmp_root),
    "spill_files": len(paths),
    "spill_total_bytes": int(sum(p.stat().st_size for p in paths)),
    "spill_write_seconds": t1 - t0,
    "stream_read_seconds": t3 - t2,
}
(out / "chunk_spill_probe.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
PY
