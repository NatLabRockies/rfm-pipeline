#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-artifacts/hpc_discovery}"
mkdir -p "$OUT_DIR"

pixi run python - <<'PY'
from __future__ import annotations
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

out = Path(os.environ.get("DISCOVERY_OUT_DIR", "artifacts/hpc_discovery"))
out.mkdir(parents=True, exist_ok=True)

rows = int(os.environ.get("DISC_ROWS", "200000"))
cols = int(os.environ.get("DISC_COLS", "200"))
row_group_size = int(os.environ.get("DISC_ROW_GROUP", "50000"))
target = out / "parquet_bench.parquet"

rng = np.random.default_rng(123)
data = {f"f{i}": rng.standard_normal(rows).astype(np.float32) for i in range(cols)}
df = pd.DataFrame(data)

t0 = time.perf_counter()
table = pa.Table.from_pandas(df, preserve_index=False)
pq.write_table(table, target, compression="zstd", row_group_size=row_group_size)
t1 = time.perf_counter()

t2 = time.perf_counter()
tbl2 = pq.read_table(target)
_ = (tbl2.num_rows, tbl2.num_columns)
t3 = time.perf_counter()

file_mib = target.stat().st_size / (1024**2)
result = {
    "rows": rows,
    "cols": cols,
    "row_group_size": row_group_size,
    "file_bytes": target.stat().st_size,
    "write_seconds": t1 - t0,
    "read_seconds": t3 - t2,
    "write_mib_per_s": file_mib / max(t1 - t0, 1e-9),
    "read_mib_per_s": file_mib / max(t3 - t2, 1e-9),
}
(out / "parquet_bench.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
PY
