#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=hpc_kestrel_config.sh
source "${SCRIPT_DIR}/hpc_kestrel_config.sh"

OUT_DIR="${1:-${DISCOVERY_OUT_DIR}}"
OUT_DIR="$(kestrel_realpath "${OUT_DIR}")"
mkdir -p "${OUT_DIR}"
export DISCOVERY_OUT_DIR="${OUT_DIR}"

if ! command -v pixi >/dev/null 2>&1; then
  echo "ERROR: pixi not found. Run ./install_pixi_project_env.sh first or source ${OUT_DIR}/pixi_kestrel_env.sh." >&2
  exit 2
fi

# Default benchmark targets compare durable project storage, high-throughput
# scratch, and node-local TMPDIR only when TMPDIR appears to be real disk.
TARGETS=(
  "${KESTREL_SCRATCH_ROOT}/${SLURM_JOB_ID:-manual}/parquet_bench"
  "${KESTREL_PROJECT_ROOT}/hpc_discovery/${USER}/${SLURM_JOB_ID:-manual}/parquet_bench"
)
if kestrel_is_real_disk_tmpdir; then
  TARGETS+=("${TMPDIR}/bsm_parquet_bench")
fi

if [[ -n "${DISCOVERY_PARQUET_TARGETS:-}" ]]; then
  # Colon-separated override.
  IFS=':' read -r -a TARGETS <<< "${DISCOVERY_PARQUET_TARGETS}"
fi

TARGET_LIST_FILE="${OUT_DIR}/parquet_bench_targets.txt"
printf '%s\n' "${TARGETS[@]}" > "${TARGET_LIST_FILE}"
export DISCOVERY_PARQUET_TARGETS_FILE="${TARGET_LIST_FILE}"

pixi run python - <<'PY'
from __future__ import annotations
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

out = Path(os.environ["DISCOVERY_OUT_DIR"])
out.mkdir(parents=True, exist_ok=True)

targets = [Path(line.strip()) for line in Path(os.environ["DISCOVERY_PARQUET_TARGETS_FILE"]).read_text().splitlines() if line.strip()]
rows = int(os.environ.get("DISC_ROWS", "200000"))
cols = int(os.environ.get("DISC_COLS", "200"))
projection_cols = int(os.environ.get("DISC_PROJECTION_COLS", "50"))
compression = os.environ.get("DISC_PARQUET_COMPRESSION", "zstd")
target_row_group_mib = int(os.environ.get("DISC_TARGET_ROW_GROUP_MIB", "512"))

# PyArrow row_group_size is rows, not bytes. Estimate rows per row group from
# float32 payload width so ultra-wide matrices do not accidentally create giant
# row groups.
if "DISC_ROW_GROUP" in os.environ:
    row_group_size = int(os.environ["DISC_ROW_GROUP"])
else:
    bytes_per_row = max(1, cols * np.dtype(np.float32).itemsize)
    row_group_size = max(1, (target_row_group_mib * 1024**2) // bytes_per_row)
    row_group_size = min(rows, row_group_size)

rng = np.random.default_rng(123)
data = {f"f{i}": rng.standard_normal(rows).astype(np.float32) for i in range(cols)}
df = pd.DataFrame(data)
table = pa.Table.from_pandas(df, preserve_index=False)
columns = [f"f{i}" for i in range(min(projection_cols, cols))]

results = []
for target_dir in targets:
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"parquet_bench_{os.environ.get('SLURM_JOB_ID', 'manual')}.parquet"

    t0 = time.perf_counter()
    pq.write_table(table, target, compression=compression, row_group_size=row_group_size)
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    full = pq.read_table(target)
    _ = (full.num_rows, full.num_columns)
    t3 = time.perf_counter()

    t4 = time.perf_counter()
    projected = pq.read_table(target, columns=columns)
    _ = (projected.num_rows, projected.num_columns)
    t5 = time.perf_counter()

    pf = pq.ParquetFile(target)
    file_mib = target.stat().st_size / (1024**2)
    usage = shutil.disk_usage(target_dir)
    safe_name = str(target_dir).strip('/').replace('/', '_').replace(':', '_') or 'root'
    result = {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "target_dir": str(target_dir),
        "target_file": str(target),
        "rows": rows,
        "cols": cols,
        "projection_cols": len(columns),
        "compression": compression,
        "target_row_group_mib": target_row_group_mib,
        "row_group_size_rows": row_group_size,
        "row_groups": pf.metadata.num_row_groups,
        "file_bytes": target.stat().st_size,
        "write_seconds": t1 - t0,
        "full_read_seconds": t3 - t2,
        "projected_read_seconds": t5 - t4,
        "write_mib_per_s": file_mib / max(t1 - t0, 1e-9),
        "full_read_mib_per_s": file_mib / max(t3 - t2, 1e-9),
        "projected_read_apparent_mib_per_s": file_mib / max(t5 - t4, 1e-9),
        "disk_total_bytes": usage.total,
        "disk_used_bytes": usage.used,
        "disk_free_bytes": usage.free,
    }
    results.append(result)
    (out / f"parquet_bench_{safe_name}.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

summary = {"results": results}
(out / "parquet_bench_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
