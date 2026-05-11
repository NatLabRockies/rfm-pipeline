# Runtime API Bundle: Dask + dask-jobqueue on Kestrel

**Purpose:** API and launch guidance for Dask+dask-jobqueue as the recommended distributed dataframe runtime on Kestrel.\
**Retrieval date:** 2026-05-10.\
**Default account:** `bsm`.

## 1. Kestrel-specific Dask evidence

**Verified external fact.** NLR documents Dask on Kestrel, including Dask local clusters, `dask-jobqueue`, `SLURMCluster`, `SLURMRunner`, `account`, `walltime`, `queue`, `processes`, `memory`, and `interface='hsn0'`. Source URL: `https://nrel.github.io/HPC/Documentation/Development/Languages/Python/dask/`; retrieved 2026-05-10.

**Verified external fact.** NLR recommends Conda/Mamba environments outside `/home`, especially for Dask. For this repo, apply the same placement rule to Pixi and package caches under `/projects/bsm`. Source URL: `https://nrel.github.io/HPC/Documentation/Development/Languages/Python/dask/`; retrieved 2026-05-10.

## 2. Install strategy with Pixi

```bash
export PIXI_HOME="/projects/bsm/.pixi"
export PIXI_CACHE_DIR="/projects/bsm/.cache/pixi"
export CONDA_PKGS_DIRS="/projects/bsm/.conda/pkgs"
export XDG_CACHE_HOME="/projects/bsm/.cache"

pixi add dask distributed dask-jobqueue pyarrow pandas
```

Smoke import:

```bash
pixi run python - <<'PY'
import dask
import distributed
import dask_jobqueue
print("dask", dask.__version__)
print("distributed", distributed.__version__)
print("dask_jobqueue", dask_jobqueue.__version__)
PY
```

## 3. Current `SLURMCluster` API syntax

**Verified API fact.** dask-jobqueue documents `SLURMCluster` parameters including `queue`, `account`, `cores`, `memory`, `processes`, `interface`, `local_directory`, `death_timeout`, `walltime`, `job_script_prologue`, and `job_extra_directives`. It marks older parameters such as `project`, `env_extra`, and `job_extra` as deprecated. Source URL: `https://jobqueue.dask.org/en/stable/generated/dask_jobqueue.SLURMCluster.html`; retrieved 2026-05-10.

Recommended CPU debug smoke script:

```python
# scripts/dask_jobqueue_debug_smoke.py
from __future__ import annotations

from pathlib import Path

from dask.distributed import Client
from dask_jobqueue import SLURMCluster

scratch = Path("/scratch") / Path.home().name / "bsm_dask_smoke"
scratch.mkdir(parents=True, exist_ok=True)

cluster = SLURMCluster(
    queue="debug",
    account="bsm",
    cores=4,
    memory="8GB",
    processes=2,
    walltime="00:30:00",
    interface="hsn0",
    local_directory=str(scratch / "worker-space"),
    log_directory=str(scratch / "logs"),
    job_script_prologue=[
        'export PIXI_HOME="/projects/bsm/.pixi"',
        'export PIXI_CACHE_DIR="/projects/bsm/.cache/pixi"',
        'export CONDA_PKGS_DIRS="/projects/bsm/.conda/pkgs"',
        'export XDG_CACHE_HOME="/projects/bsm/.cache"',
        'export OMP_NUM_THREADS="1"',
        'export MKL_NUM_THREADS="1"',
        'export OPENBLAS_NUM_THREADS="1"',
        'export NUMEXPR_NUM_THREADS="1"',
    ],
)

print(cluster.job_script())
cluster.scale(jobs=1)
client = Client(cluster)
future = client.submit(lambda x: x + 1, 41)
print("result", future.result())
client.close()
cluster.close()
```

## 4. `SLURMRunner`

**Verified API fact.** dask-jobqueue documents `SLURMRunner`, with a default scheduler file pattern like `scheduler-{job_id}.json`. Source URL: `https://jobqueue.dask.org/en/stable/generated/dask_jobqueue.SLURMRunner.html`; retrieved 2026-05-10.

**Verified Kestrel example.** NLR’s Dask page shows `SLURMRunner` using scheduler files under `/scratch`, `interface='hsn0'`, worker/scheduler options, and `srun`. Source URL: `https://nrel.github.io/HPC/Documentation/Development/Languages/Python/dask/`; retrieved 2026-05-10.

Use `SLURMRunner` for batch-only workflows that should start after all requested nodes are allocated. Use `SLURMCluster` for dynamic jobqueue-managed workers.

## 5. Dask DataFrame API relevance

**Verified API fact.** Dask DataFrame is a parallel dataframe made from many pandas DataFrames and supports larger-than-memory tabular workflows using lazy task graphs. Source URL: `https://docs.dask.org/en/stable/dataframe.html`; retrieved 2026-05-10.

Example Parquet read:

```python
from __future__ import annotations

import dask.dataframe as dd

ddf = dd.read_parquet(
    "/scratch/$USER/bsm_runs/example/input_parquet",
    engine="pyarrow",
    columns=["sample_id", "feature_001", "feature_002"],
)
summary = ddf.describe().compute()
print(summary)
```

Avoid large shuffles and avoid converting full distributed dataframes to local pandas objects.

## 6. Spill and local directory policy

**Verified Kestrel fact.** NLR warns that `$TMPDIR` can consume RAM on nodes without local disk. Source URL: `https://nrel.github.io/HPC/Documentation/Systems/Kestrel/Filesystems/`; retrieved 2026-05-10.

```python
from __future__ import annotations

import os
import subprocess
from pathlib import Path


def disk_backed_tmpdir() -> Path | None:
    tmpdir = os.environ.get("TMPDIR")
    if not tmpdir:
        return None
    try:
        output = subprocess.check_output(["df", "-T", tmpdir], text=True)
    except Exception:
        return None
    if "tmpfs" in output:
        return None
    return Path(tmpdir)


def choose_dask_local_directory(run_id: str) -> str:
    local = disk_backed_tmpdir()
    if local is not None:
        path = local / "bsm_dask_spill"
    else:
        path = Path("/scratch") / os.environ["USER"] / "bsm_runs" / run_id / "dask_spill"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)
```

## 7. Required tests

1. Dask config uses `account=bsm`.
1. CPU smoke queue is `debug`.
1. Production queue can be `shared` or `nvme`.
1. Network interface defaults to `hsn0`.
1. `local_directory` comes from spill selector.
1. Rendered code avoids deprecated `project`, `env_extra`, and `job_extra`.
1. Local integration test with `distributed.LocalCluster`.
1. Manual Kestrel smoke test with `SLURMCluster(queue="debug", account="bsm", ...)`.

## 8. Recommended default config

```yaml
distributed:
  default_backend: dask_jobqueue
  dask_jobqueue:
    account: bsm
    cpu_probe_queue: debug
    production_queue: shared
    network_interface: hsn0
    cores_per_job: 4
    processes_per_job: 2
    memory_per_job: 8GB
    walltime: "00:30:00"
    durable_root: /projects/bsm
    scratch_root_template: /scratch/{user}/bsm_runs/{run_id}
```
