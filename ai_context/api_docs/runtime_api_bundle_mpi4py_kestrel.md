# Runtime API Bundle: MPI / mpi4py on Kestrel

**Purpose:** API and launch guidance for mpi4py as a Kestrel distributed runtime option.\
**Retrieval date:** 2026-05-10.\
**Default account:** `bsm`.

## 1. Kestrel MPI environment guidance

**Verified external fact.** NLR recommends Cray MPICH on Kestrel through `PrgEnv-*` programming environments and compiler wrappers such as `cc`, `CC`, and `ftn`. NLR states OpenMPI does not currently run performantly/stably on Kestrel and recommends avoiding OpenMPI unless instructed otherwise. Source URL: `https://nrel.github.io/HPC/Documentation/Development/Programming-Environments/`; retrieved 2026-05-10.

**Verified external fact.** NLR’s MPI documentation describes Cray MPICH and MPI compiler wrappers on Kestrel. Source URL: `https://nrel.github.io/HPC/Documentation/Development/MPI/`; retrieved 2026-05-10.

Probe commands:

```bash
module list
module avail PrgEnv 2>&1 | head -100
module avail cray-mpich 2>&1 | head -100
which cc || true
which mpicc || true
```

## 2. Install strategy with Pixi

**Assumption.** The repo will use Pixi under `/projects/bsm`. Validate mpi4py installation against Kestrel’s Cray MPICH before production MPI use.

```bash
export PIXI_HOME="/projects/bsm/.pixi"
export PIXI_CACHE_DIR="/projects/bsm/.cache/pixi"
export CONDA_PKGS_DIRS="/projects/bsm/.conda/pkgs"
export XDG_CACHE_HOME="/projects/bsm/.cache"

module load PrgEnv-gnu 2>/dev/null || true
module load cray-mpich 2>/dev/null || true

pixi add mpi4py
pixi run python - <<'PY'
from mpi4py import MPI
print("mpi4py import ok")
print("rank", MPI.COMM_WORLD.Get_rank(), "size", MPI.COMM_WORLD.Get_size())
PY
```

If source build is needed, build against the active Kestrel MPI wrappers and document loaded modules.

## 3. Minimal mpi4py program

```python
# scripts/mpi_smoke.py
from __future__ import annotations

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

local_value = rank + 1
total = comm.reduce(local_value, op=MPI.SUM, root=0)

print(f"rank={rank} size={size} local_value={local_value}", flush=True)
if rank == 0:
    print(f"total={total}", flush=True)
```

**Verified API fact.** mpi4py communicator objects expose rank and size methods and collective operations. Source URL: `https://mpi4py.readthedocs.io/en/stable/tutorial.html`; retrieved 2026-05-10.

## 4. Kestrel CPU debug smoke batch script

```bash
#!/usr/bin/env bash
#SBATCH --job-name=mpi4py-smoke
#SBATCH --account=bsm
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=logs/mpi4py-smoke-%j.out
#SBATCH --error=logs/mpi4py-smoke-%j.err

set -euo pipefail

module load PrgEnv-gnu 2>/dev/null || true
module load cray-mpich 2>/dev/null || true

export PIXI_HOME="/projects/bsm/.pixi"
export PIXI_CACHE_DIR="/projects/bsm/.cache/pixi"
export CONDA_PKGS_DIRS="/projects/bsm/.conda/pkgs"
export XDG_CACHE_HOME="/projects/bsm/.cache"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

srun -n "${SLURM_NTASKS}" pixi run python scripts/mpi_smoke.py
```

## 5. mpi4py communication API notes

**Verified API fact.** mpi4py has lower-case methods such as `send`, `recv`, `bcast`, `scatter`, and `gather` for generic Python objects serialized with pickle, and upper-case methods such as `Send`, `Recv`, `Bcast`, `Scatter`, and `Gather` for buffer-like objects and arrays. Source URL: `https://mpi4py.readthedocs.io/en/stable/tutorial.html`; retrieved 2026-05-10.

Recommended repo use:

```python
# Lower-case object communication for small metadata only.
metadata = comm.bcast(metadata if rank == 0 else None, root=0)

# Upper-case buffer communication for numeric arrays where needed.
# Prefer explicit NumPy arrays and MPI datatypes for large data.
```

Do not use pickle-based communication for large dataframe or matrix payloads unless benchmarked and justified.

## 6. Fit for this repo

Good fit:

```text
- deterministic rank-sharded processing
- rank-local Parquet block processing
- explicit reductions of metrics or model summaries
- simple multi-node algorithms with predictable communication
```

Poor first-default fit:

```text
- dynamic task graphs
- dataframe query planning
- automatic spill-to-disk
- automatic retry or scheduler-level fault tolerance
```

## 7. Required tests

1. Unit test rank-to-shard mapping as a pure function.
1. Unit test output path generation by rank and shard.
1. Integration smoke script: `srun -n 4 pixi run python scripts/mpi_smoke.py`.
1. Failure test: rerun after deleting one rank’s output and verify only missing shard is recomputed.
1. Generated MPI script uses `srun`, not OpenMPI-specific `mpirun` assumptions.
