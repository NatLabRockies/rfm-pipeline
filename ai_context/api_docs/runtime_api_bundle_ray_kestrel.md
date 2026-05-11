# Runtime API Bundle: Ray on Kestrel SLURM

**Purpose:** API and launch guidance for Ray as an experimental Kestrel runtime.\
**Retrieval date:** 2026-05-10.\
**Default account:** `bsm`.

## 1. Status

Ray is **experimental** for this repo on Kestrel.

**Verified external fact.** Ray has official SLURM deployment documentation and states that Ray 2.49+ includes `ray symmetric-run`, which starts a Ray cluster on SLURM nodes and runs the entrypoint only on the head node. Source URL: `https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm.html`; retrieved 2026-05-10.

**Risk / unknown.** No Kestrel-specific Ray documentation was found in public NLR HPC docs during this research pass. Do not enable Ray as a default backend until Kestrel smoke tests pass.

## 2. Install strategy with Pixi

```bash
export PIXI_HOME="/projects/bsm/.pixi"
export PIXI_CACHE_DIR="/projects/bsm/.cache/pixi"
export CONDA_PKGS_DIRS="/projects/bsm/.conda/pkgs"
export XDG_CACHE_HOME="/projects/bsm/.cache"

pixi add "ray-default>=2.49" pyarrow pandas
```

If the package name differs in the selected Pixi/Conda channel, resolve it explicitly in the repo lockfile and document the source. Do not silently substitute packages.

Smoke import:

```bash
pixi run python - <<'PY'
import ray
print("ray", ray.__version__)
PY
```

## 3. Ray Core API facts

**Verified API fact.** Ray tasks are Python functions decorated with `@ray.remote`; calling `.remote()` returns an object reference, and `ray.get()` retrieves results. Source URL: `https://docs.ray.io/en/latest/ray-core/tasks.html`; retrieved 2026-05-10.

```python
# scripts/ray_smoke.py
from __future__ import annotations

import ray


@ray.remote
def add_one(x: int) -> int:
    return x + 1


def main() -> None:
    ray.init(address="auto")
    refs = [add_one.remote(i) for i in range(8)]
    print(ray.get(refs))


if __name__ == "__main__":
    main()
```

## 4. Ray on SLURM launch pattern

**Verified API fact.** Ray’s SLURM example uses `scontrol show hostnames "$SLURM_JOB_NODELIST"` to derive the head node and runs `ray symmetric-run` under `srun` with one task per node. Source URL: `https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm-basic.html`; retrieved 2026-05-10.

CPU debug smoke script:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=ray-smoke
#SBATCH --account=bsm
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/ray-smoke-%j.out
#SBATCH --error=logs/ray-smoke-%j.err

set -euo pipefail

export PIXI_HOME="/projects/bsm/.pixi"
export PIXI_CACHE_DIR="/projects/bsm/.cache/pixi"
export CONDA_PKGS_DIRS="/projects/bsm/.conda/pkgs"
export XDG_CACHE_HOME="/projects/bsm/.cache"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

nodes="$(scontrol show hostnames "$SLURM_JOB_NODELIST")"
head_node="$(echo "$nodes" | head -n 1)"
ip_head="${head_node}:6379"

cleanup() {
  pixi run ray stop --force || true
}
trap cleanup EXIT

srun --nodes="${SLURM_JOB_NUM_NODES}" \
  --ntasks="${SLURM_JOB_NUM_NODES}" \
  pixi run ray symmetric-run \
    --address "${ip_head}" \
    --min-nodes "${SLURM_JOB_NUM_NODES}" \
    --num-cpus "${SLURM_CPUS_PER_TASK}" \
    -- \
    pixi run python scripts/ray_smoke.py
```

## 5. Short GPU smoke script

**Verified Kestrel fact.** Kestrel documents short H100 GPU partition `gpu-h100s` and GPU jobs request GPUs with `--gpus=<quantity>`. Source URL: `https://nrel.github.io/HPC/Documentation/Systems/Kestrel/Running/`; retrieved 2026-05-10.

```bash
#!/usr/bin/env bash
#SBATCH --job-name=ray-gpu-smoke
#SBATCH --account=bsm
#SBATCH --partition=gpu-h100s
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --output=logs/ray-gpu-smoke-%j.out
#SBATCH --error=logs/ray-gpu-smoke-%j.err

set -euo pipefail

export PIXI_HOME="/projects/bsm/.pixi"
export PIXI_CACHE_DIR="/projects/bsm/.cache/pixi"
export CONDA_PKGS_DIRS="/projects/bsm/.conda/pkgs"
export XDG_CACHE_HOME="/projects/bsm/.cache"

nvidia-smi

nodes="$(scontrol show hostnames "$SLURM_JOB_NODELIST")"
head_node="$(echo "$nodes" | head -n 1)"
ip_head="${head_node}:6379"

cleanup() {
  pixi run ray stop --force || true
}
trap cleanup EXIT

srun --nodes="${SLURM_JOB_NUM_NODES}" \
  --ntasks="${SLURM_JOB_NUM_NODES}" \
  pixi run ray symmetric-run \
    --address "${ip_head}" \
    --min-nodes "${SLURM_JOB_NUM_NODES}" \
    --num-cpus "${SLURM_CPUS_PER_TASK}" \
    --num-gpus 1 \
    -- \
    pixi run python scripts/ray_gpu_smoke.py
```

Minimal GPU task:

```python
# scripts/ray_gpu_smoke.py
from __future__ import annotations

import ray


@ray.remote(num_gpus=1)
def gpu_probe() -> str:
    import os
    return os.environ.get("CUDA_VISIBLE_DEVICES", "unset")


def main() -> None:
    ray.init(address="auto")
    print(ray.get(gpu_probe.remote()))


if __name__ == "__main__":
    main()
```

## 6. Acceptance gates before enabling Ray backend

1. `ray --version` reports Ray >= 2.49.
1. CPU debug one-node `ray symmetric-run` smoke test passes.
1. CPU multi-node smoke test passes on a permitted partition.
1. GPU short `gpu-h100s` smoke test passes if GPU support is in scope.
1. Logs confirm only the head node runs the entrypoint.
1. Cleanup leaves no orphan Ray processes.
1. Repo integration tests disable Ray unless `BSM_ENABLE_RAY_EXPERIMENTAL=1`.
