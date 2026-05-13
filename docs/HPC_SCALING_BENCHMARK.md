# HPC Scaling Benchmark & Compute Calculator

This guide explains how to benchmark the BSM manuscript pipeline on a single
HPC node to build a runtime scaling model, then use that model to estimate
full-run times for arbitrary configurations.

## Overview

The benchmark sweeps four key configuration axes:

| Axis             | Variable             | What it controls                         |
| ---------------- | -------------------- | ---------------------------------------- |
| **Cores**        | `n_jobs`             | Parallel workers → Amdahl scaling        |
| **Features**     | `max_retained_terms` | O(N²) pair expansion → dominant cost     |
| **Permutations** | `n_permutations`     | Null distribution accuracy → linear cost |
| **Trees**        | `n_tree_estimators`  | Model accuracy → linear cost             |

Output is a CSV with measured wall-clock time for each cell. A log-linear
scaling model is fitted to that data and produces a calculator that takes any
configuration and returns an estimated runtime — including a decision table
across core counts.

______________________________________________________________________

## Quick Start

### On an HPC node (Kestrel example)

```bash
# 1. Get an interactive node
salloc --account=bsm --partition=shared --nodes=1 \
       --cpus-per-task=104 --mem=240G --time=08:00:00

# 2. Run the full benchmark (copies to /scratch automatically)
cd /projects/bsm/bsm-public-rf
bash scripts/hpc_node_benchmark.sh \
    --output-root /scratch/$USER/bsm_benchmark \
    --dataset-path /projects/bsm/bsm-public-rf/artifacts/test_dataset_3k

# 3. View results
cat /scratch/$USER/bsm_benchmark/compute_report.md
```

### Quick smoke test (3 cells, ~5 minutes)

```bash
bash scripts/hpc_node_benchmark.sh --quick
```

### As a batch job

```bash
sbatch scripts/hpc_node_benchmark.sh \
    --output-root /scratch/$USER/bsm_benchmark
```

______________________________________________________________________

## Pixi Commands

Both tools are available as Pixi tasks:

```bash
# Run the full benchmark grid
pixi run hpc-scaling-benchmark

# Options
pixi run hpc-scaling-benchmark -- \
    --output-root /scratch/$USER/bsm_bench \
    --dataset-path /path/to/dataset \
    --max-cores 64 \
    --quick          # 3-cell smoke test
    --dry-run        # print grid without running

# Run specific sweeps only
pixi run hpc-scaling-benchmark -- \
    --sweeps core_scaling feature_scaling
```

```bash
# Fit model and print tables
pixi run hpc-compute-calculator -- \
    --results artifacts/hpc_scaling_benchmark/scaling_results.csv

# Point estimate
pixi run hpc-compute-calculator -- \
    --results artifacts/hpc_scaling_benchmark/scaling_results.csv \
    --n-jobs 64 \
    --max-retained-terms 350 \
    --n-permutations 21 \
    --n-tree-estimators 100 \
    --n-samples 30000

# Save report + model
pixi run hpc-compute-calculator -- \
    --results artifacts/hpc_scaling_benchmark/scaling_results.csv \
    --save-model artifacts/hpc_scaling_benchmark/compute_model.json \
    --report artifacts/hpc_scaling_benchmark/compute_report.md
```

______________________________________________________________________

## Benchmark Grid Structure

The full benchmark (~35 unique cells after deduplication) runs in sweeps:

### Core Scaling Sweep

Fixes all settings at moderate values and varies only `n_jobs`. Measures
parallel efficiency (how close to linear scaling you get as cores increase).

| n_jobs | max_ret | n_perm | n_trees |
| ------ | ------- | ------ | ------- |
| 1      | 100     | 21     | 100     |
| 2      | 100     | 21     | 100     |
| 4      | 100     | 21     | 100     |
| 8      | 100     | 21     | 100     |
| 16     | 100     | 21     | 100     |
| 32     | 100     | 21     | 100     |
| 64     | 100     | 21     | 100     |
| max    | 100     | 21     | 100     |

### Feature Scaling Sweep

Fixes `n_jobs=max`, varies `max_retained_terms`. This is the **most important
axis** because pair count grows as O(N²): doubling features quadruples cost.

| max_ret  | Pairs (C(N,2))     |
| -------- | ------------------ |
| 20       | 190                |
| 40       | 780                |
| 75       | 2,775              |
| 100      | 4,950              |
| 150      | 11,175             |
| 200      | 19,900             |
| 300      | 44,850             |
| uncapped | depends on dataset |

### Permutation Scaling Sweep

Linear cost: `n_permutations` times as many model fits.

### Tree Scaling Sweep

Linear cost: more trees = more model accuracy and more compute.

### 2D Cross Sweep

Selected (n_jobs, max_retained_terms) pairs to capture parallelism × feature
interaction effects (important for fitting the joint model).

______________________________________________________________________

## Scaling Model

The fitted model is:

```
T = α × C(N,2)^β₁ × B^β₂ × E^β₃ / n_jobs^γ × (S/S_ref)^δ
```

| Parameter | Expected value    | Meaning                           |
| --------- | ----------------- | --------------------------------- |
| `α`       | very small float  | seconds per unit (intercept)      |
| `β₁`      | ≈ 1.0             | pair-count scaling (linear)       |
| `β₂`      | ≈ 1.0             | permutation scaling (linear)      |
| `β₃`      | ≈ 1.0             | tree scaling (linear)             |
| `γ`       | 0.8–1.0           | parallel efficiency (1.0 = ideal) |
| `δ`       | 1.0–1.5           | sample scaling exponent           |
| `R²`      | > 0.95 = reliable | goodness of fit                   |

**A `γ < 1.0` means parallelism is less than perfectly efficient.** Common
causes: Python GIL, joblib overhead on small tasks, memory bandwidth
saturation. The model will capture this empirically.

______________________________________________________________________

## Calculator Outputs

### Point estimate

```
Config   : n_jobs=64, max_retained=350, n_perm=21, n_trees=100, n_samples=30000
Pairs    : 61,075
Estimated: 4.2h (15,048s)
```

### Decision table (varies cores)

```
  Cores      Pairs     Estimated time   Speedup
  -----------------------------------------------
      1     61,075          5d 3h          1.0×
      2     61,075          2d 16h         2.0×
      4     61,075          1d 8h          3.9×
      8     61,075         16.3h           7.5×
     16     61,075          8.8h          14.1×
     32     61,075          4.8h          25.8×
     64     61,075          2.6h          46.9×
    104     61,075          1.7h          72.4×
```

### Feature count → runtime table (varies features at fixed cores)

```
  Features      Pairs     Estimated time
  ----------------------------------------
        20        190              8.3s
        50      1,225              3.1m
       100      4,950             12.6m
       150     11,175             28.4m
       200     19,900             50.6m
       300     44,850          1h 53.9m
       350     61,075          2h 35.2m
```

______________________________________________________________________

## Interpreting the Results

### "How many cores do I need?"

Use the decision table. As a rule of thumb:

- Doubling cores roughly halves runtime (if `γ ≈ 0.9`)
- Past ~64 cores, returns diminish if `γ < 0.85`

### "How tight is the feature cap?"

The feature count is the most powerful knob. Capping at 100 features vs
uncapped 350 can be a **10–150× speedup** depending on dataset.

### "Is my `R²` good enough?"

- `R² > 0.95`: model is reliable for interpolation; extrapolation within 2×
- `R² 0.90–0.95`: useful directionally, ±50% uncertainty on estimates
- `R² < 0.90`: too few cells or high noise; add more benchmark points

### "My node is different from the benchmark node"

The model captures parallelism efficiency empirically. If you run on a node
with different core count or memory bandwidth, the `α` and `γ` will differ.
**Always benchmark on the same hardware class you plan to use for production.**

______________________________________________________________________

## Files

| File                                                  | Purpose                                                           |
| ----------------------------------------------------- | ----------------------------------------------------------------- |
| `tools/hpc_scaling_benchmark.py`                      | Benchmark runner — generates config grid, runs cells, writes CSV  |
| `tools/hpc_compute_calculator.py`                     | Calculator — reads CSV, fits model, produces estimates and tables |
| `scripts/hpc_node_benchmark.sh`                       | Self-contained HPC wrapper — copy to node, run directly           |
| `artifacts/hpc_scaling_benchmark/scaling_results.csv` | Timing data (generated)                                           |
| `artifacts/hpc_scaling_benchmark/compute_model.json`  | Fitted model JSON (generated)                                     |
| `artifacts/hpc_scaling_benchmark/compute_report.md`   | Human-readable report (generated)                                 |

______________________________________________________________________

## Sharing Results

To help collaborators on other HPC systems estimate their runtimes:

1. Copy `scaling_results.csv` from the HPC node to your local machine
1. Run the calculator locally: `pixi run hpc-compute-calculator -- --results scaling_results.csv`
1. Or share `compute_report.md` directly as a reference

The model JSON can also be used programmatically:

```python
import json, math

model = json.loads(Path("compute_model.json").read_text())

def estimate_seconds(model, n_features, n_perm, n_trees, n_jobs, n_samples):
    pairs = n_features * (n_features - 1) // 2
    ref = model["reference_n_samples"]
    log_t = (
        model["log_alpha"]
        + model["b_pairs"] * math.log(pairs)
        + model["b_perm"] * math.log(n_perm)
        + model["b_trees"] * math.log(n_trees)
        - model["gamma"] * math.log(n_jobs)
        + model["delta"] * math.log(n_samples / ref)
    )
    return math.exp(log_t)

# Example: 350 features, 21 perms, 100 trees, 64 cores, 30k samples
t = estimate_seconds(model, 350, 21, 100, 64, 30_000)
print(f"Estimated: {t/3600:.1f} hours")
```
