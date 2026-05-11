# Phase 8: Distributed Execution and Out-of-Core Processing

**Status**: Planning (validated environment + method manifest completed)
**Priority**: High — required for scaling to full dataset (30k+ samples)
**Scope**: Config-driven distributed execution + out-of-core/chunked processing
**Platform Focus**: NREL Kestrel SLURM + Pixi, generic for other HPC environments
**Target Completion**: Fully autonomously executable on Kestrel; portable to other clusters via config

______________________________________________________________________

## 1. Objective

Scale manuscript workflow execution from local multi-threaded Python (Phase 5-7) to distributed HPC execution:

- **Input**: 30k+ sample datasets (millions of rows × millions of columns in Parquet)
- **Output**: Full manuscript workflow artifacts matching single-node validation
- **Constraints**: Config-only (no source code edits); numeric drift ≤ 5 decimal places acceptable
- **Validation**: Bit-exact reproducibility where feasible; statistical equivalence otherwise

______________________________________________________________________

## 2. Architecture Overview

### 2.1 Four-Layer Abstraction

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: Config (cluster/job/resource specs)               │
│  - account, partition, queue, walltime, memory              │
│  - Pixi environment paths (/projects/bsm)                   │
│  - spill/scratch policies                                   │
│  - runtime thresholds (OOM, temp-file limits)               │
└─────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────┐
│  Layer 2: Execution Adapters (pluggable)                    │
│  - SLURM array shard runner (primary)                       │
│  - Dask + dask-jobqueue (secondary)                         │
│  - MPI / mpi4py (tertiary, for collectives)                 │
│  - Ray experimental (opt-in, BSM_ENABLE_RAY=1)              │
└─────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────┐
│  Layer 3: In-Memory Operations                              │
│  - Chunked dataframe I/O (Parquet row-group aware)          │
│  - Streaming aggregations (no full materialization)         │
│  - Spill-to-disk reductions (when memory threshold hit)     │
│  - Checkpoint/recovery markers (_SUCCESS, attempt logs)     │
└─────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────┐
│  Layer 4: Manuscript Workflow (unmodified)                  │
│  - Stage chain: output_conditioning → interaction → ...     │
│  - Uses chunked I/O + streaming reduction from Layer 3      │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Implementation Strategy

**Phase 8 = 3 concrete sub-phases:**

1. **Phase 8a**: Config schema + SLURM array baseline (deterministic production foundation)
1. **Phase 8b**: Out-of-core chunk processing + checkpoint/recovery
1. **Phase 8c**: Optional Dask/MPI adapters + Ray experimental (tested but opt-in)

______________________________________________________________________

## 3. Phase 8a: SLURM Array Baseline (Week 1-2)

### 3.1 Deliverables

- [ ] **New `src/bsm_rfm/distributed/` module**:

  - `config_distributed.py` — typed config for SLURM account, queue, partition, walltime, memory, spill paths
  - `slurm_array_runner.py` — manifest-driven shard execution with `SLURM_ARRAY_TASK_ID` lookup
  - `spill.py` — intelligent `/scratch` vs `$TMPDIR` selection + free-space monitoring
  - `checkpoint.py` — idempotent output layout with `_SUCCESS.json` markers and attempt tracking

- [ ] **Manifest schema (`src/bsm_rfm/distributed/manifest.py`)**:

  ```python
  @dataclass
  class ShardManifest:
      shard_id: str                           # e.g. "task-1234"
      input_paths: List[str]                  # Parquet files
      output_path: str                        # durable output dir
      expected_rows: int
      expected_columns: int
      feature_block_id: Optional[str]
      scenario_id: Optional[str]
      year: Optional[int]
      status: Literal["pending", "running", "completed", "failed"]
  ```

- [ ] **Script templates** (Jinja2-rendered from config):

  - `scripts/sbatch_templates/slurm_array_stage.sbatch` — array job template
  - `scripts/sbatch_templates/slurm_array_reduce.sbatch` — dependency-driven reduce/merge job
  - `scripts/sbatch_templates/slurm_array_diagnostic.sbatch` — telemetry collection

- [ ] **Tests**:

  - Unit tests for spill path selection, checkpoint semantics
  - Smoke test: create small manifest, render SLURM script, verify task ID parsing
  - Integration test (requires Kestrel access or mock SLURM_ARRAY_TASK_ID): run 1-task array locally

- [ ] **Config additions** (`src/bsm_rfm/config.py`):

  ```python
  @dataclass
  class DistributedExecutionConfig:
      enabled: bool = False
      backend: Literal["slurm_array", "dask", "mpi", "ray"] = "slurm_array"

  @dataclass
  class SlurmConfig:
      account: str = "bsm"
      partition: str = "shared"
      queue: Optional[str] = None
      walltime: str = "01:00:00"
      cpus_per_task: int = 1
      memory_per_task: str = "8G"
      max_array_size: int = 11000
      output_log_dir: Optional[str] = None  # defaults to /projects/bsm/<run_id>/logs

  @dataclass
  class SpillConfig:
      scratch_root: str = "/scratch"  # will add $USER/bsm_<run_id>
      projects_root: str = "/projects/bsm"
      max_spill_mb: int = 500000
      prefer_nvme: bool = True
  ```

- [ ] **Documentation**:

  - `docs/DISTRIBUTED_EXECUTION_GUIDE.md` — architecture, config examples, troubleshooting
  - `docs/KESTREL_SLURM_QUICKSTART.md` — copy-paste examples for Kestrel users
  - Inline docstrings for `distributed/` module APIs

### 3.2 Implementation Approach

1. **Design config schema** first; write tests for config loading/validation
1. **Implement spill path logic**; test on Kestrel with live `df -h` + `lfs quota` probes
1. **Implement shard manifest and checkpoint logic**; verify idempotent semantics (rerun same task, should skip or overwrite cleanly)
1. **Create SLURM template renderer**; test script generation with example config
1. **Write integration smoke test** that can run locally (set SLURM_ARRAY_TASK_ID manually, validate output)
1. **Document workflow**: how user creates shard manifest, submits array job, monitors progress, collects results

### 3.3 Example Workflow (for documentation)

```bash
# 1. Create shard manifest from data
pixi run python tools/create_shard_manifest.py \
  --input-root /projects/bsm/data \
  --output-manifest /projects/bsm/runs/run-001/manifest.parquet \
  --shard-size 1000 rows

# 2. Create array job script from config
pixi run python tools/render_slurm_array_job.py \
  --config configs/kestrel_distributed.yml \
  --manifest /projects/bsm/runs/run-001/manifest.parquet \
  --output-script /projects/bsm/runs/run-001/job_array.sbatch

# 3. Submit and monitor
sbatch -A bsm /projects/bsm/runs/run-001/job_array.sbatch
watch squeue -A bsm
watch cat /projects/bsm/runs/run-001/progress.json

# 4. Collect results after completion
pixi run python tools/collect_shard_results.py \
  --shard-root /projects/bsm/runs/run-001/outputs \
  --output /projects/bsm/runs/run-001/final_artifacts.parquet
```

______________________________________________________________________

## 4. Phase 8b: Out-of-Core Processing (Week 2-3)

### 4.1 Deliverables

- [ ] **Chunked I/O layer** (`src/bsm_rfm/chunked_io.py`):

  ```python
  class ChunkedParquetReader:
      """Iterate large Parquet files in chunks without full materialization"""
      def __init__(self, path: str, chunk_size_mb: int = 512, ...)
      def __iter__(self) -> Iterator[pd.DataFrame]

  class ChunkedAggregation:
      """Streaming reduction for operations like sum, mean, quantile"""
      def __init__(self, operation: Literal["sum", "mean", "concat"], ...)
      def add_chunk(self, chunk: pd.DataFrame)
      def finalize(self) -> Any
  ```

- [ ] **Spill-to-disk operations** (`src/bsm_rfm/spill_ops.py`):

  - Temporary Parquet file accumulation when memory threshold hit
  - Atomic promotion to final location after validation
  - `$TMPDIR` vs `/scratch` intelligent selection

- [ ] **Integration with workflow**:

  - Modify `manuscript_stages.py` interaction/nonlinear/sparse stages to accept chunked I/O
  - Add `use_chunked_io: bool` config flag (default False for backward compat)
  - Verify numerical equivalence between chunked and full-load paths on small test cases

- [ ] **Tests**:

  - Unit: ChunkedParquetReader on small (10 MB) test file, verify iteration correctness
  - Unit: ChunkedAggregation operations vs pandas equivalents
  - Integration: run 300-sample with `use_chunked_io=true`, compare artifacts against non-chunked baseline
  - Stress: synthetic 10 GB Parquet file, verify memory stays below threshold

- [ ] **Documentation**:

  - `docs/CHUNKED_IO_DESIGN.md` — when/why/how to use, memory behavior, numerical properties

### 4.2 Implementation Approach

1. **Start with read-only chunking** (Parquet → memory chunks)
1. **Implement streaming reductions** (sum, mean, count without full materialization)
1. **Add spill-to-disk for aggregations** that exceed memory threshold
1. **Test on real Parquet data** (use existing 300-sample validation set as baseline)
1. **Integrate into one slow stage** (e.g., interaction scoring) as proof of concept
1. **Validate numerical equivalence** before rolling out to all stages

______________________________________________________________________

## 5. Phase 8c: Optional Dask / MPI / Ray Adapters (Week 3-4)

### 5.1 Dask + dask-jobqueue (Secondary Baseline)

**Objective**: Support dynamic task scheduling when SLURM arrays become a bottleneck.

- [ ] Create `src/bsm_rfm/distributed/dask_runner.py`:

  - Use Dask DataFrame for large tabular operations
  - `SLURMCluster` for on-Kestrel execution
  - Partition-aware operations to avoid full materialization
  - Graceful fallback if Dask not available

- [ ] Config additions:

  ```python
  @dataclass
  class DaskConfig:
      n_workers: int = 4
      memory_per_worker: str = "8G"
      network_interface: str = "hsn0"  # Kestrel high-speed network
      scheduler: str = "threads"  # or "processes" / "distributed"
  ```

- [ ] Tests: smoke test on local Dask scheduler with small dataset

- [ ] Documentation: when to use Dask vs SLURM arrays

### 5.2 MPI / mpi4py (Tertiary)

**Objective**: Support low-level rank-based communication when collectives or all-reduce needed.

- [ ] Create `src/bsm_rfm/distributed/mpi_runner.py`
- [ ] Rank-based shard assignment + collective reductions
- [ ] Documentation + environment setup (module load mpi, etc.)

### 5.3 Ray Experimental (Opt-In)

**Objective**: Experimental support; disabled by default.

- [ ] Create `src/bsm_rfm/distributed/ray_runner_experimental.py`
- [ ] Environment variable gate: `BSM_ENABLE_RAY_EXPERIMENTAL=1`
- [ ] Documentation: known limitations, why disabled by default

______________________________________________________________________

## 6. Config-Driven Usage Examples

### 6.1 Local Single-Machine (No Distributed)

```yaml
# configs/local_single_machine.yml
distributed_execution:
  enabled: false

runtime:
  n_jobs: -1  # use all cores
```

### 6.2 Kestrel SLURM Arrays

```yaml
# configs/kestrel_distributed_slurm.yml
distributed_execution:
  enabled: true
  backend: slurm_array

slurm:
  account: bsm
  partition: shared
  walltime: "02:00:00"
  cpus_per_task: 4
  memory_per_task: "16G"
  output_log_dir: /projects/bsm/runs/run-001/logs

spill:
  scratch_root: /scratch
  projects_root: /projects/bsm
  max_spill_mb: 1000000
  prefer_nvme: true
```

### 6.3 Kestrel Dask

```yaml
# configs/kestrel_distributed_dask.yml
distributed_execution:
  enabled: true
  backend: dask

dask:
  n_workers: 8
  memory_per_worker: "8G"
  network_interface: "hsn0"
```

______________________________________________________________________

## 7. Testing Strategy

### Unit Tests

- Config loading/validation for all new `distributed_*` fields
- Spill path selection logic (correct fallback behavior)
- Checkpoint/recovery semantics (idempotency)
- Manifest parsing

### Integration Tests (can run on Kestrel)

- Submit minimal 1-task SLURM array; verify task ID parsing + output structure
- Create small manifest (10 shards), run locally with mock `SLURM_ARRAY_TASK_ID=0..9`
- Verify all shards complete and reduce correctly

### End-to-End (Kestrel-only)

- Run 300-sample dataset via SLURM array baseline; compare artifacts to local run
- Verify runtime scaling with 1k, 3k, 10k sample subsets
- Run full 30k dataset; validate completion + artifact integrity

### CI/CD

- Unit tests run on every commit
- Integration tests gated to Kestrel-capable runners (skip in generic CI)
- Create separate `.github/workflows/kestrel_ci.yml` for Kestrel-specific smoke tests

______________________________________________________________________

## 8. Implementation Sequence

| Week | Sub-Phase   | Deliverables                                 | Validation                     |
| ---- | ----------- | -------------------------------------------- | ------------------------------ |
| 1    | 8a Part 1   | Config schema + spill logic                  | Unit tests pass                |
| 1    | 8a Part 2   | Manifest + checkpoint layer                  | Smoke test (local)             |
| 2    | 8a Part 3   | SLURM template renderer                      | Script generation verified     |
| 2    | 8a Part 4   | Integration test on Kestrel                  | 1-task array runs              |
| 3    | 8b Part 1   | ChunkedParquetReader + streaming aggregation | Unit tests pass                |
| 3    | 8b Part 2   | Spill-to-disk operations                     | Stress test (10 GB) passes     |
| 3    | 8b Part 3   | Workflow integration (1 stage)               | Numerical equivalence verified |
| 4    | 8c Optional | Dask/MPI/Ray runners (secondary)             | Smoke tests pass               |

______________________________________________________________________

## 9. Risks & Mitigations

| Risk                                             | Mitigation                                                                           |
| ------------------------------------------------ | ------------------------------------------------------------------------------------ |
| SLURM job array size limit (11k) exceeded        | Phase 8a uses `%N` throttling; manifest pre-split into ≤10k shards                   |
| Memory blowup from unchecked I/O                 | ChunkedParquetReader + streaming aggregation (Phase 8b) + runtime thresholds         |
| Numeric drift across distributed stages          | Accept ≤5 decimal place drift; validate on 300-sample baseline first                 |
| Filesystem contention (many nodes writing)       | Use `/scratch` for temp; atomic promotion to `/projects` only for final              |
| Preemption/node failure (though PreemptMode=OFF) | Idempotent task outputs + checkpoint markers allow re-run without full restart       |
| Inter-node communication overhead                | Start with embarrassingly parallel (SLURM arrays); add collective ops only if needed |
| Kestrel module/environment changes               | Pin Pixi env versions; document live `module load` commands in discovery bundle      |

______________________________________________________________________

## 10. Success Criteria

- [ ] Config file can specify all distributed execution parameters without code changes
- [ ] SLURM array baseline successfully runs 300-sample dataset on Kestrel
- [ ] Full 30k-sample dataset completes with acceptable runtime (≤ 24 hours for full stage chain)
- [ ] Artifacts match local-run validation within 5 decimal places
- [ ] Out-of-core processing passes numerical equivalence test on 300-sample with full dataset chunking
- [ ] Documentation enables self-service execution on Kestrel + generic HPC clusters
- [ ] All tests pass in CI + Kestrel-specific integration tests pass

______________________________________________________________________

## 11. Unresolved Questions (TBD)

1. **Optimal SLURM shard size** — is 1k rows/shard, 10k rows/shard, or 100k rows/shard best for runtime + memory trade-off?
1. **Network I/O bottleneck** — will Lustre contention limit scalability at 10k+ tasks?
1. **Dask vs SLURM decision boundary** — at what dataset size does Dask distributed dataframe become more efficient than array shard collection?
1. **Full-dataset artifact validation** — does 30k dataset produce feature counts matching manuscript expectations, or do we have new correctness issues?

______________________________________________________________________

## 12. References

- `ai_context/methods/kestrel_slurm_distributed_compute_method_manifest.md` — detailed method guidance
- `kestrel_bsm_hpc_discovery_answers.md` — live Kestrel system information
- `docs/RUNNING_MANUSCRIPT_REPRODUCTION.md` — single-machine workflow reference
- NLR Kestrel documentation: https://natlabrockies.github.io/HPC/
