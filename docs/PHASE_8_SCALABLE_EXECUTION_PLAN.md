# Phase 8: Scalable Execution — Local Out-of-Core + Optional Distributed HPC

**Status**: Planning (validated environment + method manifest completed)
**Priority**: High — required for scaling to full dataset (30k+ samples) locally and on HPC
**Scope**: Out-of-core/chunked processing (primary) + optional distributed execution (secondary)
**Target Users**:

- Local workflows (no HPC) — can process 30k+ samples with streaming/spill-to-disk
- HPC workflows (Kestrel + others) — drastically reduce runtime with multi-node execution

______________________________________________________________________

## 1. Objective

Enable manuscript workflow execution on datasets of any size:

1. **Local + Laptop Friendly** (PRIMARY):

   - Out-of-core/chunked processing for giant arrays/tables
   - Streaming sparse/final stages over chunks
   - Spill-to-disk matrix operations
   - Config-driven memory thresholds (no code changes)

1. **Optional HPC Acceleration** (SECONDARY):

   - Distributed execution on NREL Kestrel + generic HPC
   - Multi-node job arrays to drastically reduce compute time
   - Config-only parameters (account, partition, walltime, etc.)
   - Opt-in: local single-machine is default

______________________________________________________________________

## 2. Architecture Overview

### 2.1 Five-Layer Model

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: Config (runtime + optional HPC specs)             │
│  - memory_budget_mb, chunk_size_mb, spill_policy           │
│  - temp_dir, scratch_root, projects_root                    │
│  - [OPTIONAL] distributed_execution.enabled, backend        │
│  - [OPTIONAL] slurm.account, partition, walltime, etc.      │
└─────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────┐
│  Layer 2a: Out-of-Core Processing (ALWAYS)                  │
│  - ChunkedParquetReader (stream Parquet row-groups)         │
│  - ChunkedAggregation (streaming sum, mean, concat)         │
│  - SpillToDisk operations (overflow to $TMPDIR/$scratch)    │
│  - LargeArrayHandler (lazy-load, chunked matrix ops)        │
│  - ProgressReporting (live chunk counts)                    │
└─────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────┐
│  Layer 2b: Distributed Execution Adapters (OPTIONAL)        │
│  - SLURM array shard runner                                 │
│  - Dask + dask-jobqueue                                     │
│  - MPI / mpi4py                                             │
│  - Ray experimental (opt-in)                                │
└─────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────┐
│  Layer 3: Manuscript Stages (chunked-ready)                 │
│  - output_conditioning: stream X/Y                          │
│  - empirical_null_screen: stream permutations               │
│  - interaction_discovery: chunk-wise scoring → reduce       │
│  - nonlinear_discovery: chunk-wise transforms → reduce      │
│  - sparse_selection: chunked stabilization resamples        │
│  - final_artifacts: stream bootstrap resamples              │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Implementation Strategy

**Phase 8 = 3 sequential sub-phases:**

1. **Phase 8a** (Weeks 1-2): Out-of-Core Foundation — chunked I/O, streaming aggregations, spill-to-disk (LOCAL FIRST)
1. **Phase 8b** (Weeks 2-3): Stage Integration — modify sparse/final stages to use chunked I/O; validate numerical equivalence
1. **Phase 8c** (Weeks 3-5): Optional Distributed HPC — SLURM array baseline, Dask/MPI/Ray adapters

______________________________________________________________________

## 3. Phase 8a: Out-of-Core Foundation (Weeks 1-2)

### 3.1 Objective

Create reusable out-of-core building blocks that work on ANY machine (laptop → HPC node).

### 3.2 Deliverables

#### Core Module: `src/bsm_rfm/out_of_core/`

**3.2.1 `chunked_io.py`**

```python
class ChunkedParquetReader:
    """Stream large Parquet files without full materialization"""
    def __init__(self, path: str, chunk_size_mb: int = 512, columns=None)
    def __iter__(self) -> Iterator[pd.DataFrame]
    def __len__(self) -> int  # total row count

class ChunkedCSVReader:
    """Stream large CSV files (Parquet preferred for efficiency)"""
    def __init__(self, path: str, chunk_size: int = 10000, ...)
    def __iter__(self) -> Iterator[pd.DataFrame]
```

**3.2.2 `streaming_ops.py`**

```python
class StreamingAggregation:
    """Streaming reduction for sum, mean, count, percentile, concat"""
    def __init__(self, operation: Literal["sum", "mean", "count", "concat"],
                 output_dtype=None)
    def add_chunk(self, chunk: pd.DataFrame)
    def finalize(self) -> Any

class StreamingQuantile:
    """Approximate quantiles from streaming chunks (t-digest or similar)"""
    def __init__(self, percentiles: List[float], ...)
    def add_chunk(self, chunk: pd.DataFrame)
    def finalize(self) -> pd.DataFrame
```

**3.2.3 `spill_ops.py`**

```python
class SpillToDiskBuffer:
    """Accumulate chunks to disk when memory threshold hit"""
    def __init__(self, temp_dir: str, max_memory_mb: int, schema=None)
    def add_chunk(self, chunk: pd.DataFrame)
    def get_final_dataframe(self) -> pd.DataFrame  # reads back from disk if spilled

class LargeArrayWriter:
    """Write large arrays in chunks (row-major + column-major aware)"""
    def __init__(self, output_path: str, dtype, shape, chunk_size_mb=512)
    def write_chunk(self, chunk: np.ndarray, row_offset: int)
    def finalize(self)

class DiskSpilledDask:
    """Lazy Dask-like interface over disk-spilled matrices (optional)"""
    # For operations like matrix multiplies, t-SNE, etc. that can't fit in memory
```

**3.2.4 `memory.py`**

```python
class MemoryBudget:
    """Monitor and enforce memory thresholds"""
    def __init__(self, budget_mb: int, reserve_mb: int = 500)
    def available_mb(self) -> float
    def should_spill(self) -> bool
    def check_fit(self, estimated_mb: float) -> bool

def choose_temp_dir(preferred_root: str, fallback_root: str) -> str:
    """Intelligent temp dir selection (prefers NVMe, rejects RAM-backed tmpfs)"""
```

**3.2.5 `progress.py`**

```python
class ChunkProgress:
    """Report streaming progress (chunks completed, total, ETA)"""
    def __init__(self, total_chunks: Optional[int] = None)
    def update(self, chunks_completed: int, bytes_read: int = 0)
    def report(self) -> dict  # {chunks_done, total, pct, eta_seconds}
```

#### Config Schema Additions

```python
# In src/bsm_rfm/config.py

@dataclass
class OutOfCoreConfig:
    enabled: bool = True  # always on by default
    chunk_size_mb: int = 512  # Parquet row-group friendly
    memory_budget_mb: int = 8000  # total RAM budget for workflow
    spill_policy: Literal["disk", "error"] = "disk"  # fail or spill on OOM
    temp_dir: str = "${TMPDIR}"  # or /scratch/$USER/bsm_<run_id>
    track_progress: bool = True
    log_spill_events: bool = True
```

#### Tests

```python
# tests/test_chunked_io.py
- test_chunked_parquet_reader_iteration()
- test_chunked_reader_matches_full_load()
- test_streaming_aggregation_sum_equals_pandas()
- test_streaming_aggregation_concat_preserves_schema()
- test_spill_to_disk_buffer_overflow()
- test_memory_budget_enforcement()
- test_temp_dir_selection_prefers_nvme()

# tests/test_large_array_io.py
- test_large_array_writer_chunked()
- test_spill_ops_read_back()
```

#### Documentation

- `docs/OUT_OF_CORE_DESIGN.md` — design choices, memory model, spill strategies
- `docs/CHUNKED_IO_COOKBOOK.md` — usage patterns (per-stage)
- Inline docstrings with examples

### 3.3 Implementation Notes

1. **Parquet-optimized**: Use `pyarrow.parquet.read_table(..., memory_map=True, pre_buffer=False)` for efficient row-group streaming
1. **Row-group alignment**: Compute row-group boundaries from file metadata; iterate whole row-groups to avoid fragmentation
1. **Spill strategy**: Write overflow chunks to Parquet (not CSV) for efficient re-reading
1. **Memory tracking**: Use `psutil.Process().memory_info()` to monitor actual usage vs. budget
1. **Progress reporting**: Emit JSON telemetry compatible with `watch -n 1 cat` monitoring

______________________________________________________________________

## 4. Phase 8b: Stage Integration (Weeks 2-3)

### 4.1 Objective

Integrate out-of-core operations into sparse selection and final artifact stages (the memory-intensive ones).

### 4.2 Target Stages

1. **Sparse Selection Stage** (`discover_manuscript_sparse_selection`):

   - Input: 30k samples × 50k candidate features (too large for full memory on laptop)
   - Operation: stability resamples (per-feature + cross-feature correlations)
   - Solution: chunk-wise resampling + spill correlation matrix to disk

1. **Final Artifacts Stage** (`create_manuscript_final_artifacts`):

   - Input: 30k samples × final features (managed), 200+ bootstrap resamples
   - Operation: bootstrap inference on each resample
   - Solution: stream resamples; batch inference on manageable chunks

1. **Interaction Discovery** (`discover_manuscript_interactions`):

   - Input: 30k samples × C(n,2) interaction pairs (potentially huge)
   - Operation: SHAP scoring per permutation
   - Solution: batch-wise SHAP scores; stream to reduction layer

### 4.3 Implementation Strategy

#### 4.3.1 Sparse Selection Rewrite

```python
def discover_manuscript_sparse_selection(
    output_path: str,
    case_study: CaseStudyArtifacts,
    use_chunked_io: bool = True,  # NEW FLAG
    chunk_size_mb: int = 512,
    memory_budget_mb: int = 8000,
    ...
):
    """
    New implementation:
    1. Read candidate features in chunks
    2. For each chunk: compute stability resamples (streaming aggregation)
    3. Spill correlation/covariance matrices to disk if needed
    4. Aggregate results at end
    """
    if use_chunked_io:
        # Use ChunkedParquetReader + ChunkedAggregation
        features_reader = ChunkedParquetReader(candidate_features_path, chunk_size_mb)
        agg = StreamingAggregation("concat")

        for feature_chunk in features_reader:
            resample_results = _compute_resamples_chunk(feature_chunk, ...)
            agg.add_chunk(resample_results)

        final_results = agg.finalize()
    else:
        # Original full-load implementation
        final_results = _original_sparse_selection_impl(...)

    return final_results
```

#### 4.3.2 Final Artifacts Streaming

```python
def create_manuscript_final_artifacts(
    output_path: str,
    ...,
    use_chunked_io: bool = True,  # NEW FLAG
    bootstrap_batch_size: int = 20,  # process N bootstraps at a time
    ...
):
    """
    Streaming bootstrap inference:
    1. Read bootstrap resamples in batches (don't load all 200+ at once)
    2. Run inference on each batch
    3. Stream results to output (Parquet with appending)
    """
    if use_chunked_io:
        bootstrap_reader = ChunkedParquetReader(bootstrap_path,
                                              chunk_size=bootstrap_batch_size)
        output_buffer = SpillToDiskBuffer(temp_dir, max_memory_mb=memory_budget_mb)

        for bootstrap_batch in bootstrap_reader:
            batch_results = _run_bootstrap_inference(bootstrap_batch, ...)
            output_buffer.add_chunk(batch_results)

        final_df = output_buffer.get_final_dataframe()
    else:
        final_df = _original_final_artifacts_impl(...)

    return final_df
```

### 4.4 Config Flag Integration

```yaml
# configs/local_laptop.yml
algorithm: ...
stages:
  sparse_selection:
    use_chunked_io: true
  final_manuscript_artifacts:
    use_chunked_io: true

runtime:
  memory_budget_mb: 4000  # laptop
  chunk_size_mb: 256     # smaller for low-memory machines
  spill_policy: "disk"
  temp_dir: "${TMPDIR}"

# configs/local_workstation.yml
# Same but:
runtime:
  memory_budget_mb: 16000  # workstation
  chunk_size_mb: 1024      # larger chunks for efficiency
  spill_policy: "disk"
  temp_dir: "/tmp/bsm_run"  # or NVMe partition
```

### 4.5 Validation

- [ ] Numerical equivalence test: run 300-sample with `use_chunked_io=true`, compare artifacts to non-chunked baseline
- [ ] Memory ceiling test: artificially set `memory_budget_mb=2000`, verify spill to disk, artifact correctness unchanged
- [ ] Stress test: run 10k-sample with 512 MB budget, confirm completion without OOM
- [ ] Performance test: compare runtime (chunked vs. non-chunked) on different dataset sizes

### 4.6 Documentation

- `docs/OUT_OF_CORE_INTEGRATION_GUIDE.md` — how out-of-core is used in each stage
- Update `docs/RUNNING_MANUSCRIPT_REPRODUCTION.md` with memory budget examples
- Troubleshooting guide for spill/OOM scenarios

______________________________________________________________________

## 5. Phase 8c: Optional Distributed HPC (Weeks 3-5)

### 5.1 Objective

Enable opt-in multi-node execution on HPC clusters (Kestrel primary, others secondary).

### 5.2 Deliverables

#### 5.2.1 SLURM Array Baseline (Primary)

```python
# src/bsm_rfm/distributed/slurm_array_runner.py
class SlurmArrayRunner:
    """Manifest-driven shard execution"""
    def __init__(self, manifest_path: str, config: SlurmConfig, ...)
    def render_sbatch_script(self) -> str  # Jinja2 template
    def submit_job(self) -> str  # returns job ID
    def monitor_progress(self)
    def collect_results(self)
```

**Use case**: 30k samples → 30 shards (1k samples each) → 30 array tasks

#### 5.2.2 Dask + dask-jobqueue (Secondary)

```python
# src/bsm_rfm/distributed/dask_runner.py
class DaskSlurmRunner:
    """Distributed dataframe operations on Kestrel SLURM"""
    def __init__(self, n_workers: int, memory_per_worker: str,
                 account: str = "bsm", partition: str = "shared")
    def run_stage_distributed(self, stage_name: str, ...)
```

#### 5.2.3 Config Extensions

```python
@dataclass
class DistributedExecutionConfig:
    enabled: bool = False  # default: local single-machine
    backend: Literal["slurm_array", "dask", "mpi", "ray"] = "slurm_array"

@dataclass
class SlurmConfig:
    account: str = "bsm"
    partition: str = "shared"
    walltime: str = "02:00:00"
    cpus_per_task: int = 4
    memory_per_task: str = "8G"
    max_array_size: int = 11000
    # ... (rest as documented in Phase 8 plan)
```

### 5.3 Workflow (Kestrel Users)

```bash
# 1. Create manifest + config
pixi run python tools/create_shard_manifest.py --input-root ... --output-manifest ...
pixi run python tools/render_slurm_array_job.py --config kestrel_distributed.yml ...

# 2. Submit
sbatch -A bsm job_array.sbatch

# 3. Monitor (local)
watch cat /projects/bsm/runs/run-001/progress.json

# 4. Collect
pixi run python tools/collect_shard_results.py --shard-root ...
```

### 5.4 Documentation

- `docs/DISTRIBUTED_EXECUTION_GUIDE.md` — architecture + config
- `docs/KESTREL_SLURM_QUICKSTART.md` — copy-paste examples
- `docs/COMPARING_DISTRIBUTED_BACKENDS.md` — SLURM vs. Dask vs. MPI

______________________________________________________________________

## 6. Config-Driven Usage Examples

### 6.1 Laptop (Small Memory Budget)

```yaml
# configs/laptop_small.yml
out_of_core:
  enabled: true
  chunk_size_mb: 128
  memory_budget_mb: 2000
  spill_policy: "disk"
  temp_dir: "${TMPDIR}"

runtime:
  n_jobs: 2  # small CPU count
  max_loaded_table_mb: 1500
  oom_output_cap: 100

distributed_execution:
  enabled: false  # local only
```

### 6.2 Workstation (Medium Memory)

```yaml
# configs/workstation_medium.yml
out_of_core:
  enabled: true
  chunk_size_mb: 512
  memory_budget_mb: 12000
  spill_policy: "disk"
  temp_dir: "/nvme/bsm_tmp"

runtime:
  n_jobs: -1  # all cores
  max_loaded_table_mb: 10000
  oom_output_cap: 500

distributed_execution:
  enabled: false
```

### 6.3 Kestrel CPU Node (Distributed)

```yaml
# configs/kestrel_distributed_slurm.yml
out_of_core:
  enabled: true
  chunk_size_mb: 1024  # large chunks for efficiency
  memory_budget_mb: 32000
  spill_policy: "disk"
  temp_dir: "/scratch/$USER/bsm_<run_id>"

runtime:
  n_jobs: 4  # per-task parallelism
  max_loaded_table_mb: 30000

distributed_execution:
  enabled: true
  backend: slurm_array

slurm:
  account: bsm
  partition: shared
  walltime: "01:00:00"
  cpus_per_task: 4
  memory_per_task: "16G"
```

______________________________________________________________________

## 7. Implementation Sequence

| Week | Task      | Deliverables                           | Validation                     |
| ---- | --------- | -------------------------------------- | ------------------------------ |
| 1    | 8a Part 1 | ChunkedParquetReader, streaming ops    | Unit tests pass                |
| 1    | 8a Part 2 | SpillToDisk, memory budget tracking    | Stress test (10 GB)            |
| 2    | 8a Part 3 | Integration into config, documentation | Smoke test passes              |
| 2    | 8b Part 1 | Sparse selection chunked rewrite       | Numerical equiv. on 300-sample |
| 3    | 8b Part 2 | Final artifacts streaming              | Memory ceiling test passes     |
| 3    | 8c Part 1 | SLURM array baseline + manifest        | Script generation works        |
| 4    | 8c Part 2 | Dask/MPI smoke tests                   | Dask cluster initializes       |
| 4-5  | 8c Part 3 | End-to-end on Kestrel                  | 300-sample array job completes |

______________________________________________________________________

## 8. Testing Strategy

### Unit Tests (Phase 8a)

- ChunkedParquetReader on small (10 MB) test file
- Streaming aggregation operations vs. pandas
- Memory budget enforcement + spill triggering

### Integration Tests (Phase 8b)

- Numerical equivalence: chunked read ≡ full read on 300-sample
- Memory ceiling: artificially set low budget, verify spill works
- Laptop simulation: run with 2 GB budget, confirm completion

### Stress Tests (Phase 8b)

- 10 GB synthetic Parquet file, verify memory stays \<4 GB
- 100k × 10k feature matrix, stability resamples with spill

### E2E Tests (Phase 8c, requires Kestrel access)

- Submit 1-task SLURM array; verify output structure
- Scale to 30 tasks; verify collection and final merge

______________________________________________________________________

## 9. Risks & Mitigations

| Risk                                              | Mitigation                                                        |
| ------------------------------------------------- | ----------------------------------------------------------------- |
| Streaming overhead (small chunks = many I/O ops)  | Align to Parquet row-group boundaries; benchmark chunk sizes      |
| Spill-to-disk latency                             | Use fast local NVMe where available; monitor disk I/O             |
| Numerical drift (approx quantiles, streaming sum) | Accept ≤5 decimal places; test on baseline first                  |
| TMPDIR exhaustion (no space for spill)            | Graceful degradation: fall back to error if no space              |
| Distributed job network contention                | SLURM arrays first (embarrassingly parallel); Dask only if needed |
| Kestrel module/env changes                        | Pin Pixi versions; document live `module load` commands           |

______________________________________________________________________

## 10. Success Criteria

- [ ] Out-of-core foundation passes all unit + stress tests
- [ ] Sparse selection + final artifacts run with `use_chunked_io=true` on 10k sample
- [ ] Numerical equivalence verified: chunked ≡ full-load within 5 decimals
- [ ] Laptop with 2-4 GB budget can process full 30k sample dataset
- [ ] Kestrel SLURM array baseline successfully completes 300-sample job
- [ ] Config files document usage for laptop → workstation → HPC progression
- [ ] All tests pass in CI; Kestrel tests pass on integration branch

______________________________________________________________________

## 11. Unresolved Questions

1. **Optimal chunk size** — is 256 MB, 512 MB, or 1 GB best for Parquet row-group alignment?
1. **Spill bottleneck** — will disk I/O dominate on machines with slow storage?
1. **Streaming approximation accuracy** — how many decimal places of drift for quantile-based operations?
1. **SLURM vs. Dask decision** — when does dynamic scheduling beat embarrassingly parallel?
1. **Memory profiling** — what's the actual per-stage memory footprint on real data?

______________________________________________________________________

## 12. References

- `ai_context/methods/kestrel_slurm_distributed_compute_method_manifest.md` — method guidance
- `kestrel_bsm_hpc_discovery_answers.md` — Kestrel system details
- Apache Parquet row-groups: https://parquet.apache.org/docs/file-format/configurations/
- Dask out-of-core: https://docs.dask.org/en/latest/
