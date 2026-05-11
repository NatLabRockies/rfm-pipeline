# HPC Implementation Readiness Summary

**Status**: Phase 8 planning complete; environment discovery finalized; ready for implementation
**Date**: 2026-05-10
**Target Environment**: NREL Kestrel (primary); generic HPC (secondary)

______________________________________________________________________

## 1. Environment Discovery Status

✅ **Complete** — Live Kestrel system probed and documented

### Key Findings

| Item                     | Value                                                        | Source                         |
| ------------------------ | ------------------------------------------------------------ | ------------------------------ |
| Scheduler                | Slurm 25.05.5                                                | Live `scontrol --version`      |
| Account (project)        | `bsm`                                                        | Live `sacctmgr` + user request |
| Max array size           | 11,000                                                       | Live `scontrol show config`    |
| Preemption               | OFF                                                          | Live config: `PreemptMode=OFF` |
| Auto-requeue             | OFF                                                          | Live config: `JobRequeue=0`    |
| Partition (dev)          | `debug`                                                      | Public docs + user choice      |
| Partition (prod array)   | `shared`, `short`, `standard`, `nvme`                        | Public docs                    |
| Partition (GPU)          | `gpu-h100s` (short), `gpu-h100` (medium), `gpu-h100l` (long) | Public docs                    |
| CPU nodes                | 104 cores, ~240 GB RAM                                       | Public docs                    |
| GPU nodes                | 4 H100s per node, 80 GB/GPU; 128 CPU cores; local NVMe       | Public docs                    |
| ProjectFS                | `/projects`, 200 GB/s, 68 PB                                 | Public docs                    |
| ScratchFS                | `/scratch`, 354 GB/s, 27 PB                                  | Public docs                    |
| Local NVMe               | 1.7 TB (256 CPU nodes); 3.4-14 TB (GPU nodes)                | Public docs                    |
| Network interface (Dask) | `hsn0`                                                       | NLR Dask docs                  |
| Python execution model   | Pixi (recommended)                                           | NLR Python docs                |
| Failure restart model    | Idempotent task outputs (no auto-requeue)                    | Live config analysis           |

### Discovery Artifacts

- **`kestrel_bsm_hpc_discovery_answers.md`** — 400+ lines of live + documented configuration
- **`ai_context/methods/kestrel_slurm_distributed_compute_method_manifest.md`** — implementation method guidance
- **`scripts/run_kestrel_discovery_suite.sh`** — automated probing suite for other environments
- **`scripts/discovery_*.sh`** — individual CPU/GPU/spill/Parquet probes
- **`README_kestrel_discovery_reporting.md`** — how to use discovery suite

______________________________________________________________________

## 2. Phase 8 Planning Status

✅ **Complete** — Detailed 12-section plan finalized

### Deliverables

- **`docs/PHASE_8_DISTRIBUTED_HPC_PLAN.md`** (15.7 KB)

  - 2. Architecture overview (4-layer abstraction model)
  - 3. Phase 8a: SLURM array baseline (2 weeks)
  - 4. Phase 8b: Out-of-core processing (2 weeks)
  - 5. Phase 8c: Optional adapters (1 week)
  - 6. Config-driven usage examples
  - 7. Testing strategy (unit, integration, E2E)
  - 8. Implementation sequence (week-by-week breakdown)
  - 9. Risks & mitigations (8 key risks identified)
  - 10. Success criteria (7 checkpoints)
  - 11. Unresolved questions (4 TBD)

- **Updated `docs/ENGINEERING_MANIFEST.md`**

  - Phase 8 section added with sub-phase breakdown and design principles
  - Success criteria and unresolved questions documented

- **Updated `docs/AGENT_SYNC.md`**

  - Phase 8 overview and key documents listed
  - Kestrel-specific details (account, partitions, filesystems, network)
  - Phase 8a/b/c deliverables enumerated with status
  - SQL todo items created (14 tasks)

### Config Schema Preview

```yaml
# distributed execution (all parameters, no code edits)
distributed_execution:
  enabled: true
  backend: slurm_array  # or: dask, mpi, ray_experimental

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

dask:  # optional
  n_workers: 8
  memory_per_worker: "8G"
  network_interface: "hsn0"
```

______________________________________________________________________

## 3. Implementation Readiness Checklist

### Prerequisites (all satisfied)

- [x] External research completed (NLR Kestrel documentation reviewed)
- [x] Live system probed (Slurm config, filesystem, network verified)
- [x] Method guidance documented (Phase 8 plan with 7 decision points)
- [x] Config schema designed (typed dataclasses ready)
- [x] Risk analysis completed (8 risks × mitigations)
- [x] Testing strategy defined (unit → integration → E2E)

### Ready to Start (Phase 8a)

- [x] Environment documentation sufficient to begin Phase 8a
- [x] Config schema finalized and documented
- [x] Kestrel account/partition/filesystem choices made
- [x] Example SLURM scripts available (can be templated)
- [x] Checkpoint/recovery semantics defined

### Blockers (before Phase 8 implementation)

- [ ] Uncapped 300-sample run must complete to provide baseline timing data
- [ ] User confirmation on Phase 8a timeline and resource allocation

______________________________________________________________________

## 4. Key Design Decisions

### Why SLURM Arrays First?

1. **Deterministic**: NLR documents SLURM arrays well; `MaxArraySize=11k` is a hard limit
1. **Simple**: Embarrassingly parallel (no inter-task communication required)
1. **Proven**: Common HPC pattern; easy to debug (one task at a time)
1. **Scalable**: Can handle up to ~11k shards per job; multi-job chains via dependencies

### Why Config-Only?

1. **Self-Service**: Users can target different clusters (Kestrel, other NSF HPC, cloud) without modifying code
1. **Reproducibility**: All execution parameters captured in version-controlled YAML
1. **Audit Trail**: Config file documents resource allocation, account, filesystem paths
1. **Flexibility**: Can test locally (`enabled=false`), then scale to Kestrel (`backend=slurm_array`) by changing one file

### Why Multi-Runtime Support?

1. **Dask**: When dynamic scheduling / worker reuse becomes beneficial (e.g., iterative ML-style workloads)
1. **MPI**: When rank-based collectives are simpler than task scheduling (e.g., all-reduce operations)
1. **Ray**: Experimental; documented but disabled by default until Kestrel deployment confirms stability

### Why Out-of-Core in Phase 8b?

1. **Proven baseline first**: SLURM arrays (8a) + local chunking can handle 30k samples without full OOM
1. **Minimize risk**: Chunk I/O is independent of distributed backend; can be validated on local machine
1. **Numerical validation**: Can verify chunked-read ≡ full-read on small datasets before deploying to production

______________________________________________________________________

## 5. Integration with Existing Codebase

### Zero Impact on Phase 5-7 Workflow

- Phase 8 features are **opt-in**: `distributed_execution.enabled=false` (default) runs existing single-machine workflow unchanged
- Existing `configs/validation_300_sample_*.yml` files continue to work
- `tools/run_manuscript_pipeline.py` gains optional `--distributed-config` flag; backward compatible without it

### New Code Locations

```
src/bsm_rfm/
  ├── config.py (expanded with distributed_* config classes)
  ├── chunked_io.py (new)
  ├── spill_ops.py (new)
  └── distributed/ (new module)
      ├── __init__.py
      ├── config_distributed.py
      ├── slurm_array_runner.py
      ├── spill.py
      ├── checkpoint.py
      ├── dask_runner.py
      ├── mpi_runner.py
      └── ray_runner_experimental.py

scripts/sbatch_templates/
  ├── slurm_array_stage.sbatch (Jinja2 template)
  ├── slurm_array_reduce.sbatch
  └── slurm_array_diagnostic.sbatch

configs/
  ├── kestrel_distributed_slurm.yml (new example)
  ├── kestrel_distributed_dask.yml (new example)
  └── local_single_machine.yml (updated with distributed=false)

tests/
  ├── test_distributed_config.py (new)
  ├── test_chunked_io.py (new)
  ├── test_spill_ops.py (new)
  ├── test_slurm_array_runner.py (new)
  └── test_distributed_integration.py (new, requires Kestrel)

docs/
  ├── PHASE_8_DISTRIBUTED_HPC_PLAN.md (new)
  ├── DISTRIBUTED_EXECUTION_GUIDE.md (new)
  ├── KESTREL_SLURM_QUICKSTART.md (new)
  ├── CHUNKED_IO_DESIGN.md (new)
  └── ENGINEERING_MANIFEST.md (updated)
```

______________________________________________________________________

## 6. Discovery to Implementation Mapping

| Discovery Finding                 | Phase 8 Implementation                       | Config Parameter         |
| --------------------------------- | -------------------------------------------- | ------------------------ |
| MaxArraySize=11k                  | Shard manifest split logic                   | `slurm.max_array_size`   |
| PreemptMode=OFF                   | Idempotent task design (no retry)            | N/A (design principle)   |
| JobRequeue=0                      | Checkpoint + `_SUCCESS` markers              | N/A (design principle)   |
| `/projects/bsm` for durable state | Manifests, final artifacts stored there      | `spill.projects_root`    |
| `/scratch/$USER` for temp         | Working set, logs, intermediate files        | `spill.scratch_root`     |
| `$TMPDIR` risky on non-NVMe       | `choose_spill_root()` rejects RAM-backed tmp | N/A (auto-handled)       |
| hsn0 network interface            | Dask cluster config                          | `dask.network_interface` |
| 1.7 TB local NVMe on 256 nodes    | Prefer NVMe when available for spill         | `spill.prefer_nvme`      |
| Pixi in `/projects/bsm`           | KESTREL_PROJECT_ROOT, PIXI_HOME env          | N/A (manual setup)       |
| 30-second accounting frequency    | Runtime metric granularity expectation       | N/A (operational note)   |

______________________________________________________________________

## 7. User Deployment Pathway

### For Kestrel Users (Self-Service)

1. **Setup** (one-time):

   ```bash
   cd /projects/bsm
   git clone <repo>
   cd bsm-public-rf
   export PIXI_HOME=/projects/bsm/.pixi
   export PIXI_CACHE_DIR=/projects/bsm/.cache/pixi
   ./scripts/install_pixi_project_env.sh
   source /projects/bsm/.pixi_kestrel_env.sh
   ```

1. **Local validation** (optional):

   ```bash
   pixi run python tools/run_manuscript_pipeline.py \
     configs/validation_300_sample_smoke.yml
   ```

1. **Create shard manifest** (from data):

   ```bash
   pixi run python tools/create_shard_manifest.py \
     --input-root /projects/bsm/data/year_2024 \
     --output-manifest /projects/bsm/runs/run-001/manifest.parquet \
     --shard-size 1000
   ```

1. **Generate SLURM job script** (from config):

   ```bash
   pixi run python tools/render_slurm_array_job.py \
     --config configs/kestrel_distributed_slurm.yml \
     --manifest /projects/bsm/runs/run-001/manifest.parquet \
     --output-script /projects/bsm/runs/run-001/job.sbatch
   ```

1. **Submit and monitor**:

   ```bash
   sbatch -A bsm /projects/bsm/runs/run-001/job.sbatch
   watch squeue -A bsm
   watch cat /projects/bsm/runs/run-001/progress.json
   ```

1. **Collect results** (after completion):

   ```bash
   pixi run python tools/collect_shard_results.py \
     --shard-root /projects/bsm/runs/run-001/outputs \
     --output /projects/bsm/runs/run-001/final_artifacts.parquet
   ```

### For Other HPC Clusters

1. Run discovery suite from `scripts/run_kestrel_discovery_suite.sh` (adapts to local cluster)
1. Update `configs/hpc_cluster_config.yml` with discovered parameters
1. Follow same Kestrel pathway, substituting custom config file

______________________________________________________________________

## 8. Risk Mitigation Summary

| Risk                                | Mitigation                                 | Config Parameter                |
| ----------------------------------- | ------------------------------------------ | ------------------------------- |
| Array size limit exceeded (11k)     | Pre-split manifest into ≤10k shards        | `slurm.max_array_size`          |
| Memory blowup                       | Chunked I/O + runtime thresholds           | `runtime.max_loaded_table_mb`   |
| Numeric drift                       | Accept ≤5 decimals; validate on baseline   | N/A (acceptance criterion)      |
| Filesystem contention               | Use `/scratch` for temp; atomic promotion  | `spill.scratch_root`            |
| Node failure (no auto-requeue)      | Idempotent outputs + `_SUCCESS` markers    | N/A (design principle)          |
| Inter-node comm overhead            | Start with embarrassingly parallel (SLURM) | N/A (design choice)             |
| Kestrel module changes              | Pin Pixi versions; document live probes    | N/A (operational guide)         |
| Dask cluster initialization timeout | Start with SLURM arrays first; defer Dask  | `distributed_execution.backend` |

______________________________________________________________________

## 9. Next Actions (Blocked on 300-sample completion)

1. **Monitor uncapped 300-sample validation** (in background):

   - Extract per-stage timing data from `runtime_diagnostics/stage_runtime_summary.csv`
   - Use timings to calibrate Phase 8 scaling predictions

1. **Post-completion (when 300-sample done)**:

   - Validate artifacts (feature counts, schema)
   - Compare timing vs. capped runs to assess parallelization gains
   - Use baseline to estimate 30k-sample runtime under Phase 8 architectures

1. **Phase 8a kickoff** (after baseline data available):

   - Implement config schema + validation
   - Implement spill path selection logic
   - Create manifest + checkpoint layer
   - Render first SLURM script template
   - Run smoke test locally

1. **User feedback loop**:

   - Deploy Phase 8a alpha on Kestrel
   - Collect telemetry from small (1-10 task) test jobs
   - Iterate on config defaults and error handling
   - Document deployment guide with real examples

______________________________________________________________________

## 10. Success Criteria for Phase 8 Implementation

- [ ] Config file enables all distributed parameters without source code changes
- [ ] SLURM array baseline runs 300-sample dataset on Kestrel successfully
- [ ] Artifacts from distributed run match local-machine baseline ≤ 5 decimal places
- [ ] Full 30k-sample dataset completes in ≤ 24 hours (accounting for scaling overhead)
- [ ] Out-of-core processing passes numerical equivalence test (chunked read ≡ full read)
- [ ] Documentation enables self-service: users can run workflow on Kestrel without assistance
- [ ] Generic HPC support demonstrated: at least one other cluster successfully runs via config

______________________________________________________________________

## 11. References

- **Discovery documents**:

  - `kestrel_bsm_hpc_discovery_answers.md` — live Kestrel system configuration
  - `ai_context/methods/kestrel_slurm_distributed_compute_method_manifest.md` — method guidance
  - `scripts/discovery_*.sh` — automated probing tools

- **Planning documents**:

  - `docs/PHASE_8_DISTRIBUTED_HPC_PLAN.md` — 12-section detailed plan
  - `docs/ENGINEERING_MANIFEST.md` — project status (Phase 8 section)
  - `docs/AGENT_SYNC.md` — task tracking and next steps

- **External references**:

  - NLR Kestrel docs: https://natlabrockies.github.io/HPC/
  - Slurm job arrays: https://slurm.schedmd.com/job_array.html
  - Apache Parquet row groups: https://parquet.apache.org/docs/file-format/configurations/
  - Dask on SLURM: https://docs.dask.org/en/stable/deploying-slurm.html
