# Kestrel SLURM Distributed Compute Engineering Manifest

**Purpose:** Test-first implementation plan for Kestrel SLURM distributed compute support.
**Default account:** `bsm`.

## 1. Required context artifacts

```text
ai_context/literature/kestrel_slurm_distributed_compute_literature_bundle.md
ai_context/methods/kestrel_slurm_distributed_compute_method_manifest.md
ai_context/api_docs/kestrel_slurm_scheduler_interfaces_api_docs.md
ai_context/api_docs/kestrel_distributed_runtime_decision_matrix.md
ai_context/api_docs/runtime_api_bundle_mpi4py_kestrel.md
ai_context/api_docs/runtime_api_bundle_dask_jobqueue_kestrel.md
ai_context/api_docs/runtime_api_bundle_ray_kestrel.md
```

## 2. Implementation goals

Add distributed execution support without locking the repo to a single runtime. Support:

1. Manifest-driven SLURM array execution.
1. Dask+dask-jobqueue runtime configuration and smoke scripts.
1. MPI/mpi4py rank-sharded smoke path.
1. Ray experimental smoke path, disabled by default.
1. Kestrel-specific resource presets for `bsm`, CPU debug probes, and short GPU probes.
1. Filesystem-aware spill and scratch path selection.
1. Idempotent restart semantics through output validation and `_SUCCESS` markers.

## 3. Non-negotiable constraints

**Verified external fact.** NLR examples require/assume SLURM project allocation/account and walltime fields. Source URL: `https://nrel.github.io/HPC/Documentation/Slurm/batch_jobs/`; retrieved 2026-05-10.

**Verified live fact.** Kestrel live config from 2026-05-10 reports:

```text
slurm 25.05.5
MaxArraySize=11000
JobRequeue=0
PreemptMode=OFF
KillWait=30 sec
JobAcctGatherFrequency=30
SelectType=select/cons_tres
SelectTypeParameters=CR_CORE_MEMORY
TaskPlugin=task/cgroup,task/affinity
```

**Verified external fact.** NLR documents `/projects`, `/scratch`, and `$TMPDIR` behavior and warns that `$TMPDIR` can consume RAM on nodes without local disk. Source URL: `https://nrel.github.io/HPC/Documentation/Systems/Kestrel/Filesystems/`; retrieved 2026-05-10.

**Verified external fact.** NLR Lustre best-practices warn against metadata-heavy operations and many-small-file patterns. Source URL: `https://nrel.github.io/HPC/Documentation/Systems/Kestrel/Filesystems/lustre/`; retrieved 2026-05-10.

## 4. Proposed code structure

Adapt exact package paths to the live repo after audit.

```text
src/bsm_rfm/distributed/
  __init__.py
  config.py
  resources.py
  manifests.py
  paths.py
  slurm.py
  backends/
    __init__.py
    slurm_array.py
    dask_jobqueue.py
    mpi4py_backend.py
    ray_experimental.py

scripts/kestrel/
  install_pixi_bsm.sh
  probe_slurm_config.sh
  submit_cpu_debug_probe.sh
  submit_gpu_short_probe.sh
  submit_slurm_array_smoke.sh
  submit_dask_jobqueue_debug_smoke.sh
  submit_mpi4py_debug_smoke.sh
  submit_ray_debug_smoke.sh
  submit_ray_gpu_short_smoke.sh

tests/
  test_distributed_config.py
  test_distributed_paths.py
  test_slurm_script_rendering.py
  test_shard_manifest.py
  test_success_marker_semantics.py
  test_dask_jobqueue_config.py
  test_mpi_rank_mapping.py
  test_ray_experimental_config.py
```

## 5. Configuration model

Create typed config objects or validated dataclasses equivalent to:

```yaml
kestrel:
  account: bsm
  project_root: /projects/bsm
  scratch_root_template: /scratch/{user}/bsm_runs/{run_id}
  cpu_probe_partition: debug
  gpu_probe_partition: gpu-h100s
  production_array_partition: shared
  dask_network_interface: hsn0
  max_array_size: 11000
  job_requeue_default: false
  preemption_enabled: false
  kill_wait_seconds: 30
  job_accounting_frequency_seconds: 30
```

Acceptance criteria:

- Missing account is an error.
- `bsm` is the Kestrel preset default.
- Non-Kestrel tests do not require Kestrel availability.
- Generated scripts expose account, walltime, logs, CPU, memory, and partition explicitly.

## 6. Implementation slices

### Slice 1: Kestrel config and SLURM rendering

Tests first:

```text
- test_default_kestrel_config_uses_bsm_account
- test_cpu_probe_uses_debug_partition
- test_gpu_probe_uses_gpu_h100s_partition
- test_every_rendered_sbatch_has_account_and_time
- test_rendered_scripts_do_not_reference_home_for_pixi_cache
```

Implement config model, sbatch header renderer, Pixi env/cache exports, and thread-control exports.

### Slice 2: Filesystem and spill path selection

Tests first:

```text
- test_tmpdir_missing_falls_back_to_scratch
- test_tmpdir_tmpfs_falls_back_to_scratch
- test_disk_backed_tmpdir_is_used_for_local_spill
- test_scratch_path_includes_user_run_id_and_no_flat_global_dir
```

Implement `choose_spill_root()`, `choose_scratch_root()`, and `choose_log_root()`.

### Slice 3: Manifest-driven SLURM arrays

Tests first:

```text
- test_manifest_schema_required_columns
- test_array_index_selects_expected_shard
- test_success_marker_written_after_validation_only
- test_completed_shard_skipped_on_rerun
- test_incomplete_attempt_not_promoted
```

Implement shard selection by `SLURM_ARRAY_TASK_ID`, attempt directory writes, validation, atomic promotion, and `_SUCCESS.json`.

### Slice 4: Dask + dask-jobqueue backend

Tests first:

```text
- test_dask_config_uses_account_bsm
- test_dask_cpu_probe_queue_debug
- test_dask_production_queue_can_be_shared_or_nvme
- test_dask_uses_hsn0_interface_by_default
- test_dask_local_directory_uses_spill_selector
- test_no_deprecated_dask_jobqueue_parameters
```

Use `account`, `job_script_prologue`, and `job_extra_directives`; do not use deprecated `project`, `env_extra`, or `job_extra`. Source URL: `https://jobqueue.dask.org/en/stable/generated/dask_jobqueue.SLURMCluster.html`; retrieved 2026-05-10.

### Slice 5: MPI/mpi4py backend

Tests first:

```text
- test_rank_to_shard_mapping_is_deterministic
- test_rank_output_paths_are_unique
- test_mpi_backend_requires_explicit_manifest
- test_mpi_script_uses_srun_not_mpirun
```

Use Cray MPICH-oriented environment by default; avoid OpenMPI unless explicitly approved by HPC support. Source URL: `https://nrel.github.io/HPC/Documentation/Development/Programming-Environments/`; retrieved 2026-05-10.

### Slice 6: Ray experimental backend

Tests first:

```text
- test_ray_backend_disabled_by_default
- test_ray_requires_enable_env_var
- test_ray_script_uses_symmetric_run
- test_ray_gpu_script_uses_gpu_h100s
- test_ray_version_constraint_documented
```

Keep runtime disabled unless `BSM_ENABLE_RAY_EXPERIMENTAL=1`. Ray `symmetric-run` is documented for Ray 2.49+ SLURM launches. Source URL: `https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm.html`; retrieved 2026-05-10.

### Slice 7: Observability and accounting

Tests first:

```text
- test_sacct_command_contains_required_fields
- test_logs_include_job_id_and_array_task_id
- test_accounting_summary_parser_handles_missing_disk_fields
```

Add `sacct` command builder, parser for pipe-delimited `sacct` output, and job metrics artifact writer.

## 7. Manual validation sequence on Kestrel

```bash
bash scripts/kestrel/probe_slurm_config.sh
bash scripts/kestrel/install_pixi_bsm.sh
bash scripts/kestrel/submit_cpu_debug_probe.sh
bash scripts/kestrel/submit_slurm_array_smoke.sh
bash scripts/kestrel/submit_dask_jobqueue_debug_smoke.sh
bash scripts/kestrel/submit_mpi4py_debug_smoke.sh
BSM_ENABLE_RAY_EXPERIMENTAL=1 bash scripts/kestrel/submit_ray_debug_smoke.sh
BSM_ENABLE_RAY_EXPERIMENTAL=1 bash scripts/kestrel/submit_ray_gpu_short_smoke.sh
```

## 8. Completion definition

Complete only when local tests pass, script rendering tests prove required Slurm fields, manual CPU debug probes pass, SLURM array smoke validates `_SUCCESS` behavior, Dask-jobqueue smoke passes, MPI smoke passes, Ray remains disabled by default, and documentation separates verified facts, assumptions, and unresolved Kestrel-only checks.

## 9. Explicit non-goals

```text
- Do not implement production Ray before smoke tests.
- Do not require GPUs for CPU workflow tests.
- Do not assume /home can host Pixi envs or large caches.
- Do not use $TMPDIR as disk without checking filesystem type.
- Do not rely on automatic SLURM requeue for correctness.
- Do not generate huge small-file directories.
- Do not use deprecated dask-jobqueue API parameters.
```
