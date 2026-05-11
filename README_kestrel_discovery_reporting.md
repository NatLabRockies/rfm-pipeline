# Kestrel BSM discovery scripts with status reporting

This bundle updates all Kestrel discovery scripts to report their current phase.

## What changed

- Shared status helper in `hpc_kestrel_config.sh`.
- Shared Slurm monitor helper reports:
  - `SUBMITTED`
  - `PENDING` with Slurm reason, usually waiting for node/resources
  - `RUNNING` with allocated node list and elapsed time
  - periodic output tail while running
  - `FINAL` with `sacct` summary
- CPU probes default to `--account=bsm --partition=debug`.
- GPU probes default to `--account=bsm --partition=gpu-h100s`.
- Python/Pixi, chunk spill, Parquet, environment, Pixi install, and suite scripts now emit explicit `step=...` status lines.

## Useful runtime knobs

```bash
MONITOR_INTERVAL_SECONDS=30 ./discovery_collect_slurm_job.sh
MONITOR_TAIL_LINES=20 ./discovery_submit_python_probes.sh
```

## Recommended order

```bash
./install_pixi_project_env.sh
./discovery_collect_env.sh
./discovery_collect_slurm_job.sh
./discovery_submit_python_probes.sh
./discovery_gpu_probe.sh
```

or run the suite:

```bash
./run_kestrel_discovery_suite.sh
```
