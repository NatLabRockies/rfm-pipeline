# HPC and Distributed Execution Guide

The `rfm-pipeline` package itself does not ship cluster-specific HPC
orchestration. It provides the building blocks (`rfm-hpc-submit`,
`rfm-hpc-worker`, `rfm-hpc-reduce` console entry points) that downstream
projects can wire into their own SLURM / cluster workflows.

## Console entry points

After `pixi install`, the following commands are available:

| Command          | Purpose                                                 |
| ---------------- | ------------------------------------------------------- |
| `rfm-hpc-submit` | Generate SLURM array job scripts for a configured stage |
| `rfm-hpc-worker` | Execute one shard of work on a compute node             |
| `rfm-hpc-reduce` | Aggregate shard outputs into canonical stage artifacts  |

Run any command with `--help` for full usage.

## Companion HPC orchestration repo

The BSM manuscript companion repository
[`NatLabRockies/bsm-public-rf`](https://github.com/NatLabRockies/bsm-public-rf)
provides a complete reference implementation of an end-to-end HPC
workflow that drives this pipeline, including:

- NREL Kestrel SLURM templates and submission wrappers
  (`scripts/kestrel/`)
- A local orchestration entry point (`scripts/hpc_workflow.py`) that
  ties submit / status / collect into one local command
- A four-step publication-grade run suite under
  `scripts/publication-run/`
- Six-stage cascade configuration for the manuscript study
  (`configs/hpc/kestrel_publication_*.yml`)

If you are adapting this pipeline to a new cluster, start from
`bsm-public-rf`'s scripts and configs as a template rather than from
this repository.

## Programmatic shard / reduce usage

The HPC entry points are thin wrappers around the importable functions
in `rfm_pipeline.hpc_submit`, `rfm_pipeline.hpc_shard_worker`, and
`rfm_pipeline.hpc_reduce`. Custom cluster integrations can call these
directly. See the module docstrings for argument contracts.
