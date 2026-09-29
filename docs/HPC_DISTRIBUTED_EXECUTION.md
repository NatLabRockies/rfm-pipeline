# HPC and distributed execution

Start with the canonical API or a small staged run. Use distributed execution
only after the data contract and configuration pass locally.

## Package entry points

| Command          | Role                                                      |
| ---------------- | --------------------------------------------------------- |
| `rfm-hpc-submit` | Build a shard manifest and scheduler submission artifacts |
| `rfm-hpc-worker` | Execute one stage shard                                   |
| `rfm-hpc-reduce` | Verify and combine a complete shard set                   |

After installation, run each command with `--help` for its exact arguments:

```bash
rfm-hpc-submit --help
rfm-hpc-worker --help
rfm-hpc-reduce --help
```

The corresponding Python modules are
`rfm_pipeline.hpc_submit`, `rfm_pipeline.hpc_shard_worker`, and
`rfm_pipeline.hpc_reduce`.

## What the package does not assume

The package does not assume a particular cluster, account, filesystem, queue,
or allocation. A project must provide its own scheduler configuration,
storage paths, resource limits, submission authorization, and artifact
collection policy.

## Reference integration

The
[`examples/bsm-manuscript` case study](https://github.com/NatLabRockies/rfm-pipeline/tree/main/examples/bsm-manuscript)
contains a complete SLURM/Kestrel integration, including configs,
orchestration wrappers, recovery controls, and validation records. It is an
example to adapt, not a portable default and not required for ordinary package
use.

The case study is fail-closed: its local checks do not authorize or submit a
production campaign. Follow its own execution guide and scientific gates when
working on that study.

## Recommended sequence

1. Validate a small local run.
1. Freeze the config, data identities, output root, and expected shard set.
1. Run submission generation or scheduler dry-run checks.
1. Execute workers into isolated shard directories.
1. Reduce only a complete, identity-matched shard set.
1. verify terminal markers, manifests, and hashes before downstream stages.

For out-of-core execution on one machine, use
`runtime.out_of_core` in the [Configuration reference](configuration_reference.md);
no scheduler is required.
