# Search-Chat Prompt: HPC/SLURM Environment Discovery

I need environment details to implement large-scale out-of-core + spill-to-disk + SLURM distributed execution for a Python/Pixi pipeline processing parquet data (target: millions of rows and millions of columns). Please answer with citations and concrete commands/limits for my target HPC environment.

1. Scheduler + policy

- SLURM version? Any known version-specific limitations?
- Required account/partition/QOS fields for batch jobs?
- Max walltime per partition/QOS?
- Max jobs per user and max array size?
- Preemption behavior and signal timing before kill?
- Requeue policies and whether `--requeue` is supported/enforced?
- Job dependency support and common patterns used on this cluster?

2. Node topology

- CPU architecture(s), sockets/cores/threads per node?
- RAM per node (and per core limits if enforced)?
- NUMA layout and recommended binding options?
- Local NVMe/scratch availability and size?
- Ephemeral vs persistent node-local storage behavior?

3. Storage + I/O

- Recommended paths for high-throughput temporary files?
- Parallel filesystem type(s) and best practices for metadata-heavy workloads?
- Typical read/write throughput and IOPS characteristics per filesystem tier?
- Quotas/inode limits on home/scratch/project filesystems?
- Any anti-patterns explicitly prohibited (small-file storms, etc.)?

4. Python execution model

- Recommended way to run Python on compute nodes (module/conda/container)?
- Any restrictions on internet/package install at runtime?
- MPI/UCX/Dask/Ray support available and preferred?
- Container runtime availability (Apptainer/Singularity) and policy?

5. Monitoring + observability

- Preferred tools for job-level metrics (CPU, RSS, I/O, GPU)?
- Can we collect per-task memory and I/O from `sacct`/`sstat` reliably?
- Log retention policy and recommended log directory strategy?
- Any cluster-native monitoring endpoints/APIs users can query?

6. Reliability + restart

- Recommended checkpointing patterns for long jobs?
- Best practices for idempotent reruns after preemption/node failure?
- Any known filesystem consistency caveats affecting checkpoint files?

7. Security/compliance constraints

- Any restrictions on temporary data at node-local scratch?
- Encryption-at-rest/in-transit requirements?
- Data locality constraints (must stay on-cluster partitions)?

8. Scaling guidance

- Official guidance for workloads with huge feature matrices and parquet?
- Recommended chunk sizes / row-group sizes / file counts for parquet on this cluster?
- Practical limits observed by users for Python dataframe/array workloads?

Please provide:

- concrete values where possible,
- shell commands to verify each claim,
- and a final **implementation constraints summary** section I can hand to an engineer.
