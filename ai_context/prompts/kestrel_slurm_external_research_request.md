# External Research Request: NREL Kestrel SLURM Distributed Compute Support

Copy/paste this prompt into a web-enabled research assistant.

______________________________________________________________________

You are an external web-enabled research assistant. Your task is to produce **repo-local artifacts only** for planning HPC/distributed compute support on **NREL Kestrel** using **SLURM**.

## Hard constraints

1. Do **not** provide only chat prose. Write outputs as artifact-ready markdown content mapped to exact filenames listed below.
1. Include citations/URLs and retrieval dates for all external claims.
1. Distinguish verified facts vs assumptions.
1. Do not invent package/API behavior; cite official docs.
1. Scope is **multi-runtime comparison first** (do not lock to one runtime without evidence).

## Research objectives

1. Identify Kestrel + SLURM operational constraints relevant to distributed Python workflows.
1. Compare candidate distributed runtimes for this repo context (at minimum: MPI/mpi4py, Dask+dask-jobqueue, Ray on SLURM).
1. Produce implementation-oriented method guidance and risk notes suitable for a local Copilot agent.
1. Provide current API syntax and version-sensitive caveats for shortlisted runtimes and scheduler interfaces.
1. Convert research findings into an implementation-ready engineering manifest and a direct Copilot execution prompt.

## Required artifact outputs

Generate content for all files below:

1. `ai_context/literature/kestrel_slurm_distributed_compute_literature_bundle.md`
1. `ai_context/methods/kestrel_slurm_distributed_compute_method_manifest.md`
1. `ai_context/api_docs/kestrel_slurm_scheduler_interfaces_api_docs.md`
1. `ai_context/api_docs/kestrel_distributed_runtime_decision_matrix.md`
1. `ai_context/api_docs/runtime_api_bundle_mpi4py_kestrel.md`
1. `ai_context/api_docs/runtime_api_bundle_dask_jobqueue_kestrel.md`
1. `ai_context/api_docs/runtime_api_bundle_ray_kestrel.md`
1. `ai_context/manifests/kestrel_slurm_distributed_compute_engineering_manifest.md`
1. `ai_context/prompts/kestrel_slurm_distributed_compute_copilot_execution_prompt.md`

## Required content quality checks

- Every API claim has a source URL and date.
- Runtime comparison includes selection criteria, trade-offs, and recommended default + fallback.
- Kestrel/SLURM guidance includes job submission/launch patterns, environment/module assumptions, filesystem/IO considerations, and scaling/failure caveats as documented by cited sources.
- Engineering manifest is concrete enough for test-first local implementation slices.
- Copilot execution prompt references the produced artifacts explicitly and forbids scope drift.

## Output format

Return your answer in this exact structure:

1. `FILE: <path>` followed by complete markdown content for that file.
1. Repeat for every required file.
1. End with a short unresolved-questions list (if any) and confidence notes.

______________________________________________________________________
