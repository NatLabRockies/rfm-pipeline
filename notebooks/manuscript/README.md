# Manuscript notebooks

This directory contains the tracked manuscript notebook entrypoints for the complete JDS BSM
manuscript-reproduction workflow. The Phase 3 notebooks call source-backed stage functions and
write deterministic handoff artifacts under the resolved manuscript output root.

The execution order is frozen in `configs/manuscript_runtime.yml`.

During automated validation, the notebooks resolve placeholder paths to a deterministic demo
dataset through `bsm_rfm.resolve_manuscript_runtime(...)`. For the real case study, create
`configs/local/manuscript_paths.local.yml` from `configs/manuscript_paths.template.yml` and update
the paths to your local files.
