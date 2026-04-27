# Manuscript runtime and notebook skeletons

Phase 2 adds the execution layer that makes the manuscript notebooks runnable in both CI and local
real-data mode.

## Runtime resolution policy

Use `bsm_rfm.resolve_manuscript_runtime(repo_root)` to determine notebook inputs.

- When `configs/local/manuscript_paths.local.yml` exists and every required artifact file is
  present, the runtime resolves to `mode = "real"`.
- Otherwise the runtime falls back to `mode = "demo"` and writes a deterministic toy dataset to a
  temporary directory.

The deterministic demo path exists so the notebook gate can execute the full notebook stack without
shipping the real BSM case-study files.

## Data-intake helpers

The public helpers for the notebook layer are:

- `bsm_rfm.load_manuscript_runtime_manifest(...)`
- `bsm_rfm.load_manuscript_paths_template(...)`
- `bsm_rfm.load_manuscript_local_override(...)`
- `bsm_rfm.resolve_manuscript_runtime(...)`
- `bsm_rfm.load_manuscript_artifact_tables(...)`
- `bsm_rfm.validate_manuscript_artifact_tables(...)`
- `bsm_rfm.build_manuscript_notebook_context(...)`
- `bsm_rfm.manuscript_runtime_summary_table(...)`

## Notebook skeletons

The tracked manuscript notebooks now exist at the frozen Phase 1 paths under
`notebooks/manuscript/`:

- `00_case_study_data_intake.ipynb`
- `01_candidate_library_audit.ipynb`
- `02_output_conditioning.ipynb`
- `03_empirical_null_screen.ipynb`
- `04_interaction_discovery.ipynb`
- `05_nonlinear_discovery.ipynb`
- `06_sparse_selection_and_stability.ipynb`
- `07_final_ols_and_bundle_export.ipynb`
- `08_manuscript_tables_and_figures.ipynb`

## Phase 3 output-conditioning stage

The first Phase 3 source-backed notebook stage is `02_output_conditioning.ipynb`. It calls
`bsm_rfm.run_output_conditioning_stage(...)`, which reads the frozen output-conditioning settings
from `configs/manuscript_case_study.yml`, applies train-only output filters, computes the PCA
reduced-response representation, and writes deterministic CSV handoff artifacts under
`{output_root}/output_conditioning/`.

The stage currently writes:

- `output_filter_diagnostics.csv`
- `pca_scores.csv`
- `pca_loadings.csv`
- `pca_explained_variance.csv`
- `output_conditioning_summary.csv`

The remaining manuscript notebooks still provide the Phase 2 executable skeletons until their
scientific stages are promoted into source-backed functions in later Phase 3 slices.
