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

The tracked manuscript notebooks now exist at the frozen Phase 1 paths under `notebooks/manuscript/`:

- `00_case_study_data_intake.ipynb`
- `01_candidate_library_audit.ipynb`
- `02_output_conditioning.ipynb`
- `03_empirical_null_screen.ipynb`
- `04_interaction_discovery.ipynb`
- `05_nonlinear_discovery.ipynb`
- `06_sparse_selection_and_stability.ipynb`
- `07_final_ols_and_bundle_export.ipynb`
- `08_manuscript_tables_and_figures.ipynb`

These notebooks are intentionally lightweight in Phase 2. They prove that the runtime/path layer is
working and establish the exact manuscript-stage entrypoints that later phases must fill in with the
full real-data workflow.
