# Manuscript data contract

Phase 1 freezes the real-data artifact and notebook manifests for the complete manuscript
reproduction package.

Required real-data artifacts:

- `input_metadata`
- `output_metadata`
- `case_study_input_matrix`
- `case_study_output_matrix`
- `manuscript_feature_catalog`
- `fixed_holdout_assignments`

Placeholder-path policy:

- tracked template: `configs/manuscript_paths.template.yml`
- local override: `configs/local/manuscript_paths.local.yml`
- placeholder token: `REPLACE_WITH_REAL_FILE/`

Notebook execution order (frozen in `configs/manuscript_runtime.yml`):

- `00_case_study_data_intake.ipynb`
- `01_candidate_library_audit.ipynb`
- `02_output_conditioning.ipynb`
- `03_empirical_null_screen.ipynb`
- `04_interaction_discovery.ipynb`
- `05_nonlinear_discovery.ipynb`
- `06_sparse_selection_and_stability.ipynb`
- `07_final_ols_and_bundle_export.ipynb`
- `08_manuscript_tables_and_figures.ipynb`
