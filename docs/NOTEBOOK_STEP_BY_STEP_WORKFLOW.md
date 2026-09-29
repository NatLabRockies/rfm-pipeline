# Notebook Step-by-Step Workflow

This notebook sequence is the supported stage-by-stage debugging path for manuscript reproduction.

## Notebook stage map

| Notebook                                  | Canonical call                                                                     |
| ----------------------------------------- | ---------------------------------------------------------------------------------- |
| `00_case_study_data_intake.ipynb`         | `build_manuscript_notebook_context(...)` + `manuscript_runtime_summary_table(...)` |
| `01_candidate_library_audit.ipynb`        | `build_manuscript_notebook_context(...)`                                           |
| `02_output_conditioning.ipynb`            | `run_output_conditioning_stage(context)`                                           |
| `03_empirical_null_screen.ipynb`          | `run_empirical_null_screening_stage(context)`                                      |
| `04_interaction_discovery.ipynb`          | `run_interaction_discovery_stage(context)`                                         |
| `05_nonlinear_discovery.ipynb`            | `run_nonlinear_discovery_stage(context)`                                           |
| `06_sparse_selection_and_stability.ipynb` | `run_sparse_selection_stability_stage(context)`                                    |
| `07_final_ols_and_bundle_export.ipynb`    | `run_final_manuscript_artifacts_stage(context)`                                    |
| `08_manuscript_tables_and_figures.ipynb`  | `run_final_manuscript_artifacts_stage(context)`                                    |

All notebooks should construct context with:

```python
from rfm_pipeline import build_manuscript_notebook_context
context = build_manuscript_notebook_context(REPO_ROOT, "<notebook_name>.ipynb")
```

## Validation guardrail

Notebook stage-call correctness is checked by:

```bash
pixi run notebook-workflow-check
```

This check is part of `pixi run gate`.

## Recommended debug workflow

1. Run notebook `00` and verify runtime tables and paths.
1. Execute notebooks `02` → `08` in order.
1. If a stage fails, inspect that stage artifact directory first.
1. Re-run the equivalent pipeline config with tracked runner for full logs:

```bash
pixi run workflow-run -- --config configs/validation_300_sample_no_caps.yml
```

Tracked logs and summaries are written under `artifacts/workflow_runs/`.
