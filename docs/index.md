# rfm-pipeline documentation

Use `rfm-pipeline` to fit and export reduced-form models. If you only need
predictions from the released BSM model, use
[`bsm-public-rf`](https://github.com/NatLabRockies/bsm-public-rf) instead.

## Start by outcome

| Goal                             | Start here                                                                                                 |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| Verify an installation           | [Setup and first run](setup_and_first_run.md)                                                              |
| Fit a model from four DataFrames | [Python API quickstart](quickstart.md)                                                                     |
| Run the full staged workflow     | [Configuration reference](configuration_reference.md)                                                      |
| Locate or interpret outputs      | [Artifact reference](artifact_reference.md)                                                                |
| Scale beyond one process         | [HPC and distributed execution](HPC_DISTRIBUTED_EXECUTION.md)                                              |
| Explore the BSM implementation   | [BSM workflow case study](https://github.com/NatLabRockies/rfm-pipeline/tree/main/examples/bsm-manuscript) |

```{toctree}
:maxdepth: 2
:caption: Start here

overview
setup_and_first_run
quickstart
reproducibility_example
```

```{toctree}
:maxdepth: 2
:caption: Use the workflow

configuration_reference
artifact_reference
interpreting_results
export_bundle
HPC_DISTRIBUTED_EXECUTION
troubleshooting
```

```{toctree}
:maxdepth: 2
:caption: Reference

scope_boundary
api
```

## Project records

Scientific audits, design records, historical implementation plans, and the
BSM publication handoff remain in the repository for traceability. They are
maintainer records, not prerequisites for using the package, and are omitted
from the primary navigation.

```{toctree}
:hidden:

manuscript_data_contract
debiased_lasso_contract
debiased_lasso_exactness_audit
manuscript_runtime
ENGINEERING_MANIFEST
CAVEMAN_CONTEXT
decision_log
manuscript_alignment_audit
review_register
scope_backlog
workflow_audit
module_plan
manuscripts/full_dataset_run_revision_notes
manuscripts/track_b_analytic_baselines
manuscripts/manuscript_impact_log
manuscript_summary_log
3K_TEST_VALIDATION_REPORT
CATALOG_GENERATION_GUIDE
CATALOG_STRUCTURE_FINDINGS
CATALOG_UTILITY_SUMMARY
DOCUMENTATION_AUDIT
DOCUMENTATION_IMPROVEMENT_PLAN
RUNNING_MANUSCRIPT_REPRODUCTION
NOTEBOOK_STEP_BY_STEP_WORKFLOW
SESSION_SUMMARY_2026-05-08
SESSION_SUMMARY_2026_05_09
WORKFLOW_FIX_PLAN
MANUSCRIPT_WORKFLOW_REFERENCE
PARALLEL_RUN_VALIDATION_CHECKLIST
RUNTIME_INVESTIGATION_WORKFLOW
REFACTOR_CONFIG_DRIVEN_DESIGN
PHASE5_CONFIG_DRIVEN_IMPLEMENTATION
PHASE_8_SCALABLE_EXECUTION_PLAN
PHASE_8_DISTRIBUTED_HPC_PLAN
HPC_IMPLEMENTATION_READINESS_SUMMARY
HPC_SCALING_BENCHMARK
INTERACTION_DISCOVERY_OPTIMIZATION
```
