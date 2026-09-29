# Workflow configurations

Most users should begin with the Python API in
[`docs/quickstart.md`](../docs/quickstart.md); it does not require YAML.

For the full staged workflow:

1. Copy `workflow.template.yml`.
1. Set `dataset.path` and `output.artifact_dir`.
1. Run the staged entry point.

```bash
cp configs/workflow.template.yml configs/my-workflow.yml
pixi run python tools/run_manuscript_pipeline.py configs/my-workflow.yml
```

The runner filename is retained for compatibility, but the typed config is
reusable. See
[`docs/configuration_reference.md`](../docs/configuration_reference.md) for
the dataset layout, fields, and resume controls.

## Directory map

| Path                    | Purpose                                                          |
| ----------------------- | ---------------------------------------------------------------- |
| `workflow.template.yml` | Annotated starting point for a new staged run                    |
| `validation_*.yml`      | Repository test and performance fixtures                         |
| `sensitivity_study/`    | Sensitivity-study specifications                                 |
| `datasets/`             | Legacy BSM-compatible direct-path adapter                        |
| `hpc/`                  | Study-specific scheduler configuration retained for traceability |
| `manuscript_*.yml`      | Historical BSM stage and notebook contracts                      |

Validation, HPC, and manuscript configs are not general-purpose templates.
Use them only with the data and execution contracts they name.
