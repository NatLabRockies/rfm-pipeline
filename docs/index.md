# rfm-pipeline documentation

Use `rfm-pipeline` to fit, evaluate, and export reduced-form models. For a
ready-to-use fitted model, see
[`bsm-public-rf`](https://github.com/NatLabRockies/bsm-public-rf).

## Start by outcome

| Goal                             | Start here                                                    |
| -------------------------------- | ------------------------------------------------------------- |
| Verify an installation           | [Setup and first run](setup_and_first_run.md)                 |
| Fit a model from four DataFrames | [Python API quickstart](quickstart.md)                        |
| Run the staged workflow          | [Configuration reference](configuration_reference.md)         |
| Locate or interpret outputs      | [Artifact reference](artifact_reference.md)                   |
| Scale beyond one process         | [HPC and distributed execution](HPC_DISTRIBUTED_EXECUTION.md) |

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
