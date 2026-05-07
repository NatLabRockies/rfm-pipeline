______________________________________________________________________

## applyTo: "artifacts/**,outputs/**"

# Artifacts and Outputs

Treat `artifacts/` and `outputs/` as generated or semi-generated unless repository docs say otherwise.

Do not edit generated outputs by hand unless the task explicitly targets them.

Do not commit large generated artifacts unless the manifest says they are tracked deliverables.

If outputs change, document:

- command that generated them
- input data/config
- expected reproducibility
- whether they should be committed
