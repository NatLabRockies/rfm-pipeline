______________________________________________________________________

## applyTo: "analysis/**/\*.py,analysis/**/*.ipynb,reports/\*\*/*.md,reports/\*\*/\*.qmd"

# Analysis and Reports

Use repo-local Pixi. Do not use bare Python or global environments.

Keep analysis reproducible:

- no absolute user paths
- no hidden manual steps
- config comes from `config/`
- intermediate products go to `outputs/` only when intentionally generated
- final reports go to `reports/`

Do not make analytical claims that are not backed by code, tests, or documented inputs.

If analysis logic becomes reusable, move it to `src/` with tests rather than duplicating notebook code.
