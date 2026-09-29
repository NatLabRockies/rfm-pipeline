______________________________________________________________________

## applyTo: "scripts/**/\*.py,scripts/**/\*.sh,tools/\*\*/\*.py"

# Scripts and Validation

Scripts must be deterministic and repo-local.

For Python/tooling, use Pixi from repo root:

- `pixi run ...`

Shell scripts:

- use `set -euo pipefail`
- avoid destructive cleanup by default
- provide `--help` when user-facing
- support check-only behavior by default when applicable

`pixi run gate` is the authoritative local gate. CI should call the matching
`pixi run ci` task.
