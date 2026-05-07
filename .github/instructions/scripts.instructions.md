______________________________________________________________________

## applyTo: "scripts/**/\*.py,scripts/**/\*.sh,test_repo.sh"

# Scripts and Validation

Scripts must be deterministic and repo-local.

For Python/tooling, use Pixi from repo root:

- `pixi run ...`

Shell scripts:

- use `set -euo pipefail`
- avoid destructive cleanup by default
- provide `--help` when user-facing
- support check-only behavior by default when applicable

`test_repo.sh` is the authoritative local gate when present. CI should call it.
