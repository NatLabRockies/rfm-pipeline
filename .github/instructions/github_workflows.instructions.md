______________________________________________________________________

## applyTo: ".github/workflows/**/\*.yml,.github/workflows/**/*.yaml,.github/hooks/\*\*/*.json"

# GitHub Workflows and Hooks

Local and CI validation must align.

Prefer CI calling:

- `bash test_repo.sh --ci`

Do not create duplicate CI entrypoints.

Hook configs under `.github/hooks/*.json` enforce Copilot policy. Do not weaken them to make agent work easier.

Changes to hooks/CI are protected-surface changes: require targeted validation and clear rationale.
