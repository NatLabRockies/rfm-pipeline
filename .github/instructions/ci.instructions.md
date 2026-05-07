______________________________________________________________________

## applyTo: ".github/workflows/\*\*"

# CI instructions

Keep local and CI validation aligned.

Prefer CI calling:

```bash
bash test_repo.sh --ci
```

Do not add duplicate workflows or hidden validation paths.
Do not weaken CI to pass.
