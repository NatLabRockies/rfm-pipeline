______________________________________________________________________

## applyTo: "tests/\*\*/\*.py"

# Tests

Tests define behavior. Do not weaken them.

Required:

- focused tests before implementation
- negative and edge cases for validation logic
- no local absolute paths
- no network dependency unless marked live/integration
- no broad xfail/skip to pass

Do not commit failing tests merely to prove red state. Observe red locally, then implement.
