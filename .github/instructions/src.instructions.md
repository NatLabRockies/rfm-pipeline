______________________________________________________________________

## applyTo: "src/\*\*/\*.py"

# Source Code

Protect public APIs, config schemas, data contracts, and canonical interfaces.

Before implementation:

- identify module contract
- write tests first
- preserve existing public behavior unless manifest requires change
- avoid broad abstractions and new extension points unless justified

Do not add compatibility shims unless explicitly requested.

Use clear errors and tests for invalid input.
