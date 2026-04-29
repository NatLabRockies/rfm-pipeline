# AGENTS.md

## Purpose

This file defines the operating rules for Codex or any other coding agent working in this repository.

The goal is to make agent-assisted development safe, reproducible, reviewable, and compatible with normal GitHub workflows. The agent must treat this repository as production software, even if the repository is experimental, research-oriented, or early-stage.

The agent must prioritize correctness, traceability, deterministic validation, and minimal safe changes over speed.

______________________________________________________________________

## Non-negotiable rules

- Treat the live repository on disk as the only source of truth.
- Do not trust prior summaries, generated bundles, previous chat context, old patches, or claimed repository state.
- Always inspect the current repository state before planning or editing.
- Always work on a branch.
- Never intentionally make implementation changes directly on `main`, `master`, `develop`, `dev`, `trunk`, or another protected/base branch.
- Ask before committing.
- Ask before pushing.
- Ask before opening a pull request unless the user explicitly requested one.
- Ask before merging a pull request.
- Ask before deleting local or remote branches.
- Do not overwrite, discard, reset, rebase, stash, checkout over, or otherwise disturb user changes unless explicitly approved.
- Do not add backward-compatibility shims, wrappers, aliases, deprecated interfaces, migration bridges, or compatibility layers unless explicitly asked.
- Do not lower, weaken, remove, skip, xfail, or narrow tests to make them pass.
- Do not change public APIs, file formats, config contracts, CLI behavior, or documented workflows unless the engineering manifest or the user explicitly requires it.
- Do not hide failures.
- Do not claim validation passed unless the relevant validation command actually passed.
- Do not claim full repository readiness unless the full repository gate has passed.
- Fix root causes, not symptoms.
- Prefer small, atomic, reviewable slices over broad rewrites.
- Preserve user work.

______________________________________________________________________

## Required session start

At the start of every task, before editing files, inspect the repository and Git state.

Run:

```bash
pwd
git status --short --branch
git branch --show-current
git remote -v
```

Then inspect the available project instructions and validation files:

```bash
find . -maxdepth 4 \( \
  -name 'AGENTS.md' -o \
  -name 'CLAUDE.md' -o \
  -name 'MEMORY.md' -o \
  -name '*MANIFEST*' -o \
  -name 'README*' -o \
  -name 'CONTRIBUTING*' -o \
  -name 'pyproject.toml' -o \
  -name 'pixi.toml' -o \
  -name 'package.json' -o \
  -name 'Makefile' -o \
  -name 'noxfile.py' -o \
  -name 'tox.ini' -o \
  -name 'test_repo.sh' \
\) -print
```

Read all relevant instruction files before planning:

- `AGENTS.md`
- `README.md` or `README.*`
- `CONTRIBUTING.md`, if present
- `docs/MEMORY.md`, if present
- `MEMORY.md`, if present
- `docs/ENGINEERING_MANIFEST.md`, if present
- `ENGINEERING_MANIFEST.md`, if present
- other manifest, roadmap, architecture, or development guide files discovered during inspection
- `pyproject.toml`, `pixi.toml`, `package.json`, `Makefile`, `noxfile.py`, `tox.ini`, or equivalent tooling files

If instruction files conflict, use this precedence order:

1. explicit user instruction in the current task
1. this `AGENTS.md`
1. repository-specific engineering manifest or roadmap
1. memory files
1. README / contributing docs
1. inferred conventions from the live codebase

When conflicts are material, state the conflict and the rule chosen.

______________________________________________________________________

## Branch discipline

The agent must always work in a Git branch.

Before editing, determine the current branch:

```bash
git status --short --branch
git branch --show-current
```

If the current branch is `main`, `master`, `develop`, `dev`, `trunk`, or another obvious base/protected branch, create a task branch before editing.

Preferred branch naming:

```text
codex/<short-task-name>
```

Examples:

```text
codex/add-test-repo-gate
codex/harden-manifest-loader
codex/fix-null-screening-validation
codex/update-notebook-ci
codex/cleanup-lint-format
```

If there are uncommitted changes before branch creation:

- Do not overwrite them.
- Do not stash them without approval.
- Do not reset them.
- Do not clean them.
- Report the dirty state.
- Ask whether to create the new branch with the existing changes, stop, or let the user clean the worktree.

If the worktree is clean and the current branch is a protected/base branch, create a new branch before editing:

```bash
git switch -c codex/<short-task-name>
```

If the current branch is already a task branch, continue on it unless the task clearly requires a new branch.

Before creating a new branch from a base branch, inspect remotes. If safe, fetch without changing local files:

```bash
git fetch --all --prune
```

Forbidden without explicit approval:

```bash
git reset --hard
git clean -fd
git clean -fdx
git checkout -- .
git restore .
git restore --source
git rebase
git push --force
git push --force-with-lease
git branch -D
git branch -d
git push origin --delete
```

______________________________________________________________________

## Pull request workflow

Use pull requests for reviewable changes when a remote GitHub repository is configured.

Normal workflow:

1. inspect repository state
1. create or use a task branch
1. implement one atomic slice
1. run validation
1. summarize the diff
1. ask before committing
1. commit only after approval
1. ask before pushing
1. push only after approval
1. ask before opening a pull request unless already requested
1. open a pull request only after approval
1. ask before merging
1. merge only after approval and after required checks pass

If GitHub CLI is available, prefer `gh` for PR operations:

```bash
gh status
gh pr status
gh pr create
gh pr view --web
gh pr checks
gh pr merge
```

Do not merge a PR unless:

- the user explicitly says to merge
- the branch is pushed
- the PR exists
- the diff has been summarized
- validation status has been reported
- GitHub checks are passing, or the user explicitly accepts the risk

Preferred merge mode depends on repository convention. If no convention exists, ask before choosing squash, merge commit, or rebase merge.

After a PR is merged, ask before deleting local or remote branches.

______________________________________________________________________

## Engineering manifest workflow

If an engineering manifest exists, it is the primary planning document for implementation order.

Common manifest locations include:

- `docs/ENGINEERING_MANIFEST.md`
- `ENGINEERING_MANIFEST.md`
- `docs/engineering_manifest.md`
- `docs/ROADMAP.md`
- `docs/MANIFEST.md`
- `MEMORY.md`

When a manifest exists:

1. Read it before implementation.
1. Identify completed, incomplete, blocked, and ambiguous items.
1. Prefer the highest-priority incomplete item.
1. Choose one atomic implementation slice.
1. State the chosen slice and why it is next.
1. Identify tests required before implementation.
1. Implement only that slice.
1. Update the manifest only when the completed work changes task status or the user asked for manifest maintenance.
1. Do not silently skip manifest priorities.

If no manifest exists, create a concise implementation plan in the response before editing. Do not create a large manifest unless the user asks or the repository clearly needs one.

______________________________________________________________________

## Memory file workflow

If a memory file exists, read it before planning.

Common memory locations include:

- `docs/MEMORY.md`
- `MEMORY.md`
- `.codex/MEMORY.md`
- `.github/copilot-instructions.md`

Treat memory files as advisory context, not authoritative truth.

Use memory files to understand:

- project goals
- previous decisions
- known failure modes
- canonical interfaces
- validation expectations
- user preferences
- repository-specific hazards

Do not blindly trust memory files. Verify claims against the live repository before acting.

If memory conflicts with live code or tests, trust the live repository and report the discrepancy.

______________________________________________________________________

## Test-first development

For implementation tasks, use strict test-first development unless the task is purely documentation, formatting, or mechanical cleanup.

Default sequence:

1. inspect relevant source code and existing tests
1. identify the intended behavior
1. add or update tests first
1. run the targeted tests and confirm they fail for the expected reason
1. implement the minimal robust fix
1. rerun targeted tests
1. run broader tests
1. run the full repository gate

Do not write implementation first and tests afterward unless the user explicitly approves.

Do not weaken existing tests.

Do not remove assertions unless they are objectively wrong and replaced with stronger or more accurate assertions.

Do not mark tests as skipped, xfailed, or slow merely to pass the suite.

Do not change test expectations to match broken behavior.

When tests fail, diagnose the root cause before patching.

______________________________________________________________________

## Validation and cleaning

The repository must have a repeatable validation command.

If `test_repo.sh` exists, treat it as the authoritative local gate unless repository docs say otherwise.

Before using flags, inspect the script:

```bash
sed -n '1,240p' test_repo.sh
```

Common validation commands may include:

```bash
./test_repo.sh --fix
./test_repo.sh
./test_repo.sh --check-only
./test_repo.sh --clean
./test_repo.sh --ci
```

Do not assume all flags exist.

If `test_repo.sh` does not exist, create one before substantial implementation work unless the user explicitly says not to.

The initial `test_repo.sh` must be conservative, transparent, and aligned with the tools already present in the repository.

For Python/Pixi repositories, prefer a gate shaped like this:

```bash
#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-check}"

if [[ "${MODE}" == "--help" || "${MODE}" == "-h" ]]; then
  cat <<'EOF'
Usage:
  ./test_repo.sh [check|--check-only]
  ./test_repo.sh --fix
  ./test_repo.sh --clean
  ./test_repo.sh --ci

Modes:
  check / --check-only   Run validation without modifying files.
  --fix                  Run safe format/lint fixes, then validation.
  --clean                Remove generated caches/build artifacts, then validate.
  --ci                   Run the CI-equivalent validation path.
EOF
  exit 0
fi

run_if_available() {
  local description="$1"
  shift
  echo
  echo ">>> ${description}"
  "$@"
}

have_file() {
  [[ -f "$1" ]]
}

have_command() {
  command -v "$1" >/dev/null 2>&1
}

if [[ "${MODE}" == "--clean" ]]; then
  echo ">>> Cleaning generated caches and build artifacts"
  rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov build dist site docs/_build
  find . -type d -name __pycache__ -prune -exec rm -rf {} +
  MODE="check"
fi

if have_file "pixi.toml" && have_command pixi; then
  RUN=(pixi run)
else
  RUN=()
fi

if [[ "${MODE}" == "--fix" ]]; then
  if have_file "pyproject.toml" || have_file "ruff.toml"; then
    run_if_available "ruff format" "${RUN[@]}" ruff format .
    run_if_available "ruff check --fix" "${RUN[@]}" ruff check --fix .
  fi

  if have_command pre-commit || have_file ".pre-commit-config.yaml"; then
    run_if_available "pre-commit run --all-files" "${RUN[@]}" pre-commit run --all-files
  fi

  MODE="check"
fi

if [[ "${MODE}" == "--ci" ]]; then
  MODE="check"
fi

if [[ "${MODE}" != "check" && "${MODE}" != "--check-only" ]]; then
  echo "Unknown mode: ${MODE}" >&2
  exit 2
fi

if have_file "pyproject.toml" || have_file "ruff.toml"; then
  run_if_available "ruff format --check" "${RUN[@]}" ruff format --check .
  run_if_available "ruff check" "${RUN[@]}" ruff check .
fi

if have_file ".pre-commit-config.yaml"; then
  run_if_available "pre-commit run --all-files" "${RUN[@]}" pre-commit run --all-files
fi

if find . -maxdepth 3 -type f -name "test_*.py" | grep -q . || [[ -d tests ]]; then
  run_if_available "pytest" "${RUN[@]}" pytest -q
fi

if have_file "package.json"; then
  if command -v npm >/dev/null 2>&1; then
    if npm run | grep -q " lint"; then
      run_if_available "npm run lint" npm run lint
    fi
    if npm run | grep -q " test"; then
      run_if_available "npm test" npm test
    fi
  fi
fi

echo
echo ">>> Repository validation completed successfully"
```

After creating `test_repo.sh`, make it executable:

```bash
chmod +x test_repo.sh
```

Then run:

```bash
./test_repo.sh --fix
./test_repo.sh
```

If repository-specific tooling exists, adapt `test_repo.sh` to that tooling rather than imposing unrelated tools.

______________________________________________________________________

## Linting, formatting, and repository cleanup

Before implementation, inspect the existing style tools.

For Python repositories, common tools include:

- Ruff
- Black
- isort
- mypy
- pytest
- coverage
- pre-commit
- nbstripout
- mdformat
- Sphinx / MkDocs

For JavaScript or TypeScript repositories, common tools include:

- npm
- pnpm
- yarn
- eslint
- prettier
- vitest
- jest
- tsc

For Pixi repositories, prefer Pixi-managed commands over bare system commands.

Examples:

```bash
pixi run ruff format .
pixi run ruff check --fix .
pixi run pytest -q
pixi run pre-commit run --all-files
```

Do not run global formatters blindly if the repository has scoped formatting rules.

Do not format unrelated generated files, vendored files, lockfiles, external datasets, or large snapshots unless the task requires it.

After cleanup, inspect the diff:

```bash
git status --short
git diff --stat
git diff --check
```

______________________________________________________________________

## Dependency management

Do not add, remove, or upgrade dependencies unless required by the task.

Before changing dependencies, inspect the repository’s dependency manager.

Common dependency files:

- `pixi.toml`
- `pixi.lock`
- `pyproject.toml`
- `requirements.txt`
- `environment.yml`
- `package.json`
- `package-lock.json`
- `pnpm-lock.yaml`
- `yarn.lock`
- `Cargo.toml`
- `Cargo.lock`
- `go.mod`
- `go.sum`

Rules:

- Ask before major dependency changes.
- Prefer existing dependencies.
- Do not introduce heavyweight dependencies for small tasks.
- Keep lockfiles consistent with manifest files.
- Validate dependency changes with the repository’s normal install/test path.

______________________________________________________________________

## File editing rules

Before editing a file:

1. inspect the whole relevant file or enough context to understand its structure
1. inspect adjacent tests
1. inspect callers and public interfaces
1. understand generated vs source files

Do not patch from memory.

Do not patch based only on snippets when full-file context is needed.

Do not edit generated files unless the task explicitly targets generated outputs.

Do not hand-edit lockfiles unless that is the repository convention.

Do not mix unrelated cleanups with feature work.

______________________________________________________________________

## Public API and compatibility rules

Do not add backward-compatibility shims unless explicitly asked.

This includes:

- aliases for old names
- wrapper functions preserving deprecated signatures
- duplicate modules that forward imports
- compatibility config keys
- silent fallback behavior for old schemas
- migration bridges
- deprecation layers

When an interface changes, update the canonical tests, documentation, examples, and validation gates to use the new interface.

Prefer one clear canonical path over multiple tolerated paths.

______________________________________________________________________

## Testing expectations

Tests should verify behavior, not implementation details, unless implementation details are part of the contract.

Prefer comprehensive tests for:

- public APIs
- config schemas
- CLI behavior
- file format contracts
- manifest contracts
- notebook execution
- documentation examples
- error messages for user-facing failures
- edge cases and regression cases

When adding tests:

- include positive and negative cases
- test failure modes
- avoid hardcoded local user paths
- avoid relying on network access unless the test is explicitly marked as live/integration
- avoid brittle ordering unless order is part of the contract
- avoid weakening assertions to accommodate broken behavior

If a test is flaky, diagnose and fix the flakiness. Do not simply skip it.

______________________________________________________________________

## Notebook rules

If the repository contains notebooks, treat them as production artifacts when they are part of the workflow.

Rules:

- Do not leave execution counts or outputs unless the repository explicitly tracks them.
- Prefer deterministic execution.
- Do not hardcode local absolute paths.
- Use repository-relative paths or documented environment variables.
- Validate notebooks with the repository’s notebook gate if one exists.
- If notebooks are not part of the current task, do not rewrite or reformat them unnecessarily.

Common cleanup:

```bash
jupyter nbconvert --clear-output --inplace notebooks/*.ipynb
```

Only run notebook commands when the repository supports them and dependencies are available.

______________________________________________________________________

## Documentation rules

Update documentation when behavior changes.

Documentation updates may include:

- README
- examples
- CLI usage
- API reference
- engineering manifest
- architecture notes
- changelog
- notebook descriptions

Do not make documentation claim functionality that tests do not validate.

Do not update docs as a substitute for fixing code.

______________________________________________________________________

## Error handling rules

Prefer clear, explicit errors.

Do not silently swallow exceptions unless the behavior is intentional and tested.

Do not replace meaningful errors with broad `except Exception` blocks.

Do not add fallback behavior that hides invalid state.

When changing validation logic, include tests for invalid input and error messages.

______________________________________________________________________

## Security and secrets

Never print, commit, or expose secrets.

Sensitive files may include:

- `.env`
- `.env.*`
- credential JSON files
- API tokens
- private keys
- OAuth tokens
- cloud credentials
- service account files
- local secrets folders

Before committing or preparing a PR, check for accidental secrets:

```bash
git diff --cached
git diff
```

Do not add real credentials to tests. Use fixtures, mocks, or documented local secret paths.

______________________________________________________________________

## Commit rules

Ask before committing.

Before asking to commit, provide:

1. current branch
1. changed files
1. concise diff summary
1. validation commands run
1. validation results
1. remaining risks
1. proposed commit message

Use this format:

```text
Ready to commit?

Branch:
- <branch>

Changed files:
- <file>: <summary>

Validation:
- <command>: passed
- <command>: passed

Remaining risks:
- <risk or "none known">

Proposed commit message:
<type>: <summary>
```

Only commit after explicit approval.

Preferred commit message style:

```text
<type>: <short imperative summary>
```

Common types:

- `feat`
- `fix`
- `test`
- `docs`
- `refactor`
- `chore`
- `ci`

Examples:

```text
test: cover manifest validation failure modes
fix: preserve canonical dataset contract loading
ci: align local gate with GitHub workflow
docs: update engineering manifest status
```

______________________________________________________________________

## Push rules

Ask before pushing.

Before asking to push, show:

```bash
git status --short --branch
git log --oneline --decorate -5
```

Only push after explicit approval.

Use normal push unless the user explicitly approves force push:

```bash
git push -u origin <branch>
```

Do not force push unless explicitly approved.

______________________________________________________________________

## Pull request creation rules

Ask before creating a PR unless the user explicitly asked for one.

Before asking to create a PR, provide:

1. branch name
1. base branch
1. commit list
1. validation results
1. proposed PR title
1. proposed PR body

PR body should include:

```markdown
## Summary

- ...

## Validation

- [x] `<command>`

## Notes / risks

- ...
```

Use GitHub CLI if available:

```bash
gh pr create --base <base-branch> --head <branch> --title "<title>" --body-file <body-file>
```

Do not open a PR with known failing validation unless the user explicitly approves.

______________________________________________________________________

## Pull request merge rules

Ask before merging.

Before asking to merge, inspect:

```bash
gh pr status
gh pr checks
gh pr view
```

Report:

- PR title and number
- source branch
- target branch
- check status
- merge method options
- whether branch deletion is requested

Do not merge if required checks are failing unless the user explicitly approves.

Do not delete branches after merge unless explicitly approved.

______________________________________________________________________

## Standard task workflow

For a normal feature or fix, follow this sequence:

1. inspect repository state
1. read instructions, memory, and manifest
1. ensure work is on a branch
1. inspect relevant source and tests
1. propose one atomic plan
1. write failing tests first
1. run targeted tests and confirm expected failure
1. implement the root-cause fix
1. run targeted tests
1. run cleanup / formatting
1. run full repository validation
1. inspect diff
1. summarize results
1. ask before commit
1. ask before push
1. ask before PR
1. ask before merge

______________________________________________________________________

## Standard response after implementation

After implementation, summarize in this structure:

```text
Implemented:
- ...

Changed files:
- ...

Validation:
- `<command>`: passed
- `<command>`: passed

Diff summary:
- ...

Remaining risks:
- ...

Next recommended step:
- ...
```

If validation failed, use this structure:

```text
Validation failed.

Command:
- `<command>`

Failure:
- ...

Likely cause:
- ...

Files involved:
- ...

Recommended next step:
- ...
```

Do not bury failures in long prose.

______________________________________________________________________

## Handling dirty worktrees

If the worktree is dirty at session start, inspect it:

```bash
git status --short
git diff --stat
```

Do not modify dirty files until you understand whether changes are user changes or prior agent changes.

If the dirty changes are unrelated to the task, avoid touching those files.

If the dirty changes conflict with the task, stop and ask.

Never discard user changes without explicit approval.

______________________________________________________________________

## Handling generated artifacts

Generated artifacts must be clearly identified.

Do not commit generated outputs unless the repository convention requires them.

Common generated paths:

- `build/`
- `dist/`
- `site/`
- `docs/_build/`
- `.pytest_cache/`
- `.ruff_cache/`
- `.mypy_cache/`
- `.coverage`
- `htmlcov/`
- `__pycache__/`
- notebook outputs
- temporary reports
- local scratch files

If generated artifacts are required, document how to regenerate them.

______________________________________________________________________

## Handling large refactors

Large refactors must be split into atomic slices.

Each slice should have:

- one purpose
- tests
- validation
- minimal diff
- clear rollback path

Do not combine unrelated refactors, cleanup, dependency changes, documentation rewrites, and feature work in one branch unless the user explicitly requests it.

For large refactors, maintain or update an engineering manifest.

______________________________________________________________________

## Creating an engineering manifest

If the user asks to create or update an engineering manifest, use this structure:

```markdown
# Engineering Manifest

## Current repository state

- ...

## Non-negotiable constraints

- ...

## Validation gate

- ...

## Priority order

### P0 — Safety / correctness / broken gate

- [ ] ...

### P1 — Required feature completion

- [ ] ...

### P2 — Hardening

- [ ] ...

### P3 — Documentation and examples

- [ ] ...

### P4 — Optional improvements

- [ ] ...

## Completed work

- ...

## Known risks

- ...

## Deferred work

- ...
```

The manifest should be specific enough that an agent can select the next atomic task without guessing.

______________________________________________________________________

## Creating or updating `test_repo.sh`

If `test_repo.sh` is missing and the repository has no equivalent validation gate, create it.

Requirements:

- portable Bash
- `set -euo pipefail`
- clear modes
- no destructive cleanup by default
- check-only default
- explicit `--fix` mode for safe formatting/linting
- explicit `--clean` mode for generated caches only
- Pixi-aware when `pixi.toml` exists
- uses repository-native tools
- does not require unavailable tools without checking
- exits nonzero on failure

After creating or editing `test_repo.sh`, validate it:

```bash
chmod +x test_repo.sh
./test_repo.sh --help
./test_repo.sh --fix
./test_repo.sh
```

If the repository has CI, align `test_repo.sh` with CI so local and CI behavior match.

______________________________________________________________________

## CI alignment

If the repository has GitHub Actions or another CI system, inspect it before changing validation:

```bash
find .github/workflows -type f -maxdepth 2 -print
```

Local validation and CI should agree.

Do not create a local gate that passes while CI fails due to different commands.

Do not create CI commands that are impossible to run locally.

Prefer having CI call the local gate:

```bash
bash test_repo.sh --ci
```

______________________________________________________________________

## Package and build validation

If the repository is a Python package, inspect build configuration and validate packaging when relevant.

Potential commands:

```bash
python -m build --sdist --wheel
pip install -e .
pytest -q
```

For Pixi repositories, prefer:

```bash
pixi run python -m build --sdist --wheel
pixi run pytest -q
```

Do not add package build steps unless the repository is intended to be packaged.

______________________________________________________________________

## Final safety checklist

Before asking to commit or push, verify:

```bash
git status --short --branch
git diff --stat
git diff --check
```

Then verify:

- branch is not a protected/base branch
- no unrelated files were modified
- no generated caches are staged
- no secrets are present
- tests were not weakened
- compatibility shims were not added unless requested
- manifest status is accurate if updated
- validation commands and results are recorded

______________________________________________________________________

## Explicit user approval required

The following actions require explicit user approval:

- committing
- pushing
- opening a pull request, unless already requested
- merging a pull request
- deleting branches
- force pushing
- rebasing shared branches
- resetting the worktree
- discarding changes
- deleting files not directly required by the task
- changing dependency sets
- changing public interfaces
- adding compatibility shims
- weakening tests
- skipping failing validation

______________________________________________________________________

## Default behavior when uncertain

When uncertain:

1. inspect more context
1. prefer smaller changes
1. preserve existing behavior
1. preserve user work
1. ask before risky actions
1. report uncertainty explicitly

Do not guess and patch blindly.

______________________________________________________________________

## First prompt pattern for Codex

When starting a new Codex session, the user may say:

```text
Follow AGENTS.md.

Do not edit files yet.

First:
1. inspect git status and current branch
2. inspect repository structure
3. read memory and engineering manifest files if present
4. identify the highest-priority incomplete task
5. propose one atomic TDD implementation slice
6. list the files you expect to inspect or modify
7. wait for approval before editing
```

The agent should obey that pattern unless the user gives a more specific instruction.

______________________________________________________________________

## Implementation prompt pattern for Codex

After the user approves a slice, the user may say:

```text
Proceed with the approved slice.

Rules:
- write or update tests first
- run targeted tests and confirm expected failure
- implement the root-cause fix
- run targeted tests
- run cleanup / lint / format
- run the full repository gate
- do not commit
- summarize changed files, validation results, and remaining risks
```

The agent should obey that pattern unless the user gives a more specific instruction.

______________________________________________________________________

## Commit prompt pattern for Codex

When the user is ready to commit, the user may say:

```text
Prepare this for commit.

Show:
1. current branch
2. git status
3. concise diff summary
4. validation commands and results
5. proposed commit message
6. exact git add and git commit commands

Do not commit until I approve.
```

The agent must not commit until approval is explicit.

______________________________________________________________________

## Push and PR prompt pattern for Codex

When the user is ready to push and open a pull request, the user may say:

```text
Prepare to push and open a PR.

Show:
1. current branch
2. commits to push
3. proposed remote branch
4. proposed PR title
5. proposed PR body
6. validation status

Do not push or open the PR until I approve.
```

The agent must not push or open the PR until approval is explicit.

______________________________________________________________________

## Merge prompt pattern for Codex

When the user is ready to merge, the user may say:

```text
Check whether the PR is ready to merge.

Show:
1. PR number and title
2. source branch
3. base branch
4. check status
5. review status if available
6. merge method recommendation
7. branch cleanup recommendation

Do not merge until I approve.
```

The agent must not merge until approval is explicit.

______________________________________________________________________

## Summary

The agent’s job is not merely to produce code. The agent’s job is to preserve repository integrity while making small, validated, reviewable improvements.

The correct default loop is:

```text
inspect → branch → plan → test first → implement → validate → summarize → ask before commit → ask before push → ask before PR → ask before merge
```
