**Audit Summary**

Current Git state:

- Location: `/Users/dhetting/src/bsm-public-rf`
- Branch: `main`
- Protected/base branch: yes, `main`
- Remote: `origin git@github.com:NatLabRockies/bsm-public-rf.git`
- Worktree: dirty, with untracked `AGENTS.md`
- No tracked diff: `git diff --stat` was empty
- I did not edit, delete, stash, reset, rebase, commit, push, or run mutating validation.

Because the repo is on `main` and has uncommitted user work, AGENTS.md requires approval before creating a task branch. If we proceed, I recommend creating `codex/update-session-manifest-plan` with the existing untracked `AGENTS.md` left in place.

**Discovered Planning Files**

Memory:

- `docs/MEMORY.md`

No file named:

- `ENGINEERING_MANIFEST.md`
- `docs/ENGINEERING_MANIFEST.md`
- `docs/MANIFEST.md`
- `docs/ROADMAP.md`

Closest live planning/status files:

- `docs/module_plan.md`
- `docs/manuscript_alignment_audit.md`
- `docs/workflow_audit.md`
- `docs/debiased_lasso_contract.md`
- `docs/manuscript_contract.md`

This is a drift point: the user prompt and memory expect an engineering manifest, but the live repo does not contain one under the expected names.

**Repository State**

Main source:

- `src/bsm_rfm/`

Tests:

- `tests/`, with broad coverage for public API, workflow, manuscript stages, docs, CI contract, gate contract, examples, notebooks, and artifact contracts.

Docs:

- `docs/`
- `notebooks/manuscript/README.md`

Notebooks:

- `notebooks/manuscript/00_...` through `08_...`

Tooling:

- `pixi.toml`
- `pyproject.toml`
- `.pre-commit-config.yaml`
- `test_repo.sh`
- `.github/workflows/ci.yml`

Generated/ignored outputs visible:

- `artifacts/`
- `bsm-public-rf-v0.1.zip`
- `bsm-public-rf-v0.2.zip`
  These are ignored by `.gitignore`.

No refined merge-conflict marker search hits were found. No `pytest.mark.skip` or `pytest.mark.xfail` usage was found in `tests/`.

**Validation Gate**

Authoritative local gate:

- `./test_repo.sh --check`

Mutating/fix gate:

- `./test_repo.sh --fix`

CI gate:

- `.github/workflows/ci.yml` runs `./test_repo.sh --ci`

Pixi tasks include:

- `pixi run lint`
- `pixi run test`
- `pixi run notebook-tests`
- `pixi run docs`
- `pixi run package-build`
- `pixi run manuscript-reproduction-smoke`

Local and CI are mostly aligned because CI calls `./test_repo.sh --ci`. One minor duplication: CI also runs `pixi run docs` after the gate, while the gate already includes `docs`.

I did not run the gate because `docs/MEMORY.md` recommends full validation, but your current instruction says no edits, and `test_repo.sh --fix` mutates by design while `--check` still includes transient cleanup.

**Manifest / Live Repo Drift**

Complete or apparently current:

- Phase 0/1/2/3 manuscript scaffold exists in source, docs, notebooks, configs, and tests.
- Reproduction audit and gate-level manuscript smoke check exist.
- Public docs explicitly avoid claiming exact manuscript reproduction.
- Tests enforce public API, docs, CI, gate, and manuscript-stage contracts.

Incomplete or intentionally non-exact:

- Interaction discovery remains a residualized-product public surrogate, not tree-SHAP exact.
- Nonlinear discovery remains a residualized parametric-transform surrogate, not GAM EDF/p-value exact.
- Sparse selection remains EBIC/L1 surrogate, not validated as de-biased LASSO exact.
- Final HC3 filter exists but still needs real-data verification against private manuscript outputs.
- Final tables/figures are deterministic public artifacts, not verified manuscript-exact reproductions.

Ambiguous/stale:

- Expected engineering manifest is absent.
- `docs/MEMORY.md` contains older continuation notes that say “next Phase 3 slice” even though later notes and live repo indicate those slices are implemented.
- `docs/module_plan.md` still says remaining reconciliation includes feature-expansion defaults and provenance boundaries, but it is not structured as the engineering manifest AGENTS.md expects.

**Priorities**

P0:

- Resolve branch/worktree hygiene before edits: current branch is `main`, and `AGENTS.md` is untracked.
- Establish an authoritative engineering manifest or explicitly designate/update the existing planning document.
- Run a non-mutating baseline gate once branch/worktree handling is approved.

P1:

- Update the planning/status ledger to reflect live repo state: completed reproduction audit, smoke gate, implemented Phase 3 chain, and remaining exactness gaps.
- Add tests that lock the manifest/status document into docs navigation and prevent stale “next Phase 3 slice” wording.

P2:

- Reduce CI/local duplication if desired, but current alignment is acceptable.
- Harden docs around exactness boundaries and real-data verification requirements.

P3:

- Future scientific work: tree-SHAP interaction implementation or external artifact contract; GAM nonlinear implementation or external artifact contract; de-biased-LASSO port/validation; real-data HC3/table/figure verification.

**Recommended First Atomic Slice**

First slice: create/update an authoritative engineering manifest/status plan from the live repo.

Why first:

- The repo instructions say the engineering manifest is the primary planning document when present.
- The user expected one, but the live repo lacks the expected manifest file.
- This will prevent future agents from following stale memory notes over live status.

Tests to add/update first:

- Add/update a docs/planning contract test, likely in `tests/test_manuscript_documentation_contract.py` or a new `tests/test_engineering_manifest.py`, to assert:
  - the authoritative manifest exists, probably `docs/ENGINEERING_MANIFEST.md`;
  - `docs/index.md` exposes it;
  - it records current exactness gaps and does not say the already-implemented Phase 3 stages are still “next” work.

Likely files to inspect/modify:

- `docs/ENGINEERING_MANIFEST.md` new or `docs/module_plan.md` if you prefer not to add a new file
- `docs/index.md`
- `tests/test_manuscript_documentation_contract.py` or new `tests/test_engineering_manifest.py`
- Possibly `docs/MEMORY.md` only if you want memory refreshed too

Validation after implementation:

- targeted: `pixi run pytest tests/test_manuscript_documentation_contract.py -q` or new test file
- broader: `pixi run lint`
- docs: `pixi run docs`
- full gate: `./test_repo.sh --check`

**Open Questions**

1. Should I create `docs/ENGINEERING_MANIFEST.md`, or treat `docs/module_plan.md` as the manifest and update it?
1. Since we are on `main` with untracked `AGENTS.md`, do you approve creating a task branch with the current untracked file left untouched?

Approve proceeding with the first atomic slice: branch setup, tests first, then manifest/status update.
