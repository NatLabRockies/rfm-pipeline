# Caveman Context Compression

## Purpose

Caveman support is optional. The toolkit uses Caveman-style concise responses by default through Copilot instructions. File compression is an optional extra step for repositories with long, repeatedly read context files.

## Install model

Install Caveman once at user/global scope. Do not install Caveman in a way that writes repo-local rule files into managed repositories.

Recommended command:

```bash
npx skills add JuliusBrussee/caveman -a github-copilot -g -y
```

This targets GitHub Copilot and installs the skill globally for the current user. Avoid Caveman's broader `--all` or `--with-init` setup inside managed repositories unless you intentionally want Caveman to write repo-local instruction files such as `.github/copilot-instructions.md` or `AGENTS.md`. This toolkit manages those files.

## Refresh model

After a repo is prepared with this toolkit, refresh compressed context files with:

```bash
cop-caveman-refresh /path/to/repo
```

Equivalent direct command from the toolkit checkout:

```bash
./scripts/caw-refresh-caveman /path/to/repo
```

The refresh helper searches for the installed Caveman `caveman-compress` skill files and runs the documented local compression entry point from that skill:

```bash
cd /path/to/caveman-compress
python3 -m scripts /absolute/path/to/file
```

The helper compresses temporary copies, then writes sibling agent-facing `*.caveman.md` files. Canonical verbose files are not overwritten.

If the skill files are installed somewhere non-standard, set:

```bash
export CAVEMAN_COMPRESS_DIR=/path/to/caveman-compress
```

Targets:

- `AGENTS.md`
- `.github/copilot-instructions.md`
- `MEMORY.md`
- `docs/ENGINEERING_MANIFEST.md`
- `docs/AGENT_SYNC.md`
- `docs/review_register.md`
- `docs/scope_backlog.md`
- `docs/CAVEMAN_CONTEXT.md`

Outputs are sibling files such as:

- `AGENTS.caveman.md`
- `MEMORY.caveman.md`
- `docs/AGENT_SYNC.caveman.md`

## Authority

Verbose originals are canonical. Compressed files are agent-facing convenience copies.

Agents should read `*.caveman.md` first when present, then fall back to verbose originals when:

- compressed files are missing
- compression appears stale
- compressed wording is ambiguous
- exact wording matters
- public API, schema, validation, security, or publication behavior is at stake

## Files not to compress

Do not compress:

- source code
- tests
- diffs
- tracebacks
- raw API documentation bundles
- raw literature bundles
- notebooks
- manuscripts
- legal/compliance text
- artifacts where exact wording is the artifact
