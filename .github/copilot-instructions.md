# Repository Copilot Instructions

Read `MEMORY.md`, `docs/ENGINEERING_MANIFEST.md`, `docs/AGENT_SYNC.md`, `config/agent_policy.yaml`, and `pixi.toml` before substantial work.

Use compact Caveman-style responses.

Copilot manages repo state and next-step selection from the manifest/sync file. Do not ask what next when the repo state is sufficient.

Use Pixi for all project tooling when `pixi.toml` exists. Do not use bare Python/tooling or global Pixi environments.

Follow focused TDD. Do not commit failing tests to prove red state. Do not weaken tests. Do not add compatibility shims unless explicitly requested.

New functionality and scope expansion require a formal scope-increase request with justification.

PR reviews, bug checks, code-quality audits, and security audits must update `docs/review_register.md` and `docs/AGENT_SYNC.md`. Required follow-up findings must also be reflected in `docs/ENGINEERING_MANIFEST.md`; non-blocking out-of-scope findings belong in `docs/scope_backlog.md`.

External research: this repo-local Copilot environment should not perform web research. If a task depends on current literature, method identification, package/API documentation, standards, or version-sensitive guidance, route it through external research first and require completed artifacts under `ai_context/` before implementation. Agents must understand and use completed research/method/API packets, not invent them.

Caveman: prefer local `*.caveman.md` context files when present, then fall back to verbose originals. Verbose originals remain authoritative on ambiguity.
