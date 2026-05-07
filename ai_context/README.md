# AI Context

This directory is the bridge between external web-enabled research and local repo-native Copilot implementation.

Copilot agents in this repo should not invent current literature, current package APIs, method recommendations, standards, or version-sensitive guidance. If a task depends on those topics, the work must first be routed to an external web-enabled research environment. The result of that external work must be converted into durable local artifacts under this directory before local implementation begins.

## Standard structure

```text
ai_context/
  README.md
  _templates/
  literature/
  methods/
  api_docs/
  manifests/
  prompts/
```

## Artifact types

- `literature/`: literature bundles and source syntheses.
- `methods/`: method manifests that translate research into chosen methods, assumptions, limitations, and validation plans.
- `api_docs/`: current package/API documentation bundles with version assumptions and usage examples.
- `manifests/`: feature-specific engineering manifests derived from external research.
- `prompts/`: execution prompts or external-research request prompts.

## Local-agent policy

Repo-local Copilot agents may read and use completed artifacts in this directory. They may scaffold missing artifact templates or prepare a request for an external web-enabled assistant. They may not perform the web research themselves unless the active Copilot environment explicitly has web access and the repo policy allows it.

The live repo, tests, root `MEMORY.md`, root engineering manifest, and explicit user instructions remain authoritative for repo state. `ai_context/` artifacts are authoritative for the external research claims they document, subject to human approval and the scope of the artifact.
