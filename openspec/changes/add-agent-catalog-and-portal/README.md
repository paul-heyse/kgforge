# Agent Catalog & Agent Portal

Purpose: provide one machine entry-point (Agent Catalog JSON) and one human entry-point (Agent Portal HTML) that enumerate, link, and contextualize all documentation artifacts for agents and developers.

## What you get
- `docs/_build/agent_catalog.json` — single JSON describing every relevant artifact, with deep links and link policy (editor/GitHub).
- `site/_build/agent/index.html` — single HTML page rendering a clear hierarchy and unified quick links/search.
- `docs/_build/schema_agent_catalog.json` — JSON Schema for validation and consumers.

## Contents of the catalog (high-level)
- `version`, `generated_at`, `repo.sha`
- `link_policy` — `mode` and link templates for editor/GitHub
- `artifacts` — paths to DocFacts, NavMap, Symbols indexes, Sphinx HTML/JSON, Schemas, Graphs, Observability, Search
- `packages[*].modules[*]` — per-module source path, rendered page, fjson, anchors, and DocFacts symbol list

Example (abbreviated):

```json
{
  "version": "1.0",
  "generated_at": "2025-10-29T12:34:56Z",
  "repo": { "sha": "abcdef1", "root": "." },
  "link_policy": { "mode": "editor", "editor_template": "vscode://file/{path}:{line}" },
  "artifacts": { "docfacts": "docs/_build/docfacts.json", "navmap": "site/_build/navmap/navmap.json" },
  "packages": [ { "name": "kgfoundry", "modules": [ { "module": "kgfoundry.orchestration.flows", "paths": {"source": "src/kgfoundry/orchestration/flows.py", "page": "site/_build/html/api/orchestration/flows.html" }, "docfacts": ["kgfoundry.orchestration.flows._e2e_flow_impl"] } ] } ]
}
```

## How agents should consume
1. Read `docs/_build/agent_catalog.json`.
2. Use `artifacts` to locate global data (DocFacts, indices, schemas, observability).
3. For a given module, prefer `paths.page` (human) or `paths.fjson` (machine) and follow anchors to source with `link_policy`.
4. Use `docfacts` qnames to cross-reference symbol metadata.

## How humans should consume
- Open `site/_build/agent/index.html` for an overview, quick links, and search.
- Navigate to package/module cards and jump to rendered pages or source.

## Link modes
- `DOCS_LINK_MODE=editor` (default) → `vscode://file/{path}:{line}`
- `DOCS_LINK_MODE=github` → requires `DOCS_GITHUB_ORG`, `DOCS_GITHUB_REPO`, `DOCS_GITHUB_SHA`

## Make targets (planned)
- `make artifacts` — runs agent catalog and portal generation at the end.

## Validation
- The generator validates the JSON against `schema_agent_catalog.json` each run.
- CI should fail when the catalog is missing required fields or contains broken links.


