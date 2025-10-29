## Why
Agents need one stable entry point to discover and navigate all documentation artifacts without crawling. Today we produce multiple outputs (DocFacts, NavMap, Sphinx JSON, graphs, schemas, observability), but there is no single, versioned catalog tying them together with consistent link policies. A consolidated Agent Catalog (JSON) and a human-friendly Agent Portal (HTML) will make our documentation system maximally useful to AI programming agents and humans alike, improving comprehension, discovery, and change impact analysis.

## What Changes
- Agent Catalog (JSON)
  - New generator `tools/docs/build_agent_catalog.py` emits `docs/_build/agent_catalog.json`.
  - Unifies pointers to: DocFacts, NavMap, Symbols index, by_file/by_module indexes, Sphinx HTML/JSON pages, JSON Schemas, graphs, observability, and search index.
  - Encodes link policy (editor vs GitHub) and provides per-symbol deep links for both rendered docs and source lines.
  - Includes provenance: repo SHA, generation timestamp, tool versions.

- Agent Portal (HTML)
  - New renderer `tools/docs/render_agent_portal.py` produces `site/_build/agent/index.html` from the catalog.
  - Clear hierarchy and quick links; single search box over combined indices; package/module cards; symbol lists with actions: open in editor/GitHub.
  - Offline-friendly static HTML (no framework); progressive enhancement optional.

- Orchestrator integration
  - Update `tools/docs/build_artifacts.py` to run the catalog builder and portal renderer after existing steps (docstrings → navmap → testmap → observability → schemas → agent-catalog → agent-portal).
  - Export a JSON Schema for the catalog at `docs/_build/schema_agent_catalog.json` and validate during generation.

- Documentation
  - Add `docs/agent_portal_readme.md` (generated or maintained) that explains how to consume the catalog and portal (for humans and agents), link modes, and example code snippets.

### Best-in-class enhancements (included in this change)
- Catalog enrichments
  - Symbol graph: import and call-graph edges per module/symbol; dependency visualization pointers.
  - Quality signals: mypy status, ruff rule hits, pydoclint parity, interrogate coverage, doctest outcomes.
  - Metrics & provenance: complexity (mccabe), churn, LOC, last_modified, codeowners, stability/deprecation flags.
  - Stable identifiers and anchors: `symbol_id` (content hash of normalized AST) and CST/AST spans with fuzzy remap across line drift.

- Typed clients and query surface
  - Python client (Pydantic models + helpers) and TypeScript client for Node/Deno; both consume the catalog with strong types.
  - `agentctl` CLI for jq-like queries (e.g., list-modules, find-callers, show-quality for a symbol).
  - Optional GraphQL/JSONata local query interface over the static JSON for advanced agents.

- Search and ranking
  - Semantic index: embeddings for symbols/docstrings with a FAISS store and metadata; hybrid search (lexical + vector rerank).
  - Facets: filter by package, stability, churn, coverage, parity status; breadcrumb navigation in the portal.

- Portal UX and content
  - Task playbooks (add API, refactor module, fix parity) and quick actions (safe checks) linked to relevant symbols/tests.
  - Usage examples and performance metrics per module; dependency graphs embedded; responsive layout and ARIA roles.
  - Tutorials and feedback mechanism (local-only by default; server mode optional).

- Security, governance, performance
  - Drift gates and CI annotations; analytics JSON summarizing usage.
  - Optional hosted mode with RBAC and audit logs; privacy guardrails for user data in feedback.
  - Content-addressed shards for large catalogs; caching strategies for fast portal loads.

## Technical architecture details (summary)
- Symbol graph extraction
  - Imports: build per-module import graph from Sphinx indices and Python AST; normalize aliases; exclude stdlib by default.
  - Calls: static call-graph via AST (direct calls only), limited by Python’s dynamic nature; annotate edges with confidence (static|heuristic).
  - Storage: adjacency lists under `graph.imports` and `graph.calls` keyed by module/symbol.

- Stable IDs and anchors
  - `symbol_id = sha256(qname + normalized_AST(code_without_ws/comments))`.
  - Anchors include `start_line`, `end_line`. Fuzzy remap on link resolve: (1) `symbol_id` match using AST hash; (2) name + arity match; (3) nearest-text search fallback.

- Quality signals
  - mypy: `ok|error|unknown`; ruff: list of rule codes hit; doc parity: boolean from pydoclint; coverage: docstring coverage from interrogate; doctest: `ok|fail|skip|unknown`.
  - Complexity: mccabe; churn: last-N commit count; last_modified: ISO timestamp; stability: `experimental|stable|internal`; deprecated: boolean.

- Semantic index
  - Model: MiniLM or equivalent (configurable); dimension ~384–768; tokenized on symbol doc+context.
  - Index: FAISS IVF/Flat (configurable); files: `agent_index.faiss` + `agent_index.json` mapping `symbol_id`→row.
  - Query: lexical candidate set → vector encode → rerank; weights configurable.

- Link policy resolution
  - Precedence: CLI/env overrides → catalog `link_policy` → defaults.
  - Templates: editor `vscode://file/{path}:{line}`, GitHub `https://github.com/{org}/{repo}/blob/{sha}/{path}#L{line}`.

- Sharding and performance
  - Threshold: shard when catalog > 20MB or modules > 2000; per-package shard naming `agent_catalog.pkg.<name>.json` with root index.
  - Budgets: portal cold-load < 1.5s; first query < 300ms; generation reuses content-addressed cache where inputs unchanged.

- Clients and CLI
  - Python/TS typed clients mirroring schema; `agentctl` supports: `list-modules`, `find-callers`, `show-quality`, `search` (hybrid).

- Analytics and governance
  - `docs/_build/analytics.json`: anonymized portal usage counters; CI PR annotations summarizing coverage/parity/high-churn deltas.


## JSON: Agent Catalog format (overview)
- Stable top-level fields: `version`, `generated_at`, `repo` (sha, root), `link_policy` (mode, templates), `artifacts` (global artifact paths).
- Per-package and per-module entries with deep links to rendered Sphinx page (`site/_build/html/**`), raw Sphinx JSON (`site/_build/json/**`), editor/GitHub anchors, and list of symbol qualified names present in DocFacts.
- Deterministic, relative paths; absolute URLs when `DOCS_LINK_MODE=github` and repo metadata provided.

Example (abbreviated):

```json
{
  "version": "1.0",
  "generated_at": "2025-10-29T12:34:56Z",
  "repo": { "sha": "abcdef1", "root": "." },
  "link_policy": {
    "mode": "editor",
    "editor_template": "vscode://file/{path}:{line}",
    "github_template": "https://github.com/{org}/{repo}/blob/{sha}/{path}#L{line}"
  },
  "artifacts": {
    "docfacts": "docs/_build/docfacts.json",
    "navmap": "site/_build/navmap/navmap.json",
    "symbols": "site/_build/symbols.json",
    "by_file": "site/_build/by_file.json",
    "by_module": "site/_build/by_module.json",
    "schemas": {
      "docstrings": "docs/_build/schema_docstrings.json",
      "agent_catalog": "docs/_build/schema_agent_catalog.json"
    },
    "graphs": ["docs/_build/graphs/kgfoundry-imports.svg", "docs/_build/graphs/kgfoundry-uml.svg"],
    "observability": "docs/_build/observability_docstrings.json",
    "search": "site/_build/search/index.json"
  },
  "packages": [
    {
      "name": "kgfoundry",
      "modules": [
        {
          "module": "kgfoundry.orchestration.flows",
          "paths": {
            "source": "src/kgfoundry/orchestration/flows.py",
            "page": "site/_build/html/api/orchestration/flows.html",
            "fjson": "site/_build/json/api/orchestration/flows.fjson"
          },
          "anchors": {
            "open_in_editor": "vscode://file/src/kgfoundry/orchestration/flows.py:75",
            "open_in_github": "https://github.com/org/repo/blob/abcdef1/src/kgfoundry/orchestration/flows.py#L75"
          },
          "docfacts": ["kgfoundry.orchestration.flows._e2e_flow_impl", "..."]
        }
      ]
    }
  ]
}
```

Validation schema will be emitted to `docs/_build/schema_agent_catalog.json` and verified during generation.

## HTML: Agent Portal format (overview)
- Static HTML page at `site/_build/agent/index.html` consuming the catalog:
  - Header with title and global search (merging `by_file`, `by_module`, `symbols`).
  - Quick links to top artifacts (DocFacts, NavMap, Schemas, Graphs, Observability).
  - Package and module sections with cards; each module shows: summary, open-in-editor/GitHub, links to page and JSON, and contained symbols.
  - “What changed since” control when git data is available.
  - Minimal CSS for readability; no build-time JS bundling required.

## Impact
- Improves agent onboarding: single JSON fetch and one portal page cover all documentation.
- Encourages consistent link handling and provenance across artifacts.
- Adds two new scripts and one orchestrator step; does not change existing builder outputs.

## Non-Goals
- No change to how Sphinx/MkDocs build content; we only aggregate and link.
- No search indexing changes beyond wiring existing indices together on the portal.


