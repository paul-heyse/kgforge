## Context
We generate numerous artifacts for documentation consumers: `docs/_build/docfacts.json`, `site/_build/navmap/navmap.json`, Sphinx HTML/JSON pages under `site/_build/{html,json}/`, symbol indices (`site/_build/by_file.json`, `site/_build/by_module.json`, `site/_build/symbols.json`), graphs (`docs/_build/graphs/*.svg`), schemas, and observability JSON. Agents lack a single, stable catalog that ties these together with link policies and provenance.

## Goals / Non-Goals
- Goals
  - G1: Provide a single JSON (Agent Catalog) describing all documentation artifacts with deep links and provenance.
  - G2: Provide a single HTML (Agent Portal) that renders a clear hierarchy, offers a unified search box over combined indices, and provides open-in-editor/GitHub actions.
  - G3: Keep the system deterministic, offline-friendly, and easy to integrate (no server required).
  - G4: Validate the catalog against a JSON Schema each generation.
- Non-Goals
  - N1: Do not change how existing artifacts are produced.
  - N2: Do not introduce new search backends; reuse existing indices.

## Decisions
1) Catalog schema and versioning
   - Use a stable top-level structure with `version`, `generated_at`, `repo.sha`, `link_policy`, `artifacts`, and `packages[*].modules[*]` entries.
   - Emit a JSON Schema (`docs/_build/schema_agent_catalog.json`) and validate during generation.
   - Bump a semantic `version` when breaking changes occur; keep additive changes backwards-compatible.

2) Data sources and mapping
   - Load:
     - DocFacts from `docs/_build/docfacts.json` (for symbol qnames and metadata)
     - Sphinx indexes from `site/_build/by_file.json`, `by_module.json`, `symbols.json`
     - NavMap from `site/_build/navmap/navmap.json`
     - Graphs, Schemas, Observability via known paths
   - Map module paths to Sphinx HTML/JSON using established folder layouts (`site/_build/html`, `site/_build/json`).

3) Link policy
   - Respect `DOCS_LINK_MODE=editor|github` and environment variables `DOCS_GITHUB_ORG`, `DOCS_GITHUB_REPO`, `DOCS_GITHUB_SHA`.
   - Provide templates in the catalog for both editor and GitHub URLs and compute per-item anchors when line numbers are available.

4) Portal rendering
   - Render a static, self-contained HTML page from the catalog. Minimal JS to enable search and quick filtering.
   - No dependency on node or bundlers; pure Python template or string builder.

5) Enriched catalog data model (best-in-class)
   - Symbol graph: encode import edges (module → module) and call-graph edges (symbol → symbol). Provide adjacency lists per package/module.
   - Quality signals: per-symbol fields capturing `mypy_status`, `ruff_rules` (list), `doc_parity_ok` (bool), `coverage_pct`, `doctest_status`.
   - Metrics & provenance: `complexity`, `loc`, `last_modified`, `codeowners`, `stability` (e.g., experimental/stable), `deprecated` (bool).
   - Stable IDs & anchors: `symbol_id` = sha256 of (qname + normalized AST). `anchors` include `start_line`, `end_line`, and optional character offsets; fuzzy remap uses CST/AST matching when lines drift.
   - Shards: allow splitting the catalog into per-package JSON shards with a tiny root index; the portal and clients load shards on demand.
   - Semantic index: store embeddings metadata in `artifacts.agent_index` (FAISS paths + sidecar JSON) with mapping to `symbol_id`.

6) Typed clients and query surface
   - Python client with Pydantic models matching the catalog schema; helpers for common queries (e.g., `find_callers(symbol_id)`, `list_modules(pkg)`).
   - TypeScript client for Node/Deno with generated types (e.g., via quicktype or hand-written models).
   - `agentctl` CLI: jq-like queries over the catalog; supports output (JSON/table) and convenience commands; runs entirely locally.
   - Optional local query layer: expose GraphQL/JSONata against the loaded catalog for richer agent interactions without a server.

7) Search and ranking
   - Hybrid search: lexical (combined indices) + vector rerank using the semantic index; configurable weights.
   - Faceted filters: package, stability, churn, coverage, parity; expose facets in the portal and client API.
   - Breadcrumbs: derive from module path segments and package structure to orient navigation.

8) Portal UX
   - Task playbooks: curated links and checklists for common tasks (add API, refactor module, fix parity) with deep links.
   - Usage examples & metrics: embed example snippets and performance snapshots per module.
   - Dependency visualization: inline SVG graph or link to pre-rendered graphs with focused highlights per module.
   - Tutorials and feedback: offline tutorials packaged with the portal; optional feedback JSON drop (local file) with redaction.
   - Accessibility: ARIA landmarks/labels; keyboard navigation; responsive layout; no-JS fallback of core hierarchy.

9) Security & governance
   - CI drift gates: schema validation and link checks fail the pipeline; annotate PRs with summary changes and risk signals.
   - Optional hosted mode: RBAC (viewer/contributor/admin), audit logs for actions (e.g., feedback submissions), privacy guardrails.
   - SBOM and license page references from the portal; do not embed secrets.

10) Performance & caching
   - Content-addressed caching for catalog shards and portal HTML sections; only rewrite changed pieces.
   - Budget targets: cold-load portal < 1.5s locally; catalog resolution (first query) < 300ms.
   - Load testing: ensure portal handles large catalogs (e.g., 10k symbols) with responsive search.

## Implementation details
Files to add:
- `tools/docs/build_agent_catalog.py`
  - Inputs: paths above; optional env for link mode/org/repo/sha
  - Steps:
    1. Read artifacts and collect provenance: repo SHA (`git rev-parse`), generated_at (UTC), tool versions as available.
    2. Build global `artifacts` map (DocFacts, NavMap, Symbols/by_file/by_module, Schemas, Graphs, Observability, Search).
    3. Construct package/module list:
       - From `by_module.json` and `by_file.json`, derive module → source path and HTML/JSON page paths.
       - For each module, aggregate DocFacts symbols present.
       - Compute anchors: open-in-editor/GitHub using link templates and best-effort line numbers (from DocFacts when available).
    4. Assemble catalog dict and validate against in-memory JSON Schema.
    5. Write `docs/_build/agent_catalog.json` and `docs/_build/schema_agent_catalog.json`.

- `tools/docs/render_agent_portal.py`
  - Inputs: `docs/_build/agent_catalog.json`
  - Render to `site/_build/agent/index.html`:
    - Header, quick-links section, search input
    - Packages and modules as cards with deep links
    - Optional section summarizing observability (counts/timings)
     - Faceted filters and breadcrumbs
     - Inline dependency graphs per module (optional) or links to focused views

- `tools/docs/build_artifacts.py`
  - Append steps to invoke the two scripts and print status lines on success/failure.

## Agent Catalog JSON Schema (v1.0, abbreviated)
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://kgfoundry/docs/_build/schema_agent_catalog.json",
  "title": "Agent Catalog",
  "type": "object",
  "required": ["version", "generated_at", "repo", "link_policy", "artifacts", "packages"],
  "properties": {
    "version": {"type": "string"},
    "generated_at": {"type": "string", "format": "date-time"},
    "repo": {
      "type": "object",
      "required": ["sha"],
      "properties": {"sha": {"type": "string"}, "root": {"type": "string"}}
    },
    "link_policy": {
      "type": "object",
      "required": ["mode"],
      "properties": {
        "mode": {"type": "string", "enum": ["editor", "github"]},
        "editor_template": {"type": "string"},
        "github_template": {"type": "string"}
      }
    },
    "artifacts": {
      "type": "object",
      "required": ["docfacts", "navmap"],
      "properties": {
        "docfacts": {"type": "string"},
        "navmap": {"type": "string"},
        "symbols": {"type": "string"},
        "by_file": {"type": "string"},
        "by_module": {"type": "string"},
        "schemas": {"type": "object"},
        "graphs": {"type": "array", "items": {"type": "string"}},
        "observability": {"type": "string"},
        "search": {"type": "string"},
        "agent_index": {"type": "string"}
      }
    },
    "packages": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["name", "modules"],
        "properties": {
          "name": {"type": "string"},
          "modules": {
            "type": "array",
            "items": {
              "type": "object",
              "required": ["module", "paths"],
              "properties": {
                "module": {"type": "string"},
                "paths": {
                  "type": "object",
                  "properties": {
                    "source": {"type": "string"},
                    "page": {"type": "string"},
                    "fjson": {"type": "string"}
                  }
                },
                "anchors": {
                  "type": "object",
                  "properties": {
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"}
                  }
                },
                "docfacts": {"type": "array", "items": {"type": "string"}},
                "symbol_id": {"type": "string"},
                "quality": {
                  "type": "object",
                  "properties": {
                    "mypy_status": {"type": "string"},
                    "ruff_rules": {"type": "array", "items": {"type": "string"}},
                    "doc_parity_ok": {"type": "boolean"},
                    "coverage_pct": {"type": "number"},
                    "doctest_status": {"type": "string"}
                  }
                },
                "metrics": {
                  "type": "object",
                  "properties": {
                    "complexity": {"type": "integer"},
                    "loc": {"type": "integer"},
                    "last_modified": {"type": "string", "format": "date-time"}
                  }
                },
                "graph": {
                  "type": "object",
                  "properties": {
                    "imports": {"type": "array", "items": {"type": "string"}},
                    "calls": {"type": "array", "items": {"type": "string"}}
                  }
                },
                "stability": {"type": "string"},
                "deprecated": {"type": "boolean"}
              }
            }
          }
        }
      }
    }
  }
}
```

## HTML structure (outline)
```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Agent Portal</title>
    <style>/* minimal, accessible styles */</style>
  </head>
  <body>
    <header>
      <h1>Agent Portal</h1>
      <input id="search" placeholder="Search modules, symbols" />
      <nav id="quick-links"><!-- links to DocFacts, NavMap, Schemas, Graphs --></nav>
    </header>
    <main id="content"><!-- packages/modules rendered here --></main>
    <script>/* read agent_catalog.json, render, enable search */</script>
  </body>
  </html>
```

## Risks / Trade-offs
- Schema drift: mitigate with schema validation and tests.
- Path mapping assumptions: keep logic configurable and base on existing indices.
- Link policy correctness: centralize template application and add unit tests.

## Open Questions
- Should we also generate a minimal combined search index specifically for the portal?
- Preferred placement of `agent_portal_readme.md` (under `docs/` vs generated to `site/_build/agent/`)?


