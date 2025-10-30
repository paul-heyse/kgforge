## 0. Foundations & Scoping
- [x] 0.1 Confirm scope and objectives
    - [x] 0.1.1 Review proposal/design/spec; sign off on V1 in-scope features.
    - [x] 0.1.2 Create example catalog fixture under `tests/fixtures/agent/` for local dev.
    - AC:
        - Scope doc updated; fixture checked in and referenced by tests.

- [x] 0.2 Schema draft and sample
    - [x] 0.2.1 Draft `docs/_build/schema_agent_catalog.json` (first pass) aligning with design.
    - [x] 0.2.2 Author a small sample catalog JSON that validates against the draft.
    - [x] 0.2.3 Encode ordering semantics: use arrays for any ordered data (e.g., `anchors.remap_order`, `exemplars`).
    - AC:
        - `jsonschema` validation passes for sample; schema committed.

## 1. Agent Catalog Generator (Core)
- [x] 1.1 Scaffolding
    - [x] 1.1.1 Create `tools/docs/build_agent_catalog.py` entrypoint and CLI args (mode, org, repo, sha overrides).
    - [x] 1.1.2 Implement file discovery and input checks for required artifacts.
    - AC:
        - Running with `--help` shows options; missing inputs produce actionable errors.

- [x] 1.2 Provenance and link policy
    - [x] 1.2.1 Resolve `repo.sha` from `git rev-parse --short HEAD`; compute `generated_at` UTC.
    - [x] 1.2.2 Resolve link policy precedence (CLI > env > defaults); validate GitHub variables in github mode.
    - AC:
        - Policy set correctly; github mode fails fast when required env missing.

- [x] 1.3 Module mapping
    - [x] 1.3.1 Load `site/_build/by_module.json` / `site/_build/by_file.json` / `site/_build/symbols.json` and map module → source path, HTML, FJSON.
    - [x] 1.3.2 Attach DocFacts symbols per module from `docs/_build/docfacts.json`.
    - AC:
        - ≥95% of modules have correct source/page/fjson paths (spot check).

## 2. Graphs, IDs, Anchors, Quality
- [x] 2.1 Import/call graph extraction
    - [x] 2.1.1 Parse AST for `Import`/`ImportFrom` nodes per module; normalize names; exclude stdlib by default.
    - [x] 2.1.2 Walk AST calls; resolve function names within module; tag edges with `static` confidence.
    - AC:
        - Graph nodes/edges populated for representative modules; unit tests cover import and call detection.

- [x] 2.2 Stable IDs and anchors
    - [x] 2.2.1 Normalize AST (strip comments/docstrings/whitespace) and compute `symbol_id = sha256(qname + ast_text)`.
    - [x] 2.2.2 Record anchors: `start_line`/`end_line` per symbol; compute `anchors.cst_fingerprint` (token trigrams).
    - [x] 2.2.3 Implement `anchors.remap_order` (array) and fuzzy remap resolver: `[symbol_id, cst_fingerprint, name_arity, nearest_text]`.
    - AC:
        - IDs stable across runs; remap survives ±20-line drift and docstring/formatting churn.

- [x] 2.3 Quality signals and metrics
    - [x] 2.3.1 Aggregate mypy status, ruff rules, pydoclint parity, interrogate coverage, doctest status.
    - [x] 2.3.2 Compute mccabe complexity, LOC; extract `last_modified` via git; map `codeowners` (if present).
    - AC:
        - Quality/metrics present for sample modules; missing inputs handled gracefully.

- [x] 2.4 Agent Hints, Change Impact, Exemplars
    - [x] 2.4.1 Populate `agent_hints` per module/symbol: `intent_tags`, `safe_ops`, `tests_to_run`, `perf_budgets`, `breaking_change_notes`.
    - [x] 2.4.2 Populate `change_impact`: `callers`, `callees`, `tests`, `codeowners`, `churn_last_n`; optionally write as a separate shard for large repos.
    - [x] 2.4.3 Populate `exemplars` per symbol with `{title, language, snippet, counter_example?, negative_prompts?, context_notes?}`.
    - AC:
        - Fields present in catalog; portal renders them later; clients can query them.

## 3. Semantic Index & SQLite Catalog
- [x] 3.1 Embedding and index build
    - [x] 3.1.1 Choose embedding model (MiniLM or configured); implement encoder; generate vectors for symbols.
    - [x] 3.1.2 Build FAISS index and sidecar mapping `symbol_id`→row; persist paths in catalog.
    - AC:
        - Index build completes; lookup by symbol_id returns vector row.

- [x] 3.2 Hybrid search API
    - [x] 3.2.1 Implement lexical candidate selection then vector rerank with weight α (configurable).
    - [x] 3.2.2 Provide CLI function to query top-K with optional facets.
    - AC:
        - Internal benchmark achieves MRR@10 ≥ target; lexical fallback works when index absent.

- [x] 3.3 Optional SQLite catalog
    - [x] 3.3.1 Define DDL for `catalog.sqlite` (read-only): `symbols`, `calls`, `anchors`, `fts`, `ranking_features`.
    - [x] 3.3.2 Export SQLite during generation; loader tries SQLite first, falls back to JSON shards.
    - AC:
        - Loader uses SQLite when present; fallback works.

## 4. Catalog Assembly & Validation
- [x] 4.1 Compose catalog
    - [x] 4.1.1 Assemble `artifacts`, per-package modules, graphs, quality, metrics, anchors (with `cst_fingerprint`/`remap_order`), IDs, hints/impact/exemplars.
    - [x] 4.1.2 Validate against schema; produce human-readable errors on failure.
    - AC:
        - `docs/_build/agent_catalog.json` validates; top-level pointers resolvable.

- [x] 4.2 Sharding
    - [x] 4.2.1 Implement thresholds (size/modules) and write per-package shards with root index.
    - [x] 4.2.2 Add shard loader helper used by clients.
    - AC:
        - Large catalogs produce shards; root references resolve in tests.

## 5. Typed Clients, CLI, and Stdio Session Server
- [x] 5.1 Python client
    - [x] 5.1.1 Define Pydantic models for catalog; implement shard-aware loader.
    - [x] 5.1.2 Helpers: `list_packages`, `list_modules`, `get_module`, `find_callers`, `search`.
    - AC:
        - Type-checked client; helper tests pass.

- [x] 5.2 TypeScript client
    - [x] 5.2.1 Define types and functions matching Python client; publish as local package.
    - AC:
        - Unit tests pass in Node/Deno; importable by examples.

- [x] 5.3 `catalogctl` CLI (serverless)
    - [x] 5.3.1 Implement commands: `capabilities`, `symbol`, `find-callers`, `find-callees`, `search`, `open-anchor`, `change-impact`, `suggest-tests`, `explain-ranking` with exit codes 0/2/3 and JSON/NDJSON output.
    - AC:
        - CLI returns expected outputs and exit codes; help docs present.

- [x] 5.4 Editor-activated stdio session server (no daemon)
    - [x] 5.4.1 Implement MCP/JSON-RPC over stdio process `catalogctl-mcp` (or equivalent) spawned by editor/agent; handshake `capabilities`.
    - [x] 5.4.2 Maintain warm in-memory cache during session; exit on stdin EOF or explicit shutdown.
    - AC:
        - Session queries are faster due to warm cache; process exits cleanly; no TCP ports opened.

- [x] 5.5 OpenAPI & SDKs
    - [x] 5.5.1 Emit OpenAPI 3.2 doc for callable procedures `docs/_build/agent_api_openapi.json`.
    - [x] 5.5.2 Generate Python/TS SDKs; errors follow RFC9457 Problem Details.
    - AC:
        - OpenAPI validates; SDKs compile; Problem Details contract verified in tests.

## 6. Agent Portal
- [x] 6.1 MVP rendering
    - [x] 6.1.1 Build `tools/docs/render_agent_portal.py`; output `site/_build/agent/index.html`.
    - [x] 6.1.2 Render header, quick links, packages/modules, simple search (lexical), actions.
    - AC:
        - Portal loads offline with working links; snapshot test passes.

- [x] 6.2 Enriched UX
    - [x] 6.2.1 Add facets (package, stability, churn, coverage, parity) and breadcrumbs.
    - [x] 6.2.2 Inline dependency graphs per module (or link to focused views).
    - [x] 6.2.3 Show examples and metrics panels when available; render `agent_hints`; add "Edit here" impact card and "Insert exemplar" buttons.
    - AC:
        - Filters limit results correctly; breadcrumbs reflect hierarchy; graphs render; hints/impact/exemplars visible and actionable.

- [x] 6.3 Accessibility & responsiveness
    - [x] 6.3.1 Add ARIA roles/labels; ensure keyboard navigation; responsive CSS.
    - [x] 6.3.2 No-JS fallback for core hierarchy; disable search with guidance if JS off.
    - AC:
        - a11y audit passes; tab order functional; layout adapts to narrow screens.

- [x] 6.4 Tutorials & feedback (local-only)
    - [x] 6.4.1 Add tutorials/playbooks section; local feedback form writing to JSON with redaction.
    - AC:
        - Tutorial links work; feedback stored locally; PII redaction tested.

## 7. Analytics, Orchestrator, Performance
- [ ] 7.1 Analytics JSON
    - [ ] 7.1.1 Emit `docs/_build/analytics.json` with usage counters and generation stats.
    - AC:
        - File written with `generated_at`, counters, and version.

- [x] 7.2 Orchestrator integration
    - [x] 7.2.1 Update `tools/docs/build_artifacts.py` to run catalog then portal; stop on failure.
    - AC:
        - `make artifacts` prints agent steps; exits non-zero on validation failure.

- [ ] 7.3 Performance & caching
    - [ ] 7.3.1 Implement content-addressed caching for catalog shards and portal sections.
    - [ ] 7.3.2 Verify budgets: cold-load < 1.5s; first search < 300ms; subsequent < 150ms.
    - AC:
        - Load tests demonstrate budgets met on representative hardware.

- [ ] 7.4 Docker integration (serverless)
    - [ ] 7.4.1 Multi-stage Dockerfile builds wheel with `uv build`; install into slim runtime.
    - [ ] 7.4.2 COPY `catalog_artifacts/` → `/srv/catalog`; set `CATALOG_ROOT=/srv/catalog`.
    - [ ] 7.4.3 Ensure no daemons/ports; editor/agent calls `catalogctl` or spawns stdio process on demand.
    - AC:
        - Container executes CLI/stdio flows with no network exposure.

## 8. Hosted Mode (Optional)
- [ ] 8.1 RBAC and audit
    - [ ] 8.1.1 Implement role checks (viewer/contributor/admin) behind feature flag.
    - [ ] 8.1.2 Write audit log entries for sensitive actions (e.g., feedback submissions).
    - AC:
        - Role denial tested; audit entries recorded.

## 9. Documentation
- [ ] 9.1 Agent & Human README
    - [ ] 9.1.1 Expand `docs/agent_portal_readme.md` with schema fields, link modes, client/CLI examples, portal usage.
    - [ ] 9.1.2 Add CLI reference for `catalogctl` and stdio/MCP usage examples.
    - AC:
        - Docs build cleanly; examples runnable.

## 10. Testing & CI
- [ ] 10.1 Schema tests
    - [ ] 10.1.1 Positive/negative validation for catalog; sharding root index test.
- [ ] 10.2 Link tests
    - [ ] 10.2.1 Resolve a sample of source/page/fjson/editor/GitHub links.
- [ ] 10.3 Portal tests
    - [ ] 10.3.1 Snapshot and interaction tests for search, facets, breadcrumbs, hints/impact/exemplars.
- [x] 10.4 Client & CLI tests
    - [x] 10.4.1 Python/TS client unit tests; CLI exit codes and output contract; `catalogctl` parity with stdio methods.
- [ ] 10.5 Search quality
    - [ ] 10.5.1 Evaluate hybrid search; assert MRR@10 ≥ target on benchmark set.
- [ ] 10.6 CI integration
    - [ ] 10.6.1 Add jobs for schema validation, linkcheck, analytics write, PR annotations.
- [ ] 10.7 OpenAPI/SDK validation
    - [ ] 10.7.1 Validate OpenAPI 3.2; compile generated SDKs; Problem Details tests.
- [ ] 10.8 Stdio server tests
    - [ ] 10.8.1 Spawn stdio process; handshake `capabilities`; run queries; ensure clean shutdown.
- [ ] 10.9 Ordering semantics tests
    - [ ] 10.9.1 Verify arrays used for ordered sequences (e.g., `remap_order`); ensure consumers do not rely on object key order.
- [ ] 10.10 CST remap tests
    - [ ] 10.10.1 Verify anchor remap order and CST fingerprint fallback under docstring/formatting churn.
    - AC:
        - CI fails on catalog/schema/link/quality/API regressions with clear messages.

## 11. Acceptance
- [ ] `agent_catalog.json` present and valid (or sharded with root index); portal navigable offline.
- [ ] Links open correct targets for `editor` and `github` modes.
- [ ] Clients/CLI and stdio server work; performance budgets met; quality targets satisfied; OpenAPI/SDK validated.

