## 0. Foundations & Scoping
- [ ] 0.1 Confirm scope and objectives
    - [ ] 0.1.1 Review proposal/design/spec; sign off on V1 in-scope features.
    - [ ] 0.1.2 Create example catalog fixture under `tests/fixtures/agent/` for local dev.
    - AC:
        - Scope doc updated; fixture checked in and referenced by tests.

- [ ] 0.2 Schema draft and sample
    - [ ] 0.2.1 Draft `docs/_build/schema_agent_catalog.json` (first pass) aligning with design.
    - [ ] 0.2.2 Author a small sample catalog JSON that validates against the draft.
    - AC:
        - `jsonschema` validation passes for sample; schema committed.

## 1. Agent Catalog Generator (Core)
- [ ] 1.1 Scaffolding
    - [ ] 1.1.1 Create `tools/docs/build_agent_catalog.py` entrypoint and CLI args (mode, org, repo, sha overrides).
    - [ ] 1.1.2 Implement file discovery and input checks for required artifacts.
    - AC:
        - Running with `--help` shows options; missing inputs produce actionable errors.

- [ ] 1.2 Provenance and link policy
    - [ ] 1.2.1 Resolve `repo.sha` from `git rev-parse --short HEAD`; compute `generated_at` UTC.
    - [ ] 1.2.2 Resolve link policy precedence (CLI > env > defaults); validate GitHub variables in github mode.
    - AC:
        - Policy set correctly; github mode fails fast when required env missing.

- [ ] 1.3 Module mapping
    - [ ] 1.3.1 Load `site/_build/by_module.json` / `site/_build/by_file.json` / `site/_build/symbols.json` and map module → source path, HTML, FJSON.
    - [ ] 1.3.2 Attach DocFacts symbols per module from `docs/_build/docfacts.json`.
    - AC:
        - ≥95% of modules have correct source/page/fjson paths (spot check).

## 2. Graphs, IDs, Anchors, Quality
- [ ] 2.1 Import/call graph extraction
    - [ ] 2.1.1 Parse AST for `Import`/`ImportFrom` nodes per module; normalize names; exclude stdlib by default.
    - [ ] 2.1.2 Walk AST calls; resolve function names within module; tag edges with `static` confidence.
    - AC:
        - Graph nodes/edges populated for representative modules; unit tests cover import and call detection.

- [ ] 2.2 Stable IDs and anchors
    - [ ] 2.2.1 Normalize AST (strip comments/docstrings/whitespace) and compute `symbol_id = sha256(qname + ast_text)`.
    - [ ] 2.2.2 Record anchors: `start_line`/`end_line` per symbol; implement fuzzy remap resolver.
    - AC:
        - IDs stable across runs; fuzzy remap resolves after ±20-line drift in tests.

- [ ] 2.3 Quality signals and metrics
    - [ ] 2.3.1 Aggregate mypy status, ruff rules, pydoclint parity, interrogate coverage, doctest status.
    - [ ] 2.3.2 Compute mccabe complexity, LOC; extract `last_modified` via git; map `codeowners` (if present).
    - AC:
        - Quality/metrics present for sample modules; missing inputs handled gracefully.

## 3. Semantic Index
- [ ] 3.1 Embedding and index build
    - [ ] 3.1.1 Choose embedding model (MiniLM or configured); implement encoder; generate vectors for symbols.
    - [ ] 3.1.2 Build FAISS index and sidecar mapping `symbol_id`→row; persist paths in catalog.
    - AC:
        - Index build completes; lookup by symbol_id returns vector row.

- [ ] 3.2 Hybrid search API
    - [ ] 3.2.1 Implement lexical candidate selection then vector rerank with weight α (configurable).
    - [ ] 3.2.2 Provide CLI function to query top-K with optional facets.
    - AC:
        - Internal benchmark achieves MRR@10 ≥ target; lexical fallback works when index absent.

## 4. Catalog Assembly & Validation
- [ ] 4.1 Compose catalog
    - [ ] 4.1.1 Assemble `artifacts`, per-package modules, graphs, quality, metrics, anchors, IDs.
    - [ ] 4.1.2 Validate against schema; produce human-readable errors on failure.
    - AC:
        - `docs/_build/agent_catalog.json` validates; top-level pointers resolvable.

- [ ] 4.2 Sharding
    - [ ] 4.2.1 Implement thresholds (size/modules) and write per-package shards with root index.
    - [ ] 4.2.2 Add shard loader helper used by clients.
    - AC:
        - Large catalogs produce shards; root references resolve in tests.

## 5. Typed Clients & CLI
- [ ] 5.1 Python client
    - [ ] 5.1.1 Define Pydantic models for catalog; implement shard-aware loader.
    - [ ] 5.1.2 Helpers: `list_packages`, `list_modules`, `get_module`, `find_callers`, `search`.
    - AC:
        - Type-checked client; helper tests pass.

- [ ] 5.2 TypeScript client
    - [ ] 5.2.1 Define types and functions matching Python client; publish as local package.
    - AC:
        - Unit tests pass in Node/Deno; importable by examples.

- [ ] 5.3 agentctl CLI
    - [ ] 5.3.1 Implement commands: `list-modules`, `find-callers`, `show-quality`, `search` with exit codes 0/2/3.
    - AC:
        - CLI returns expected outputs and exit codes; help docs present.

## 6. Agent Portal
- [ ] 6.1 MVP rendering
    - [ ] 6.1.1 Build `tools/docs/render_agent_portal.py`; output `site/_build/agent/index.html`.
    - [ ] 6.1.2 Render header, quick links, packages/modules, simple search (lexical), actions.
    - AC:
        - Portal loads offline with working links; snapshot test passes.

- [ ] 6.2 Enriched UX
    - [ ] 6.2.1 Add facets (package, stability, churn, coverage, parity) and breadcrumbs.
    - [ ] 6.2.2 Inline dependency graphs per module (or link to focused views).
    - [ ] 6.2.3 Show examples and metrics panels when available.
    - AC:
        - Filters limit results correctly; breadcrumbs reflect hierarchy; graphs render for sample modules.

- [ ] 6.3 Accessibility & responsiveness
    - [ ] 6.3.1 Add ARIA roles/labels; ensure keyboard navigation; responsive CSS.
    - [ ] 6.3.2 No-JS fallback for core hierarchy; disable search with guidance if JS off.
    - AC:
        - a11y audit passes; tab order functional; layout adapts to narrow screens.

- [ ] 6.4 Tutorials & feedback (local-only)
    - [ ] 6.4.1 Add tutorials/playbooks section; local feedback form writing to JSON with redaction.
    - AC:
        - Tutorial links work; feedback stored locally; PII redaction tested.

## 7. Analytics, Orchestrator, Performance
- [ ] 7.1 Analytics JSON
    - [ ] 7.1.1 Emit `docs/_build/analytics.json` with usage counters and generation stats.
    - AC:
        - File written with `generated_at`, counters, and version.

- [ ] 7.2 Orchestrator integration
    - [ ] 7.2.1 Update `tools/docs/build_artifacts.py` to run catalog then portal; stop on failure.
    - AC:
        - `make artifacts` prints agent steps; exits non-zero on validation failure.

- [ ] 7.3 Performance & caching
    - [ ] 7.3.1 Implement content-addressed caching for catalog shards and portal sections.
    - [ ] 7.3.2 Verify budgets: cold-load < 1.5s; first search < 300ms; subsequent < 150ms.
    - AC:
        - Load tests demonstrate budgets met on representative hardware.

## 8. Hosted Mode (Optional)
- [ ] 8.1 RBAC and audit
    - [ ] 8.1.1 Implement role checks (viewer/contributor/admin) behind feature flag.
    - [ ] 8.1.2 Write audit log entries for sensitive actions (e.g., feedback submissions).
    - AC:
        - Role denial tested; audit entries recorded.

## 9. Documentation
- [ ] 9.1 Agent & Human README
    - [ ] 9.1.1 Expand `docs/agent_portal_readme.md` with schema fields, link modes, client/CLI examples, and portal usage.
    - AC:
        - Docs build cleanly; examples runnable.

## 10. Testing & CI
- [ ] 10.1 Schema tests
    - [ ] 10.1.1 Positive/negative validation for catalog; sharding root index test.
- [ ] 10.2 Link tests
    - [ ] 10.2.1 Resolve a sample of source/page/fjson/editor/GitHub links.
- [ ] 10.3 Portal tests
    - [ ] 10.3.1 Snapshot and interaction tests for search, facets, breadcrumbs.
- [ ] 10.4 Client & CLI tests
    - [ ] 10.4.1 Python/TS unit tests; CLI exit codes and output contract.
- [ ] 10.5 Search quality
    - [ ] 10.5.1 Evaluate hybrid search; assert MRR@10 ≥ target on benchmark set.
- [ ] 10.6 CI integration
    - [ ] 10.6.1 Add jobs for schema validation, linkcheck, analytics write, and PR annotations summary.
    - AC:
        - CI fails on catalog/schema/link/quality regressions with clear messages.

## 11. Acceptance
- [ ] `agent_catalog.json` present and valid (or sharded with root index); portal navigable offline.
- [ ] Links open correct targets for `editor` and `github` modes.
- [ ] Clients/CLI queries return correct results; performance budgets met; quality targets satisfied.

