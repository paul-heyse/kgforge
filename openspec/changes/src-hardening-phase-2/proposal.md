## Why
Agent catalog search and the search API host the bulk of remaining Ruff/mypy/pyrefly errors: FAISS integrations return `Any`, SQL is constructed via string concatenation, CLI/MCP surfaces lack schema validation, and error handling relies on blind `except Exception` blocks. These gaps threaten correctness (possible SQL injection), hinder observability, and block strict type adoption. Phase 2 applies the Phase 1 infrastructure to the agent catalog + search stack, delivering typed interfaces, secure data access, and schema-backed APIs.

## What Changes
- [x] **ADDED**: Typed Protocols and TypedDicts for FAISS, SPLADE, BM25 contexts, agent catalog search results, and MCP payloads.
- [x] **ADDED**: JSON Schemas for agent search responses, CLI/MCP envelopes, and registry operations under `schema/search/`.
- [x] **MODIFIED**: `kgfoundry/agent_catalog/*` to adopt typed models, parameterized queries, structured logging, metrics, and Problem Details emission.
- [x] **MODIFIED**: `search_api/*` (FastAPI app, service layer, adapters) to eliminate `Any`, secure SQL, enforce timeouts, and validate HTTP responses against schemas.
- [x] **MODIFIED**: `registry/*` DuckDB integrations to use typed wrappers with sanitized inputs and typed exceptions.
- [ ] **REMOVED**: Legacy search CLI/JSON formats and blind exception handlers once rollout completes (tracked in tasks).
- [ ] **RENAMED**: _None._
- [ ] **BREAKING**: Public HTTP API remains functionally compatible; JSON responses gain versioned metadata and Problem Details envelope (documented in changelog).

## Impact
- **Affected specs:** `src-agent-search` (see `specs/src-agent-search/spec.md`).
- **Affected code paths:**
  - `src/kgfoundry/agent_catalog/search.py`, `session.py`, `cli.py`, `mcp.py`, `audit.py`
  - `src/search_api/app.py`, `service.py`, `faiss_adapter.py`, `bm25_index.py`, `splade_index.py`, `fixture_index.py`
  - `src/registry/*.py`
  - New schemas under `schema/search/`
  - Tests under `tests/search_api/**`, `tests/agent_catalog/**`
- **Rollout:** Feature flags `AGENT_SEARCH_TYPED=0|1` and `SEARCH_API_TYPED=0|1` guard new pathways. CLI gains `--legacy-json` toggle for one release. Stage with telemetry dashboards (latency, error rate, SQL errors) before flipping default.
- **Risks:** Potential performance regressions (mitigated with benchmarks), schema drift (controlled via tests + fixtures), third-party plugin breakage (compatibility shim provided).

## Acceptance
- Quality gates (Ruff, pyrefly, mypy, pytest, doctests/xdoctests, import linter, pip-audit, artifacts, openspec validate) pass for affected modules.
- All SQL uses parameterized queries; static analysis (S608) reports zero issues.
- HTTP responses validate against new schemas; Problem Details returned for failures with schema compliance examples.
- CLI/MCP outputs match envelopes and include correlation IDs + metrics updates.
- No blind `except Exception`; exceptions preserve causes and map to typed taxonomy.
- Benchmarks recorded for search operations (FAISS/BM25/SPLADE) with budgets documented; regressions noted.
- Feature flags documented; rollout plan executed with telemetry checkpoints.

