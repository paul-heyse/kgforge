# Src Hardening Phase 2 — Agent Catalog & Search APIs

Phase 2 tackles the highest volume Ruff/mypy/pyrefly violations in the agent catalog and search services (FAISS/SPLADE/BM25/fixture indexes, FastAPI app, registry adapters). This guide complements `proposal.md`, `design.md`, `tasks.md`, and the capability deltas under `specs/src-agent-search/`.

## Quick Start Checklist
1. Run `scripts/bootstrap.sh` to ensure toolchain parity (Python 3.13.9 + `uv`).
2. Read:
   - `proposal.md` — scope, impact, acceptance
   - `design.md` — architecture plan, typed APIs, schema strategy
   - `tasks.md` — ordered runbook + appendices
   - `specs/src-agent-search/spec.md` — normative requirements and scenarios
   - `openspec/changes/src-hardening-phase-1/` outputs (Phase 1 foundations)
3. Inspect canonical modules:
   - `src/kgfoundry/agent_catalog/search.py`, `session.py`, `cli.py`, `mcp.py`, `audit.py`
   - `src/search_api/app.py`, `service.py`, `faiss_adapter.py`, `bm25_index.py`, `splade_index.py`, `fixture_index.py`
   - `src/registry/*.py` for DuckDB integrations
   - `tests/search_api/**` and data fixtures
4. Capture baseline diagnostics (attach to execution note):
   ```bash
   uv run ruff check src/kgfoundry/agent_catalog src/search_api
   uv run pyrefly check src/kgfoundry/agent_catalog src/search_api
   uv run mypy src/kgfoundry/agent_catalog src/search_api
   python tools/check_imports.py
   uv run pip-audit --strict
   ```
5. Collect representative sample payloads for API/CLI responses (for schema validation tests).

6. Packaging sanity checks (local):
```bash
pip wheel .
python -m venv /tmp/v && /tmp/v/bin/pip install .[faiss,duckdb,splade]
```

## Deliverables Snapshot
- Typed FAISS/SPLADE/BM25 interfaces via Protocols and TypedDicts; removal of `Any` flows.
- Secure subprocess & SQL execution (parameterized queries, sanitized inputs) across search and registry modules.
- Search API returns RFC 9457 Problem Details with schema validation; structured logs & metrics across failure paths.
- CLI and MCP interfaces emit versioned JSON envelopes aligning with `schema/tools/cli_envelope.json` + new agent catalog schemas.
- Expanded pytest suites covering search edge cases, plugin regression, SQL injection attempts, and error paths.
- Updated documentation (schemas, API reference, migration notes) and optional extras for dependencies (FAISS, DuckDB, numpy).

## Junior playbook: implementing Phase 2

1) Typed interfaces & schemas
- Create `search_api/types.py` with `FaissIndexProtocol`, `SearchResultRecord`, etc.; annotate numpy types via `NDArray[np.float32]`.
- Add JSON Schemas under `schema/search/` (`search_response.json`, `catalog_cli.json`, `mcp_payload.json`) and matching example payloads in `docs/examples/search/`.
- Update `stubs/faiss/__init__.pyi` and `stubs/duckdb/__init__.pyi` to expose the protocol shapes used by adapters.

2) Correlation ID middleware & logging
- Implement middleware (e.g., `search_api/middleware.py`) that reads/creates `X-Correlation-ID`, stores in a `ContextVar`, and ensures `with_fields(logger, correlation_id=...)` usage in handlers/services.
- Wrap blocking FAISS/DuckDB calls in `run_in_executor` to avoid event-loop blocking; propagate correlation IDs.

3) SQL hardening
- Introduce `registry/duckdb_helpers.py` with `run_query(conn, sql, params, timeout_s)` that enforces parameterized statements, allowlists table names, and sets `statement_timeout`.
- Update `faiss_adapter.py`, `bm25_index.py`, `splade_index.py`, `fixture_index.py`, and registry modules to route all queries through helpers; add unit tests covering SQL injection attempts.

4) HTTP/CLI/MCP contracts
- Update FastAPI handlers to serialize responses via Pydantic models mirroring `search_response.json`; enable optional response validation (`SEARCH_API_VALIDATE=1`) in dev/staging.
- Refactor agent catalog CLI/MCP to emit base envelope + search payload (`schema/tools/cli_envelope.json` + `schema/search/catalog_cli.json`), including Problem Details on failure.

5) Testing & doctests
- Expand pytest suites (`tests/search_api/test_endpoints.py`, `tests/agent_catalog/test_cli.py`, etc.) with table-driven cases: happy path, invalid payload, SQL injection, timeout, missing index.
- Ensure doctests/xdoctests run by adding copy-ready examples to public docstrings and referencing schema fixtures.

6) Observability & performance
- Emit metrics (`search_requests_total`, `search_duration_seconds`, `search_errors_total`); verify stub mode still works.
- Add pytest-benchmark coverage for FAISS/BM25/SPLADE search hot paths and record baseline numbers (include in execution note).

7) Import rules, OpenAPI, packaging
- Add import-linter contracts (`search-api-no-upwards`, `agent-catalog-no-upwards`) and fix any violations.
- Lint the generated OpenAPI schema with Spectral (or equivalent) as part of CI.
- Verify wheel/install with extras (`pip wheel .`, `pip install .[faiss,duckdb,splade]`) in a clean venv.

## Acceptance Gates (run before submission)
```bash
uv run ruff format && uv run ruff check --fix src/kgfoundry/agent_catalog src/search_api src/registry
uv run pyrefly check src/kgfoundry/agent_catalog src/search_api
uv run mypy src/kgfoundry/agent_catalog src/search_api src/registry
uv run pytest -q tests/search_api tests/agent_catalog
python tools/check_imports.py
uv run pip-audit --strict
make artifacts && git diff --exit-code
openspec validate src-hardening-phase-2 --strict
```

## Questions & Support
- FAISS/SPLADE APIs → coordinate with ML platform owners.
- SQL/registry changes → sync with data platform team for DuckDB guidelines.
- API contracts → align with API guild for Problem Details envelope & schema additions.

Agents must log progress via the execution note template referenced in `tasks.md`.

