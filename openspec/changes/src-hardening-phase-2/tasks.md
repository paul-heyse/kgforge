## 0. Orientation (AI agent pre-flight)
- [ ] 0.1 Run `scripts/bootstrap.sh`; confirm toolchain parity.
- [ ] 0.2 Cache `openspec/changes/src-hardening-phase-2/` docs + `openspec/AGENTS.md` locally.
- [ ] 0.3 Review Phase 1 outputs (`src-hardening-phase-1`) to understand shared infrastructure contracts.
- [ ] 0.4 Capture baseline diagnostics (attach logs):
  - `uv run ruff check src/kgfoundry/agent_catalog src/search_api src/registry`
  - `uv run pyrefly check src/kgfoundry/agent_catalog src/search_api`
  - `uv run mypy src/kgfoundry/agent_catalog src/search_api src/registry`
  - `python tools/check_imports.py`
  - `uv run pip-audit --strict`
- [ ] 0.5 Draft four-item design note (Summary, API sketch, Data contracts, Test plan).

## 1. Typed Interfaces & Schemas
- [ ] 1.1 Define Protocols/TypedDicts in `kgfoundry/search_api/types.py` (FAISS, SPLADE, BM25, search result models).
- [ ] 1.2 Author JSON Schemas:
  - `schema/search/search_response.json`
  - `schema/search/catalog_cli.json`
  - `schema/search/mcp_payload.json`
  - Example fixtures under `docs/examples/search/`
- [ ] 1.3 Update stubs (`stubs/faiss/__init__.pyi`, `stubs/duckdb/__init__.pyi`) to expose typed APIs.
- [ ] 1.4 Add tests verifying schema meta-validation.

## 2. FAISS & Index Adapter Refactor
- [ ] 2.1 Refactor `search_api/faiss_adapter.py`:
  - adopt Protocol types, remove `Any`
  - handle GPU availability with typed exceptions
  - parameterize SQL operations (no `str` concatenation)
- [ ] 2.2 Refactor `search_api/bm25_index.py`, `search_api/splade_index.py`, `search_api/fixture_index.py` to use typed numpy arrays, sanitized SQL.
- [ ] 2.3 Add unit tests for index building/loading/search (happy path + failure + injection attempts).
- [ ] 2.4 Add pytest benchmarks for search operations (FAISS, BM25, SPLADE) and record baseline.

## 3. Agent Catalog Service
- [ ] 3.1 Update `agent_catalog/search.py` to use typed interfaces, structured logging, metrics, Problem Details.
- [ ] 3.2 Refactor `session.py`, `cli.py`, `mcp.py`, `audit.py` to adopt typed models, sanitized inputs, and JSON envelopes.
- [ ] 3.3 Implement compatibility shim for CLI `--legacy-json`; default to typed envelope when feature flag enabled.
- [ ] 3.4 Tests: `tests/agent_catalog/test_search.py`, `test_cli.py`, `test_mcp.py` covering success, invalid input, Problem Details, schema validation.

## 4. Search API (FastAPI)
- [ ] 4.1 Update `search_api/app.py` routes with typed request/response models, Problem Details mapping using Phase 1 helpers.
- [ ] 4.2 Refactor `search_api/service.py` to remove unused args, integrate typed search results, structured logs, metrics.
- [ ] 4.3 Add middleware or response hook to validate JSON against schema in dev/staging (configurable).
- [ ] 4.4 Tests: `tests/search_api/test_endpoints.py` with table-driven cases (success, invalid body, timeout, SQL injection attempt) ensuring Problem Details responses.

## 5. Registry & DuckDB Integrations
- [ ] 5.1 Create `registry/duckdb_helpers.py` providing typed query execution with parameter binding.
- [ ] 5.2 Refactor `registry/*.py` to use helpers, handle errors via typed exceptions (`RegistryError`).
- [ ] 5.3 Add tests for migrations and helper functions, including failure paths (missing table, invalid schema) and SQL injection attempts.

## 6. Observability & Security
- [ ] 6.1 Instrument search operations with metrics (`search_requests_total`, `search_duration_seconds`, `search_errors_total`).
- [ ] 6.2 Propagate correlation IDs via logging context; ensure CLI/MCP include correlation in output envelope.
- [ ] 6.3 Enforce request timeouts (HTTP client calls, DuckDB operations) and log slow queries.
- [ ] 6.4 Validate and sanitize all external inputs (query text, IDs, file paths) before use.
- [ ] 6.5 Run `pip-audit --strict`; address any vulnerabilities introduced by optional dependencies.

## 7. Documentation & Rollout
- [ ] 7.1 Update docs: search API reference, CLI usage, schema docs (link to new JSON Schemas), metrics descriptions.
- [ ] 7.2 Document feature flags `AGENT_SEARCH_TYPED`, `SEARCH_API_TYPED` with rollout steps.
- [ ] 7.3 Update changelog with migration notes (CLI JSON envelope, Problem Details fields).
- [ ] 7.4 Prepare telemetry dashboards for rollout (latency, error rate, SQL errors).

## 8. Validation & Sign-off
- [ ] 8.1 Run acceptance gates (see README) and attach outputs to execution note.
- [ ] 8.2 Verify zero Ruff/pyrefly/mypy diagnostics remain for target modules.
- [ ] 8.3 Ensure tests (unit, integration, doctest, benchmark) pass; review coverage for new code paths.
- [ ] 8.4 Execute rollout checklist: enable flags in staging, monitor metrics, flip default when stable.
- [ ] 8.5 Archive execution note summarizing outcomes, telemetry results, and follow-up tasks for Phase 3.

## Appendix A — Module Migration Guide

### `kgfoundry/agent_catalog/search.py`
- Replace `Any` with typed numpy arrays and `VectorSearchResult` models.
- Use `with_fields(logger, correlation_id=..., command="catalog-search")` to log.
- Parameterize SQL via DuckDB helper; avoid concatenation.
- Return typed `AgentSearchResponse`; on failure, raise `AgentCatalogSearchError` (subclass of `KgFoundryError`).

### `kgfoundry/search_api/faiss_adapter.py`
- Define `FloatArray = NDArray[np.float32]`.
- Ensure `normalize_L2` returns typed arrays; annotate accordingly.
- Replace `try/except Exception: pass` with targeted errors (`FaissOperationError`) + logging.
- Write tests verifying `add/search` operations and error handling.

### `search_api/app.py`
- Use FastAPI Pydantic models mirroring `search_response` schema.
- Map exceptions via Phase 1 `problem_from_exception` helper.
- Validate outgoing responses against schema when `SEARCH_API_VALIDATE=1` (dev/staging).

### CLI & MCP
- CLI `--json` outputs base envelope + search payload; validate before printing.
- MCP messages conform to `schema/search/mcp_payload.json`.
- Add tests invoking CLI command with `click.testing.CliRunner` and verifying output.

### Security Defaults
- SQL timeouts default 10 s; configurable via settings.
- All HTTP client calls include `timeout=5.0` (configurable) and structured error handling.
- File paths resolved via `Path.resolve(strict=True)` and limited to configured directories.

## Appendix B — CI & Packaging Checklist
- [ ] Add import-linter contracts (`search-api-no-upwards`, `agent-catalog-no-upwards`) to `importlinter.cfg`; verify they pass.
- [ ] Enable doctests/xdoctests in CI (already in `pytest.ini`); ensure examples execute.
- [ ] Packaging: `pip wheel .` succeeds; `pip install .[faiss,duckdb,splade]` in a clean venv succeeds.
- [ ] OpenAPI linter (Spectral) passes against generated API spec.
- [ ] Security: `uv run pip-audit --strict` passes; SQL Bandit rule S608 cleared.
- [ ] Performance: run pytest-benchmark for FAISS/BM25/SPLADE and record baseline numbers.

## Appendix B — CI & Extras
- [ ] Ensure `pyproject.toml` defines extras for optional deps: `faiss`, `duckdb`, `splade`.
- [ ] Add import-linter contract `agent-catalog-no-upwards` to prevent circular imports.
- [ ] Enable new benchmark suite in CI (non-blocking) or provide instructions for manual execution.
- [ ] Update `.pre-commit-config.yaml` if required (e.g., add SQL lint rule).

