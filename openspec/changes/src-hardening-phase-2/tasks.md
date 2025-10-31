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
- Replace `Any` with typed numpy arrays (`NDArray[np.float32]`) and TypedDict models for search results.
- Convert helper functions (`_scores_from_index`, `compute_vector_scores`, etc.) to return typed mappings and raise `AgentCatalogSearchError` with Problem Details.
- When opening FAISS artifacts, resolve paths with `Path.resolve(strict=True)` and validate they stay under `repo_root`.
- Route any DuckDB access through the new helper (`registry.duckdb_helpers.run_query`) to enforce parameterization.
- Update `_SimpleFaissModule` to satisfy `FaissModuleProtocol` during tests.
- Tests: ensure `tests/agent_catalog/test_search.py` covers lexical-only, semantic, invalid embedding model, missing artifacts, SQL injection attempt.

### `kgfoundry/agent_catalog/session.py`
- Replace raw `int(status)` casts with validated conversions; raise `CatalogSessionError` with Problem Details on invalid data.
- Use Phase 1 logging helpers (`with_fields`) around subprocess lifecycle; ensure `run_tool`-style wrapper enforces timeouts/env allowlist.
- Validate JSON-RPC IDs and results strictly (`JsonObject`/`JsonValue` TypeAliases).
- Tests: `tests/agent_catalog/test_session.py` covering success path, invalid JSON response, Problem Details propagation, process failure.

### `kgfoundry/agent_catalog/cli.py`
- Add `--json` flag that emits base CLI envelope + `catalog_cli.json` payload; validate before write.
- Emit Problem Details to stderr when command fails, including correlation ID and exit code.
- Use typed dataclasses for CLI outputs and avoid `asdict` on dataclass-of-dataclass (prefer `.model_dump()` from Pydantic models once typed conversions done).
- Update `_parse_facets` to enforce allowed facet keys (package/module/kind/stability/deprecated) and surface friendly errors.
- Tests: expand `tests/agent_catalog/test_cli.py` using `CliRunner` for search/explain/failure cases.

### `kgfoundry/agent_catalog/mcp.py`
- Define typed MCP request/response models that mirror `schema/search/mcp_payload.json`.
- Validate incoming params (`k`, `facets`) and convert to `SearchOptions`; enforce `k` bounds.
- Emit Problem Details via Phase 1 helper on failure; ensure logs include correlation ID.
- Tests: new `tests/agent_catalog/test_mcp.py` verifying schema compliance and failure responses.

### `kgfoundry/agent_catalog/audit.py`
- Replace `dict[str, Any]` payloads with explicit dataclasses/TypedDicts for audit rows.
- Parameterize file accesses; log operations with correlation IDs.
- Tests: `tests/agent_catalog/test_audit.py` ensuring audit export success/failure and Problem Details on IO errors.

### `search_api/faiss_adapter.py`
- Introduce typed aliases (`FloatArray`, `IndexArray`) and ensure FAISS interactions use Protocol methods only.
- Replace f-string SQL with helper-based parameterized queries; enforce `statement_timeout` via helper.
- Normalize vectors using typed numpy operations and raise `FaissOperationError` with Problem Details.
- Tests: `tests/search_api/test_faiss_adapter.py` covering build/load/search, missing vectors, fallback to CPU, and SQL injection rejection.

### `search_api/bm25_index.py`
- Type annotate BM25 data structures (`csr_matrix`, `BM25Model` dataclass); sanitize SQL/Parquet paths.
- Use helper for DuckDB access; ensure Problem Details on missing indexes.
- Tests: `tests/search_api/test_bm25_index.py` for load/build, invalid schema, injection attempt.

### `search_api/splade_index.py`
- Guard torch/transformers imports behind optional extras; use typed dataclasses for metadata.
- Parameterize file access and ensure `Path.resolve(strict=True)` checks.
- Tests: `tests/search_api/test_splade_index.py` covering encode success/failure, missing encoder, injection attempt.

### `search_api/fixture_index.py`
- Replace dynamic dicts with typed fixture records; avoid SQL concatenation.
- Provide typed conversion functions for fixture JSON to dataclasses.
- Tests: `tests/search_api/test_fixture_index.py` verifying fixture load, invalid data, injection rejection.

### `search_api/service.py`
- Introduce typed service-layer functions returning `AgentSearchResponse`; remove unused parameters.
- Ensure lexical/vector merge uses typed dataclasses and logs metrics (`search_requests_total`, `search_duration_seconds`).
- Wrap FAISS/DuckDB calls in threadpool with timeouts; surface Problem Details on failure.
- Tests: `tests/search_api/test_service.py` covering success, timeout, FAISS load failure, SQL error, injection attempt.

### `search_api/app.py`
- Install correlation ID middleware and ensure endpoints use `with_fields`.
- Replace placeholder docstrings with real examples referencing schema fixtures; ensure doctests run.
- Configure optional response validation (`SEARCH_API_VALIDATE=1`) and log validation failures with Problem Details.
- Tests: `tests/search_api/test_endpoints.py` verifying HTTP status codes, schema validation, feature flag behavior.

### `search_api/cli.py` & MCP integration (if applicable)
- Align CLI outputs with `catalog_cli.json`; ensure Problem Details printed on failure with exit code semantics.
- Validate `--json` envelope via schema before writing.
- Tests: included with CLI suite as above.

### `vectorstore_faiss/gpu.py`
- Ensure GPU wrapper implements `FaissIndexProtocol`; remove `Any` via typed numpy arrays.
- Parameterize DuckDB interactions and guard GPU-owned resources (CUDA) with typed checks + Problem Details on unsupported hardware.
- Tests: `tests/search_api/test_faiss_gpu.py` for CPU/GPU load, fallback behavior, error logging.

### `registry/migrate.py` and new `registry/duckdb_helpers.py`
- Move direct SQL execution to helper ensuring parameterization + timeouts; helper raises typed `RegistryError` with Problem Details.
- Update CLI to output Problem Details on failure and include structured logs.
- Tests: `tests/registry/test_migrate.py` + `test_duckdb_helpers.py` covering success, failure, injection attempt.

### Tests & benchmarks
- Create/expand test modules:
  - Agent catalog: `test_search.py`, `test_cli.py`, `test_mcp.py`, `test_session.py`, `test_audit.py`
  - Search API: `test_endpoints.py`, `test_service.py`, `test_bm25_index.py`, `test_splade_index.py`, `test_fixture_index.py`, `test_faiss_adapter.py`, `test_faiss_gpu.py`
  - Registry: `test_duckdb_helpers.py`, `test_migrate.py`
- Add pytest-benchmark modules for FAISS/BM25/SPLADE operations; store baseline metrics in execution note.
- Ensure new fixtures live under `tests/fixtures/search/` mirroring JSON Schema examples.
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
- [ ] Schema meta-validation: run helper (`python -m kgfoundry_common.schema_helpers validate schema/search`) and ensure all new schemas pass.
- [ ] `make artifacts && git diff --exit-code` remains clean after updating docs/schemas/nav maps.

## Appendix C — Extras & Tooling
- [ ] Ensure `pyproject.toml` defines extras for optional deps: `faiss`, `duckdb`, `splade`.
- [ ] Add import-linter contract `agent-catalog-no-upwards` to prevent circular imports.
- [ ] Enable new benchmark suite in CI (non-blocking) or provide instructions for manual execution.
- [ ] Update `.pre-commit-config.yaml` if required (e.g., add SQL lint rule).

