## Context
Agent catalog search and the search API integrate FAISS, SPLADE, BM25, DuckDB, and FastAPI endpoints. The current implementation relies on dynamic dicts, handwritten SQL strings, untyped numpy arrays, and blind exception handling. Phase 1 established shared infrastructure (Problem Details, structured logging, observability, typed settings). Phase 2 applies those foundations to agent catalog + search modules, eliminating `Any`, securing data access, and enforcing schema-backed APIs.

## Goals / Non-Goals
- **Goals**
  - Introduce typed data models and Protocol interfaces for search pipelines (FAISS, SPLADE, BM25, fixture indexes).
  - Secure all SQL interactions with parameterized queries and input sanitization.
  - Ensure HTTP/CLI/MCP responses validate against JSON Schemas and emit Problem Details on failure.
  - Provide structured logging + metrics (latency, error counters) using Phase 1 adapters.
  - Expand tests to cover edge cases (invalid inputs, SQL injection attempts, plugin failures) and ensure doctests run.
- **Non-Goals**
  - Redesign ranking algorithms or vector store logic (behavior remains consistent).
  - Replace FAISS or DuckDB dependencies (treat as externals with typed facades).
  - Modify Phase 1 shared modules beyond necessary wiring.

## Architecture Overview

| Layer | Responsibility | Deliverables |
| --- | --- | --- |
| Domain models | Typed representations of indexes, search results, configuration | `TypedDict`/dataclasses for `VectorSearchResult`, `CatalogEntry`, etc. |
| Interfaces | Protocols describing FAISS/SPLADE/BM25 operations | `FaissIndexProtocol`, `SpladeEncoderProtocol`, `BM25IndexProtocol` |
| Adapters | Implement Protocols via third-party libs with parameter checks | `faiss_adapter.py`, `bm25_index.py`, `splade_index.py` refactors |
| Services | Agent catalog + API business logic using typed models | `agent_catalog/search.py`, `search_api/service.py` |
| HTTP/CLI | FastAPI app, CLI commands, MCP surfaces with schema validation | Response schemas, Problem Details emission, CLI envelopes |

### Typed Interfaces
- Define `VectorArray = NDArray[np.float32]` using `numpy.typing`; expose aliases (`IndexArray = NDArray[np.int64]`).
- `FaissIndexProtocol` exposes `add`, `search`, `reset`, `is_trained`, and optional `add_with_ids`; adapters must use attribute checks with typed fallbacks.
- Additional public models:
  - `AgentSearchQuery` dataclass (`query: str`, `k: int`, `facets: Mapping[str, str]`, `explain: bool`).
  - `VectorSearchResult` TypedDict (`symbol_id`, `score`, `lexical_score`, `vector_score`, `package`, `module`, `qname`, `kind`, `anchor`, `metadata`).
  - `AgentSearchResponse` TypedDict (`results: list[VectorSearchResult]`, `total: int`, `took_ms: int`, `metadata: Mapping[str, JsonValue]`).
  - CLI/MCP envelope payloads referencing `schema/search/catalog_cli.json` and `schema/search/mcp_payload.json`.
- Publish these types via `search_api/types.py` and `agent_catalog/models.py` with explicit `__all__`; update modules to import from these definitions rather than ad-hoc dicts.
- Provide Protocols for SPLADE encoder (`encode(texts: Sequence[str]) -> NDArray[np.float32]`), BM25 index, and registry helpers (typed DuckDB connection wrapper).

### SQL & Data Access
- Replace string concatenation with parameterized queries via DuckDB placeholders or binder objects.
- Validate inputs (IDs, table names) against allowlists; use `Path.resolve()` for file paths.
- Centralize SQL helpers in `registry/duckdb_helpers.py` with typed exceptions.
- Tests include SQL injection attempts verifying sanitized behavior.

#### Typed helper sketch
```python
class SqlError(SearchError): ...

def run_query(conn: duckdb.DuckDBPyConnection, sql: str, params: Mapping[str, object], *, timeout_s: float) -> duckdb.DuckDBPyRelation:
    if "?" not in sql and ":" not in sql:
        raise SqlInjectionAttemptError("Query must be parameterized")
    conn.execute(f"SET statement_timeout={int(timeout_s * 1000)}")
    return conn.execute(sql, params)
```
- Provide convenience wrappers (`fetch_all`, `fetch_one`, `stream_parquet`) that log slow queries (`duration_ms` > threshold) and attach correlation IDs.
- All adapters (FAISS, BM25, SPLADE, fixture index, registry migrate) MUST route queries through these helpers; direct string interpolation is prohibited.
- Maintain allowlists for schema/table names; reject user-provided identifiers that fall outside configured sets.

### HTTP & CLI Contracts
- JSON Schemas:
  - `schema/search/search_response.json` — HTTP response structure (results, metadata, pagination, Problem Details). 
  - `schema/search/catalog_cli.json` — machine-readable CLI output.
  - `schema/search/mcp_payload.json` — MCP communication envelope.
- FastAPI routes wrap responses with validation helper (optional post-response validation in dev/staging).
- CLI uses base envelope schema from Phase 1 + search-specific payloads.

#### OpenAPI & schema validation strategy
- FastAPI models mirror `schema/search/search_response.json`; enable an optional dev/staging response validator.
- Run Spectral (or equivalent) as a CI OpenAPI linter against the generated OpenAPI.
- Generate schemas from Pydantic models and export under `schema/openapi/search_api.v1.json`; treat linter warnings as blockers.

### Logging & Observability
- Each operation logs via `get_logger(__name__)`, using `with_fields` for correlation IDs, durations, command names.
- Metrics via `MetricsProvider` (counters for `search_requests_total`, histograms for `search_duration_seconds`, counters for `sql_errors_total`).
- Add OpenTelemetry spans around search requests.

#### Correlation ID middleware (FastAPI)
Use middleware to inject `X-Correlation-ID` into a `ContextVar` and ensure all logs include it. For async endpoints, wrap blocking FAISS/SQL calls using a threadpool and respect cancellations.

#### Minimal structured fields
Log at minimum: `correlation_id`, `operation`, `status`, `duration_ms`, `command`, and search‑specific fields (`k`, `alpha`, `index_name`).
- CLI/MCP logs should include `command`, `result_count`, `error_type`; metrics should tag `backend` (`faiss`, `bm25`, `splade`).
- Increment `search_errors_total{error_type=...}` alongside Problem Details emission.

### Testing Strategy
- Unit tests for adapters (FAISS/SPLADE/BM25) using deterministic fixtures + fakes.
- Integration tests for FastAPI endpoints using `TestClient`, parameterized over success/failure/invalid input.
- CLI tests using `subprocess.run` (with sanitized environment), validating JSON output against schema.
- Regression tests for SQL injection attempts (ensuring sanitized behavior).
- Doctests for high-level API usage.

#### Table-driven coverage (examples)
- Endpoints: valid/invalid input, SQL injection attempts, timeout, missing index, schema mismatch.
- Adapters: FAISS add/search/load, SPLADE encode failure, BM25 corners.
- CLI/MCP: `--json` envelope validation with Problem Details on error.
- Session client: JSON-RPC success/error, invalid payload, process timeout.
- Registry helpers: successful migration, failed statement, injection attempt, timeout.
- Benchmarks: measure FAISS/BM25/SPLADE operations; assert results logged in execution note.

### Typed API Sketches
```python
class VectorSearchResult(TypedDict):
    id: str
    score: float
    metadata: Mapping[str, JsonValue]

class AgentSearchResponse(TypedDict):
    results: list[VectorSearchResult]
    total: int
    took_ms: int
    metadata: Mapping[str, JsonValue]

class FaissIndexProtocol(Protocol):
    dim: int
    def is_trained(self) -> bool: ...
    def add(self, x: NDArray[np.float32], ids: NDArray[np.int64]) -> None: ...
    def search(self, x: NDArray[np.float32], k: int) -> tuple[NDArray[np.int64], NDArray[np.float32]]: ...

def search_agents(query: AgentSearchQuery, *, index: FaissIndexProtocol, logger: Logger) -> AgentSearchResponse: ...
```

### Security Considerations
- Parameterize all SQL; reject table name overrides from user input.
- Validate request payloads against schemas; return Problem Details for invalid data.
- Enforce request timeouts (HTTP client + DuckDB operations) and log warnings on slow queries.
- Avoid storing secrets in logs; redact sensitive fields.

#### Supply-chain and input safety
- Use `yaml.safe_load`; reject naive datetime/`os.path` in new code; validate user inputs (lengths, enums, allowlists).

### Performance & Benchmarks
- Add pytest-benchmark for FAISS search, BM25 scoring, SPLADE encoding. Document baseline and acceptable regression thresholds.
- Use numpy vectorization and `normalize_L2` correctly typed to avoid copies.
- Monitor metrics (p95 latency, error rates) during rollout.

#### Bench harness
Use pytest-benchmark for FAISS/BM25/SPLADE hotspots; document baseline locally and track regressions in CI (non-gating initially).
- Example command: `pytest tests/benchmarks -k "faiss or bm25 or splade" --benchmark-json benchmarks/search.json`.
- Attach benchmark JSON + interpretation to execution note; flag regressions >10% for follow-up.

## Detailed Implementation Plan

| Step | Description | Acceptance |
| --- | --- | --- |
| 1 | Author typed Protocols, TypedDicts, and schema files | mypy passes on definitions; schemas validate |
| 2 | Refactor FAISS adapter and index builders to use Protocols + sanitized SQL | No `Any`; S608 cleared; tests cover add/search/save/load |
| 3 | Update SPLADE/BM25 modules with typed numpy arrays and safe serialization | Remove `Any`; parameterized I/O; tests for encode/search |
| 4 | Refactor agent catalog search service with typed models, Problem Details, metrics | CLI/MCP outputs validated |
| 5 | Harden FastAPI app/service: typed request/response models, Problem Details, schema checks | HTTP tests verifying success/failure |
| 6 | Update registry helpers for typed DuckDB access | Parameterized queries; typed exceptions; tests |
| 7 | Expand tests (unit + integration + doctest) and benchmarks | pytest suite green; doctests/xdoctests run |
| 8 | Documentation + changelog updates; feature flag guidance | `make artifacts` clean |

### Layering & Import Rules
Add import-linter contracts:
```ini
[contract:search-api-no-upwards]
type = forbidden
name = search_api must not import higher app layers
source_modules = src.search_api
forbidden_modules =
    src.registry
    src.orchestration

[contract:agent-catalog-no-upwards]
type = forbidden
name = agent_catalog must not import higher app layers
source_modules = src.kgfoundry.agent_catalog
forbidden_modules =
    src.search_api
```

### Dependencies & Stubs
- Provide `stubs/faiss/__init__.pyi` and `stubs/duckdb/__init__.pyi` updates with Protocol definitions.
- Ensure numpy typing via `numpy.typing` strategies (`NDArray`).
- For optional dependencies (FAISS, SPLADE), define extras in `pyproject` and guard imports with typed fallbacks.

### Packaging & Extras
Declare extras in `pyproject.toml`:
```toml
[project.optional-dependencies]
faiss = ["faiss-cpu>=1.7"]
duckdb = ["duckdb>=1.0"]
splade = ["torch>=2.2", "transformers>=4.44"]
```
CI commands:
```bash
pip wheel .
python -m venv /tmp/v && /tmp/v/bin/pip install .[faiss,duckdb,splade]
```

## Rollout Plan
1. Land typed interfaces + adapters behind feature flags; run in shadow mode in staging while comparing results against legacy outputs.
2. Enable `AGENT_SEARCH_TYPED=1`, `SEARCH_API_TYPED=1` in staging; monitor dashboards (latency, error rate, SQL errors, GPU utilization) and log correlation IDs.
3. Roll to production incrementally (10%, 50%, 100%), capturing metrics + sample payloads at each step and verifying schema compliance.
4. Confirm CLI/MCP consumers accept new envelopes; once telemetry is stable for ≥7 days, remove legacy fallbacks and deprecate `--legacy-json` flag in Phase 3.
5. Document rollout timeline, feature flag toggles, and rollback steps in execution note and changelog.

## Migration / Backout
- Backout by toggling feature flags to `0`; keep legacy paths until Phase 3 cleanup.
- Schema version constants allow quick downgrade by referencing prior version if clients fail.
- Document `git revert` sequences for critical files in design appendix.

