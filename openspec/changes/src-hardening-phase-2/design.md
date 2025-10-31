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
- Define `VectorArray = NDArray[np.float32]` (with `numpy.typing`).
- `FaissIndexProtocol` exposes `add(vectors: VectorArray, ids: NDArray[np.int64]) -> None`, `search(vectors: VectorArray, k: int) -> tuple[ndarray, ndarray]` etc.
- `SearchResult` TypedDict with fields (`id`, `score`, `metadata`).
- Use Protocols to allow test stubs/mocks without `Any`.

### SQL & Data Access
- Replace string concatenation with parameterized queries via DuckDB placeholders or binder objects.
- Validate inputs (IDs, table names) against allowlists; use `Path.resolve()` for file paths.
- Centralize SQL helpers in `registry/duckdb_helpers.py` with typed exceptions.
- Tests include SQL injection attempts verifying sanitized behavior.

### HTTP & CLI Contracts
- JSON Schemas:
  - `schema/search/search_response.json` — HTTP response structure (results, metadata, pagination, Problem Details). 
  - `schema/search/catalog_cli.json` — machine-readable CLI output.
  - `schema/search/mcp_payload.json` — MCP communication envelope.
- FastAPI routes wrap responses with validation helper (optional post-response validation in dev/staging).
- CLI uses base envelope schema from Phase 1 + search-specific payloads.

### Logging & Observability
- Each operation logs via `get_logger(__name__)`, using `with_fields` for correlation IDs, durations, command names.
- Metrics via `MetricsProvider` (counters for `search_requests_total`, histograms for `search_duration_seconds`, counters for `sql_errors_total`).
- Add OpenTelemetry spans around search requests.

### Testing Strategy
- Unit tests for adapters (FAISS/SPLADE/BM25) using deterministic fixtures + fakes.
- Integration tests for FastAPI endpoints using `TestClient`, parameterized over success/failure/invalid input.
- CLI tests using `subprocess.run` (with sanitized environment), validating JSON output against schema.
- Regression tests for SQL injection attempts (ensuring sanitized behavior).
- Doctests for high-level API usage.

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

### Performance & Benchmarks
- Add pytest-benchmark for FAISS search, BM25 scoring, SPLADE encoding. Document baseline and acceptable regression thresholds.
- Use numpy vectorization and `normalize_L2` correctly typed to avoid copies.
- Monitor metrics (p95 latency, error rates) during rollout.

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

## Dependencies & Stubs
- Provide `stubs/faiss/__init__.pyi` and `stubs/duckdb/__init__.pyi` updates with Protocol definitions.
- Ensure numpy typing via `numpy.typing` strategies (`NDArray`).
- For optional dependencies (FAISS, SPLADE), define extras in `pyproject` and guard imports with typed fallbacks.

## Rollout Plan
1. Land typed interfaces + adapters behind feature flags; run in shadow mode in staging.
2. Enable `AGENT_SEARCH_TYPED=1`, `SEARCH_API_TYPED=1` in staging; monitor metrics & logs.
3. Roll to production in phases (10%, 50%, 100%) with telemetry dashboards for errors/latency.
4. Once stable, remove legacy fallback and `--legacy-json` options in Phase 3.

## Migration / Backout
- Backout by toggling feature flags to `0`; keep legacy paths until Phase 3 cleanup.
- Schema version constants allow quick downgrade by referencing prior version if clients fail.
- Document `git revert` sequences for critical files in design appendix.

